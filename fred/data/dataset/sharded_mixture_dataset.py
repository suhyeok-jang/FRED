from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import ctypes
import gc
import time

import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from fred.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from fred.data.dataset.robot_human_mix import interleave_pattern, robot_human_split_counts


def merge_statistics(
    per_dataset_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    """Merge {joint_group: {stat: values}} dicts of several datasets by sampling weight: mean/std are the weighted
    mixture moments (Var = Σ w_i (σ_i² + μ_i²) − (Σ w_i μ_i)²), min/q01 the minimum and max/q99 the maximum
    over datasets. Values are (D,) vectors or (horizon, D) tables (relative_action)."""
    dataset_sampling_weights = np.array(dataset_sampling_weights)
    normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    overall_stats: dict[str, dict[str, list[float]]] = {}

    for modality in per_dataset_stats[0]:
        # dimensionality from the first dataset (assumed consistent)
        dim = np.asarray(per_dataset_stats[0][modality]["mean"]).shape

        weighted_means = np.zeros(dim)
        weighted_squares = np.zeros(dim)

        min_list = []
        max_list = []
        q01_list = []
        q99_list = []

        for dataset_idx, dataset_stats in enumerate(per_dataset_stats):
            w_i = normalized_weights[dataset_idx]
            stats = dataset_stats[modality]
            means = np.array(stats["mean"])
            stds = np.array(stats["std"])

            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)

            min_list.append(stats["min"])
            max_list.append(stats["max"])
            q01_list.append(stats["q01"])
            q99_list.append(stats["q99"])

        overall_mean = weighted_means.tolist()
        overall_variance = weighted_squares - weighted_means**2
        overall_std = np.sqrt(overall_variance).tolist()

        overall_min = np.min(np.array(min_list), axis=0).tolist()
        overall_max = np.max(np.array(max_list), axis=0).tolist()

        # conservative bounds: min of the q01s, max of the q99s
        q01_array = np.array(q01_list)
        q99_array = np.array(q99_list)
        weighted_q01 = np.min(q01_array, axis=0).tolist()
        weighted_q99 = np.max(q99_array, axis=0).tolist()

        overall_stats[modality] = {
            "min": overall_min,
            "max": overall_max,
            "mean": overall_mean,
            "std": overall_std,
            "q01": weighted_q01,
            "q99": weighted_q99,
        }

    return overall_stats


class ShardedMixtureDataset(IterableDataset):
    """Infinite IterableDataset over several ShardedSingleStepDatasets: per epoch it draws num_shards_per_epoch
    (dataset, shard) pairs weighted by ``weights`` (normalized by average shard size), splits them across
    ranks x workers, prefetches the next shard in a background thread and yields the current shard's steps in
    random order. Also merges the per-dataset statistics and installs them on the processor.

    Example:
        >>> mixture = ShardedMixtureDataset(
        ...     datasets=[dataset1, dataset2, dataset3],
        ...     weights=[0.5, 0.3, 0.2],
        ...     processor=my_processor,
        ...     num_shards_per_epoch=10000,
        ... )
        >>> for batch in mixture:
        ...     # batch contains processed data from mixed datasets
        ...     pass
    """

    def __init__(
        self,
        datasets: list[ShardedSingleStepDataset],
        weights: list[float],
        processor,
        seed: int = 42,
        num_shards_per_epoch: int = int(1e5),
        shard_prefetch_timeout_s: int = 1800,
        configure_processor: bool = True,   # set the merged statistics on the processor (sub-mixtures pass False)
    ):
        self.shard_prefetch_timeout_s = int(shard_prefetch_timeout_s)
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        self.num_shards_per_epoch = num_shards_per_epoch
        self.epoch = 0
        self.processor = processor
        self.configure_processor = bool(configure_processor)

        self.shard_sampling_schedule = self.generate_shard_sampling_schedule()

        self.merge_statistics()

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

        self.curr_shard = None
        self._executor = None
        self._cache_job: Future | None = None

    def merge_statistics(self):
        """Weighted-merge the per-dataset statistics (all datasets share the one embodiment tag), install them on
        the processor (unless configure_processor=False) and hand the processor to every dataset."""
        emb = self.datasets[0].embodiment_tag.value
        assert all(ds.embodiment_tag.value == emb for ds in self.datasets), "all datasets must share the embodiment tag"
        stats = [ds.get_dataset_statistics() for ds in self.datasets]
        self.global_stats = {
            emb: {
                modality: merge_statistics([s[modality] for s in stats], self.weights)
                for modality in ["state", "action", "relative_action"]
                if modality in stats[0]
            }
        }
        if self.configure_processor:
            self.processor.set_statistics(self.global_stats)
        for ds in self.datasets:
            ds.set_processor(self.processor)

    def get_dataset_statistics(self):
        """Get the merged dataset statistics."""
        return self.global_stats

    def generate_shard_sampling_schedule(self) -> list[tuple[int, int]]:
        """Epoch schedule of (dataset_index, shard_index): datasets drawn with weight / average shard size (so
        datasets with smaller shards are not under-represented), shards of each dataset cycled in shuffled order."""
        rng = np.random.default_rng(self.seed + self.epoch)

        average_shard_sizes = []
        for dataset in self.datasets:
            average_shard_size = sum(
                dataset.get_shard_length(i) for i in range(len(dataset))
            ) / len(dataset)
            average_shard_sizes.append(average_shard_size)

        normalized_weights = np.array(
            [w / s for w, s in zip(self.weights, average_shard_sizes)]
        )
        normalized_weights = normalized_weights / normalized_weights.sum()

        dataset_sampling_schedule = rng.choice(
            len(self.datasets), size=self.num_shards_per_epoch, p=normalized_weights
        )

        shard_sampling_schedule = []
        shards_to_sample = []
        for dataset in self.datasets:
            shard_ids = list(range(len(dataset)))
            rng.shuffle(shard_ids)
            shards_to_sample.append(shard_ids)

        for i in dataset_sampling_schedule:
            if len(shards_to_sample[i]) == 0:
                shard_ids = list(range(len(self.datasets[i])))
                rng.shuffle(shard_ids)
                shards_to_sample[i] = shard_ids
            shard_idx = shards_to_sample[i].pop(0)
            shard_sampling_schedule.append((i, shard_idx))

        return shard_sampling_schedule

    def filter_shard_sample_schedule(self):
        """This (rank, worker)'s slice of the schedule: entry i goes to rank * num_workers + worker_id == i mod
        (world_size * num_workers)."""
        filtered_schedule = []
        worker_info = get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # the worker identity must not change between epochs (the schedule split depends on it)
        if self.worker_id is None:
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert self.worker_id == worker_id and self.num_workers == num_workers, (
                "Worker ID or number of workers has been changed since it was set. This is not allowed."
            )

        for i, shard in enumerate(self.shard_sampling_schedule):
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        return filtered_schedule

    def __iter__(self):
        """Wait for the prefetched shard, start prefetching the next one, yield the current shard's steps shuffled;
        a new schedule is generated when the epoch is exhausted (never terminates)."""
        self._executor = ThreadPoolExecutor(max_workers=1)

        self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
        if len(self.worker_shard_sampling_schedule) == 0:
            self._executor.shutdown(wait=False)
            return

        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed + self.epoch)

        while True:
            self.curr_shard_index += 1

            wait_start = time.time()
            self.finish_cache_shard()
            wait_end = time.time()

            dataset_index, shard_index = self.worker_shard_sampling_schedule[self.curr_shard_index]
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: Wait for shard {shard_index} in dataset {dataset_index} in {wait_end - wait_start:.2f} seconds"
            )

            self.cache_next_shard()   # prefetch the next shard while this one is consumed

            assert self.curr_shard is not None
            indices_in_shard = np.arange(len(self.curr_shard))
            rng.shuffle(indices_in_shard)
            for index in indices_in_shard:
                yield self.curr_shard[index]

            self.delete_cached_shard()

    def cache_next_shard(self):
        """Submit get_shard() of the next scheduled shard to the background thread; starts a new epoch (fresh
        schedule) when the current one is exhausted."""
        assert self._executor is not None
        if self.curr_shard_index + 1 >= len(self.worker_shard_sampling_schedule):
            self.epoch += 1
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
            self.curr_shard_index = -1

        print(f"Rank {self.rank}, Worker {self.worker_id}: Caching shard...")
        next_dataset_idx, next_shard_idx = self.worker_shard_sampling_schedule[
            self.curr_shard_index + 1
        ]
        self._cache_job = self._executor.submit(
            self.datasets[next_dataset_idx].get_shard, next_shard_idx
        )

    def finish_cache_shard(self):
        """Wait for the prefetched shard, bounded by data.shard_prefetch_timeout_s. A stalled network-FS read /
        video decode would otherwise block this rank silently until the other ranks' NCCL watchdog (~30 min)
        aborts the job with a confusing SIGABRT; failing loud here lets the Job recycle the pod cleanly. Keep the
        timeout above a legitimately slow read (minutes) and below the NCCL watchdog."""
        assert self._cache_job is not None
        wait_timeout = int(self.shard_prefetch_timeout_s)
        try:
            self.curr_shard = self._cache_job.result(timeout=wait_timeout)
        except FuturesTimeoutError as e:
            raise RuntimeError(
                f"finish_cache_shard: background get_shard() did not return within {wait_timeout}s "
                f"(rank {self.rank}, worker {self.worker_id}, shard schedule index "
                f"{self.curr_shard_index + 1}). Almost certainly a stalled DDN read / video decode. "
                f"Failing loud so the Job recycles this pod cleanly instead of hanging into an NCCL "
                f"watchdog SIGABRT. (raise data.shard_prefetch_timeout_s if this fires on legitimately slow I/O.)"
            ) from e
        self._cache_job = None

    def delete_cached_shard(self):
        """Drop the consumed shard and return its memory to the OS: freed shard tensors and video-decoder buffers
        are not promptly released by glibc (decoder objects with cyclic refs wait for a GC pass), so long runs
        creep on host RAM until the cgroup OOM-kills the worker. Once per shard, so the cost is negligible."""
        del self.curr_shard
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    def reset_seed(self, seed: int):
        """Reset the seed and regenerate the schedule from epoch 0 (deterministic restarts)."""
        self.seed = seed
        self.epoch = 0
        self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
        self.curr_shard_index = -1
        self.curr_shard = None
        self._cache_job = None


class ExactRobotHumanMixDataset(ShardedMixtureDataset):
    """Exact per-batch robot/human split (data.batch_group_fractions; built by factory.py when it is set).

    The DataLoader assembles each per-GPU batch from PER_DEV consecutive samples of one worker's iterator, so it
    suffices that every worker's stream follows a fixed group pattern of length PER_DEV (r,h,r,h,... for 0.5/0.5):
    every pattern-aligned window holds exactly n_robot robot + n_human human clips.

    Two independent ShardedMixtureDataset streams (robot datasets / human datasets, each with its own schedule,
    rank/worker filtering and prefetch) are zipped according to the pattern. Statistics come from the full mixture
    (super().__init__); the sub-mixtures use configure_processor=False so they do not touch the processor stats.
    """

    def __init__(
        self,
        datasets: list[ShardedSingleStepDataset],
        weights: list[float],
        processor,
        is_human: list[bool],
        per_device_batch_size: int,
        human_frac: float,
        seed: int = 42,
        num_shards_per_epoch: int = int(1e5),
        shard_prefetch_timeout_s: int = 1800,
    ):
        super().__init__(
            datasets=datasets, weights=weights, processor=processor, seed=seed,
            num_shards_per_epoch=num_shards_per_epoch, shard_prefetch_timeout_s=shard_prefetch_timeout_s,
        )
        assert len(is_human) == len(datasets)
        r_idx = [i for i, h in enumerate(is_human) if not h]
        h_idx = [i for i, h in enumerate(is_human) if h]
        assert r_idx and h_idx, (
            f"data.batch_group_fractions needs ≥1 robot AND ≥1 human dataset (got robot={len(r_idx)}, human={len(h_idx)}); "
            f"human datasets are marked by meta/wam_human.json + data.human_datasets."
        )
        n_robot, n_human = robot_human_split_counts(int(per_device_batch_size), float(human_frac))
        self.pattern = interleave_pattern(int(per_device_batch_size), n_human)
        # sub-stream seeds are offset so the robot / human schedules are decorrelated
        self._sub = {
            "robot": ShardedMixtureDataset(
                datasets=[datasets[i] for i in r_idx], weights=[weights[i] for i in r_idx],
                processor=processor, seed=seed,
                num_shards_per_epoch=num_shards_per_epoch, shard_prefetch_timeout_s=shard_prefetch_timeout_s, configure_processor=False,
            ),
            "human": ShardedMixtureDataset(
                datasets=[datasets[i] for i in h_idx], weights=[weights[i] for i in h_idx],
                processor=processor, seed=seed + 7919,
                num_shards_per_epoch=num_shards_per_epoch, shard_prefetch_timeout_s=shard_prefetch_timeout_s, configure_processor=False,
            ),
        }
        print(
            f"[batch_group_fractions] per-GPU batch {per_device_batch_size} = {n_robot} robot + {n_human} human "
            f"clips, pattern={''.join('r' if g == 'robot' else 'h' for g in self.pattern)}; robot datasets="
            f"{[str(datasets[i].dataset_path) for i in r_idx]} human datasets={[str(datasets[i].dataset_path) for i in h_idx]}",
            flush=True,
        )

    def __iter__(self):
        its = {g: iter(ds) for g, ds in self._sub.items()}
        pos = 0
        while True:
            yield next(its[self.pattern[pos % len(self.pattern)]])
            pos += 1

    def reset_seed(self, seed: int):
        super().reset_seed(seed)
        self._sub["robot"].reset_seed(seed)
        self._sub["human"].reset_seed(seed + 7919)
