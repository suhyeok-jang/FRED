"""FredTrainer: HF ``Trainer`` with a resume-aware dataloader (reseed instead of data skip), optional per-module
LR groups, and the EMA / GC-freeze / resume / checkpoint-format callbacks."""

from __future__ import annotations

from pathlib import Path
import shutil
import logging
import os
from typing import Any, Optional

import torch
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback


logger = logging.getLogger(__name__)


class _GCFreezeCallback(TrainerCallback):
    """Once, right after the first optimizer step of this process, move every live object
    into the permanent generation so later gc passes skip the steady-state heap (dataloader-heavy real-robot
    runs spent seconds per step in gc)."""

    done = False  # one-shot per process (resume-safe: first completed step of THIS process)

    def on_step_end(self, args, state, control, **kwargs):
        if self.done:
            return
        self.done = True
        import gc
        gc.collect()
        gc.freeze()
        print(f"[gc_freeze] gc.freeze() after first step of this process "
              f"(global_step={int(state.global_step)}, frozen={gc.get_freeze_count()})", flush=True)


def _worker_gc_unfreeze(_worker_id):
    """DataLoader worker_init_fn: workers keep default GC (no-op unless frozen)."""
    import gc
    gc.unfreeze()


class _ResumeCallback(TrainerCallback):
    """on_train_begin fires after the resume load and before the loop, which makes it the hook for two things:

    * the current ``args`` save/log cadence is re-applied to the restored TrainerState (HF restores
      save_steps / logging_steps from trainer_state.json and DefaultFlowCallback keys off ``state.*``, so a
      changed CLI value would otherwise be a silent no-op);
    * after a DeepSpeed resume (``deepspeed_load_checkpoint`` bypasses ``_load_optimizer_and_scheduler``) the
      load's transient GPU buffers are freed, or they linger into the first forward and can OOM it.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps = args.save_steps
        state.logging_steps = args.logging_steps
        if state.global_step > 0:  # resume only (global_step preloaded from trainer_state)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[i] resume: cadence from args, GPU buffers cleared at global_step={state.global_step}", flush=True)


class CheckpointFormatCallback(TrainerCallback):
    """Make every checkpoint-{step}/ standalone: copy experiment_cfg/, the saved processor files and
    wandb_config.json from the run dir into it after each save."""

    def __init__(self, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None):
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class _EMACallback(TrainerCallback):
    """After every optimizer step: video_ema <- video DiT, action_ema <- action head (fred/model/modules/ema.py).
    ``state.global_step`` (restored on resume) drives the decay schedule. Sharded video EMA (14B): the per-rank
    fp32 master shard is not in the model state_dict, so it is saved/restored here per rank."""

    def __init__(self):
        self.resume_dir = None  # resolved resume dir (set by FredTrainer.train)

    @staticmethod
    def _model_and_extractor(model):
        m = getattr(model, "module", model)
        return m, m.backbone.extractor

    def on_save(self, args, state, control, model=None, **kwargs):
        _, ext = self._model_and_extractor(model)
        if not ext.video_ema.sharded:
            return
        path = ext.video_ema.shard_save(os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(state.global_step)}"))
        if path is not None and args.local_rank in (-1, 0):
            print(f"[EMA] step={int(state.global_step)}: fp32 shard -> {path} (one file per rank)", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        _, ext = self._model_and_extractor(model)
        if not ext.video_ema.sharded or int(state.global_step) <= 0:
            return
        ckpt_dir = self.resume_dir or os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(state.global_step)}")
        if ext.video_ema.shard_load(ckpt_dir) and args.local_rank in (-1, 0):
            print(f"[EMA] resume step={int(state.global_step)}: fp32 shard restored from {ckpt_dir}", flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        m, ext = self._model_and_extractor(model)
        step = int(state.global_step)
        beta = ext.video_ema.update(ext.video_dit, step)
        beta_act = m.action_ema.update(m.action_head, step)
        if (step <= 20 or step % 500 == 0) and args.local_rank in (-1, 0):
            print(f"[EMA] step={step} beta video={beta:.6f} action={beta_act:.6f}", flush=True)


class FredTrainer(Trainer):
    """HF Trainer with a resume-aware dataloader (reseed instead of data skip), optional per-module LR groups,
    and the EMA / GC-freeze / resume callbacks."""

    def __init__(
        self,
        *args: Any,
        multiprocessing_context: str,
        prefetch_factor: int | None,
        lr_per_module_enabled: bool,
        lr_base: float,
        lr_backbone: float,
        lr_action_head: float,
        **kwargs: Any,
    ) -> None:
        assert multiprocessing_context in {"fork", "spawn", "forkserver"}, f"multiprocessing_context must be fork|spawn|forkserver, got {multiprocessing_context!r}"
        self.multiprocessing_context = multiprocessing_context   # DataLoader worker start method (data.multiprocessing_context)
        self.prefetch_factor = prefetch_factor
        self.lr_per_module_enabled = lr_per_module_enabled
        self.lr_base = lr_base
        self.lr_backbone = lr_backbone
        self.lr_action_head = lr_action_head
        super().__init__(*args, **kwargs)
        self.add_callback(_ResumeCallback())
        self.add_callback(_GCFreezeCallback())
        self.add_callback(_EMACallback())

    def create_optimizer(self):
        """With ``lr_per_module_enabled``, DiT4DiT-style param groups (first match wins per trainable parameter):
        name contains "backbone.extractor.video_dit" -> lr_backbone; name starts with "action_head." ->
        lr_action_head; everything else -> lr_base. ``TrainingArguments.weight_decay`` applies to every group
        (DeepSpeed ZeRO keeps per-group LRs)."""
        if not self.lr_per_module_enabled:
            return super().create_optimizer()
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model_wrapped
        backbone_params, head_params, base_params = [], [], []
        for name, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone.extractor.video_dit" in name:
                backbone_params.append(p)
            elif name.startswith("action_head."):
                head_params.append(p)
            else:
                base_params.append(p)

        wd = float(self.args.weight_decay)
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.lr_backbone, "weight_decay": wd, "name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.lr_action_head, "weight_decay": wd, "name": "action_head"})
        if base_params:
            param_groups.append({"params": base_params, "lr": self.lr_base, "weight_decay": wd, "name": "base"})

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        optimizer_kwargs.pop("lr", None)   # per-group lrs above take precedence
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        if self.args.local_rank in (-1, 0):
            logger.info(
                "[per-module LR] base=%g (#=%d) backbone=%g (#=%d) action_head=%g (#=%d)",
                self.lr_base, len(base_params),
                self.lr_backbone, len(backbone_params),
                self.lr_action_head, len(head_params),
            )

        return self.optimizer

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):
        """DataLoader over the ShardedMixtureDataset (an IterableDataset that orders its own samples): on resume
        the dataset is reseeded with ``seed + global_step`` instead of skipping data."""
        curr_global_step = self.state.global_step
        logger.info(f"[i] Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            logger.info(f"[i] Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible.")

        logger.info("[i] Creating custom train dataloader")
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_num_workers > 0,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context
            logger.info("[i] DataLoader multiprocessing_context=%s", self.multiprocessing_context)
            # data.prefetch_factor (PyTorch default 2): batches staged per worker, at
            # num_workers * prefetch_factor * batch_size of host memory.
            if self.prefetch_factor:
                dataloader_params["prefetch_factor"] = int(self.prefetch_factor)
            dataloader_params["worker_init_fn"] = _worker_gc_unfreeze

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Resolve the resume dir and preload its TrainerState so get_train_dataloader sees global_step."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logger.info(f"[w] No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None:
            logger.info(f"[i] Resuming from checkpoint {resume_from_checkpoint}")
            # HF loads the state later, inside the training loop
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            for _cb in self.callback_handler.callbacks:   # sharded video EMA: the per-rank fp32 shard files live there
                if isinstance(_cb, _EMACallback):
                    _cb.resume_dir = resume_from_checkpoint

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # GA-aware SelfDistill centering: stamp BEFORE the forward. transformers 4.57.0 `_inner_training_loop`
        # calls `accelerator.gradient_state._set_sync_gradients(do_sync_step)` right before every training_step,
        # DeepSpeed included, so `accelerator.sync_gradients` is True exactly on the last micro-batch of the
        # accumulation window. `model` is the wrapped model (DeepSpeed engine / DDP); `.module` is the WAMModel.
        um = getattr(model, "module", model)
        um.grad_accum_is_last_micro_batch = bool(self.accelerator.sync_gradients)
        um.grad_accum_steps = int(self.args.gradient_accumulation_steps)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """``Trainer.compute_loss`` with ``return_outputs=True``; on logging steps the model's ``outputs["logs"]``
        scalars are logged."""
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if (
            isinstance(outputs, dict)
            and "logs" in outputs
            and self.state.global_step % self.args.logging_steps == 0
            and model.training
            and self.args.local_rank in (-1, 0)
        ):
            self.log({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in outputs["logs"].items() if v is not None})

        return (loss, outputs) if return_outputs else loss
