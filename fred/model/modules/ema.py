"""EMA copies: `EMA` wraps a frozen fp32 copy of a module and updates it after every optimizer step.

Two instances exist in a WAMModel: `backbone.extractor.video_ema` (the self-distillation teacher; its weights are
what evaluation runs) and `action_ema` (the action head evaluation runs). Both live inside the single checkpoint as
`<name>.model.*`.

Why a wrapper module: DeepSpeed casts the whole model to bf16 at engine init and bf16 underflows the (1−β)≈1e-4 EMA
increments, so `_apply` re-asserts fp32 after any cast. In sharded mode (14B) the runnable copy stays bf16 on the
GPU and the fp32 master is this rank's 1/W slice of the flattened parameters (`ShardedFP32EMALayout`), persisted per rank
as `<ckpt>/teacher_fp32_shard_rank{r}.pt` by the trainer callback.

Decay schedule (diffusers `EMAModel(use_ema_warmup=True, inv_gamma=1, power=3/4, decay=0.9999)`):
β(step) = min(1 − (1 + (step−1))^(−3/4), 0.9999), β(1) = 0 so the first update copies the live weights.
"""
from __future__ import annotations

import bisect
import copy
import os
from typing import Optional

import torch
import torch.nn as nn
from fred.utils.dist import get_global_rank, get_world_size


def ema_decay(step: int, inv_gamma: float = 1.0, power: float = 0.75, max_decay: float = 0.9999) -> float:
    s = max(0, int(step) - 1)
    if s <= 0:
        return 0.0
    return min(1.0 - (1.0 + s / inv_gamma) ** (-power), max_decay)


class EMA(nn.Module):
    """Frozen copy of `module` (eval mode, no grad, hooks cleared) updated by `update(live, step)`."""

    def __init__(self, module: nn.Module, sharded: bool = False):
        super().__init__()
        model = copy.deepcopy(module)
        model.eval().requires_grad_(False)
        for m in model.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()
        self.model = model.to(torch.bfloat16) if sharded else model.float()
        self.sharded = sharded
        self._layout: Optional[ShardedFP32EMALayout] = None
        if sharded:
            self.register_buffer("fp32_shard", None, persistent=False)   # rank slice; never in the model state_dict

    # ── dtype pinning ──
    def _apply(self, *args, **kwargs):
        if self.sharded:
            return self._apply_sharded(*args, **kwargs)
        super()._apply(*args, **kwargs)
        for t in list(self.parameters()) + list(self.buffers()):
            if t.is_floating_point() and t.dtype != torch.float32:
                t.data = t.data.float()
        return self

    def _apply_sharded(self, *args, **kwargs):
        """The fp32 master shard never passes through a dtype cast (only the device move); the runnable copy stays bf16."""
        fn = args[0] if args else kwargs["fn"]
        had = "fp32_shard" in self._buffers
        shard = self._buffers.pop("fp32_shard", None)
        try:
            super()._apply(*args, **kwargs)
        finally:
            if had:
                if shard is not None:
                    probe = fn(torch.zeros(1, dtype=torch.float32, device=shard.device))
                    if probe.device != shard.device:
                        shard = shard.to(probe.device)
                self._buffers["fp32_shard"] = shard
        for p in self.model.parameters():
            if p.is_floating_point() and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    # ── update ──
    @torch.no_grad()
    def update(self, live: nn.Module, step: int) -> float:
        """EMA ← β·EMA + (1−β)·live over the parameters (frozen live params are copied), buffers copied. Returns β."""
        decay = ema_decay(step)
        if self.sharded:
            self._sharded_step(list(live.parameters()), decay)
        else:
            for s, p in zip(self.model.parameters(), live.parameters()):
                if p.requires_grad:
                    s.sub_((1.0 - decay) * (s - p))
                else:
                    s.copy_(p)
        for bt, bs in zip(self.model.buffers(), live.buffers()):
            bt.copy_(bs)
        return decay

    # ── sharded fp32 master (14B) ──
    def layout(self) -> "ShardedFP32EMALayout":
        if self._layout is None:
            self._layout = ShardedFP32EMALayout([p.numel() for p in self.model.parameters()], get_world_size(), get_global_rank())
        return self._layout

    @torch.no_grad()
    def _shard_reset_from(self, src_params) -> None:
        params = list(self.model.parameters())
        self.fp32_shard = self.layout().new_shard(list(src_params), device=params[0].device)

    @torch.no_grad()
    def _sharded_step(self, live_params, decay: float) -> None:
        """fp32 shard ← EMA(shard, live slice), then all_gather(bf16) in place into the runnable copy."""
        params = list(self.model.parameters())
        if self.fp32_shard is None:
            self._shard_reset_from(params)   # fresh run: the copy == live at build, same start as the fp32 path
        lay = self.layout()
        lay.ema_step(self.fp32_shard, live_params, decay)
        lay.gather_into_params(self.fp32_shard, params)

    def shard_save(self, ckpt_dir: str) -> Optional[str]:
        if self.fp32_shard is None:
            return None
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, self.layout().shard_filename())
        self.layout().save_shard(self.fp32_shard, path)
        return path

    def shard_load(self, ckpt_dir: str) -> bool:
        """Resume: restore this rank's fp32 shard; a missing or mismatched file (e.g. another world size) rebuilds
        the master from the bf16 copy with a warning."""
        params = list(self.model.parameters())
        path = os.path.join(ckpt_dir, self.layout().shard_filename())
        try:
            assert os.path.isfile(path), path
            self.fp32_shard = self.layout().load_shard(path, device=params[0].device)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[EMA] WARNING rank{self.layout().rank}: fp32 shard not restored from {path!r} ({e!r}); rebuilding from the bf16 copy", flush=True)
            self._shard_reset_from(params)
            return False


class ShardedFP32EMALayout:
    """Layout + update math for a data-parallel-sharded fp32 EMA master of a bf16 runnable copy.

    Flat vector = concat(p.view(-1) for p in module.parameters()) (parameter order), padded to W*S with
    S = ceil(N/W); rank r owns flat[r*S:(r+1)*S] as one fp32 tensor (the padded tail is unused zeros).
    Per optimizer step:
      1. ``ema_step``: the rank's fp32 slice <- EMA(slice, live-param slice) with exactly the per-param
         expression of ``EMA.update`` (``s.sub_((1-β)*(s-p))`` for requires_grad params, ``s.copy_(p)``
         otherwise), so the sharded master is bit-identical to the unsharded fp32 copy. β comes from the
         caller (one schedule: ``ema_decay``). Live params are replicated under ZeRO-2, so slicing is local.
      2. ``gather_into_params``: bf16-cast slice -> bucketed all_gather -> written in place into the bf16
         module params (no full fp32 copy, no second bf16 copy).
    Only numels are stored; parameter lists are passed per call because ``load_state_dict(assign=True)``
    replaces the Parameter objects and cached references would go stale.
    """

    FILE_FMT = "teacher_fp32_shard_rank{rank}.pt"

    def __init__(self, numels, world_size: int, rank: int, bucket_numel: int = 8 * 1024 * 1024):
        self.numels = [int(n) for n in numels]
        self.offsets = []
        acc = 0
        for n in self.numels:
            self.offsets.append(acc)
            acc += n
        self.total = acc
        self.world_size = int(world_size)
        self.rank = int(rank)
        assert 0 <= self.rank < self.world_size, f"ShardedFP32EMALayout: rank {self.rank} outside world {self.world_size}"
        self.shard_len = (-(-self.total // self.world_size)) if self.total else 0
        self.lo = self.rank * self.shard_len
        self.hi = min(self.lo + self.shard_len, self.total)
        self.pieces = self._pieces(self.lo, self.hi)  # [(param_idx, a, b, off_in_shard)]
        self.bucket_numel = max(1, min(int(bucket_numel), self.shard_len)) if self.shard_len else 1
        self.n_buckets = (-(-self.shard_len // self.bucket_numel)) if self.shard_len else 0
        # gather plan: bucket j (shard[c0:c1] on every rank) -> [(recv_off, param_idx, a, b)]
        plan = []
        for j in range(self.n_buckets):
            c0 = j * self.bucket_numel
            c1 = min(self.shard_len, c0 + self.bucket_numel)
            n = c1 - c0
            entries = []
            for r in range(self.world_size):
                flo = r * self.shard_len + c0
                fhi = min(r * self.shard_len + c1, self.total)
                for (i, a, b, off) in self._pieces(flo, fhi):
                    entries.append((r * n + off, i, a, b))
            plan.append((c0, c1, entries))
        self.gather_plan = plan

    def _pieces(self, lo: int, hi: int):
        """Parameter pieces covering flat[lo:hi] -> [(param_idx, a, b, off)], off relative to lo."""
        out = []
        if hi <= lo:
            return out
        i = max(bisect.bisect_right(self.offsets, lo) - 1, 0)
        while i < len(self.numels) and self.offsets[i] < hi:
            po, pn = self.offsets[i], self.numels[i]
            if pn > 0:
                a = max(lo, po) - po
                b = min(hi, po + pn) - po
                if b > a:
                    out.append((i, a, b, max(lo, po) - lo))
            i += 1
        return out

    def check(self, params):
        params = list(params)
        assert len(params) == len(self.numels) and all(p.numel() == n for p, n in zip(params, self.numels)), (
            f"ShardedFP32EMALayout: parameter layout changed (got {len(params)} params vs layout "
            f"{len(self.numels)}) — the shard layout is bound to the teacher's parameter list."
        )
        return params

    @torch.no_grad()
    def new_shard(self, src_params, device=None) -> torch.Tensor:
        """Fresh fp32 shard = this rank's slice of ``src_params`` (upcast on copy; pad tail = 0)."""
        src = self.check(src_params)
        dev = device if device is not None else (src[0].device if src else "cpu")
        shard = torch.zeros(self.shard_len, dtype=torch.float32, device=dev)
        for (i, a, b, off) in self.pieces:
            shard[off:off + (b - a)].copy_(src[i].detach().reshape(-1)[a:b])
        return shard

    @torch.no_grad()
    def ema_step(self, shard: torch.Tensor, live_params, decay: float) -> None:
        """diffusers EMAModel.step (foreach=False) math on this rank's fp32 slice only."""
        live = self.check(live_params)
        one_minus_decay = 1 - float(decay)
        for (i, a, b, off) in self.pieces:
            s_param = shard[off:off + (b - a)]
            p = live[i]
            param = p.detach().reshape(-1)[a:b]
            if p.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

    @torch.no_grad()
    def gather_into_params(self, shard: torch.Tensor, params) -> None:
        """all_gather (NCCL) the bf16-cast shard slices and write them IN PLACE into ``params`` (bucketed)."""
        params = self.check(params)
        if not params:
            return
        dtype = params[0].dtype
        flats = [p.data.view(-1) for p in params]  # view(-1) FAILS LOUD on a non-contiguous param
        dist_on = self.world_size > 1
        if dist_on:
            import torch.distributed as dist  # noqa: PLC0415
        for (c0, c1, entries) in self.gather_plan:
            n = c1 - c0
            send = shard[c0:c1].to(dtype)
            if not dist_on:
                recv = send
            else:
                recv = torch.empty(self.world_size * n, dtype=dtype, device=send.device)
                dist.all_gather_into_tensor(recv, send)
            for (ro, i, a, b) in entries:
                flats[i][a:b].copy_(recv[ro:ro + (b - a)])

    # ── per-rank shard file (RESUME) ──
    def shard_filename(self) -> str:
        return self.FILE_FMT.format(rank=self.rank)

    def _meta(self):
        return {"world_size": self.world_size, "rank": self.rank, "total": self.total,
                "shard_len": self.shard_len, "n_params": len(self.numels)}

    def save_shard(self, shard: torch.Tensor, path: str) -> None:
        obj = {"format": "wam_teacher_fp32_shard_v1", "shard": shard.detach().to("cpu", copy=True)}
        obj.update(self._meta())
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def load_shard(self, path: str, device=None) -> torch.Tensor:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        meta = self._meta()
        bad = {k: (obj.get(k), v) for k, v in meta.items() if obj.get(k) != v}
        assert not bad and int(obj["shard"].numel()) == self.shard_len, f"fp32 shard file {path!r} does not match this layout: {bad or 'numel'}"
        return obj["shard"].to(device=device, dtype=torch.float32)
