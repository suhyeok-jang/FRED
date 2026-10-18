"""Self-distillation pieces shared by the student and the EMA teacher: the prototype head and the
gradient-accumulation mean accumulator used for the teacher centering / head-EMA updates."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfDistillHead(nn.Module):
    """Prototype head (the DINO/iBOT `DINOHead` role): MLP in→2048→GELU→2048→GELU→256, L2-normalized bottleneck
    (logits = cosines to the prototype rows), weight_norm(Linear(256→K, no bias)) with weight_g initialized to 1
    and left trainable.

    role="student": trainable, runs in the model's compute dtype.
    role="teacher": the EMA copy — fp32-pinned (``_apply`` re-asserts fp32 after any whole-model cast so the
    (1−m)≈4e-3 increments never accumulate in bf16), no grad, plus the ``center`` buffer of its logits.
    ``init_from(student)`` copies the student weights; ``ema_update(student, m)`` does the per-step EMA."""

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int, out_dim: int, role: str = "student"):
        super().__init__()
        assert role in ("student", "teacher"), f"role must be student|teacher, got {role!r}"
        self.role = role
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        last = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.init.trunc_normal_(last.weight, std=0.02)
        self.last_layer = nn.utils.weight_norm(last)
        self.last_layer.weight_g.data.fill_(1.0)
        if role == "teacher":
            self.register_buffer("center", torch.zeros(out_dim, dtype=torch.float32), persistent=True)
            self.float().requires_grad_(False)

    def forward(self, x):
        x = self.mlp(x.to(self.mlp[0].weight.dtype))   # the head's own dtype: compute dtype (student) or fp32 (teacher)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)

    @torch.no_grad()
    def init_from(self, student: "SelfDistillHead") -> "SelfDistillHead":
        """Teacher := student weights (load_state_dict: deepcopy of a weight_norm-ed module is refused; strict=False
        because the teacher additionally owns `center`)."""
        assert self.role == "teacher" and student.role == "student", f"SelfDistillHead.init_from: teacher <- student expected, got {self.role!r} <- {student.role!r}"
        self.load_state_dict(student.state_dict(), strict=False)
        return self.float().requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, student: "SelfDistillHead", m: float) -> None:
        assert self.role == "teacher" and student.role == "student", f"SelfDistillHead.ema_update: teacher <- student expected, got {self.role!r} <- {student.role!r}"
        for p_t, p_s in zip(self.parameters(), student.parameters()):
            p_t.mul_(m).add_((1.0 - m) * p_s.detach().float())

    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)
        if self.role == "teacher":
            for t in list(self.parameters(recurse=True)) + list(self.buffers(recurse=True)):
                if t.is_floating_point() and t.dtype != torch.float32:
                    t.data = t.data.float()
        return self


class MicroBatchMeanAccumulator:
    """Mean over the micro-batches of one optimizer step (gradient accumulation).

    With gradient_accumulation_steps = k the forward runs k times per optimizer step, so the SelfDistill center
    EMA (momentum 0.9) and the teacher-head EMA (0.996) would fire k times per step against an unchanged student.
    Both must fire once per step on the full-step batch mean so a GA=k run matches GA=1 with k× the per-device
    batch. The mean of equal-size micro-batch means is the full-step mean; unequal sizes are refused."""

    def __init__(self) -> None:
        self.reset()

    def add(self, mean: torch.Tensor, rows: int) -> None:
        assert rows > 0, f"rows must be > 0, got {rows}"
        assert self._rows in (None, rows), f"unequal micro-batch sizes within one optimizer step ({self._rows} vs {rows})"
        self._rows = rows
        self._sum = mean if self._sum is None else self._sum + mean
        self._n += 1

    def pop(self) -> torch.Tensor:
        """Full-step mean, then reset."""
        assert self._n > 0, "nothing accumulated"
        out = self._sum / self._n if self._n > 1 else self._sum
        self.reset()
        return out

    def reset(self) -> None:
        self._sum = None
        self._n = 0
        self._rows = None
