"""Continuous-time flow-matching scheduler (shift-warped σ grid) shared by the Wan2.2-5B and Wan2.1-14B extractors.

One ``shift`` drives both the training-t distribution and the bell-weight normalization grid (FastWAM coupling);
the extractors build it with ``config.wan_fm_train_shift``."""
import torch


class WanContinuousFlowMatchScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        assert num_train_timesteps > 0, f"`num_train_timesteps` must be positive, got {num_train_timesteps}"
        assert shift > 0, f"`shift` must be positive, got {shift}"
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        # bell stats (y_min, norm_const) over the shift-warped 1000-grid
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        self._y_min = float(y_grid.min().item())
        self._weight_norm_const = float((y_grid - self._y_min).mean().item())

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def sample_training_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """(B,) float32 timesteps σ·num_train_timesteps with σ = φ(u, shift), u ~ U(0,1) from the global RNG stream
        (shift=1 → uniform σ; shift=5 → 5u/(1+4u), native Wan)."""
        assert batch_size > 0, f"`batch_size` must be positive, got {batch_size}"
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        return self._phi(u, self.shift) * float(self.num_train_timesteps)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Mid-σ bell weight, shifted to min 0 and normalized to mean 1 over the shift-warped grid."""
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        return (y - self._y_min) / (self._weight_norm_const + self.eps)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """x_t = (1−σ)·x0 + σ·ε with per-sample (B,) timesteps."""
        assert timestep.ndim == 1 and int(timestep.shape[0]) == int(original_samples.shape[0]), (
            f"add_noise: `timestep` must be a per-sample (B,) vector, got {tuple(timestep.shape)} for samples {tuple(original_samples.shape)}"
        )
        sigma = (timestep / float(self.num_train_timesteps)).to(original_samples.device, dtype=original_samples.dtype)
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - sample
