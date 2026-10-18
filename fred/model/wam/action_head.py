"""Cross-attention action DiT head: flow-matching velocity prediction over the action chunk, conditioned on the
video-DiT features (the only cross-attention context)."""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from fred.model.modules.action_dit import ActionDiT


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal embedding of timesteps (DiT4DiT reference implementation): (B, T) -> (B, T, embedding_dim)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        device = timesteps.device
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """Single-embodiment action encoder (DiT4DiT style).

    Inputs:
        actions:   (B, T, action_dim)
        timesteps: (B,)   discretized scalar per batch item
    Returns:
        (B, T, hidden_size)
    """

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        assert not (timesteps.dim() != 1 or timesteps.shape[0] != B), f"Expected timesteps shape (B,) where B={B}, got {tuple(timesteps.shape)}"
        # broadcast tau across action horizon
        timesteps = timesteps.unsqueeze(1).expand(-1, T)

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = F.silu(self.layer2(x))
        x = self.layer3(x)
        return x


class WAMActionHead(nn.Module):
    """Cross-attention DiT action head (DiT4DiT-style: the video-DiT features are the only context, single
    embodiment, no state dropout/noise). Flow-matching convention (DiT4DiT / CogACT):
        noisy    = (1 - t) * actions + t * noise
        velocity = noise - actions          (target)
        inference: actions = actions - dt * pred_velocity (Euler from t=1→0)
    """

    def __init__(self, config, backbone_dim: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.action_head_hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        diffusion_cfg = dict(config.action_dit_cfg)
        self.dit = ActionDiT(
            **diffusion_cfg,
            cross_attention_dim=int(backbone_dim),   # the video DiT hidden size (from the backbone, not a config field)
        )

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # Padding action dims get no flow-matching noise, in training and inference; the valid-dim count comes from
        # the checkpoint (config.action_valid_dim) or the first training batch's action_mask.
        self.action_valid_dim = int(config.action_valid_dim) if config.action_valid_dim is not None else None

        self.state_encoder = (
            MLP(
                input_dim=config.max_state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.max_state_dim
            else None
        )
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.dit.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

    def _record_valid_dim(self, action_mask: torch.Tensor) -> None:
        """Derive the number of valid action dims from action_mask and persist it to the config.

        Inference (predict_action) has no action_mask, so the valid dim observed at training time
        is recorded in config.action_valid_dim; save_pretrained writes it to config.json and eval
        reproduces the same masking. mask is (B, T, D); padding columns are all-zero.
        """
        if self.action_valid_dim is not None:
            return
        with torch.no_grad():
            col_any = (action_mask != 0).any(dim=0).any(dim=0)     # (D,) bool
            nz = torch.nonzero(col_any, as_tuple=False)
            valid = int(nz.max().item()) + 1 if nz.numel() else 0
            # valid = max valid column + 1: per-dataset action groups may leave zero columns below it, but every
            # column above it is zero in every sample, which is all the padding mask needs.
        if valid <= 0:
            # Pathological all-zero batch mask: do not persist a wrong value, retry next step.
            return
        self.action_valid_dim = valid
        self.config.action_valid_dim = valid
        print(
            f"[WAMActionHead] action_valid_dim={valid} / max_action_dim={self.action_dim} "
            f"(padding {self.action_dim - valid} dims) — persisted to config.json.",
            flush=True,
        )

    def forward(
        self,
        vdm_embeds: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training forward (flow matching velocity prediction).

        Args:
            vdm_embeds:   (B, S, D_kv) backbone features used as cross-attention KV
            actions:     (B, T, action_dim) ground-truth actions
            action_mask: (B, T, action_dim) 1 for valid entries
            state:       (B, 1, state_dim) or None
        Returns:
            dict with `loss`, `action_mse_loss`, `action_l1_loss`, `action_t`.
        """
        device = vdm_embeds.device

        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        # Padding dims (action_mask==0) get zero noise → noisy_trajectory is exactly 0 there, so no unsupervised random
        # values enter the model; the velocity target is 0 too, consistent with the loss masking.
        noise = noise * action_mask
        self._record_valid_dim(action_mask)
        t = self.beta_dist.sample([actions.shape[0]]).to(device, dtype=actions.dtype) / self.config.timestep_sample_scale
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * actions + t * noise
        velocity = noise - actions

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = None
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = (
            torch.cat((state_features, action_features), dim=1)
            if state_features is not None
            else action_features
        )

        model_output = self.dit(hidden_states=sa_embs, encoder_hidden_states=vdm_embeds, timestep=t_discretized)

        pred = self.action_decoder(model_output)
        pred_velocity = pred[:, -actions.shape[1]:]

        squared = (pred_velocity - velocity) ** 2 * action_mask
        loss = squared.sum() / action_mask.sum()
        action_l1 = (F.l1_loss(pred_velocity, velocity, reduction="none") * action_mask).sum() / action_mask.sum()

        return {
            "loss": loss,
            "action_mse_loss": loss.detach(),
            "action_l1_loss": action_l1.detach(),
            "action_t": t.squeeze(-1).squeeze(-1).detach(),
        }

    @torch.inference_mode()
    def predict_action(
        self,
        vdm_embeds: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Inference: Euler-integrate from noise (t=1) toward clean (t=0). ``generator`` seeds the noise draw
        (the eval server owns it) so rollouts are reproducible run to run."""
        batch_size = vdm_embeds.shape[0]
        device = vdm_embeds.device

        actions = torch.randn((batch_size, self.action_horizon, self.action_dim), generator=generator, dtype=vdm_embeds.dtype, device=device)
        # Train/test parity: training masks the padding-dim noise, so inference does too. A checkpoint without
        # action_valid_dim was trained before the masking existed (noise on every dim) and keeps that behaviour.
        valid = int(self.action_valid_dim) if self.action_valid_dim is not None else self.action_dim
        if valid < self.action_dim:
            actions[..., valid:] = 0

        state_features = None
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_cont = 1.0 - step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                (batch_size,), t_discretized, device=device, dtype=torch.long
            )

            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = (
                torch.cat((state_features, action_features), dim=1)
                if state_features is not None
                else action_features
            )

            model_output = self.dit(hidden_states=sa_embs, encoder_hidden_states=vdm_embeds, timestep=timesteps_tensor)
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions - dt * pred_velocity
            if valid < self.action_dim:
                actions[..., valid:] = 0

        return actions
