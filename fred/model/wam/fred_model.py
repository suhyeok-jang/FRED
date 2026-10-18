"""WAMModel — video world model (Wan DiT) + action DiT, trained with the SelfDistill token-level
self-distillation term on the video DiT's future tokens.

Pipeline (training step):
    cond_video (+ future_video) ──► backbone (Wan DiT) ──► feature = blocks[L] tokens (σ=1.0 extraction)
                                         │                        │
                                         │           (state, noisy actions, t) ──► action DiT ──► L_action
                                         ├──► video flow matching on the live DiT ──► L_fm
                                         └──► SelfDistill: student blocks[L] future tokens (grad-on) vs the
                                              self-EMA teacher's teacher-forced future tokens ──► L_sd

    loss = action_loss_weight·L_action + video_loss_weight·(L_fm + γ·L_sd)

The action loop uses the action-repeat trick (``repeated_diffusion_steps``): the batch is replicated N×
before the action head so each sample sees N independent flow-matching timesteps.

Checkpoint key layout: backbone.extractor.* (see wan22_extractor), action_head.*, sd_student_head.* (student
prototype head, role=student), sd_teacher_head.* (its EMA copy + `center`, fp32, role=teacher), action_ema.* (EMA action head, fp32).
Evaluation loads the EMA copies into the live modules (fred/policy/eval_weights.py).
"""
from typing import Tuple

import torch
from transformers import PreTrainedModel

from fred.configs.model.wam import WAMConfig, resolve_action_dit_dims
from fred.model.wam.action_head import WAMActionHead
from fred.model.wam.backbone import BackboneOutput, WanBackbone
from fred.model.modules.ema import EMA
from fred.model.modules.self_distill import MicroBatchMeanAccumulator, SelfDistillHead


class WAMModel(PreTrainedModel):
    """See the module docstring. Video DiT + action DiT; inference runs the action-only forward."""

    config_class = WAMConfig
    supports_gradient_checkpointing = True   # read by transformers' gradient_checkpointing_enable (Trainer --gradient-checkpointing)

    def __init__(self, config: WAMConfig):
        super().__init__(config)
        self.config = config
        config.action_dit_cfg, config.input_embedding_dim = resolve_action_dit_dims(
            config.action_dit_size, config.action_dit_cfg, config.input_embedding_dim,
        )
        config.validate()

        # ── backbone + action head ──
        self.backbone = WanBackbone(config)
        backbone_dim = int(self.backbone.extractor.feature_dim)   # video-DiT hidden (5B 3072, 14B 5120)
        self.action_head = WAMActionHead(config, backbone_dim)

        # ── SelfDistill: student future tokens ↔ EMA-teacher future tokens (prototype head, centred teacher softmax) ──
        self.sd_student_head = SelfDistillHead(backbone_dim, 2048, 256, config.selfdistill_num_prototypes)
        self.sd_teacher_head = SelfDistillHead(backbone_dim, 2048, 256, config.selfdistill_num_prototypes, role="teacher").init_from(self.sd_student_head)
        # Gradient-accumulation-aware centering / head-EMA: with GA=k the forward runs k× per optimizer step; the
        # center update (+ all-reduce) and the head EMA fire ONCE per step on the full-step mean (the trainer
        # stamps grad_accum_is_last_micro_batch / grad_accum_steps). GA=1 → per-forward update.
        self.sd_teacher_mean_acc = MicroBatchMeanAccumulator()

        self.action_ema = EMA(self.action_head)   # the action head evaluation runs; updated by the trainer after each optimizer step

    # ── inputs ──

    def prepare_input(self, inputs: dict) -> dict:
        """Tensors to the model's device, floating ones to its dtype; everything else (prompt strings) as is."""
        def to_device_with_dtype(x):
            if not torch.is_tensor(x):
                return x
            return x.to(self.device, dtype=self.dtype) if torch.is_floating_point(x) else x.to(self.device)

        return {k: to_device_with_dtype(v) for k, v in inputs.items()}

    def _compute_action_loss(self, backbone_outputs: BackboneOutput, inputs: dict) -> dict:
        """Action-repeat trick: every row is replicated `repeated_diffusion_steps`× so it sees N independent timesteps."""
        n = self.config.repeated_diffusion_steps
        vdm_embeds = backbone_outputs.vdm_embeds.repeat(n, 1, 1)
        actions = inputs["action"].repeat(n, 1, 1)
        action_mask = inputs["action_mask"].repeat(n, 1, 1)
        state = inputs.get("state")
        if state is not None:
            state = state.repeat(n, 1, 1)
        return self.action_head(vdm_embeds=vdm_embeds, actions=actions, action_mask=action_mask, state=state)

    # ── SelfDistill ──

    def _compute_sd_loss(self, student_future_tokens: torch.Tensor, teacher_future_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-distillation on the future tokens: student logits = head(stu)/τ_s (grad-on);
        teacher targets = softmax((ema_head(tea) − center)/τ_t) (no grad). Center + head EMA update
        once per optimizer step (GA-aware). Returns (L_sd, teacher-entropy)."""
        student_logits = self.sd_student_head(student_future_tokens).float() / self.config.selfdistill_tau_student
        with torch.no_grad():
            def center_update(buf, batch_mean):
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(batch_mean)
                    batch_mean = batch_mean / torch.distributed.get_world_size()
                buf.mul_(0.9).add_(0.1 * batch_mean.to(buf.dtype))

            ga_last = self._is_last_ga_microbatch()  # None → per-forward update (GA=1); bool → last micro-batch?
            teacher_logits = self.sd_teacher_head(teacher_future_tokens)
            teacher_probs = torch.softmax((teacher_logits - self.sd_teacher_head.center) / self.config.selfdistill_tau_teacher, dim=-1)
            if ga_last is None:
                center_update(self.sd_teacher_head.center, teacher_logits.mean(dim=(0, 1)))
            else:
                self.sd_teacher_mean_acc.add(teacher_logits.mean(dim=(0, 1)).float(), rows=int(teacher_logits.shape[0] * teacher_logits.shape[1]))
                if ga_last:
                    center_update(self.sd_teacher_head.center, self.sd_teacher_mean_acc.pop())
            if self.training and (ga_last is None or ga_last):
                self.sd_teacher_head.ema_update(self.sd_student_head, self.config.selfdistill_head_ema_momentum)
        loss = -(teacher_probs * torch.log_softmax(student_logits, dim=-1)).sum(-1).mean()
        teacher_entropy = (-(teacher_probs * (teacher_probs + 1e-9).log()).sum(-1).mean()).detach()
        return loss, teacher_entropy

    @torch.no_grad()
    def _is_last_ga_microbatch(self):
        """None → per-forward update (eval, or GA=1); otherwise the bool the trainer stamped on the model before the
        forward (``grad_accum_is_last_micro_batch``: True only on the last micro-batch of the optimizer step; ``grad_accum_steps``)."""
        if not self.training:
            return None
        assert hasattr(self, "grad_accum_is_last_micro_batch") and hasattr(self, "grad_accum_steps"), (
            "`grad_accum_is_last_micro_batch`/`grad_accum_steps` were not stamped on the model before forward — FredTrainer.training_step does this."
        )
        if int(self.grad_accum_steps) <= 1:
            return None
        return bool(self.grad_accum_is_last_micro_batch)

    # ── forward ──

    def forward(self, inputs: dict) -> dict:
        inputs = self.prepare_input(inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone(inputs)
        with torch.autocast("cuda", dtype=torch.float32):
            action_outputs = self._compute_action_loss(backbone_outputs, inputs)
        total = action_outputs["loss"] * self.config.action_loss_weight
        logs = {
            "action_mse_loss": action_outputs["action_mse_loss"],
            "action_l1_loss": action_outputs["action_l1_loss"],
            "action_noise_level": (1 - action_outputs["action_t"]).mean().detach(),
        }
        if backbone_outputs.video_fm_loss is not None:   # a future clip was given: video FM + SelfDistill terms
            fm_loss = backbone_outputs.video_fm_loss
            total = total + fm_loss * self.config.video_loss_weight
            logs["fm_mse_loss"] = fm_loss.detach()
            student_future_tokens, teacher_future_tokens = backbone_outputs.sd_student_future, backbone_outputs.sd_teacher_future
            assert student_future_tokens is not None or not self.training, (
                "SelfDistill: the extractor returned no (student, teacher) future-token pair this step — the teacher path did not run."
            )
            if student_future_tokens is not None:
                sd_loss, teacher_entropy = self._compute_sd_loss(student_future_tokens, teacher_future_tokens)
                total = total + self.config.selfdistill_weight * sd_loss * self.config.video_loss_weight
                logs["sd_loss"] = sd_loss.detach()
                logs["sd_teacher_entropy"] = teacher_entropy
        logs["batch_loss"] = total.detach()
        return {"loss": total, "logs": logs}

    @torch.inference_mode()
    def get_action(self, inputs: dict, generator: torch.Generator | None = None) -> dict:
        """``{"action_pred": (B, action_horizon, action_dim)}`` normalized actions from the EMA action head."""
        inputs = self.prepare_input(inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone(inputs)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_head.predict_action(vdm_embeds=backbone_outputs.vdm_embeds, state=inputs.get("state"), generator=generator)
        return {"action_pred": actions}


