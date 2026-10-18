from dataclasses import dataclass
from typing import Annotated

import tyro


@dataclass
class TrainingConfig:
    """Optimizer, schedule, checkpointing, hardware."""

    # output
    output_dir: str = "./outputs"
    experiment_name: str | None = None

    # schedule
    max_steps: int = 100000
    global_batch_size: int = 64          # clips per micro-batch over all GPUs
    gradient_accumulation_steps: int = 1

    # optimization
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine_with_min_lr"
    weight_decay: float = 0.01
    warmup_steps: int = 5000
    max_grad_norm: float = 1.0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # per-module LR: video DiT / action head / rest (otherwise the single learning_rate)
    lr_per_module_enabled: bool = False
    lr_base: float = 3e-5
    lr_backbone: float = 1e-5
    lr_action_head: float = 1e-4

    base_model_path: Annotated[str | None, tyro.conf.Suppress] = None   # set by launch_train from --base-model-path

    # precision
    tf32: bool = True
    bf16: bool = True

    # logging / checkpoints
    logging_steps: int = 10
    save_steps: int = 10000
    save_total_limit: int = 5

    gradient_checkpointing: bool = False
    min_lr_ratio: float = 0.01           # cosine floor = ratio × peak lr (sim 0.01, real robot 0.1)
    warmstart_min_coverage: float = 0.5  # a warm start must fill at least this fraction of the parameters
    deepspeed_load_universal: bool = False      # DeepSpeed universal checkpoint (resume with a different GPU count)
    resume_from_checkpoint: bool = True

    # hardware
    num_gpus: int = 1
    dataloader_num_workers: int = 2

    # wandb
    use_wandb: bool = True
    wandb_project: str = "train-fred"
