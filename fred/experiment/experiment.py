import datetime
import json
import logging
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from transformers import TrainingArguments, set_seed

from fred.configs.base_config import Config
from fred.experiment.pipeline import build_dataset, build_model, build_processor
from fred.experiment.trainer import CheckpointFormatCallback, FredTrainer

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def git_commit() -> str:
    """HEAD of the repo this package is imported from, or "unknown" (bookkeeping never blocks a run)."""
    try:
        r = subprocess.run(("git", "rev-parse", "HEAD"), cwd=str(Path(__file__).resolve().parents[2]),
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def init_distributed():
    """torchrun launch (WORLD_SIZE > 1): NCCL process group with a 20 h timeout; otherwise single process."""
    if dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    timeout = datetime.timedelta(seconds=72000)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=timeout)
    # torch's new_group() (DeepSpeed ZeRO builds its grad-comm subgroup with it) would otherwise fall back to the
    # 600 s c10d default, which times out when one rank stalls at an allreduce; give it the main group's timeout
    orig_new_group = dist.new_group

    def _new_group_with_timeout(*a, **kw):
        kw.setdefault("timeout", timeout)
        return orig_new_group(*a, **kw)

    dist.new_group = _new_group_with_timeout


def run(config: Config):
    """Build model / processor / dataset from ``config`` and run FredTrainer to completion."""
    init_distributed()
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    setup_logging()
    set_seed(config.data.seed)
    config.validate()

    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    config.save(save_cfg_dir / "config.yaml")
    with open(output_dir / "wandb_config.json", "w") as f:
        json.dump({"project": config.training.wandb_project, "run_id": experiment_name}, f)
    logger.info(f"\n[i] Saved config to {save_cfg_dir}")

    if config.training.use_wandb and global_rank == 0:
        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config={**config.__dict__, "git_commit_hash": git_commit()},
            settings=wandb.Settings(init_timeout=300),   # the 90 s default times out on slow connects to wandb.ai
        )

    model = build_model(config, save_cfg_dir)
    processor = build_processor(config)
    train_dataset = build_dataset(config, processor, save_cfg_dir)
    processor.save_pretrained(processor_dir)
    logger.info(f"[i] Model type: {type(model)}")

    if config.training.num_gpus > 1:   # DeepSpeed ZeRO for every multi-GPU run; one GPU runs the plain Trainer
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None

    per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    logger.info(f"per_device_train_batch_size: {per_device_train_batch_size}")

    t = config.training
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=t.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        # cosine decay to peak × min_lr_ratio per param group (HF cosine_with_min_lr), warmup by steps
        lr_scheduler_kwargs={"min_lr_rate": float(t.min_lr_ratio)},
        weight_decay=t.weight_decay,   # applied uniformly to every param group (FredTrainer.create_optimizer)
        warmup_steps=t.warmup_steps,
        adam_beta1=t.adam_beta1,
        adam_beta2=t.adam_beta2,
        adam_epsilon=t.adam_epsilon,
        max_grad_norm=t.max_grad_norm,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        bf16=t.bf16,
        tf32=t.tf32,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",   # unfused AdamW, as every published run (fused changes the reduction order)
        dataloader_num_workers=t.dataloader_num_workers,
        report_to="wandb" if t.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_timeout=72000,
        batch_eval_metrics=True,
        remove_unused_columns=False,   # our inputs are nested dicts, not model-signature columns
        ignore_data_skip=True,
    )

    trainer = FredTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=processor.collator,
        multiprocessing_context=config.data.multiprocessing_context,
        prefetch_factor=config.data.prefetch_factor,
        lr_per_module_enabled=t.lr_per_module_enabled,
        lr_base=t.lr_base,
        lr_backbone=t.lr_backbone,
        lr_action_head=t.lr_action_head,
    )
    trainer.add_callback(CheckpointFormatCallback(exp_cfg_dir=save_cfg_dir, processor_dir=processor_dir))

    logger.info("🚀 Starting training...")
    trainer.train(resume_from_checkpoint=t.resume_from_checkpoint)

    trainer.save_model()
    logger.info(f"Model saved to {output_dir}")
    logger.info("🚀 Training completed!")
