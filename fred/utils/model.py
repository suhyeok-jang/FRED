"""Model-loading helpers: resolving a Hugging Face id to a local snapshot, reading a module's dtype."""

import os

import torch
import torch.nn as nn


def resolve_repo(model_id_or_path: str) -> str:
    """A local dir as is; a hub id resolves to the cached snapshot when complete (no API call), else the id."""
    repo = str(model_id_or_path)
    if os.path.isdir(repo):
        return repo
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        return snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001 — no complete local snapshot: let from_pretrained resolve the id
        return repo


def module_dtype(module: nn.Module, fallback: torch.dtype = torch.bfloat16) -> torch.dtype:
    for p in module.parameters():
        if p.is_floating_point():
            return p.dtype
    return fallback
