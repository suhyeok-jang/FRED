"""Array helpers shared by FredPolicy and FredSimPolicyWrapper."""

from typing import Any

import numpy as np
import torch


def to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Cast the floating-point tensors of a nested dict/list structure to ``dtype``; other values are untouched."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    if isinstance(x, dict):
        return {k: to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [to_dtype(v, dtype) for v in x]
    return x


def check_arrays(
    arrays: dict[str, Any],
    keys: list[str],
    horizon: int,
    dtype: type,
    ndim: int,
    kind: str,
    batch_size: int | None = None,
    channels: int | None = None,
) -> int:
    """Assert every key of ``keys`` is in ``arrays`` as an np.ndarray of ``dtype`` with ``ndim`` dims, batch
    ``batch_size`` (taken from the first array when None), horizon ``horizon`` and, when given, ``channels``
    last. Returns the batch size."""
    for key in keys:
        assert key in arrays, f"{kind} key '{key}' is missing"
        arr = arrays[key]
        assert isinstance(arr, np.ndarray), f"{kind} key '{key}' must be a numpy array. Got {type(arr)}"
        assert arr.dtype == dtype, f"{kind} key '{key}' must have dtype {dtype.__name__}. Got {arr.dtype}"
        assert arr.ndim == ndim, f"{kind} key '{key}' must have {ndim} dims (B, T, ...). Got shape {arr.shape}"
        if batch_size is None:
            batch_size = arr.shape[0]
        assert arr.shape[0] == batch_size, f"{kind} key '{key}' must have batch size {batch_size}. Got {arr.shape[0]}"
        assert arr.shape[1] == horizon, f"{kind} key '{key}' must have horizon {horizon}. Got {arr.shape[1]}"
        assert channels is None or arr.shape[-1] == channels, (
            f"{kind} key '{key}' must have {channels} channels. Got {arr.shape[-1]}"
        )
    return batch_size
