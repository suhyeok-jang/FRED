import builtins

import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_global_zero() -> bool:
    return get_global_rank() == 0


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


def rank_zero_print(*args, **kwargs):
    if is_global_zero():
        builtins.print(*args, **kwargs)
