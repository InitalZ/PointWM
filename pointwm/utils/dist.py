from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def init_distributed():
    """torchrun -> (on, rank, world, local_rank); single process otherwise."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world, rank=rank)
        return True, rank, world, local_rank
    return False, 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)


def per_gpu_batch(global_batch: int, world: int) -> int:
    """Batch sizes in the paper are global across the GPUs (Supp. Sec. A.3)."""
    if global_batch % world:
        raise ValueError(f"global batch {global_batch} is not divisible by {world} GPUs")
    return global_batch // world
