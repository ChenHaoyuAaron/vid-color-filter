import os
import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize torch.distributed if launched via torchrun, else single-process fallback.

    Returns:
        (rank, world_size, device)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, device


def shard_pairs(
    pairs: list[tuple[str, str]],
    rank: int,
    world_size: int,
) -> list[tuple[str, str]]:
    """Evenly split video pairs across ranks.

    Each rank gets pairs[rank::world_size], ensuring no overlap.
    """
    return pairs[rank::world_size]


def cleanup():
    """Destroy the distributed process group if active."""
    if dist.is_initialized():
        dist.destroy_process_group()
