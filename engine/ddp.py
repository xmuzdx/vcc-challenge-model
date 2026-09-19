"""DDP 启动与 rank-0 守卫：只在主进程保存 checkpoint。"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup(local_rank: int | None = None) -> tuple[int, int, torch.device]:
    """返回 (rank, world, device)。未启用时退回单卡 cuda:0。"""
    if not enabled():
        gpu = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        return 0, 1, torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    if local_rank is not None:
        local = local_rank
    torch.cuda.set_device(local)
    return rank, world, torch.device(f"cuda:{local}")


def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_flag(flag: bool, device: torch.device) -> bool:
    """rank0 的布尔决策同步到所有进程，避免早停死锁。"""
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.broadcast(t, src=0)
    return bool(int(t.item()))


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap(model: torch.nn.Module) -> torch.nn.Module:
    if not enabled() or not dist.is_initialized():
        return model
    local = int(os.environ.get("LOCAL_RANK", 0))
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local], output_device=local, find_unused_parameters=False,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_best(path, payload: dict) -> None:
    if is_rank0():
        torch.save(payload, path)
