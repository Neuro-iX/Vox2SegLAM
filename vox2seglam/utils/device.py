import multiprocessing
import os
from typing import Any

import torch

__all__ = ["to_device", "worker_start_method", "deterministic_inference"]

# deterministic cuBLAS needs a fixed workspace, read when CUDA initialises: set on import,
# before any model reaches the GPU (training/seeding.py does the same for a training run)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# A prediction has no RNG to seed - no augmentation, no shuffling - but the reductions that
# make a training step vary make a prediction vary too: about 1e-4 voxel on the landmarks
# between two runs of the same command, which is enough to move a boundary voxel and change
# a Dice in its fifth decimal. One scan is cheap, so it is made exact instead.
def deterministic_inference() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def worker_start_method() -> str:
    """forkserver where it exists (Linux, macOS), spawn otherwise (Windows)."""
    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


# moves every tensor value in a batch dict to `device`, leaving non-tensor entries
# (e.g. "pid", the subject id string) untouched; non_blocking lets the copy overlap
# with compute when the source is pinned memory (data.pin_memory=True)
def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()
    }
