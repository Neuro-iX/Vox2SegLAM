import random
from typing import Any
from collections.abc import Sequence

import numpy as np
import torch
from monai.data import set_rnd
from torch.utils.data import DataLoader, get_worker_info

from ..utils.device import worker_start_method
from .datasets import build_dataset, read_split

__all__ = ["build_dataloader", "build_dataloaders", "reseed_for_epoch"]


# without this every worker inherits the same RNG state and augments identically
def _worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)

    info = get_worker_info()
    if info is not None:
        set_rnd(info.dataset, int(seed))


def build_dataloader(cfg, split: str) -> DataLoader:
    dataset = build_dataset(cfg, split)
    is_train = split == "train"

    # train and eval have separate batch_size/num_workers; only train shuffles and drops its
    # partial last batch (eval must see every subject, in order)
    num_workers = int(cfg.data.num_workers) if is_train else cfg.data.resolved_eval_num_workers()
    kwargs: dict[str, Any] = dict(
        batch_size=int(cfg.data.batch_size if is_train else cfg.data.eval_batch_size),
        shuffle=is_train,
        num_workers=num_workers,
        # pinned memory only means anything with an accelerator to copy to; asking for it
        # without one is ignored by torch, loudly, once per worker
        pin_memory=bool(cfg.data.pin_memory) and torch.cuda.is_available(),
        drop_last=bool(cfg.data.drop_last) if is_train else False,
    )

    # only relevant with subprocess workers
    if num_workers > 0:
        # forking a process that runs CUDA/pin-memory threads can deadlock; forkserver starts clean
        # workers (seeding is unchanged: torch.initial_seed() + set_rnd in _worker_init_fn)
        kwargs["multiprocessing_context"] = worker_start_method()
        # a persistent training worker keeps its augmentation RNG across epochs, so a resumed run
        # would augment differently; deterministic runs re-create and re-seed workers every epoch
        persistent = bool(cfg.data.persistent_workers)
        if is_train and cfg.repro.deterministic:
            persistent = False
        kwargs["persistent_workers"] = persistent
        if cfg.data.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
        if cfg.repro.seed_workers:
            kwargs["worker_init_fn"] = _worker_init_fn
    elif cfg.repro.seed_workers:
        # with num_workers=0 there is no worker to seed: seed the dataset's own MONAI RNG directly
        set_rnd(dataset, int(cfg.repro.seed))

    if is_train:
        # seeds the shuffle order itself, independent of augmentation RNG seeding above
        generator = torch.Generator()
        generator.manual_seed(int(cfg.repro.seed))
        kwargs["generator"] = generator

    return DataLoader(dataset, **kwargs)


# called every epoch: shuffle order, worker base seeds and (without workers) the dataset's
# augmentation RNG depend only on (seed, epoch), so a resumed run reproduces the original
def reseed_for_epoch(loader: DataLoader, seed: int, epoch: int) -> None:
    if loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)
    if loader.num_workers == 0:
        set_rnd(loader.dataset, seed + epoch)


def build_dataloaders(cfg, splits: Sequence[str] = ("train", "val")) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in splits:
        ids = read_split(cfg.paths.resolve_split(split))
        if not ids:
            # an empty split file (e.g. no test set for this dataset) is skipped, not an error
            continue
        loaders[split] = build_dataloader(cfg, split)
    return loaders
