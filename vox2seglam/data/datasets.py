import os
from typing import Any
from collections.abc import Sequence

import numpy as np
from torch.utils.data import Dataset

from ..utils.io import load_nifti
from .augmentation import build_train_augmenter
from .prepare import ensure_prepared
from .sample import preprocess_sample_eval, preprocess_sample_train

__all__ = ["Vox2SegLAMDataset", "build_dataset", "read_split"]


# one subject id per line; blank lines and "#"-comments are skipped
def read_split(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"split file not found: {path}\n"
            "Check paths.splits_dir and paths.split_* in your configuration; leave "
            "splits_dir empty to use the subject lists of the published runs."
        )
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.lstrip().startswith("#")]


class Vox2SegLAMDataset(Dataset):
    def __init__(self, ids: Sequence[str], cfg, train: bool) -> None:
        self.ids = list(ids)
        self.train = bool(train)

        # runs skull-stripping/N4 once per subject if not already cached; returns
        # wherever the prepared (not raw) volumes ended up (see data/prepare/)
        self.image_dir, self.image_suffix = ensure_prepared(cfg, self.ids)

        # seg volumes are only looked up/loaded when the model actually has a seg head
        self.use_seg = bool(cfg.model.seg_head.enabled)
        self.seg_dir = cfg.paths.resolve_seg_dir() if self.use_seg else None
        self.landmark_dir = cfg.paths.resolve_landmark_dir()
        self.seg_suffix = cfg.paths.seg_suffix
        self.landmark_suffix = cfg.paths.landmark_suffix

        self.landmark_ids = cfg.task.resolved_landmark_ids()
        self.zscore_brain_masked = bool(cfg.preprocess.zscore_brain_masked)
        self.zscore_eps = float(cfg.preprocess.zscore_eps)
        self.max_attempts = int(cfg.aug.affine_max_attempts)

        self.augmenter = build_train_augmenter(cfg.aug) if self.train else None  # None in eval

    def __len__(self) -> int:
        return len(self.ids)

    # the on-disk path for each volume this subject needs; "seg" only when use_seg
    def _paths(self, pid: str) -> dict[str, str]:
        paths = {
            "image": os.path.join(self.image_dir, f"{pid}{self.image_suffix}"),
            "landmarks": os.path.join(self.landmark_dir, f"{pid}{self.landmark_suffix}"),
        }
        if self.use_seg:
            paths["seg"] = os.path.join(self.seg_dir, f"{pid}{self.seg_suffix}")
        return paths

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pid = self.ids[idx]
        paths = self._paths(pid)
        # fail loudly here with the actual missing path, rather than a generic
        # nibabel/IO error once load_nifti is reached below
        for kind, path in paths.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"subject {pid!r}: missing {kind} volume at {path}. Check "
                    "paths.data_root and the *_dir / *_suffix settings."
                )

        image_xyz, _ = load_nifti(paths["image"], dtype=np.float32)
        seg_xyz = load_nifti(paths["seg"])[0] if self.use_seg else None
        lm_xyz, _ = load_nifti(paths["landmarks"])

        # train applies augmentation and a retry budget (max_attempts) for affine
        # transforms that might move a landmark out of frame; eval is deterministic
        if self.train:
            sample = preprocess_sample_train(
                image_xyz,
                seg_xyz,
                lm_xyz,
                landmark_ids=self.landmark_ids,
                augmenter=self.augmenter,
                zscore_brain_masked=self.zscore_brain_masked,
                max_attempts=self.max_attempts,
                zscore_eps=self.zscore_eps,
            )
        else:
            sample = preprocess_sample_eval(
                image_xyz,
                seg_xyz,
                lm_xyz,
                landmark_ids=self.landmark_ids,
                zscore_brain_masked=self.zscore_brain_masked,
                zscore_eps=self.zscore_eps,
            )
        sample["pid"] = pid
        return sample


# only the "train" split gets augmentation/retries; everything else is eval-mode
def build_dataset(cfg, split: str) -> Vox2SegLAMDataset:
    ids = read_split(cfg.paths.resolve_split(split))
    return Vox2SegLAMDataset(ids=ids, cfg=cfg, train=(split == "train"))
