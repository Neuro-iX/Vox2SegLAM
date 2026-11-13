"""The data layer: preparation, datasets, loaders, augmentation.

Imported lazily. `vox2seglam.data.prepare.register` is what the inference scripts need, and
reaching it through this package used to pull in the augmentation module, hence MONAI, hence
torch -- three seconds of silence before a command-line tool could print its first line. The
names below still resolve on attribute access (PEP 562), so `from vox2seglam.data import
build_dataloaders` works unchanged; only the cost moved to the first use.
"""

import importlib
from typing import TYPE_CHECKING

_SOURCE = {
    "TrainAugmenter": "augmentation",
    "build_train_augmenter": "augmentation",
    "Vox2SegLAMDataset": "datasets",
    "build_dataset": "datasets",
    "read_split": "datasets",
    "ensure_prepared": "prepare",
    "build_dataloader": "loaders",
    "build_dataloaders": "loaders",
    "preprocess_sample_eval": "sample",
    "preprocess_sample_train": "sample",
}

__all__ = sorted(_SOURCE)

if TYPE_CHECKING:  # for type checkers and editors only
    from .augmentation import TrainAugmenter, build_train_augmenter  # noqa: F401
    from .datasets import Vox2SegLAMDataset, build_dataset, read_split  # noqa: F401
    from .loaders import build_dataloader, build_dataloaders  # noqa: F401
    from .prepare import ensure_prepared  # noqa: F401
    from .sample import preprocess_sample_eval, preprocess_sample_train  # noqa: F401


def __getattr__(name: str):
    module = _SOURCE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # resolved once, then a plain global
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SOURCE))
