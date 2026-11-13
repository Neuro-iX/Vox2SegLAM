"""Small shared helpers: NIfTI and JSON I/O, coordinate conventions.

Imported lazily (PEP 562). `geometry` holds two tensor helpers and so needs torch, which made
`from vox2seglam.utils.io import save_json` -- all a command line wants before it prints its
first line -- pay a second and a half for it. Every name below still resolves on attribute
access, so `from vox2seglam.utils import flip_lps_ras` works unchanged.
"""

import importlib
from typing import TYPE_CHECKING

_SOURCE = {
    "LPS_TO_RAS": "geometry",
    "clamp_coords_zyx": "geometry",
    "flip_lps_ras": "geometry",
    "patch_grid_cells": "geometry",
    "ensure_dir": "io",
    "load_nifti": "io",
    "save_json": "io",
    "save_nifti": "io",
    "to_zyx": "io",
}

__all__ = sorted(_SOURCE)

# for type checkers and editors only: the names __getattr__ serves lazily at runtime, so
# every one of them is unused here by construction
if TYPE_CHECKING:  # noqa: F401
    from .geometry import (  # noqa: F401
        LPS_TO_RAS,
        clamp_coords_zyx,
        flip_lps_ras,
        patch_grid_cells,
    )
    from .io import ensure_dir, load_nifti, save_json, save_nifti, to_zyx  # noqa: F401


def __getattr__(name: str):
    module = _SOURCE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SOURCE))
