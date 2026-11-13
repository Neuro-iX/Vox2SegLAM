import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ensure_dir", "save_json", "load_nifti", "save_nifti", "to_zyx"]


# idempotent mkdir -p; returns the Path so it can be chained
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


# NIfTI is stored XYZ, everything inside vox2seglam works in ZYX: the conversion point right
# after load_nifti, and its inverse before save_nifti
def to_zyx(arr_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(arr_xyz, (2, 1, 0))


# raw load: returns the array in nibabel's native axis order (XYZ) and the affine --
# NOT yet converted to ZYX, callers apply to_zyx themselves (see data/datasets.py)
def load_nifti(path: str | Path, dtype=None):
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr, img.affine


# expects XYZ order (the caller converts back from ZYX first); default dtype int16
# suits label/segmentation volumes, not raw intensities
def save_nifti(arr_xyz: np.ndarray, affine: np.ndarray, path: str | Path, dtype=np.int16) -> None:
    import nibabel as nib

    path = Path(path)
    ensure_dir(path.parent)
    nib.save(nib.Nifti1Image(arr_xyz.astype(dtype), affine), str(path))
