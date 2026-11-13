# `from __future__ import annotations` keeps the torch annotations below as strings, so this
# module costs numpy and nothing else to import. The two tensor helpers import torch when
# they are called: `flip_lps_ras` and LPS_TO_RAS are what the inference scripts need before
# printing their first line, and waiting a second and a half for torch to reach them is a
# cost nobody was buying anything with.
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = ["LPS_TO_RAS", "flip_lps_ras", "clamp_coords_zyx", "patch_grid_cells"]

# LPS and RAS differ in the sign of the first two axes only, so one factor converts either
# way; kept in one place since a wrong sign is a silent left/right swap
LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])


def flip_lps_ras(points_mm: np.ndarray) -> np.ndarray:
    """LPS millimetres <-> RAS millimetres (the conversion is its own inverse)."""
    return np.asarray(points_mm, dtype=float) * LPS_TO_RAS


# keeps coordinates inside [0, size-1] after every graph step and the refiner; NaNs map to
# 0 first, as in models/sampling.py normalize_coords
def clamp_coords_zyx(
    coords_zyx: torch.Tensor,
    shape_zyx: Sequence[int],
) -> torch.Tensor:
    import torch

    coords_zyx = torch.nan_to_num(coords_zyx, nan=0.0)
    Z, Y, X = (int(v) for v in shape_zyx)
    return torch.stack(
        (
            coords_zyx[..., 0].clamp(0, Z - 1),
            coords_zyx[..., 1].clamp(0, Y - 1),
            coords_zyx[..., 2].clamp(0, X - 1),
        ),
        dim=-1,
    )


# integer offsets of a PxPxP patch centred on 0 (P=3 -> -1, 0, 1 per axis), scaled by
# `pitch` where sampled; defined here because the refiner and its loss must lay out the same
# patch and neither owns the other
def patch_grid_cells(
    patch_size: int,
    device: torch.device,
    dtype: "torch.dtype | None" = None,
) -> torch.Tensor:
    import torch

    dtype = torch.float32 if dtype is None else dtype
    P = int(patch_size)
    ax = torch.arange(P, device=device, dtype=dtype) - (P - 1) / 2.0
    gz, gy, gx = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack((gz, gy, gx), dim=-1).reshape(-1, 3)
