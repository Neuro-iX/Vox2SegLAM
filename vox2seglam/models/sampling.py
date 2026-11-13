from collections.abc import Sequence

import torch

from ..utils.geometry import patch_grid_cells

__all__ = [
    "normalize_coords",
    "aggregate_from_indices",
    "sample_patches_multiscale",
]

ShapeZYX = tuple[int, int, int]


# continuous ZYX voxel coordinates -> the [-1, 1] XYZ grid of torch's grid_sample (reversed
# axes, align_corners=False: a voxel centre maps to (2*i+1)/size - 1); NaNs map to 0
def normalize_coords(
    coords_zyx: torch.Tensor,
    full_shape_zyx: ShapeZYX,
) -> torch.Tensor:
    Z, Y, X = (float(v) for v in full_shape_zyx)
    c = torch.nan_to_num(coords_zyx.float(), nan=0.0)
    grid = torch.stack(
        (
            (2.0 * c[..., 2] + 1.0) / X - 1.0,
            (2.0 * c[..., 1] + 1.0) / Y - 1.0,
            (2.0 * c[..., 0] + 1.0) / Z - 1.0,
        ),
        dim=-1,
    )
    return grid[:, :, None, None, :]


# trilinear lookup at continuous points (B, N, 3 normalised XYZ); out-of-bounds coordinates
# clamp to the border instead of returning zero. Equal to F.grid_sample(mode="bilinear",
# padding_mode="border", align_corners=False) but written with gather, whose backward has a
# deterministic CUDA implementation (grid_sample's has none; repro.deterministic)
def _sample(voxel_features: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    if voxel_features.ndim != 5:
        raise ValueError("voxel_features must be (B,C,Z,Y,X)")
    B, C, *size = voxel_features.shape
    points = grid.reshape(B, -1, 3).flip(-1)  # (B, N, 3) ZYX
    corners, weights = [], []
    for axis, n in enumerate(size):
        pos = ((points[..., axis] + 1.0) * n - 1.0) / 2.0  # voxel position
        pos = pos.clamp(0, n - 1)
        low = pos.floor()
        frac = (pos - low).to(voxel_features.dtype)
        low = low.long()
        corners.append((low, (low + 1).clamp(max=n - 1)))
        weights.append((1.0 - frac, frac))
    flat = voxel_features.reshape(B, C, -1)
    out = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                index = (corners[0][dz] * size[1] + corners[1][dy]) * size[2] + corners[2][dx]
                value = flat.gather(2, index[:, None, :].expand(B, C, -1))
                out = out + value * (weights[0][dz] * weights[1][dy] * weights[2][dx])[:, None]
    return out.transpose(1, 2)  # (B, N, C)


# samples several backbone feature maps (one per skip index) at the same K points and
# concatenates them channel-wise; feeds each GNN step (model.lam.aggregate_indices)
def aggregate_from_indices(
    voxel_features: Sequence[torch.Tensor],
    coords_zyx: torch.Tensor,
    skip_indices: Sequence[int],
    full_shape_zyx: ShapeZYX,
) -> torch.Tensor:
    if coords_zyx.ndim != 3 or coords_zyx.shape[-1] != 3:
        raise ValueError("coords_zyx must be (B,K,3)")
    grid = normalize_coords(coords_zyx, full_shape_zyx)
    return torch.cat([_sample(voxel_features[i], grid) for i in skip_indices], dim=2)


# the refiner's input: a full PxPxP patch around each landmark's current estimate at every
# requested backbone scale. centers_fullres is in backbone-input voxels; `pitch` is the
# spacing between patch cells, so a patch can cover more than 1:1 voxels
def sample_patches_multiscale(
    voxel_features: Sequence[torch.Tensor],
    centers_fullres: torch.Tensor,
    skip_indices: Sequence[int],
    patch_size: int,
    pitch: float,
    full_shape_zyx: ShapeZYX,
) -> torch.Tensor:
    if centers_fullres.ndim != 3 or centers_fullres.shape[-1] != 3:
        raise ValueError("centers_fullres must be (B,K,3)")

    B, K, _ = centers_fullres.shape
    P = int(patch_size)
    N = K * P * P * P

    cells = patch_grid_cells(P, centers_fullres.device)
    pos = centers_fullres.float().unsqueeze(2) + cells * float(pitch)
    grid = normalize_coords(pos.reshape(B, N, 3), full_shape_zyx)

    out = torch.cat([_sample(voxel_features[i], grid) for i in skip_indices], dim=2)
    # (B, K*P^3, C) -> (B, K, C, P, P, P): one multi-scale-feature cube per landmark
    return out.view(B, K, P, P, P, -1).permute(0, 1, 5, 2, 3, 4).contiguous()
