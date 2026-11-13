import torch


# z-score over every voxel, including background - used when preprocess.zscore_brain_masked=False
def normalize_zscore_volume(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(eps)


# z-score over brain voxels only (x>0, or an explicit mask)
def normalize_zscore_brain_masked(
    x: torch.Tensor, eps: float = 1e-8, mask: torch.Tensor | None = None
) -> torch.Tensor:
    x = x.float()
    brain = x[x > 0] if mask is None else x[mask]
    if brain.numel() == 0:
        return normalize_zscore_volume(x, eps)
    return (x - brain.mean()) / brain.std(unbiased=False).clamp_min(eps)
