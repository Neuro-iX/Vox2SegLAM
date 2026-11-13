from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _min_std_for(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).resolution


# robust (lo, hi) bracket over brain voxels, estimated from a strided subsample; exact if
# the stride leaves too few samples for a stable quantile
def _brain_bracket(
    x: torch.Tensor,
    mask: torch.Tensor,
    probe_voxels: int,
    min_brain_voxels: int,
    pct: tuple[float, float],
):
    step = max(1, x.numel() // probe_voxels)
    values = x.reshape(-1)[::step][mask.reshape(-1)[::step]]
    if values.numel() < min_brain_voxels:
        values = x[mask]
    probe = values.float()
    return torch.quantile(probe, pct[0]), torch.quantile(probe, pct[1])


# uniform sample in [low, high], then re-centered to zero mean
def _sample_centered_uniform(
    low: float, high: float, shape: Sequence[int], device, dtype
) -> torch.Tensor:
    sample = torch.empty(*shape, device=device, dtype=dtype).uniform_(low, high)
    return sample - (low + high) / 2.0


# a 1x1x1 "conv" is a per-voxel random affine map (no spatial mixing)
class RandomConvLayer(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _sample_centered_uniform(0.0, 2.0, (1, 1, 1, 1, 1), x.device, x.dtype)
        bias = _sample_centered_uniform(0.0, 0.01, (1,), x.device, x.dtype)
        return F.conv3d(x, weight, bias)


# random-convolution contrast as in CABLD (Salari et al., ICCV 2025): random affine remaps
# chained through LeakyReLU make a random, possibly inverted, tone curve. Input and output
# are robustly rescaled with clipping at the brain's (lo, hi) percentiles, so the result
# stays within the original brain intensity range
class RandomConvStack(nn.Module):
    def __init__(
        self,
        probe_voxels: int,
        min_brain_voxels: int,
        pct: tuple[float, float],
        num_layers: int = 4,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.probe_voxels = int(probe_voxels)
        self.min_brain_voxels = int(min_brain_voxels)
        self.pct = (float(pct[0]), float(pct[1]))
        layers = []
        for _ in range(int(num_layers)):
            layers.append(RandomConvLayer())
            layers.append(nn.LeakyReLU(negative_slope=float(negative_slope)))
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eps = _min_std_for(x.dtype)
        lo, hi = _brain_bracket(x, mask, self.probe_voxels, self.min_brain_voxels, self.pct)
        span = (hi - lo).clamp_min(eps)

        # input: brain intensities rescaled to [0,1] at the (lo, hi) percentiles, clipped
        flat = ((x - lo) / span).clamp(0.0, 1.0).reshape(1, 1, -1, 1, 1)
        y = self.net(flat).reshape(x.shape)

        # output: the same robust rescale on the curve's own brain percentiles (an inverted curve
        # stays inverted), mapped back to the original (lo, hi)
        ylo, yhi = _brain_bracket(y, mask, self.probe_voxels, self.min_brain_voxels, self.pct)
        y = ((y - ylo) / (yhi - ylo).clamp_min(eps)).clamp(0.0, 1.0)
        return (y * span + lo) * mask  # background zeroed


# MONAI-style dict transform wrapping RandomConvStack; "RC" = random-conv-stack contrast
class RandRCContrastd:
    def __init__(
        self,
        keys: Sequence[str],
        prob: float,
        mask_key: str = "brain",
        probe_voxels: int = 1 << 16,
        min_brain_voxels: int = 100,
        pct: tuple[float, float] = (0.005, 0.995),
    ) -> None:
        self.keys = tuple(keys)
        self.prob = float(prob)
        self.mask_key = str(mask_key)
        self.stack = RandomConvStack(probe_voxels, min_brain_voxels, pct)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        if self.prob <= 0.0 or torch.rand(()).item() >= self.prob:
            return d
        if self.mask_key not in d:
            raise KeyError(f"{type(self).__name__} needs the brain mask under {self.mask_key!r}")
        mask = d[self.mask_key]

        for key in self.keys:
            img = d[key]
            out = self.stack(img.float().unsqueeze(0), mask.unsqueeze(0))
            d[key] = out.squeeze(0).to(img.dtype)

        return d
