import math

import torch

__all__ = ["wing"]


# Wing loss (Feng et al., 2018): log-shaped (more gradient than L2) for small errors
# |diff| < width, linear (like L1) beyond it - more sensitive to the sub-pixel errors
# that matter for landmark localisation, without L2's blow-up on the rare large miss.
# `c` is the constant that makes the two pieces meet with equal value at |diff|=width.
def wing(diff: torch.Tensor, width: float, epsilon: float) -> torch.Tensor:
    if width <= 0.0 or epsilon <= 0.0:
        raise ValueError(f"wing needs width>0 and epsilon>0, got {width}/{epsilon}")
    d = diff.abs()
    c = width - width * math.log(1.0 + width / epsilon)
    return torch.where(d < width, width * torch.log(1.0 + d / epsilon), d - c)
