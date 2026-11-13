import torch
import torch.nn as nn

__all__ = ["FiLM"]


# Feature-wise Linear Modulation: a conditioning vector (landmark identity + pooled GNN state)
# predicts a per-channel scale and shift applied over the spatial dims, telling the refiner's
# conv tower which landmark's patch it is refining
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * num_channels)
        nn.init.normal_(
            self.proj.weight, std=0.02
        )  # small init: starts close to identity (gamma~0, beta~0)
        nn.init.zeros_(self.proj.bias)
        self.c = num_channels

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return x * (1.0 + gamma).view(-1, self.c, 1, 1, 1) + beta.view(-1, self.c, 1, 1, 1)
