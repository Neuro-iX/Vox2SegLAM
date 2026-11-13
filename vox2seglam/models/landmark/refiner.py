import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.geometry import patch_grid_cells
from .film import FiLM

__all__ = ["RefinerHead", "decode"]


# local sub-voxel correction: from a patch sampled around each landmark's current position,
# a soft heatmap over patch cells plus a per-cell offset, decoded by decode() below
class RefinerHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_landmarks: int,
        gnn_dim: int,
        hidden: int = 96,
        n_conv_layers: int = 3,
        embed_dim: int = 64,
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.K = int(n_landmarks)

        # per-landmark identity embedding (which of the K landmarks this patch is for),
        # combined with the GNN's current latent state into the FiLM conditioning vector
        self.embed = nn.Embedding(self.K, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.embed_norm = nn.LayerNorm(embed_dim)
        self.gnn_norm = nn.LayerNorm(gnn_dim)

        self.cond = nn.Sequential(
            nn.Linear(embed_dim + gnn_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.ReLU(inplace=True),
        )

        # 1x1x1 conv to mix the sampled multi-scale backbone channels down to `hidden`
        self.reduce = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.reduce_norm = nn.InstanceNorm3d(hidden, affine=True)
        self.reduce_film = FiLM(cond_dim, hidden)

        # conv tower proper: each layer is conv -> norm -> FiLM(landmark/GNN cond) -> relu
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        for _ in range(n_conv_layers):
            self.convs.append(nn.Conv3d(hidden, hidden, kernel_size=3, padding=1))
            self.norms.append(nn.InstanceNorm3d(hidden, affine=True))
            self.films.append(FiLM(cond_dim, hidden))

        # cls_head: 1 logit per patch cell (a heatmap); reg_head: a 3D offset per cell
        self.cls_head = nn.Conv3d(hidden, 1, kernel_size=1)
        self.reg_head = nn.Conv3d(hidden, 3, kernel_size=1)

        nn.init.normal_(self.cls_head.weight, std=1e-2)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.normal_(self.reg_head.weight, std=1e-3)
        nn.init.zeros_(self.reg_head.bias)

    # every (batch, landmark) patch is independent: B and K fold into one batch dimension
    def forward(self, patches: torch.Tensor, gnn_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        B, K, C, P, _, _ = patches.shape
        x = patches.reshape(B * K, C, P, P, P)

        ids = torch.arange(K, device=patches.device).repeat(B)
        cond = self.cond(
            torch.cat(
                [
                    self.embed_norm(self.embed(ids)),
                    self.gnn_norm(gnn_latent.reshape(B * K, -1).float()),
                ],
                dim=1,
            )
        )

        x = F.relu(self.reduce_film(self.reduce_norm(self.reduce(x)), cond))
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = F.relu(film(norm(conv(x)), cond))

        return {
            "logits": self.cls_head(x).reshape(B, K, P**3),
            "offsets": self.reg_head(x).reshape(B, K, 3, P**3),
        }


# soft-argmax: per-cell heatmap + offsets -> one continuous coordinate. `coarse` is the
# probability-weighted cell centre (sub-cell precision), `fine` the probability-weighted
# offset within the cell; `pitch` converts patch-cell units to backbone-input voxels. `probs`
# is also returned (the KL-loss target and the outside_frac diagnostic).
def decode(
    logits: torch.Tensor,
    offsets: torch.Tensor,
    patch_size: int,
    pitch: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    cells = patch_grid_cells(patch_size, logits.device)
    coarse = torch.einsum("bkn,nd->bkd", probs, cells)
    fine = torch.einsum("bkn,bkdn->bkd", probs, offsets.float())
    return (coarse + fine) * float(pitch), probs
