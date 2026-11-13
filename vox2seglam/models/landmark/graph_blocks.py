import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GraphConv", "Features2FeaturesResidual", "small_delta_head_init"]


# one message-passing step: output(v) = w0(v) + mean over v's neighbours of w1(neighbour).
class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w0 = nn.Linear(input_dim, output_dim)
        self.w1 = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.w0.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.w0.bias)
        nn.init.zeros_(self.w1.bias)

    def forward(self, verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if verts.is_cuda != edges.is_cuda:
            raise ValueError("verts and edges must be on the same device")
        w1 = self.w1(verts)
        # directed edge (dst, src): dst sums w1 of its sources (same as pytorch3d's gather_scatter)
        neighbours = torch.zeros_like(w1).index_add_(0, edges[:, 0], w1[edges[:, 1]])
        degree = torch.bincount(edges[:, 0], minlength=verts.shape[0]).clamp_(min=1)
        return self.w0(verts) + neighbours / degree.unsqueeze(1)


# GraphConv's equivalent of ResidualBlock: a stack of graph-conv layers (first +
# n hidden) with a residual connection from input to output (projected if the
# dimension changes)
class Features2FeaturesResidual(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layer_count: int,
        norm: str = "layer",
    ) -> None:
        if norm not in ("none", "layer", "batch"):
            raise ValueError(f"norm must be 'none', 'layer' or 'batch', got '{norm}'")
        super().__init__()

        def _norm(dim: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm1d(dim)
            if norm == "layer":
                return nn.LayerNorm(dim)
            return nn.Identity()

        self.res_proj = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else None
        )

        self.gconv_first = GraphConv(in_features, out_features)
        self.norm_first = _norm(out_features)

        self.gconv_hidden = nn.ModuleList(
            [GraphConv(out_features, out_features) for _ in range(hidden_layer_count)]
        )
        self.norm_hidden = nn.ModuleList([_norm(out_features) for _ in range(hidden_layer_count)])

    def forward(
        self,
        features: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        res = features if self.res_proj is None else self.res_proj(features)

        x = self.norm_first(self.gconv_first(features, edges))

        if not self.gconv_hidden:
            return F.relu(x + res)

        x = F.relu(x)
        for gconv, norm in zip(self.gconv_hidden[:-1], self.norm_hidden[:-1]):
            x = F.relu(norm(gconv(x, edges)))

        return F.relu(self.norm_hidden[-1](self.gconv_hidden[-1](x, edges)) + res)


# applied (via .apply()) to a head predicting a *delta* (coordinate offset): near-zero
# weights and zero bias make the head start out predicting ~0, i.e. "don't move yet",
# which is a far more stable starting point than a random large initial displacement
def small_delta_head_init(m: nn.Module, std: float = 1e-3) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, GraphConv):
        nn.init.normal_(m.w0.weight, mean=0.0, std=std)
        nn.init.constant_(m.w0.bias, 0.0)
        nn.init.normal_(m.w1.weight, mean=0.0, std=std)
        nn.init.constant_(m.w1.bias, 0.0)
