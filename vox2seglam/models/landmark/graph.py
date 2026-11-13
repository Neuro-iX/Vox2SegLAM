from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.amp import autocast

from ...utils.geometry import clamp_coords_zyx
from ...utils.logging import measure_time
from ..sampling import aggregate_from_indices
from .graph_blocks import GraphConv, Features2FeaturesResidual, small_delta_head_init
from .template import GraphTemplate

__all__ = ["GraphSpec", "LandmarkGraphNet"]


# mirrors model.lam plus the backbone channel tuples that size each step's input
@dataclass(frozen=True)
class GraphSpec:
    K: int
    patch_size: tuple[int, int, int]
    encoder_channels: tuple[int, ...]
    decoder_channels: tuple[int, ...]
    graph_channels: tuple[int, ...]
    steps: int
    aggregate_indices: tuple[tuple[int, ...], ...]
    norm: str = "layer"
    clamp_coords: bool = True
    n_residual_blocks: int = 3
    n_f2f_hidden_layer: int = 2
    detach_between_steps: bool = True


# GraphConv works on one flat graph: tile the (2, E) edge list B times with node offsets of
# K, so B disconnected K-node graphs form one B*K-node graph (block-diagonal batching)
def _edges_batched(edge_index_2e: torch.Tensor, B: int, K: int) -> torch.Tensor:
    if edge_index_2e.ndim != 2 or edge_index_2e.shape[0] != 2:
        raise ValueError("edge_index must be (2,E)")

    e = edge_index_2e.t().contiguous()
    E = e.shape[0]

    offsets = (torch.arange(B, device=e.device, dtype=torch.long) * K).view(B, 1, 1)
    eB = e.view(1, E, 2).expand(B, -1, -1) + offsets
    return eB.reshape(B * E, 2).contiguous()


# coarse-to-fine refinement by message passing: samples backbone features at the current
# coordinates, runs a graph-conv residual stack and predicts a (B, K, 3) delta
class LandmarkGraphNet(nn.Module):
    def __init__(self, cfg: GraphSpec, template: GraphTemplate):
        super().__init__()
        self.cfg = cfg
        self.K = int(cfg.K)
        K = self.K
        n_coords = 3

        if template.mean_coords_zyx.shape != (K, 3):
            raise ValueError(f"template.mean_coords_zyx must be (K,3) with K={K}")

        if len(cfg.graph_channels) != cfg.steps:
            raise ValueError("graph_channels must have one entry per step")

        if len(cfg.aggregate_indices) != cfg.steps:
            raise ValueError("aggregate_indices must have length steps")

        self.patch_size = tuple(int(x) for x in cfg.patch_size)

        self.register_buffer("mean_coords_zyx", template.mean_coords_zyx.float())

        if template.edge_index is None:
            raise ValueError(
                "template.edge_index is None. Load the template with load_edge_index=True."
            )
        self.register_buffer("edge_index_2e", template.edge_index.long())

        skip_channels = list(cfg.encoder_channels) + list(cfg.decoder_channels)
        if len(skip_channels) != 9:
            raise ValueError("Expected 9 skip feature maps (indices 0..8).")
        self.skip_channels = skip_channels
        self.n_skips = len(skip_channels)

        # per step: the backbone skip maps feeding it, and their combined channel width
        self.step_skip_indices: list[tuple[int, ...]] = []
        self.step_skip_dims: list[int] = []

        for step in range(cfg.steps):
            idx = tuple(int(i) for i in cfg.aggregate_indices[step])
            if len(idx) == 0:
                raise ValueError(f"aggregate_indices[{step}] is empty")
            if min(idx) < 0 or max(idx) >= self.n_skips:
                raise ValueError(f"aggregate_indices[{step}] contains invalid skip ids: {idx}")
            self.step_skip_indices.append(idx)
            self.step_skip_dims.append(sum(self.skip_channels[i] for i in idx))

        f2f_res_layers: list[nn.ModuleList] = []
        f2v_layers: list[nn.Module] = []

        # step 0 has only sampled features; later steps concatenate the previous latent with
        # features resampled at the updated coords
        for step in range(cfg.steps):
            latent_dim = cfg.graph_channels[step - 1] if step > 0 else 0
            in_dim = int(latent_dim + self.step_skip_dims[step])

            blocks = [
                Features2FeaturesResidual(
                    in_features=in_dim,
                    out_features=cfg.graph_channels[step],
                    hidden_layer_count=cfg.n_f2f_hidden_layer,
                    norm=cfg.norm,
                )
            ]

            for _ in range(cfg.n_residual_blocks - 1):
                blocks.append(
                    Features2FeaturesResidual(
                        in_features=cfg.graph_channels[step],
                        out_features=cfg.graph_channels[step],
                        hidden_layer_count=cfg.n_f2f_hidden_layer,
                        norm=cfg.norm,
                    )
                )

            f2f_res_layers.append(nn.ModuleList(blocks))
            # "feature-to-vertex": maps the step's latent to a 3D delta per landmark
            f2v_layers.append(GraphConv(cfg.graph_channels[step], n_coords))

        self.f2f_res_layers = nn.ModuleList(f2f_res_layers)
        self.f2v_layers = nn.ModuleList(f2v_layers)

        # each step starts by predicting ~0 delta (graph_blocks.py), so training begins near the
        # init coordinates
        self.f2v_layers.apply(lambda m: small_delta_head_init(m, std=1e-2))

    # coords_init_zyx=None means the template mean; refinement runs with autocast disabled,
    # since repeated clamping and small-delta accumulation are safer in float32 than under AMP
    @measure_time
    def forward(
        self,
        skips: Sequence[torch.Tensor],
        coords_init_zyx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        if len(skips) != self.n_skips:
            raise ValueError(f"Expected {self.n_skips} skip tensors, got {len(skips)}")

        B = int(skips[0].shape[0])
        K = self.K
        dev = skips[0].device

        skips = [s.float() for s in skips]

        if coords_init_zyx is None:
            coords = self.mean_coords_zyx.unsqueeze(0).expand(B, -1, -1).to(dev)
        else:
            coords = coords_init_zyx.to(device=dev, dtype=torch.float32)
            if coords.shape != (B, K, 3):
                raise ValueError(f"coords_init_zyx must be (B,K,3)=({B},{K},3)")

        if self.cfg.clamp_coords:
            coords = clamp_coords_zyx(coords, self.patch_size)

        edges_b = _edges_batched(self.edge_index_2e.to(dev), B=B, K=K)

        with autocast(device_type=dev.type, enabled=False):
            latent, coords, deltas, coords_steps = self._refine(skips, coords, edges_b, B, K)

        return coords, deltas, latent.reshape(B, K, -1), coords_steps

    # per step: sample -> graph-conv residual stack -> delta -> move coords -> (optionally)
    # clamp and detach
    def _refine(self, skips, coords, edges_b, B, K):
        deltas: list[torch.Tensor] = []
        coords_steps: list[torch.Tensor] = []  # actual coords after each step, for metrics
        latent = None

        for step in range(self.cfg.steps):
            sampled = aggregate_from_indices(
                voxel_features=skips,
                coords_zyx=coords,
                skip_indices=self.step_skip_indices[step],
                full_shape_zyx=self.patch_size,
            )
            sampled_packed = sampled.reshape(B * K, -1)

            x = sampled_packed if latent is None else torch.cat([latent, sampled_packed], dim=1)

            for blk in self.f2f_res_layers[step]:
                x = blk(x, edges_b)
            latent = x

            delta = self.f2v_layers[step](latent, edges_b).reshape(B, K, 3)
            deltas.append(delta)

            coords = coords + delta
            if self.cfg.clamp_coords:
                coords = clamp_coords_zyx(coords, self.patch_size)
            coords_steps.append(coords.detach())

            # stop-gradient between steps: each step trains on its own delta error, without
            # backpropagating through the earlier steps' sampling and graph convolutions. The last step
            # stays attached; detach_graph_for_refiner (models/vox2seglam.py) decides about the refiner.
            if self.cfg.detach_between_steps and step < self.cfg.steps - 1:
                latent = latent.detach()
                coords = coords.detach()

        return latent, coords, deltas, coords_steps
