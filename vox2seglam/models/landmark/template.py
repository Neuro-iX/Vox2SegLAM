from dataclasses import dataclass
import os

import numpy as np
import torch


# the fixed anatomical prior: where each landmark starts (mean position across the
# training population) and which pairs of landmarks exchange messages in the GNN
@dataclass(frozen=True)
class GraphTemplate:
    mean_coords_zyx: torch.Tensor
    edge_index: torch.Tensor | None = None


# every ordered pair (i, j), i != j - an all-to-all graph with no structural prior
def fully_connected_edge_index(k: int) -> torch.Tensor:
    idx = torch.arange(int(k))
    src = idx.repeat_interleave(int(k))
    dst = idx.repeat(int(k))
    keep = src != dst
    return torch.stack([src[keep], dst[keep]]).long()


# loads the mean coords every run needs and builds the edges per `edges`
# (model.lam.edge_source()): "fully_connected" builds every pair in code, "template" loads the
# (2, E) tensor template_dir/edge_filename (built by training/template_setup.py)
def load_graph_template(
    template_dir: str,
    device: str | None = None,
    K_expected: int = 16,
    load_edge_index: bool = True,
    edge_filename: str = "",
    edges: str = "fully_connected",
) -> GraphTemplate:
    mean_path = os.path.join(template_dir, "mean_landmark_coords_voxel_zyx.npy")

    mean_coords = torch.from_numpy(np.load(mean_path).astype(np.float32))

    if mean_coords.ndim != 2 or mean_coords.shape[1] != 3:
        raise ValueError("mean_coords must be (K,3)")
    if mean_coords.shape[0] != K_expected:
        raise ValueError(f"Expected {K_expected} landmarks, got {mean_coords.shape[0]}")

    edge_index: torch.Tensor | None = None
    if load_edge_index and edges == "fully_connected":
        edge_index = fully_connected_edge_index(K_expected)
    elif load_edge_index and edges == "template":
        edge_path = os.path.join(template_dir, edge_filename)
        if not os.path.isfile(edge_path):
            raise FileNotFoundError(f"Template edge file not found: {edge_path}")

        edge_index = torch.load(edge_path, map_location="cpu", weights_only=True).long()

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be (2,E)")

        if edge_index.numel() > 0:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= K_expected:
                raise ValueError(f"edge_index contains invalid node ids for K={K_expected}")
    elif load_edge_index:
        raise ValueError(f"edges={edges!r}; expected 'fully_connected' or 'template'")

    if device is not None:
        mean_coords = mean_coords.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)

    return GraphTemplate(
        mean_coords_zyx=mean_coords,
        edge_index=edge_index,
    )


# the template a trained model carries as buffers: lets a checkpoint be rebuilt without
# template_dir (mean coords and edge topology are saved in its state dict)
def template_from_state_dict(state: dict) -> GraphTemplate:
    try:
        return GraphTemplate(
            state["landmark_graph.mean_coords_zyx"].float().cpu(),
            state["landmark_graph.edge_index_2e"].long().cpu(),
        )
    except KeyError as e:
        raise KeyError(f"checkpoint has no landmark template buffer {e}") from None
