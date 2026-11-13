from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..utils.geometry import clamp_coords_zyx

__all__ = ["load_template_std", "build_train_init_coords"]


# per-landmark coordinate std, scaled by template_perturbation_scale; None when
# perturbation is disabled (model.lam.perturb_template=False)
def load_template_std(cfg: Config, device: torch.device) -> torch.Tensor | None:
    lam = cfg.model.lam
    if not lam.perturb_template:
        return None

    path = Path(cfg.paths.resolve_template_dir()) / lam.template_std_filename
    if not path.is_file():
        raise FileNotFoundError(f"template perturbation is enabled but {path} is missing")
    std = np.load(path).astype(np.float32) * float(lam.template_perturbation_scale)
    expected = (cfg.task.n_landmarks, 3)
    if std.shape != expected:
        raise ValueError(f"{path}: expected shape {expected}, got {std.shape}")
    return torch.from_numpy(std).to(device)


# without perturbation every sample would start the GNN from the same template coords; with
# std_coords each landmark is displaced in proportion to its own positional variance. This is
# also why loss.graph_loss_per_step="mse" requires perturb_template=True (config/root.py): MSE
# on the delta is well-scaled only once the starting offset has meaningful variance.
def build_train_init_coords(
    mean_coords_zyx: torch.Tensor,
    batch_size: int,
    device: torch.device,
    std_coords: torch.Tensor | None,
    clamp_coords: bool,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    base = mean_coords_zyx.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    if std_coords is None:
        return base

    coords = base + torch.randn_like(base) * std_coords.unsqueeze(0)
    if clamp_coords:
        coords = clamp_coords_zyx(coords, patch_size)
    return coords
