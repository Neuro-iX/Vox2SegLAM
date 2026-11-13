from collections.abc import Sequence

import torch

from ..utils.geometry import clamp_coords_zyx

from .wing import wing

__all__ = ["graph_delta_losses"]

_KINDS = ("mse", "wing")


# per-step supervision: step i predicts delta_vox[i] to close the remaining gap
# (coords_gt - coords); the error delta_vox[i] - (coords_gt - coords) is scored by "mse" or
# "wing" (loss.graph_loss_per_step), then coords advance by the actual predicted delta, so
# later steps are scored against the error earlier steps left behind
def graph_delta_losses(
    deltas_vox: Sequence[torch.Tensor],
    coords_init_vox: torch.Tensor,
    coords_gt_vox: torch.Tensor,
    patch_size_zyx: tuple[int, int, int],
    kind_per_step: Sequence[str],
    wing_epsilon: float,
    wing_width: float,
    coord_std_vox: torch.Tensor | None = None,
    clamp_coords: bool = True,
    detach_between_steps: bool = False,
) -> list[torch.Tensor]:
    S = len(deltas_vox)
    if S == 0:
        return []

    if len(kind_per_step) != S:
        raise ValueError(f"kind_per_step must have length S={S}, got {len(kind_per_step)}")

    coords_gt = coords_gt_vox.to(torch.float32)
    coords = coords_init_vox.to(torch.float32)

    if clamp_coords:
        coords = clamp_coords_zyx(coords, patch_size_zyx)

    # per-landmark coordinate std (init_coords.py), normalising the "mse" error so landmarks
    # with larger training-time perturbation do not dominate the loss
    if coord_std_vox is not None:
        K = coords_gt.shape[1]
        if coord_std_vox.shape != (K, 3):
            raise ValueError(
                f"coord_std_vox must be (K,3)=({K},3), got {tuple(coord_std_vox.shape)}"
            )
        scale = coord_std_vox.to(device=coords_gt.device, dtype=torch.float32)
        scale = scale.unsqueeze(0).clamp_min(1e-6)
    else:
        scale = None

    per_step: list[torch.Tensor] = []

    for i, delta_vox in enumerate(deltas_vox):
        delta_vox = delta_vox.to(torch.float32)
        err = delta_vox - (coords_gt - coords)

        kind = str(kind_per_step[i])
        if kind == "mse":
            if scale is not None:
                err = err / scale
            per_step.append((err**2).mean())
        elif kind == "wing":
            per_step.append(wing(err, width=wing_width, epsilon=wing_epsilon).mean())
        else:
            raise ValueError(f"graph_loss_per_step[{i}]={kind!r}; expected one of {_KINDS}")

        coords = coords + delta_vox
        if detach_between_steps:
            coords = coords.detach()
        if clamp_coords:
            coords = clamp_coords_zyx(coords, patch_size_zyx)

    return per_step
