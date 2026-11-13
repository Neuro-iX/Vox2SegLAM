import torch
import torch.nn.functional as F

from ..utils.geometry import patch_grid_cells

from .wing import wing


# soft classification target: a Gaussian heatmap over the P^3 patch cells centred on the true
# (sub-cell) position, softmax over -distance^2 so it sums to 1; `sigma` sets its spread and
# `eps` keeps every cell above zero for the KL term
def _gaussian_target(target_cells, patch_size, sigma, eps):
    cells = patch_grid_cells(patch_size, target_cells.device)
    d2 = ((cells.view(1, 1, -1, 3) - target_cells.unsqueeze(2)) ** 2).sum(-1)
    return torch.softmax(-d2 / (2.0 * sigma * sigma), dim=-1).clamp_min(eps)


# the refiner predicts a classification logit per patch cell ("cls") and a 3D offset ("reg");
# this loss supervises both, matching decode() in models/landmark/refiner.py (soft-argmax:
# probability-weighted position plus probability-weighted offset)
def refiner_losses(
    head_out: dict[str, torch.Tensor],
    probs: torch.Tensor,
    centers: torch.Tensor,
    coords_gt: torch.Tensor,
    patch_size: int,
    pitch: float,
    sigma_cells: float,
    w_cls: float,
    w_reg: float,
    wing_width: float = 1.0,
    wing_epsilon: float = 0.2,
    eps: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    logits = head_out["logits"].float()
    offsets = head_out["offsets"].float()
    P = int(patch_size)

    # true landmark position in patch-cell units, relative to the patch centre; `centers` is
    # detached so this loss does not backprop into where the graph placed the patch
    tgt_cells = (coords_gt.float() - centers.detach().float()) / float(pitch)

    # a target outside the sampled patch has no valid cell: clamp it so the math stays defined
    # and exclude the sample via `valid` (diagnostics "refine_outside_frac" / "refine_valid_frac";
    # never used to gate training)
    half = (P - 1) / 2.0
    outside = (tgt_cells.abs() > half).any(dim=-1)
    tgt_clamped = tgt_cells.clamp(-half, half)

    # classification: KL divergence between the predicted per-cell distribution and
    # the soft Gaussian target built around the true (clamped) position
    t = _gaussian_target(tgt_clamped, P, sigma_cells, eps)
    loss_cls = F.kl_div(torch.log_softmax(logits, -1), t, reduction="none").sum(-1)

    # regression: per-cell residual (true position minus the cell's location) against the predicted
    # offset via wing(), weighted by the detached predicted cell probabilities, the same
    # combination decode() uses at inference, so it is supervised where the model believes the
    # landmark is
    cells = patch_grid_cells(P, logits.device)
    residual = tgt_clamped.unsqueeze(2) - cells.view(1, 1, -1, 3)
    err = wing(offsets.permute(0, 1, 3, 2) - residual, width=wing_width, epsilon=wing_epsilon).sum(
        -1
    )
    loss_reg = (probs.detach() * err).sum(-1)

    # average only over samples whose true position actually fell inside the patch
    valid = (~outside).float()
    n = valid.sum().clamp_min(1.0)
    loss_cls = (loss_cls * valid).sum() / n
    loss_reg = (loss_reg * valid).sum() / n

    total = w_cls * loss_cls + w_reg * loss_reg
    return (
        {
            "cls": loss_cls,
            "reg": loss_reg,
            "outside_frac": outside.float().mean().detach(),
            "valid_frac": valid.mean().detach(),
            "outside_per_landmark": outside.float().mean(0).detach(),
        },
        total,
    )
