"""Metrics on a batch of tensors, inside the training loop: cheap enough to run every epoch.
metrics.py has the per-subject ones, in millimetres, for a finished prediction."""

from typing import Any

import torch

__all__ = ["dice_score", "lm_mre_vox_score", "ensure_batch_seg", "ensure_batch_coords"]


# normalises a seg tensor to (B,Z,Y,X), accepting (Z,Y,X), (1,Z,Y,X), (B,Z,Y,X) or
# (B,1,Z,Y,X) - broadcasting a single (ground-truth) sample across a batch of B
def ensure_batch_seg(seg: torch.Tensor, B: int) -> torch.Tensor:
    if seg.ndim == 3:
        seg = seg.unsqueeze(0)

    if seg.ndim == 4:
        if seg.shape[0] == 1 and B > 1:
            seg = seg.expand(B, -1, -1, -1)
        return seg.long()

    if seg.ndim == 5 and seg.shape[1] == 1:
        seg = seg[:, 0]
        return seg.long()

    raise ValueError(
        f"seg must be (Z,Y,X) or (1,Z,Y,X) or (B,Z,Y,X) or (B,1,Z,Y,X), got {tuple(seg.shape)}"
    )


# same idea for coords: normalises to (B,K,3), broadcasting a single (K,3) across B
def ensure_batch_coords(coords: torch.Tensor, B: int) -> torch.Tensor:
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be (K,3) or (B,K,3), got {tuple(coords.shape)}")

    if coords.shape[0] == 1 and B > 1:
        coords = coords.expand(B, -1, -1)

    return coords.to(torch.float32)


# hard Dice per class: argmax label map vs. the integer ground truth, one value per
# (sample, class); nan when the class is absent from both prediction and ground truth,
# so it is skipped by the nan-aware averages instead of counting as 0 or 1
def _dice_per_class_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError(f"logits must be (B,C,Z,Y,X), got {tuple(logits.shape)}")

    B, C = logits.shape[:2]
    pred_f = logits.argmax(dim=1).reshape(B, -1)
    tgt_f = target.reshape(B, -1)

    dices = torch.empty(B, C, dtype=torch.float64, device=logits.device)
    for c in range(C):
        p, g = pred_f == c, tgt_f == c
        inter = (p & g).sum(dim=1).double()
        denom = (p.sum(dim=1) + g.sum(dim=1)).double()
        dices[:, c] = torch.where(denom > 0, 2.0 * inter / denom.clamp(min=1), torch.nan)
    return dices


# eval-time wrapper: nan placeholder when the seg head is disabled (so downstream
# reporting/selection code doesn't need a special case), otherwise scores the
# full-resolution prediction (the last deep-supervision level is always full-res)
@torch.no_grad()
def dice_score(
    pred,
    sample: dict[str, Any],
    n_v_classes: int,
) -> dict[str, Any]:
    seg_out = pred["seg"]

    if not seg_out:
        return {"mean": float("nan"), "per_class": [float("nan")] * int(n_v_classes)}

    logits = seg_out[-1]
    B, C = logits.shape[:2]

    if C != int(n_v_classes):
        raise ValueError(f"Dice: logits has C={C} but n_v_classes={n_v_classes}")

    seg_gt = ensure_batch_seg(sample["seg"], B)

    # batch mean per class over the samples where the class is defined
    per_class = torch.nanmean(_dice_per_class_from_logits(logits, seg_gt), dim=0)

    return {
        "mean": float(torch.nanmean(per_class)),
        "per_class": per_class.cpu().tolist(),
    }


# coordinates after each GNN step as the model moved them (clamp included), for the
# per-step MRE of the landmark/ TensorBoard group
def _extract_coords_steps_from_pred(pred) -> list[torch.Tensor]:
    coords_final = pred["coords"]
    if not torch.is_tensor(coords_final) or coords_final.ndim != 3 or coords_final.shape[-1] != 3:
        raise ValueError(f"coords_final must be (B,K,3), got {tuple(coords_final.shape)}")
    if not isinstance(pred, dict) or "coords_steps" not in pred:
        raise KeyError("pred is missing key 'coords_steps'")

    steps = pred["coords_steps"]
    if not steps:
        return [coords_final]
    for s, c in enumerate(steps):
        if not torch.is_tensor(c) or c.shape != coords_final.shape:
            raise ValueError(
                f"coords_steps[{s}] must be {tuple(coords_final.shape)}, "
                f"got {tuple(c.shape) if torch.is_tensor(c) else type(c)}"
            )
    return [c.to(torch.float32) for c in steps]


# Mean Radial Error in voxels: per landmark (across the batch) and overall; also the error
# at each GNN step and of the starting coordinates (the template when no init is given),
# which the curriculum compares each step against (training/stages.py)
@torch.no_grad()
def lm_mre_vox_score(pred, sample: dict[str, Any]) -> dict[str, Any]:
    coords_pred = pred["coords"]
    if coords_pred.ndim != 3 or coords_pred.shape[-1] != 3:
        raise ValueError(f"coords_pred must be (B,K,3), got {tuple(coords_pred.shape)}")

    B, K, _ = coords_pred.shape
    coords_gt = ensure_batch_coords(sample["coords"], B)

    if coords_gt.shape[1] != K:
        raise ValueError(f"coords_gt has K={coords_gt.shape[1]} but pred has K={K}")

    e_final = (coords_pred.to(torch.float32) - coords_gt).norm(dim=-1)
    per_landmark = e_final.mean(dim=0)
    mean_all = per_landmark.mean()

    coords_steps = _extract_coords_steps_from_pred(pred)
    step_means_t = [
        (coords_s.to(torch.float32) - coords_gt).norm(dim=-1).mean() for coords_s in coords_steps
    ]

    init_mean = (pred["coords_init_zyx_used"].to(torch.float32) - coords_gt).norm(dim=-1).mean()

    # stack every scalar into one tensor so there's a single .cpu() sync, not one per value
    values = (
        torch.stack([mean_all, init_mean, *step_means_t, *per_landmark]).detach().cpu().tolist()
    )
    mean_val, init_val = values[0], values[1]
    step_means = values[2 : 2 + len(step_means_t)]
    per_landmark_vals = values[2 + len(step_means_t) :]

    return {
        "mean": mean_val,
        "per_landmark": per_landmark_vals,
        "step_means": step_means,
        "init_mean": init_val,
    }
