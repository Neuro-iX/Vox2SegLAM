from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .deep_supervision import deep_supervision_weights


# ground-truth seg can arrive as (B,1,Z,Y,X) or already-squeezed (B,Z,Y,X); normalises
# to whichever cross_entropy/dice indexing below expects relative to logits' ndim
def _squeeze_seg_target(seg: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return seg[:, 0] if seg.ndim == logits.ndim else seg


# soft Dice, averaged over classes 1..num_classes-1: background is excluded from the loss
# here, unlike the eval-time Dice (evaluate/batch.py), which scores every class
def dice_loss_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    target = _squeeze_seg_target(target, logits).long()
    probs = torch.softmax(logits, dim=1)

    B, C = probs.shape[:2]
    if C != int(num_classes):
        raise ValueError(f"logits has C={C}, expected num_classes={num_classes}")

    probs_f = probs.reshape(B, C, -1)
    tgt_f = target.reshape(B, -1)

    start = 1
    dices = []
    for c in range(start, num_classes):
        p = probs_f[:, c, :]
        g = (tgt_f == c).to(p.dtype)
        inter = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        dices.append((2.0 * inter + smooth) / (denom + smooth))

    return 1.0 - torch.stack(dices, dim=0).mean()


# per-voxel CE then mean: same value as reduction="mean", but CUDA's built-in mean
# reduction accumulates with atomics and is not deterministic (repro.deterministic)
def ce_loss_multiclass(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _squeeze_seg_target(target, logits).long()
    return F.cross_entropy(logits, target, reduction="none").mean()


# sums dice/CE over the deep-supervision heads (weighted by deep_supervision_weights, deeper
# heads counting less), or the full-res head alone without deep supervision; the dice and
# CE totals are returned separately so each stays a visible, weightable term under adaptive
# weighting
def deep_supervision_seg_losses(
    seg_out: Sequence[torch.Tensor],
    seg_gt: torch.Tensor,
    *,
    num_classes: int,
    dice_smooth: float,
    deep_supervision: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not seg_out:
        z = torch.tensor(0.0, device=seg_gt.device)
        return z, z

    if deep_supervision:
        weights = deep_supervision_weights(len(seg_out), seg_gt.device)
        heads = seg_out
    else:
        weights = torch.tensor([1.0], device=seg_gt.device)
        heads = [seg_out[-1]]

    dice_total = torch.tensor(0.0, device=seg_gt.device)
    ce_total = torch.tensor(0.0, device=seg_gt.device)

    for w, logits in zip(weights, heads):
        dice_total = dice_total + w * dice_loss_multiclass(
            logits=logits,
            target=seg_gt,
            num_classes=num_classes,
            smooth=dice_smooth,
        )
        ce_total = ce_total + w * ce_loss_multiclass(logits, seg_gt)

    return dice_total, ce_total
