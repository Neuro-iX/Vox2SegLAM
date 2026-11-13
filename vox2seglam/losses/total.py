from collections.abc import Collection
from typing import Any

import torch

from ..config import LossCfg
from .adaptive_weighting import AdaptiveLossWeighting
from .cls_reg_refiner import refiner_losses
from .dice_ce_seg import deep_supervision_seg_losses
from .deep_supervision import deep_supervision_weights
from .wing_graph import graph_delta_losses

__all__ = ["compute_losses", "loss_term_names", "weighted_term_names", "stage_of_term"]


# the curriculum stages this objective has terms for (Vox2SegLAM.stage_parameters keys
# without the optimizer-only "weighter")
def loss_term_names(*, steps: int, has_seg: bool, has_refine: bool) -> list[str]:
    names = ["seg"] if has_seg else []
    names += [f"gnn{i + 1}" for i in range(int(steps))]
    if has_refine:
        names.append("refine")
    return names


# a stage can map to several terms: "refine" covers the refiner's cls and reg terms; "seg"
# and each "gnnN" map to themselves
_TERMS_OF_STAGE = {"seg": ("seg",), "refine": ("refine_cls", "refine_reg")}


# expands a list of active stage names into the individual term names to optimise
def weighted_term_names(stages: list[str]) -> list[str]:
    return [t for s in stages for t in _TERMS_OF_STAGE.get(s, (s,))]


# the inverse lookup: which stage a given term name belongs to (for active_terms gating)
def stage_of_term(term: str) -> str:
    for stage, terms in _TERMS_OF_STAGE.items():
        if term in terms:
            return stage
    return term


# patch_size is recovered from whichever of image/seg is in the batch
def _infer_patch_size_from_batch(batch: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    for key in ("image", "seg"):
        t = batch.get(key, None)
        if isinstance(t, torch.Tensor) and t.ndim == 5:
            return int(t.shape[2]), int(t.shape[3]), int(t.shape[4])
    raise ValueError("Cannot infer patch_size: batch must contain 'image' or 'seg' as (B,1,Z,Y,X).")


# assembles the multi-task objective for one batch: seg (dice+ce), one term per GNN step,
# the refiner's terms and the optional protocol term, combined by the learned weighting or a
# fixed sum; returns the scalar to backprop and a flat dict of every term for logging
def compute_losses(
    pred: dict[str, Any],
    batch: dict[str, torch.Tensor],
    cfg: LossCfg,
    *,
    num_classes: int,
    K: int,
    deep_supervision: bool,
    clamp_coords: bool = True,
    coord_std_vox: torch.Tensor | None = None,
    detach_between_steps: bool = False,
    active_terms: Collection[str] | None = None,
    weighter: AdaptiveLossWeighting | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    seg_out = pred["seg"]
    deltas = pred["deltas"]
    coords_init = pred["coords_init_zyx_used"]

    coords_gt = batch["coords"]

    B = int(coords_gt.shape[0])
    if coords_gt.ndim != 3 or coords_gt.shape[1] != K or coords_gt.shape[2] != 3:
        raise ValueError(f"coords_gt must be (B,K,3)=({B},{K},3), got {tuple(coords_gt.shape)}")

    if coords_init.shape != (B, K, 3):
        raise ValueError(
            f"coords_init_zyx_used must be (B,K,3)=({B},{K},3), got {tuple(coords_init.shape)}"
        )

    for i, d in enumerate(deltas):
        if d.shape != (B, K, 3):
            raise ValueError(f"delta[{i}] must be (B,K,3)=({B},{K},3), got {tuple(d.shape)}")

    patch_size_zyx = _infer_patch_size_from_batch(batch)

    if seg_out:
        loss_dice, loss_ce = deep_supervision_seg_losses(
            seg_out,
            batch["seg"],
            num_classes=num_classes,
            dice_smooth=cfg.dice_smooth,
            deep_supervision=deep_supervision,
        )
    else:
        loss_dice = loss_ce = coords_gt.new_zeros(())
    loss_seg = loss_dice + loss_ce

    gnn_steps = graph_delta_losses(
        deltas_vox=deltas,
        coords_init_vox=coords_init,
        coords_gt_vox=coords_gt,
        patch_size_zyx=patch_size_zyx,
        clamp_coords=clamp_coords,
        kind_per_step=cfg.graph_loss_per_step,
        wing_epsilon=cfg.wing_epsilon,
        wing_width=cfg.wing_width,
        coord_std_vox=coord_std_vox,
        detach_between_steps=detach_between_steps,
    )

    # same geometric falloff as the seg heads: the last GNN step dominates this "lm_gnn"
    # diagnostic, while each step's own loss is still optimised and reported separately
    ds = deep_supervision_weights(len(gnn_steps), coords_gt.device)
    loss_lm_gnn = sum((w * l for w, l in zip(ds, gnn_steps)), coords_gt.new_zeros(()))

    loss_refine: torch.Tensor | None = None
    refine_parts: dict[str, torch.Tensor] = {}
    if "local" in pred:
        loc = pred["local"]
        refine_parts, loss_refine = refiner_losses(
            head_out={"logits": loc["logits"], "offsets": loc["offsets"]},
            probs=loc["probs"],
            centers=loc["coords_graph_last"],
            coords_gt=coords_gt,
            patch_size=int(loc["patch_size"]),
            pitch=float(loc["pitch"]),
            sigma_cells=cfg.refine_sigma_cells,
            w_cls=cfg.refine_w_cls,
            w_reg=cfg.refine_w_reg,
            wing_width=cfg.refine_wing_width,
            wing_epsilon=cfg.refine_wing_epsilon,
            eps=cfg.refine_eps,
        )

    loss_lm = loss_lm_gnn if loss_refine is None else loss_lm_gnn + loss_refine  # reporting only

    # all_terms is what actually gets weighted and summed into `total` below.
    all_terms: dict[str, torch.Tensor] = {}
    if seg_out:
        all_terms["seg"] = loss_seg
    for i, step_loss in enumerate(gnn_steps):
        all_terms[f"gnn{i + 1}"] = step_loss
    if loss_refine is not None:
        all_terms["refine_cls"] = refine_parts["cls"]
        all_terms["refine_reg"] = refine_parts["reg"]
    terms = dict(all_terms)

    # curriculum gating: only terms of active stages are weighted and backpropagated; `losses`
    # keeps everything for logging
    if active_terms is not None:
        gated = {k: v for k, v in terms.items() if stage_of_term(k) in active_terms}
        if not gated:
            raise ValueError(
                f"active_terms={sorted(active_terms)} selects none of the "
                f"objective's terms {sorted(terms)}"
            )
        terms = gated

    if cfg.adaptive_weighting:
        if weighter is None:
            raise ValueError("cfg.adaptive_weighting=True requires `weighter`.")
        total = weighter(terms)
    else:
        # static weights: named terms use their configured multiplier, gnnN the landmark weight
        # init_w_lm
        fixed = {
            "seg": cfg.init_w_seg,
            "refine_cls": cfg.init_w_lm * cfg.refine_w_cls,
            "refine_reg": cfg.init_w_lm * cfg.refine_w_reg,
        }
        total = coords_gt.new_zeros(())
        for name, value in terms.items():
            total = total + fixed.get(name, cfg.init_w_lm) * value

    # the full reporting dict: "lm_" keys go to the landmark TensorBoard group (reporting.py),
    # "refine_*" to refine/, etc.
    losses = {
        "mre": (pred["coords"].detach() - coords_gt).norm(dim=-1).mean(),
        "dice": loss_dice,
        "ce": loss_ce,
        "seg": loss_seg,
        "lm": loss_lm,
        "lm_gnn": loss_lm_gnn,
        "total": total,
    }
    losses.update({f"lm_{name}": v for name, v in all_terms.items() if name.startswith("gnn")})
    if loss_refine is not None:
        losses["lm_refine"] = loss_refine
    losses.update({f"refine_{k}": v for k, v in refine_parts.items()})
    return losses, total
