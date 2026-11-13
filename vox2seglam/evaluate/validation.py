import math
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from ..utils.device import to_device
from .batch import dice_score, lm_mre_vox_score


# ignores nan entries: a class or landmark absent from a subject must not drag the mean
def _nanmean(values: Sequence[float]) -> float:
    live = [v for v in values if not math.isnan(v)]
    return sum(live) / len(live) if live else float("nan")


# per-column average over batches weighted by batch size (a plain mean would overweight a
# short last batch)
def _weighted_column_means(
    rows: Sequence[Sequence[float]],
    weights: Sequence[int],
) -> list[float]:
    n_cols = len(rows[0])
    out: list[float] = []
    for c in range(n_cols):
        num = den = 0.0
        for row, w in zip(rows, weights):
            if not math.isnan(row[c]):
                num += row[c] * w
                den += w
        out.append(num / den if den else float("nan"))
    return out


# one pass over `loader`: Dice and landmark MRE aggregated across batches (weighted by batch
# size) into a flat dict; logged by trainer.py and used for checkpoint selection
@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_v_classes: int,
    landmark_ids: list[int],
    n_passes: int = 1,
) -> dict[str, float]:
    model.eval()

    seg_per_class_all: list[list[float]] = []
    lm_per_lm_all: list[list[float]] = []
    lm_step_means_all: list[list[float]] = []
    lm_init_all: list[list[float]] = []
    refine_valid: list[float] = []  # per batch: share of landmarks inside the refiner patch
    batch_sizes: list[int] = []

    for batch in loader:
        batch = to_device(batch, device)

        x = batch["image"]
        if x.ndim == 4:
            x = x.unsqueeze(0)

        # n_passes > 1 re-runs the model from the previous pass's landmarks, as predictor.py does
        pred = model(x)
        for _ in range(int(n_passes) - 1):
            pred = model(x, coords_init_zyx=pred["coords"])

        if not isinstance(pred, dict):
            raise TypeError(f"Expected model(x) to return dict, got {type(pred)}")

        required_keys = {"coords", "seg", "deltas", "coords_steps", "coords_init_zyx_used"}
        missing = required_keys.difference(pred.keys())
        if missing:
            raise KeyError(f"Prediction dict missing keys: {sorted(missing)}")

        if pred["seg"]:
            seg_res = dice_score(pred, batch, n_v_classes=int(n_v_classes))
            seg_per_class_all.append([float(v) for v in seg_res["per_class"]])
        batch_sizes.append(int(x.shape[0]))

        lm_res = lm_mre_vox_score(pred, batch)
        lm_per_lm_all.append([float(v) for v in lm_res["per_landmark"]])
        lm_step_means_all.append([float(v) for v in lm_res["step_means"]])
        lm_init_all.append([float(lm_res["init_mean"])])

        # same rule as the refiner loss (losses/cls_reg_refiner.py): the true landmark lies within
        # (patch - 1) / 2 cells of the patch centre on every axis; measured on unperturbed validation
        # inputs, this is what the refine curriculum gate checks
        local = pred.get("local")
        if local is not None:
            half = (int(local["patch_size"]) - 1) / 2.0
            cells = (batch["coords"].float() - local["coords_graph_last"].float()) / float(
                local["pitch"]
            )
            refine_valid.append(float((cells.abs() <= half).all(dim=-1).float().mean()))

    out: dict[str, float] = {}

    if seg_per_class_all:
        per_class_mean = _weighted_column_means(seg_per_class_all, batch_sizes)

        out["dice_mean_bg"] = _nanmean(per_class_mean)  # mean over all classes, background included
        out[
            "dice_mean_no_bg"
        ] = (  # excludes class 0; the headline metric used for scheduling/selection
            _nanmean(per_class_mean[1:]) if int(n_v_classes) > 1 else out["dice_mean_bg"]
        )

        for c in range(int(n_v_classes)):
            out[f"dice_per_class/class{c}"] = per_class_mean[c]
    else:
        out["dice_mean_bg"] = float("nan")
        out["dice_mean_no_bg"] = float("nan")
        for c in range(int(n_v_classes)):
            out[f"dice_per_class/class{c}"] = float("nan")

    if lm_per_lm_all:
        per_lm_mean = _weighted_column_means(lm_per_lm_all, batch_sizes)

        if len(per_lm_mean) != len(landmark_ids):
            raise ValueError(
                f"val: got K={len(per_lm_mean)} but landmark_ids has {len(landmark_ids)}"
            )

        out["lm_mre_vox_mean"] = _nanmean(per_lm_mean)
        for lid, v in zip(landmark_ids, per_lm_mean):
            out[f"lm_mre_vox_per_landmark/lm{int(lid)}"] = v
    else:
        out["lm_mre_vox_mean"] = float("nan")
        for lid in landmark_ids:
            out[f"lm_mre_vox_per_landmark/lm{int(lid)}"] = float("nan")

    # What the graph produced. These do not depend on the refiner: lm_mre_vox_mean_init is what
    # staging.gnn_must_beat_template compares against (trainer.py::_stage_gates), and leaving it
    # nan without a refiner kept `prev < init` false, so gnn2 could never start.
    if lm_step_means_all:
        step_mean = _weighted_column_means(lm_step_means_all, batch_sizes)
        for s, v in enumerate(step_mean):
            out[f"lm_mre_vox_mean_step{s + 1}"] = v
        # MRE of the starting coordinates (the template on a single pass)
        out["lm_mre_vox_mean_init"] = _weighted_column_means(lm_init_all, batch_sizes)[0]
        out["lm_mre_vox_mean_gnn_last"] = step_mean[-1]  # MRE right before the local refiner
    else:
        out["lm_mre_vox_mean_init"] = float("nan")
        out["lm_mre_vox_mean_gnn_last"] = float("nan")

    out["lm_mre_vox_mean_final_step"] = out["lm_mre_vox_mean"]

    # What the refiner added, on top of the graph - these alone need it to have run
    if refine_valid:
        out["refine_valid_frac_val"] = _weighted_column_means(
            [[v] for v in refine_valid], batch_sizes
        )[0]
        # negative: the refiner reduced the error relative to the graph's last step alone
        out["lm_mre_local_head_gain"] = out["lm_mre_vox_mean"] - out["lm_mre_vox_mean_gnn_last"]
    else:
        out["lm_mre_local_head_gain"] = float("nan")

    return out
