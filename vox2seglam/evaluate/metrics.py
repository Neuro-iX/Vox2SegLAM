"""Metrics on one finished subject, in millimetres: overlap, distances, landmark error and
protocol compliance. The training loop uses batch.py instead, which works on tensors."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..protocol import ProtocolSpec, plane_distance, rule_depth

__all__ = [
    "radial_errors",
    "landmark_summary",
    "protocol_violations",
    "plane_bands",
    "segmentation_metrics",
]


def radial_errors(pred: np.ndarray, gt: np.ndarray, spacing) -> np.ndarray:
    """Euclidean distance in mm between predicted and true (..., 3) voxel coordinates.

    spacing: voxel size in mm (task.voxel_spacing_mm), required: a default would report another
    grid's millimetres.
    """
    return np.linalg.norm((np.asarray(pred, float) - gt) * np.asarray(spacing), axis=-1)


def landmark_summary(errors, thresholds_mm=(2, 3, 4)) -> dict:
    """n, MRE, SD (ddof 1), median and SDR@t (%) of a set of radial errors in mm.

    Refuses an empty sample, which would summarise to nan and land in a table as a plausible
    row. SD is nan for a single error.
    """
    e = np.asarray(errors, float).ravel()
    if e.size == 0:
        raise ValueError(
            "landmark_summary: no error to summarise -- the landmark is annotated for "
            "no subject, or the group is empty"
        )
    out = {
        "n": int(e.size),
        "mre": float(e.mean()),
        "sd": float(e.std(ddof=1)) if e.size > 1 else float("nan"),
        "median": float(np.median(e)),
    }
    out.update({f"sdr{t}": float(100 * (e <= t).mean()) for t in thresholds_mm})
    return out


def protocol_violations(
    seg: np.ndarray,
    spec: ProtocolSpec,
    planes: dict,
    tolerance: dict[str, float] | None = None,
    eps: float = 1e-6,
) -> dict[str, int]:
    """{region: voxels of that region lying past a plane of its rule} (eps: rounding slack)."""
    out = {}
    for rule in spec.rules:
        vox = np.argwhere(seg == rule.region_class).astype(np.float64)
        depth = rule_depth(vox, planes, rule, tolerance) if len(vox) else np.zeros(0)
        out[rule.region_name] = int((depth > eps).sum())
    return out


def plane_bands(
    shape: tuple[int, int, int], planes: dict, voxel_mm: float, widths_mm=(1, 2, 3, 4, 5)
) -> dict[str, np.ndarray]:
    """{"band<w>": voxels within w mm of the nearest protocol plane} (isotropic grid).

    voxel_mm: task.isotropic_spacing_mm(), required for the same reason as radial_errors.
    """
    dist = plane_distance(shape, planes) * voxel_mm
    return {f"band{w}": dist <= w for w in widths_mm}


_CONN = ndi.generate_binary_structure(3, 3)
_MARGIN = 12  # voxels kept around a structure's bounding box
_NO_DISTANCE = np.empty(0)  # a structure missing from one of the two maps


@dataclass
class _Structure:
    """One structure's masks, surfaces and distance maps, cropped to where it lives."""

    crop: tuple
    pred: np.ndarray
    gt: np.ndarray
    pred_surface: np.ndarray
    gt_surface: np.ndarray
    to_gt_surface: np.ndarray  # mm, every voxel to the nearest GT surface voxel
    to_pred_surface: np.ndarray


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=_CONN, border_value=0)


def _structure(pred: np.ndarray, gt: np.ndarray, spacing) -> _Structure:
    where = np.argwhere(pred | gt)
    low = np.maximum(where.min(0) - _MARGIN, 0)
    high = where.max(0) + _MARGIN + 1
    crop = tuple(slice(a, b) for a, b in zip(low, high))
    pred, gt = pred[crop], gt[crop]
    sp, sg = _surface(pred), _surface(gt)
    edt = ndi.distance_transform_edt
    return _Structure(
        crop, pred, gt, sp, sg, edt(~sg, sampling=spacing), edt(~sp, sampling=spacing)
    )


def _counts(s: _Structure, zone: np.ndarray) -> dict:
    """Voxel counts and surface-distance samples of one structure inside a zone."""
    z = zone[s.crop]
    p, g = s.pred & z, s.gt & z
    # A surface distance needs both surfaces. When one mask is empty over the whole volume,
    # distance_transform_edt returns the distance to the crop edge (set by _MARGIN, not anatomy);
    # dropping the samples gives nan, which _nanmean skips. The test is on the full surfaces:
    # one direction empty inside a zone is legitimate and keeps its
    # distances.
    both_exist = s.pred_surface.any() and s.gt_surface.any()
    c = {
        "tp": int((p & g).sum()),
        "fp": int((p & ~g).sum()),
        "fn": int((~p & g).sum()),
        "pred_to_gt": s.to_gt_surface[s.pred_surface & z] if both_exist else _NO_DISTANCE,
        "gt_to_pred": s.to_pred_surface[s.gt_surface & z] if both_exist else _NO_DISTANCE,
    }
    return c


def _merge(counts: list[dict]) -> dict:
    return {
        k: (
            np.concatenate([c[k] for c in counts])
            if isinstance(counts[0][k], np.ndarray)
            else sum(c[k] for c in counts)
        )
        for k in counts[0]
    }


def _ratio(num, den) -> float:
    return num / den if den else float("nan")


def _nanmean(values) -> float:
    kept = [v for v in values if v == v]
    return float(np.mean(kept)) if kept else float("nan")


def _metrics(c: dict) -> dict:
    """Overlap and surface distances of one structure: six numbers, not seventeen.

    IoU is a deterministic transform of the Dice, and the NSD and Boundary IoU families, at
    five tolerances each, were found redundant with ASSD and with the Dice when the published
    tables were pared down.
    """
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    a, b = c["pred_to_gt"], c["gt_to_pred"]
    m = {
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }
    d = np.concatenate([a, b])
    if d.size:
        p95 = [np.percentile(x, 95) if x.size else 0.0 for x in (a, b)]
        m.update(assd=float(d.mean()), hd=float(d.max()), hd95=float(max(p95)))
    else:
        m.update(assd=float("nan"), hd=float("nan"), hd95=float("nan"))
    return m


def segmentation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: dict[int, str],
    zones: dict[str, np.ndarray] | None = None,
    *,
    spacing,
) -> dict:
    """{zone: {"per": {structure: metrics}, "mean": metrics, "pooled": metrics}}.

    pred, gt: (Z, Y, X) label maps; classes: {label: structure name}; zones: {name: boolean
    volume}, the whole volume if omitted. A structure absent from both maps is skipped.
    spacing: voxel size in mm (task.voxel_spacing_mm), required so no distance is reported on
    an assumed grid.
    """
    zones = zones or {"whole": np.ones(gt.shape, bool)}
    structures = {}
    for label, name in classes.items():
        p, g = pred == label, gt == label
        if p.any() or g.any():
            structures[name] = _structure(p, g, spacing)

    out = {}
    for zone_name, zone in zones.items():
        counts = {name: _counts(s, zone) for name, s in structures.items()}
        per = {name: _metrics(c) for name, c in counts.items()}
        mean = (
            {k: _nanmean(m[k] for m in per.values()) for k in next(iter(per.values()))}
            if per
            else {}
        )
        pooled = _metrics(_merge(list(counts.values()))) if counts else {}
        out[zone_name] = {"per": per, "mean": mean, "pooled": pooled}
    return out
