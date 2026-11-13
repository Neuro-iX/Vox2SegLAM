"""What the plane-shift post-processing learns, from data it is never evaluated on."""

from collections.abc import Iterable, Sequence

import numpy as np

from ..protocol import ProtocolSpec
from ..protocol.geometry import clause_depths, clause_groups, protocol_planes
from .plane_shift import SECTION_PLANES, PlaneShift

__all__ = ["plane_errors", "shift_ranges", "binding_planes", "gt_tolerances", "fit_plane_shift"]


def plane_errors(spec: ProtocolSpec, pairs: Iterable[tuple[np.ndarray, np.ndarray]]):
    """{plane: [signed error per subject]} for (predicted, ground-truth) landmark pairs."""
    errors = {p.plane_id: [] for p in spec.planes}
    for pred, gt in pairs:
        planes, truth = protocol_planes(spec, pred), protocol_planes(spec, gt)
        for pid, (centroid, normal) in planes.items():
            errors[pid].append(float((truth[pid][0] - centroid) @ normal))
    return errors


def shift_ranges(errors: dict[str, list[float]], percentiles=(25, 75)):
    """{plane: (low, high)} percentiles of the signed plane errors, in voxels."""
    lo, hi = percentiles
    return {
        pid: (float(np.percentile(e, lo)), float(np.percentile(e, hi)))
        for pid, e in errors.items()
        if e
    }


def binding_planes(spec: ProtocolSpec, seg: np.ndarray, landmarks: np.ndarray) -> set[str]:
    """Planes that at least one voxel of `seg` crosses (tolerance 0).

    A violating voxel is charged to the plane it would have to cross to comply: the worst
    clause of its least violated OR-group.
    """
    planes = protocol_planes(spec, landmarks)
    crossed = set()
    for rule in spec.rules:
        vox = np.argwhere(seg == rule.region_class).astype(np.float64)
        if not len(vox):
            continue
        groups = clause_groups(rule)
        depth = clause_depths(vox, planes, rule)  # per group: (clauses, voxels)
        worst = np.stack([d.max(0) for d in depth])  # (groups, voxels)
        violating = np.flatnonzero(worst.min(0) > 0)
        best_group = worst[:, violating].argmin(0)
        for g in np.unique(best_group):
            cols = violating[best_group == g]
            for c in np.unique(depth[g][:, cols].argmax(0)):
                crossed.add(groups[g][c].plane_id)
    return crossed


def gt_tolerances(
    spec: ProtocolSpec,
    subjects: Iterable[tuple[np.ndarray, np.ndarray]],
    min_share: float = 0.5,
    tolerance: float = 1.0,
):
    """Tolerances fitted on ground-truth (segmentation, landmarks) pairs.

    Returns ({plane: tolerance} for the planes crossed in >= min_share of the subjects,
    {plane: number of subjects crossing it}, number of subjects).
    """
    counts, n = {}, 0
    for seg, landmarks in subjects:
        n += 1
        for pid in binding_planes(spec, seg, landmarks):
            counts[pid] = counts.get(pid, 0) + 1
    chosen = {pid: float(tolerance) for pid, k in counts.items() if k >= min_share * n}
    return chosen, counts, n


# The whole fit in one call, used by the fit-postprocess command. Takes iterables, so it
# depends on neither dataloaders nor the model.
def fit_plane_shift(
    validation_pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    percentiles=(25.0, 75.0),
    section_planes: Sequence[str] | None = None,
    tolerance_vox: float = 0.0,
    training_subjects: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    min_share: float = 0.5,
    provenance: dict | None = None,
) -> PlaneShift:
    """A PlaneShift fitted on `validation_pairs` of (predicted, ground-truth) landmarks.

    tolerance_vox > 0 additionally needs `training_subjects`, pairs of (segmentation,
    landmarks) of the training split. `provenance` is merged into the result's `fitted_on`,
    for whatever the caller knows and this function does not (split names, subject counts).
    """
    base = PlaneShift({})
    ranges = shift_ranges(plane_errors(base.spec, validation_pairs), percentiles)
    if not ranges:
        raise ValueError(
            "no plane error collected on the validation split: nothing to fit. Check that "
            "the validation split is not empty and that the landmarks are predicted."
        )
    fitted_on = dict(provenance or {})
    fitted_on.setdefault("shift_range", {}).update({"percentiles": list(percentiles)})

    tolerance: dict[str, float] = {}
    if tolerance_vox > 0:
        if training_subjects is None:
            raise ValueError("tolerance_vox > 0 needs training_subjects")
        tolerance, counts, n = gt_tolerances(base.spec, training_subjects, min_share, tolerance_vox)
        fitted_on.setdefault("tolerance", {}).update(
            {"subjects": n, "min_share": min_share, "subjects_crossing": counts}
        )

    return PlaneShift(
        ranges,
        tolerance,
        fitted_on=fitted_on,
        section_planes=list(SECTION_PLANES if section_planes is None else section_planes),
    )
