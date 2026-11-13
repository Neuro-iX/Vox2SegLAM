"""Landmark ground truth, from either form it is annotated in."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..utils.geometry import flip_lps_ras

__all__ = ["extract_landmarks", "read_landmark_csv", "apply_affine_to_coords_zyx"]


# a float label volume (nibabel returns float64 when scl_slope/inter are set) must be rounded,
# not truncated: an id stored as 14.999999 would read as 14
def as_integer_labels(vol_zyx: np.ndarray) -> np.ndarray:
    if np.issubdtype(vol_zyx.dtype, np.integer):
        return vol_zyx
    return np.rint(vol_zyx).astype(np.int64)


# expects exactly one voxel per landmark id in the label volume; anything else is a data error
def extract_landmarks(lm_vol_zyx: np.ndarray, landmark_ids: Sequence[int]) -> np.ndarray:
    ids = [int(lid) for lid in landmark_ids]
    if any(lid == 0 for lid in ids):
        raise ValueError(
            "landmark id 0 is the background label and cannot be located; check task.landmark_ids."
        )

    index = np.nonzero(lm_vol_zyx)
    values = lm_vol_zyx[index]
    points = np.stack(index, axis=1)

    coords = np.zeros((len(ids), 3), dtype=np.float32)
    for i, lid in enumerate(ids):
        hit = values == lid
        n = int(hit.sum())
        if n == 0:
            raise ValueError(f"Landmark id={lid} is missing in volume.")
        if n > 1:
            raise ValueError(f"Landmark id={lid} has {n} voxels (expected exactly 1).")
        coords[i] = points[hit][0].astype(np.float32)

    return coords


# moves a landmark to match an affine image resampling (rotate/scale/translate about
# the volume center); uses R^-1 since resampling maps output coords back to input coords
def apply_affine_to_coords_zyx(
    coords_zyx: np.ndarray, A_zyx: np.ndarray, vol_shape_zyx: Sequence[int]
) -> np.ndarray:
    A = np.asarray(A_zyx, dtype=np.float64)
    R, t = A[:3, :3], A[:3, 3]
    c = (np.asarray(vol_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    out = (coords_zyx.astype(np.float64) - c - t) @ np.linalg.inv(R).T + c
    return out.astype(np.float32)


# A coordinate table is the tool-neutral form of an annotation: one row per landmark, in
# physical millimetres, exportable from 3D Slicer, ITK-SNAP or a script.
def read_landmark_csv(
    path: str | Path,
    n_landmarks: int,
    id_column: str = "participant_id",
    group_column: str = "dataset",
) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
    """({subject: (K, 3) LPS millimetres}, {subject: group}) from a coordinate table.

    Required columns: <id_column>, landmark_id (1..K), x_mm, y_mm, z_mm. Optional:
    coordinate_system per row ("LPS", the default, or "RAS", flipped to LPS) and
    <group_column>, a label to report separately (cohort, scanner, rater). A landmark with no
    row is returned as nan.
    """
    coords: dict[str, np.ndarray] = {}
    groups: dict[str, str | None] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        # a missing column reached the caller as a bare KeyError naming only the column, which
        # says nothing about which file is wrong or what it should have contained
        required = {id_column, "landmark_id", "x_mm", "y_mm", "z_mm"}
        absent = required - set(reader.fieldnames or ())
        if absent:
            raise ValueError(
                f"{path}: no column {', '.join(sorted(absent))}. A landmark table needs "
                f"{', '.join(sorted(required))}, and may add coordinate_system and "
                f"{group_column}; this one has "
                + (", ".join(reader.fieldnames) if reader.fieldnames else "no header")
            )
        for row in reader:
            subject = row[id_column]
            lid = int(row["landmark_id"])
            if not 1 <= lid <= n_landmarks:
                raise ValueError(
                    f"{path}: {subject} has landmark_id {lid}, outside 1..{n_landmarks}"
                )
            point = np.array([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])
            # any other spelling would be read as LPS, and RAS read as LPS mirrors left and right
            system = (row.get("coordinate_system") or "LPS").strip().upper()
            if system not in ("LPS", "RAS"):
                raise ValueError(
                    f"{path}: {subject} landmark {lid} has coordinate_system="
                    f"{row['coordinate_system']!r}; expected 'LPS' or 'RAS'"
                )
            if system == "RAS":
                point = flip_lps_ras(point)
            table = coords.setdefault(subject, np.full((n_landmarks, 3), np.nan))
            if not np.isnan(table[lid - 1]).all():
                raise ValueError(f"{path}: {subject} has two rows for landmark {lid}")
            table[lid - 1] = point
            groups.setdefault(subject, row.get(group_column))
    if not coords:
        raise ValueError(f"{path}: no landmark rows")
    return coords, groups
