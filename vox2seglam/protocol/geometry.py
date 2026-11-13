"""Protocol planes built from landmarks, and where voxels stand with respect to the rules."""

from pathlib import Path

import numpy as np

from .constants import PLANE_AXIAL, PLANE_CORONAL, PLANE_SAGITTAL, STRICT_EPS
from .pql_parser import ProtocolSpec, RuleSpec

__all__ = [
    "PLANE_NORMAL",
    "plane_from_landmarks",
    "protocol_planes",
    "clause_groups",
    "clause_depths",
    "rule_depth",
    "allowed",
    "plane_distance",
]

# normal of a plane built from a single landmark: the scanner axis of its type
PLANE_NORMAL = {
    PLANE_AXIAL: np.array([1.0, 0.0, 0.0]),
    PLANE_CORONAL: np.array([0.0, -1.0, 0.0]),
    PLANE_SAGITTAL: np.array([0.0, 0.0, -1.0]),
}

Plane = tuple[np.ndarray, np.ndarray]


def plane_from_landmarks(coords: np.ndarray, landmark_ids: list[int], plane_type: int) -> Plane:
    """Plane of `plane_type` through one or two landmarks (ids 1..K, rows of `coords`).

    One landmark: the plane normal to the scanner axis of its type. Two: the plane of that
    type that contains the line joining them - a sagittal plane keeps the vertical axis, an
    axial plane the left-right axis, a coronal plane is tilted along the line only.
    """
    pts = coords[[i - 1 for i in landmark_ids]]
    centroid = pts.mean(axis=0)
    axis = PLANE_NORMAL[plane_type]
    if len(pts) == 1:
        return centroid, axis.copy()

    line = pts[1] - pts[0]
    if plane_type == PLANE_SAGITTAL:
        normal = np.array([0.0, -line[2], line[1]])
    elif plane_type == PLANE_AXIAL:
        normal = np.array([line[1], -line[0], 0.0])
    else:
        unit = line / (np.linalg.norm(line) + 1e-8)
        normal = axis - float(axis @ unit) * unit

    norm = np.linalg.norm(normal)
    normal = normal / (norm + 1e-8) if norm > 1e-6 else axis.copy()
    if float(normal @ axis) < 0:
        normal = -normal
    return centroid, normal


#: mean landmark configuration of the 80 HCP training subjects (ZYX voxels), the protocol's
#: reference for orienting planes
REFERENCE_LANDMARKS = Path(__file__).resolve().parent / "data" / "reference_landmarks_zyx.npy"
_REFERENCE_PLANES: dict = {}


# The reference orients every plane consistently across subjects. Without it, two-landmark
# planes (brainstem_top) can flip and turn compliant voxels into violations, so a missing or
# malformed reference is an error, not a default.
def _reference_planes(spec: ProtocolSpec) -> dict[str, Plane]:
    key = tuple((p.plane_id, tuple(p.landmark_ids), p.plane_type) for p in spec.planes)
    if key not in _REFERENCE_PLANES:
        if not REFERENCE_LANDMARKS.is_file():
            raise FileNotFoundError(
                f"{REFERENCE_LANDMARKS} is missing: plane orientation would fall back to "
                "the scanner axis alone, which flips brainstem_top. It ships as package "
                "data (pyproject.toml, [tool.setuptools.package-data]) -- reinstall the "
                "package."
            )
        ref = np.asarray(np.load(REFERENCE_LANDMARKS), np.float64)
        n_lm = max(i for p in spec.planes for i in p.landmark_ids)
        if ref.ndim != 2 or ref.shape[1] != 3 or len(ref) < n_lm:
            raise ValueError(
                f"{REFERENCE_LANDMARKS}: expected at least ({n_lm}, 3) ZYX coordinates, "
                f"got {ref.shape}"
            )
        _REFERENCE_PLANES[key] = {
            p.plane_id: plane_from_landmarks(ref, p.landmark_ids, p.plane_type) for p in spec.planes
        }
    return _REFERENCE_PLANES[key]


def protocol_planes(spec: ProtocolSpec, coords: np.ndarray) -> dict[str, Plane]:
    """Every plane of the protocol, from (K, 3) landmark coordinates.

    Computed in float64, so voxels exactly on a plane through landmark centres stay at distance 0.

    Each normal is oriented like the same plane on the reference landmark configuration
    (REFERENCE_LANDMARKS). Orienting by the scanner axis alone is unstable for a two-landmark
    plane nearly perpendicular to it: brainstem_top, an "axial" plane through PC and PPF, is
    within 5-27 degrees of coronal and flipped whenever PPF lay slightly behind PC, inverting
    the rule.
    """
    coords = np.asarray(coords, dtype=np.float64)
    planes = {
        p.plane_id: plane_from_landmarks(coords, p.landmark_ids, p.plane_type) for p in spec.planes
    }
    ref = _reference_planes(spec)  # raises rather than skipping the correction
    for pid, (centroid, normal) in planes.items():
        if float(normal @ ref[pid][1]) < 0:
            planes[pid] = (centroid, -normal)
    return planes


# a voxel centre within this distance of a plane (voxels) lies on it: float rounding of oblique planes
ON_PLANE_EPS = 1e-9


def clause_groups(rule: RuleSpec) -> list[list]:
    """The rule as OR-groups of AND-clauses: a voxel complies if one group holds entirely."""
    groups = [list(rule.clauses)] + [list(g) for g in rule.or_groups]
    return [g for g in groups if g]


def clause_depths(
    vox: np.ndarray,
    planes: dict[str, Plane],
    rule: RuleSpec,
    tolerance: dict[str, float] | None = None,
) -> list[np.ndarray]:
    """Per OR-group of the rule, a (clauses, N) array: how far each voxel (N, 3) lies past
    each clause's plane, in voxels, minus that plane's tolerance (default 0)."""
    tolerance = tolerance or {}
    groups = []
    for group in clause_groups(rule):
        depths = []
        for clause in group:
            centroid, normal = planes[clause.plane_id]
            signed = (vox - centroid) @ normal
            signed = np.where(
                np.abs(signed) < ON_PLANE_EPS, 0.0, signed
            )  # on the plane up to rounding
            depth = -signed if clause.sign == +1 else signed
            if getattr(clause, "strict", False):  # terminal boundary: the plane itself is excluded
                depth = depth + STRICT_EPS
            depths.append(depth - tolerance.get(clause.plane_id, 0.0))
        groups.append(np.stack(depths))
    return groups


def rule_depth(
    vox: np.ndarray,
    planes: dict[str, Plane],
    rule: RuleSpec,
    tolerance: dict[str, float] | None = None,
) -> np.ndarray:
    """How far each voxel (N, 3) lies outside what its rule allows, in voxels; <= 0 complies:
    a group is as violated as its worst clause, the rule as its least violated group."""
    groups = clause_depths(vox, planes, rule, tolerance)
    return np.min([depths.max(0) for depths in groups], axis=0)


def allowed(
    vox: np.ndarray,
    planes: dict[str, Plane],
    rule: RuleSpec,
    tolerance: dict[str, float] | None = None,
) -> np.ndarray:
    """Boolean (N,): the voxels the rule lets its region occupy."""
    return rule_depth(vox, planes, rule, tolerance) <= 0


def plane_distance(shape: tuple[int, int, int], planes: dict[str, Plane]) -> np.ndarray:
    """Distance of every voxel of a (Z, Y, X) grid to the nearest plane, in voxels."""
    zz, yy, xx = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in shape), indexing="ij")
    best = np.full(shape, np.inf, np.float32)
    for (cz, cy, cx), (nz, ny, nx) in planes.values():
        dist = np.abs((zz - cz) * nz + (yy - cy) * ny + (xx - cx) * nx)
        np.minimum(best, dist, out=best)
    return best
