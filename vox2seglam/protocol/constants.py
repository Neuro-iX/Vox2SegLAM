from __future__ import annotations

import os

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "data")

PROTOCOL_LANDMARKS = os.path.join(_DATA_DIR, "landmarks.yaml")
PROTOCOL_REGIONS = os.path.join(_DATA_DIR, "regions.yaml")
PROTOCOL_REGIONS40 = os.path.join(_DATA_DIR, "regions40.yaml")


def _vocabulary(path: str, key: str) -> list[dict]:
    """The {id, short, name} entries of a vocabulary file (data/*.yaml)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)[key]


PLANE_AXIAL = 0
PLANE_CORONAL = 1
PLANE_SAGITTAL = 2

CONSTRAINT_MAP = {
    "right_to": (PLANE_SAGITTAL, +1),
    "left_to": (PLANE_SAGITTAL, -1),
    "posterior_to": (PLANE_CORONAL, +1),
    "anterior_to": (PLANE_CORONAL, -1),
    "superior_to": (PLANE_AXIAL, +1),
    "inferior_to": (PLANE_AXIAL, -1),
}
# Terminal boundaries: a structure ends before the plane (the manual stops segmenting it on the
# landmark section), so the half-space excludes the plane. "strictly_anterior_to(p)" admits
# delta > 0 instead of delta >= 0; voxels lying exactly on the plane then violate the clause.
STRICT_PREFIX = "strictly_"
CONSTRAINT_MAP.update({STRICT_PREFIX + name: v for name, v in list(CONSTRAINT_MAP.items())})
#: depth (voxels) given to a voxel lying exactly on a strict plane: > 0, so that it violates the
#: clause, and negligible (0.0007 mm) compared with any distance reported in mm
STRICT_EPS = 1e-3

PLANE_CONSTRUCTORS = {
    "sagittal_plane": PLANE_SAGITTAL,
    "coronal_plane": PLANE_CORONAL,
    "axial_plane": PLANE_AXIAL,
}

#: {structure short name: class label}, from data/regions.yaml - the only place they are defined
REGION_TO_CLASS = {e["short"]: int(e["id"]) for e in _vocabulary(PROTOCOL_REGIONS, "regions")}

#: {landmark id: full name}, from data/landmarks.yaml
LANDMARK_NAMES = {int(e["id"]): e["name"] for e in _vocabulary(PROTOCOL_LANDMARKS, "landmarks")}

#: {landmark id: short name (LM1, LM2, ...)}, from data/landmarks.yaml
LANDMARK_SHORT = {int(e["id"]): e["short"] for e in _vocabulary(PROTOCOL_LANDMARKS, "landmarks")}


def landmark_names(n_landmarks: int, n_classes: int) -> tuple[dict, dict] | None:
    """({id: short}, {id: full name}) of the protocol's landmarks, or None when this model's
    landmarks are not the protocol's.

    Decided per model, never per id: the AFIDs model predicts 32 landmarks whose ids 1..16
    collide with the protocol's and mean something else, so labelling them one by one from this
    vocabulary described the wrong anatomy on half the points. The protocol's pairing is 16
    landmarks with 17 classes; anything else keeps bare LM<id> labels. A cohort of somebody
    else's annotated with 16 landmarks and 17 classes is the one case this cannot tell apart
    (see configs/examples/own_data.yaml).
    """
    if (n_landmarks, n_classes) == (len(LANDMARK_NAMES), len(REGION_TO_CLASS) + 1):
        return LANDMARK_SHORT, LANDMARK_NAMES
    return None


def label_table(n_classes: int) -> str | None:
    """The class map's labels as a BIDS discrete-segmentation lookup table (`*_dseg.tsv`):
    `index` and `name`, which the specification requires, plus the short form and, where the
    structure has one, its FreeSurfer label. None for a label set this package does not know.

    A NIfTI of integers says nothing on its own; every viewer and every script needs this
    beside it, which is why a prediction writes one and the published record carries one.
    """
    files = {
        len(REGION_TO_CLASS) + 1: (PROTOCOL_REGIONS, "regions"),
        40: (PROTOCOL_REGIONS40, "regions40"),
    }
    if n_classes not in files:
        return None
    path, key = files[n_classes]
    entries = _vocabulary(path, key)
    freesurfer = any(e.get("freesurfer") is not None for e in entries)
    head = ["index", "name", "abbreviation"] + (["freesurfer"] if freesurfer else [])
    rows = ["\t".join(head), "\t".join(["0", "Background", "BG"] + (["0"] if freesurfer else []))]
    for e in entries:
        row = [str(e["id"]), e["name"], e["short"]]
        if freesurfer:
            row.append("n/a" if e.get("freesurfer") is None else str(e["freesurfer"]))
        rows.append("\t".join(row))
    return "\n".join(rows) + "\n"


def class_names(n_classes: int) -> dict[int, str] | None:
    """{class label: structure name} of a published label set, background included; None for
    a cohort of somebody else's, whose names this package has no way of knowing."""
    files = {
        len(REGION_TO_CLASS) + 1: (PROTOCOL_REGIONS, "regions"),
        40: (PROTOCOL_REGIONS40, "regions40"),
    }
    if n_classes not in files:
        return None
    path, key = files[n_classes]
    return {0: "Background", **{int(e["id"]): e["name"] for e in _vocabulary(path, key)}}
