"""The one file a prediction writes for its landmarks: a 3D Slicer markups list that also
carries the full record.

Slicer decides what a `.json` is by reading it, and refuses one without a `markups` key
("markups property is not found"). So `*_landmarks.mrk.json` *is* a markups document -- it opens
by drag and drop -- and everything Slicer has no place for lives beside it under one
`vox2seglam` key: which checkpoint ran, in what precision, the plane shifts the protocol
applied, and each landmark on the reference grid and in LPS as well as RAS. Slicer reads the
`markups` array and ignores the rest; a script reads `document["vox2seglam"]["landmarks"]`.

Slicer works in RAS, so the control points carry `native_ras_mm` and nothing is converted
here. The anatomical names come from the caller, which knows which vocabulary the model was
trained on; without them the points keep bare LM<id> labels, which is what a model outside the
protocol gets (see protocol.landmark_names).
"""

from typing import Any

__all__ = ["markups_document", "MARKUPS_SCHEMA"]

MARKUPS_SCHEMA = (
    "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/"
    "Markups/Resources/Schema/markups-schema-v1.0.3.json#"
)

_IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _control_point(
    lid: int, entry: dict[str, Any], key: str, names: tuple[dict, dict] | None
) -> dict[str, Any]:
    short, full = names or ({}, {})
    return {
        "id": str(lid),
        "label": short.get(lid, f"LM{lid}"),
        "description": full.get(lid, ""),
        "associatedNodeID": "",
        "position": [float(v) for v in entry[key]],
        "orientation": list(_IDENTITY),
        "selected": True,
        "locked": False,
        "visibility": True,
        "positionStatus": "defined",
    }


def _fiducials(landmarks, key: str, name: str, color, names) -> dict[str, Any]:
    points = [
        _control_point(int(lid), entry, key, names)
        for lid, entry in sorted(landmarks.items(), key=lambda kv: int(kv[0]))
    ]
    return {
        "type": "Fiducial",
        "name": name,
        "coordinateSystem": "RAS",
        "coordinateUnits": "mm",
        "locked": False,
        "fixedNumberOfControlPoints": False,
        "labelFormat": "%N-%d",
        "lastUsedControlPointNumber": len(points),
        "controlPoints": points,
        "measurements": [],
        # legible on a greyscale T1 and small enough not to hide the structure underneath
        "display": {
            "visibility": True,
            "opacity": 1.0,
            "color": list(color),
            "selectedColor": [0.2, 1.0, 0.4],
            "activeColor": [0.4, 1.0, 1.0],
            "glyphType": "Sphere3D",
            "glyphScale": 1.6,
            "glyphSize": 2.0,
            "useGlyphScale": True,
            "textScale": 2.5,
            "pointLabelsVisibility": True,
            "propertiesLabelVisibility": False,
            "sliceProjection": False,
            "snapMode": "unconstrained",
        },
    }


#: where a control point's RAS millimetres are read from, in order of preference. The two
#: entry points name them differently because they mean different spaces: predict.py
#: returns a scan's own millimetres, predict.py the reference grid's.
POSITION_KEYS = ("native_ras_mm", "world_ras_mm")


def markups_document(
    landmarks: dict[str, dict[str, Any]],
    *,
    name: str = "",
    record: dict[str, Any] | None = None,
    position_key: str | None = None,
    names: tuple[dict, dict] | None = None,
) -> dict[str, Any] | None:
    """The landmarks of one scan as a Slicer markups document.

    landmarks: {landmark id: {reference_voxel_zyx, native_lps_mm, native_ras_mm}}.
    name: the fiducial list's name in Slicer's data tree (the scan's stem is a good one).
    record: what this prediction was, stored under "vox2seglam" with the landmarks; Slicer
    ignores it, and it is what makes this file a record and not only a set of points.
    position_key: which of each entry's coordinate systems the control points take; by default
    the first of POSITION_KEYS present. None is returned when the landmarks carry no RAS
    millimetres at all, which is the one case Slicer could not place them from.
    names: ({id: short}, {id: anatomical name}) of this model's vocabulary, from
    protocol.landmark_names. Omitted, the points are labelled LM<id> and described by nothing:
    a wrong name is worse than none, and the ids of two protocols overlap.

    One fiducial list comes out, in the scan's own space -- the space the segmentation is also
    returned in, so the two overlay. The record keeps "reference_ras_mm" for the reference
    grid, which is where the prepared volume lives; nothing draws it unless you ask.
    """
    if position_key is None:
        entry = next(iter(landmarks.values()), {})
        position_key = next((k for k in POSITION_KEYS if k in entry), "")
        if not position_key:
            return None  # no RAS millimetres: nothing Slicer could place
    stem = name or "landmarks"
    markups = [_fiducials(landmarks, position_key, stem, (1.0, 0.75, 0.0), names)]
    document = {"@schema": MARKUPS_SCHEMA, "markups": markups}
    if record is not None:
        document["vox2seglam"] = {**record, "landmarks": landmarks}
    return document
