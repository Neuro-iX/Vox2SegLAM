"""Plane geometry and rule semantics of the anatomical protocol."""

import numpy as np

from vox2seglam.protocol import (
    LANDMARK_NAMES,
    PLANE_AXIAL,
    PLANE_CORONAL,
    PLANE_SAGITTAL,
    REGION_TO_CLASS,
)
from vox2seglam.protocol.geometry import (
    PLANE_NORMAL,
    plane_from_landmarks,
    protocol_planes,
    rule_depth,
)


def test_protocol_size(spec):
    assert len(spec.planes) == 21 and len(spec.rules) == 16


def test_vocabulary_comes_from_the_data_files(spec):
    """The class labels and landmark names of the papers, read from protocol/data/*.yaml."""
    assert REGION_TO_CLASS == {
        "LV_L": 1,
        "LV_R": 2,
        "V3": 3,
        "NACC_L": 4,
        "NACC_R": 5,
        "PUT_L": 6,
        "PUT_R": 7,
        "BRAINSTEM": 8,
        "TH_L": 9,
        "TH_R": 10,
        "IH_L": 11,
        "IH_R": 12,
        "VDC_A_L": 13,
        "VDC_A_R": 14,
        "VDC_P_L": 15,
        "VDC_P_R": 16,
    }
    assert list(LANDMARK_NAMES.values()) == [
        "Left Putamen Anterior Start",
        "Right Putamen Anterior Start",
        "Left NAcc-Putamen Anterior Contact",
        "Right NAcc-Putamen Anterior Contact",
        "Left NAcc-Putamen Posterior Contact",
        "Right NAcc-Putamen Posterior Contact",
        "Left NAcc Posterior Limit",
        "Right NAcc Posterior Limit",
        "Third Ventricle Anterior Start",
        "Anterior Commissure",
        "Left Mammillary Body",
        "Right Mammillary Body",
        "Left Continuity of Atrium",
        "Right Continuity of Atrium",
        "Posterior Commissure",
        "Prepontine Fissure",
    ]
    assert list(LANDMARK_NAMES) == list(range(1, 17))
    assert {r.region_name: r.region_class for r in spec.rules} == REGION_TO_CLASS


def test_one_landmark_plane_follows_the_scanner_axis():
    coords = np.array([[10.0, 20.0, 30.0]])
    for kind in (PLANE_AXIAL, PLANE_CORONAL, PLANE_SAGITTAL):
        centroid, normal = plane_from_landmarks(coords, [1], kind)
        assert np.allclose(centroid, coords[0]) and np.allclose(normal, PLANE_NORMAL[kind])


def test_two_landmark_planes_contain_the_line():
    coords = np.array([[10.0, 20.0, 30.0], [14.0, 26.0, 33.0]])
    line = coords[1] - coords[0]
    for kind, kept_axis in ((PLANE_SAGITTAL, [1, 0, 0]), (PLANE_AXIAL, [0, 0, 1])):
        _, normal = plane_from_landmarks(coords, [1, 2], kind)
        assert np.isclose(np.linalg.norm(normal), 1)
        assert np.isclose(normal @ kept_axis, 0)  # keeps that scanner axis
        assert normal @ PLANE_NORMAL[kind] > 0  # oriented like its type
        assert np.isclose(
            normal @ [0, line[1], line[2]]
            if kind == PLANE_SAGITTAL
            else normal @ [line[0], line[1], 0],
            0,
        )


def test_tolerance_moves_the_boundary(spec, landmarks16):
    planes = protocol_planes(spec, landmarks16)
    rule = next(r for r in spec.rules if r.region_name == "TH_L")  # left of the midline
    x0 = planes["midline_ac"][0][2]
    vox = np.array([[20.0, 20.0, x0 + 1], [20.0, 20.0, x0], [20.0, 20.0, x0 - 1]])
    assert list(rule_depth(vox, planes, rule) <= 0) == [True, True, False]
    assert all(rule_depth(vox, planes, rule, {"midline_ac": 1.0}) <= 0)


def test_near_coronal_plane_keeps_its_orientation(spec):
    """brainstem_top (axial plane through PC, LM15, and PPF, LM16) is nearly coronal: its
    orientation must not depend on whether PPF lies slightly in front of or behind PC."""
    from vox2seglam.protocol.geometry import REFERENCE_LANDMARKS

    ref = np.load(REFERENCE_LANDMARKS)
    normals = []
    for dy in (+3.0, -3.0):  # PPF 3 voxels anterior, then posterior, to PC
        coords = ref.copy()
        coords[15] = coords[14] + [-20.0, dy, 0.0]
        normals.append(protocol_planes(spec, coords)["brainstem_top"][1])
    assert all(n[1] > 0.9 for n in normals)  # both point anteriorly, like the reference
    rule = next(r for r in spec.rules if r.region_name == "BRAINSTEM")
    planes = protocol_planes(spec, coords)
    behind_pc = coords[14] + [-5.0, -6.0, 0.0]  # midbrain, behind PC, below PC
    assert rule_depth(behind_pc[None], planes, rule)[0] <= 0


def test_voxels_on_an_oblique_plane_are_on_both_sides(spec, landmarks16):
    """An inclusive half-space keeps voxels on its plane, even an oblique one whose signed
    distance is zero only up to float rounding (the landmarks themselves)."""
    coords = np.round(landmarks16)
    planes = protocol_planes(spec, coords)
    rules = {r.region_name: r for r in spec.rules}
    on_plane = coords[[12, 4]]  # LM13 and LM5 lie on Pi(13,5)
    centroid, normal = planes["lv_body_floor_L"]
    assert np.allclose((on_plane - centroid) @ normal, 0, atol=1e-6)
    assert all(rule_depth(on_plane[:1], planes, rules["IH_L"]) <= 0)  # LM13: IH side
    assert all(rule_depth(on_plane[:1], planes, rules["LV_L"]) <= 0)  # LV: atrium section


def test_missing_reference_landmarks_is_an_error(spec, landmarks16, monkeypatch, tmp_path):
    """Without the reference, two-landmark planes keep the scanner-axis orientation:
    brainstem_top flips and compliant voxels read as violations. That must fail loudly."""
    import pytest

    from vox2seglam.protocol import geometry

    monkeypatch.setattr(geometry, "REFERENCE_LANDMARKS", tmp_path / "absent.npy")
    monkeypatch.setattr(geometry, "_REFERENCE_PLANES", {})
    with pytest.raises(FileNotFoundError, match="is missing"):
        geometry.protocol_planes(spec, landmarks16)


def test_a_class_map_says_what_its_numbers_mean():
    """A segmentation is a volume of integers. Nothing in it says that 6 is the left putamen,
    and the second published model has 39 of them, so both prediction commands write the
    names beside the file."""
    from vox2seglam.protocol import class_names

    sixteen = class_names(17)
    assert sixteen[0] == "Background" and sixteen[6] == "Putamen Left"
    assert len(sixteen) == 17

    thirty_nine = class_names(40)
    assert len(thirty_nine) == 40
    assert thirty_nine[27] == "Left Hippocampal Formation"
    assert thirty_nine[39] == "Corpus Callosum Anterior"

    # the two sets number the same anatomy differently: naming one with the other would be
    # a silent mislabelling, so a cohort of somebody else's gets no names at all
    assert sixteen[6] != thirty_nine[6]
    assert class_names(9) is None
