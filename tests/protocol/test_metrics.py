"""Segmentation, landmark and statistical metrics on cases with a known answer."""

import numpy as np

from vox2seglam.evaluate import (
    landmark_summary,
    paired_comparison,
    radial_errors,
    segmentation_metrics,
)


def cube(shape=(30, 30, 30), start=8, size=10, shift=0) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    m[start + shift : start + shift + size, start : start + size, start : start + size] = 1
    return m


def test_identical_masks_are_perfect():
    m = segmentation_metrics(cube(), cube(), {1: "A"}, spacing=(0.7,) * 3)["whole"]["per"]["A"]
    assert m["dice"] == 1 and m["assd"] == 0 and m["hd"] == 0
    assert m["precision"] == 1 and m["recall"] == 1


def test_shifted_cube_distances():
    m = segmentation_metrics(cube(shift=2), cube(), {1: "A"}, spacing=(0.7,) * 3)["whole"]
    assert np.isclose(m["per"]["A"]["hd"], 1.4)  # 2 voxels of 0.7 mm
    assert np.isclose(m["per"]["A"]["dice"], 0.8)
    assert m["mean"]["dice"] == m["per"]["A"]["dice"] == m["pooled"]["dice"]


def test_zone_restricts_the_counts():
    """Identical masks score 1 in any zone, so they must disagree somewhere or the test would
    pass even if `zones` were ignored."""
    agree = np.zeros((30, 30, 30), bool)
    agree[10:18, :, :] = True  # where cube(shift=2) and cube() overlap
    m = segmentation_metrics(
        cube(shift=2),
        cube(),
        {1: "A"},
        {"whole": np.ones((30,) * 3, bool), "agree": agree},
        spacing=(0.7,) * 3,
    )
    assert np.isclose(m["whole"]["per"]["A"]["dice"], 0.8)  # mismatch over the volume
    assert m["agree"]["per"]["A"]["dice"] == 1  # but perfect inside the zone


def test_landmark_metrics():
    e = radial_errors(
        np.array([[0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 1], [0, 3, 4]]), spacing=(1, 1, 1)
    )
    assert np.allclose(e, [1, 5])
    s = landmark_summary(e)
    assert s["mre"] == 3 and s["sdr2"] == 50 and s["sdr4"] == 50


def test_paired_comparison():
    c = paired_comparison([1, 2, 3, 4], [2, 3, 4, 5])
    assert c["improved"] == 4 and np.isclose(c["relative_change"], 40)
    assert paired_comparison([1, 2], [1, 2])["p"] == 1.0
    assert paired_comparison([np.nan], [1.0]) is None


def test_metrics_in_mm_require_a_spacing():
    """No metric reported in millimetres may assume this study's grid (see TaskCfg)."""
    import pytest

    with pytest.raises(TypeError):
        segmentation_metrics(cube(), cube(), {1: "A"})
    with pytest.raises(TypeError):
        radial_errors(np.zeros((2, 3)), np.zeros((2, 3)))


def test_landmark_summary_refuses_an_empty_sample():
    """nan +- nan would reach a reported table looking like a measurement."""
    import pytest

    with pytest.raises(ValueError, match="no error to summarise"):
        landmark_summary(np.array([]))
    one = landmark_summary(np.array([2.0]))  # SD needs two errors
    assert one["n"] == 1 and one["mre"] == 2.0 and np.isnan(one["sd"])


def test_surface_distances_are_nan_when_a_structure_is_missing():
    """With no voxel on one side, distance_transform_edt returns the distance to the crop edge,
    set by the margin and not by anatomy. Such a structure must report nan (skipped by the
    inter-structure mean) while its overlap metrics stay 0: missing it is a real failure."""
    pred = np.zeros((40, 40, 40), np.uint8)
    gt = np.zeros((40, 40, 40), np.uint8)
    pred[10:16, 10:16, 10:16] = 1  # A: predicted, absent from the GT
    pred[25:31, 25:31, 25:31] = 2
    gt[25:31, 25:31, 25:31] = 2  # B: exact

    m = segmentation_metrics(pred, gt, {1: "A", 2: "B"}, spacing=(0.7,) * 3)["whole"]
    a, b = m["per"]["A"], m["per"]["B"]
    assert np.isnan(a["assd"]) and np.isnan(a["hd"]) and np.isnan(a["hd95"])
    assert a["dice"] == 0 and a["precision"] == 0  # overlap is still measured
    assert b["assd"] == 0 and b["dice"] == 1  # the sound structure is untouched
    assert m["mean"]["assd"] == 0  # A skipped, not averaged in

    # the symmetric case (annotated but never predicted) behaves the same
    only_gt = segmentation_metrics(np.zeros_like(gt), gt, {2: "B"}, spacing=(0.7,) * 3)["whole"][
        "per"
    ]["B"]
    assert np.isnan(only_gt["assd"]) and only_gt["recall"] == 0
