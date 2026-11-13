"""evaluate.py as a user runs it: a predict.py output scored against their own reference."""

import json
import subprocess

import nibabel as nib
import numpy as np
import pytest

from conftest import command

SCRIPT = "evaluate"
AFFINE = np.diag([-1.0, 1.0, 1.0, 1.0])


def run(*args):
    return subprocess.run(
        [*command(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=600
    )


@pytest.fixture
def prediction(tmp_path):
    """What predict.py writes: a markups file carrying the record, and a class map."""
    out = tmp_path / "results"
    out.mkdir()
    labels = np.zeros((20, 20, 20), np.uint8)
    labels[6:12, 6:12, 6:12] = 6  # PUT_L
    nib.save(nib.Nifti1Image(labels, AFFINE), str(out / "sub-01_T1w_seg.nii.gz"))

    spread = np.linspace(4.0, 16.0, 16)
    marks = {
        str(i + 1): {
            "native_lps_mm": [float(spread[i]), float(spread[-1 - i]), 8.0],
            "reference_voxel_zyx": [8.0, 8.0, 8.0],
        }
        for i in range(16)
    }
    (out / "sub-01_T1w_landmarks.mrk.json").write_text(
        json.dumps({"@schema": "markups", "markups": [], "vox2seglam": {"landmarks": marks}}),
        encoding="utf-8",
    )
    return out, labels


def test_it_scores_a_segmentation_against_a_reference(tmp_path, prediction):
    out, labels = prediction
    truth = tmp_path / "truth"
    truth.mkdir()
    nib.save(nib.Nifti1Image(labels, AFFINE), str(truth / "sub-01_T1w.nii.gz"))

    done = run("--predictions", out, "--segmentations", truth, "--out", tmp_path / "scores.json")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "segmentation" in done.stdout and "DSC 100.00" in done.stdout

    scored = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))["subjects"][
        "sub-01_T1w"
    ]
    assert scored["segmentation"]["mean"]["dice"] == 1.0


def test_a_mismatched_grid_is_refused_rather_than_scored(tmp_path, prediction):
    """Two volumes of different sizes are not the same subject; scoring them is meaningless."""
    out, _ = prediction
    truth = tmp_path / "truth"
    truth.mkdir()
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), AFFINE), str(truth / "sub-01_T1w.nii.gz")
    )
    done = run("--predictions", out, "--segmentations", truth)
    assert done.returncode != 0
    assert "same volume on the same grid" in done.stdout + done.stderr


def test_the_protocol_is_checked_on_request_and_without_any_reference(tmp_path, prediction):
    """Compliance needs only the predicted landmarks, which is what makes it usable on a
    cohort nobody has segmented. It is asked for: measured against the unshifted planes it
    is not the figure predict.py reports on the same map, and reporting it unbidden made the
    two commands look as though they disagreed."""
    out, _ = prediction
    quiet = run("--predictions", out)
    assert quiet.returncode == 0
    assert "outside the admissible zones" not in quiet.stdout  # not measured unasked
    assert "nothing to score" in quiet.stdout

    done = run("--predictions", out, "--protocol", "--out", tmp_path / "scores.json")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "protocol" in done.stdout
    scored = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    one = scored["subjects"]["sub-01_T1w"]
    assert one["protocol"]["violating_mm3"] >= 0.0
    assert "segmentation" not in one and "landmarks" not in one
    # what the console printed is in the file, not left for the reader to recompute
    assert scored["overall"]["n_subjects"] == len(scored["subjects"])
    assert scored["overall"]["protocol"]["violating_mm3"] >= 0.0


def test_a_directory_that_is_not_a_predict_output_says_so(tmp_path):
    done = run("--predictions", tmp_path)
    assert done.returncode != 0 and "predict.py output" in done.stdout + done.stderr


def test_it_reads_either_entry_point(tmp_path, prediction):
    """predict.py writes a scan's own millimetres, predict_prepared.py the model grid's
    voxels. Both are predictions to be scored, and asking for the wrong key used to raise a
    KeyError from inside a list comprehension."""
    out, labels = prediction
    marks = {
        str(i + 1): {
            "voxel_zyx": [8.0 + i % 5, 8.0 + i % 7, 8.0 + i % 3],
            "world_ras_mm": [float(i), 1.0, 2.0],
        }
        for i in range(16)
    }
    (out / "sub-02_T1w_landmarks.mrk.json").write_text(
        json.dumps({"@schema": "markups", "markups": [], "vox2seglam": {"landmarks": marks}}),
        encoding="utf-8",
    )
    nib.save(nib.Nifti1Image(labels, AFFINE), str(out / "sub-02_T1w_seg.nii.gz"))

    done = run("--predictions", out, "--protocol", "--out", tmp_path / "scores.json")
    assert done.returncode == 0, done.stdout + done.stderr
    scored = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))["subjects"]
    assert "protocol" in scored["sub-02_T1w"] and "protocol" in scored["sub-01_T1w"]


def test_landmarks_in_no_usable_system_say_so(tmp_path, prediction):
    """Reading them is what the protocol check needs, so the subject must carry a class map
    for the coordinates to be reached at all."""
    out, labels = prediction
    (out / "sub-03_T1w_landmarks.mrk.json").write_text(
        json.dumps(
            {"markups": [], "vox2seglam": {"landmarks": {"1": {"voxel_xyz": [1.0, 2.0, 3.0]}}}}
        ),
        encoding="utf-8",
    )
    nib.save(nib.Nifti1Image(labels, AFFINE), str(out / "sub-03_T1w_seg.nii.gz"))
    done = run("--predictions", out, "--protocol")
    assert done.returncode != 0
    assert "none of which is" in done.stdout + done.stderr
