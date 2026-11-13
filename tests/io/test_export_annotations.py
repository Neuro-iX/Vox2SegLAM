"""The exported coordinate table is the one the tools read back."""

import sys

import nibabel as nib
import numpy as np
import pytest

from vox2seglam.data.landmarks import read_landmark_csv

GRID = (16, 20, 16)  # ZYX
AFFINE = np.diag([-0.7, 0.7, 0.7, 1.0])  # LAS, so RAS+ x is negated
POSITIONS = [(4, 5, 6), (8, 10, 3), (11, 14, 9)]  # ZYX voxels


@pytest.fixture
def script():
    from vox2seglam.commands import export_annotations

    return export_annotations


@pytest.fixture
def volumes(tmp_path):
    directory = tmp_path / "Landmarks"
    directory.mkdir()
    for subject in ("p1", "p2"):
        volume = np.zeros(GRID, np.int16)
        for k, (z, y, x) in enumerate(POSITIONS, start=1):
            volume[z, y, x] = k
        nib.save(
            nib.Nifti1Image(volume.transpose(2, 1, 0), AFFINE),
            str(directory / f"{subject}_landmark_16.nii.gz"),
        )
    return directory


def run(script, *args) -> int:
    old, sys.argv = sys.argv, ["export_annotations.py", *args]
    try:
        return script.main()
    finally:
        sys.argv = old


def test_the_table_reads_back_as_the_landmarks_it_came_from(script, volumes, tmp_path):
    out = tmp_path / "landmarks.csv"
    assert run(script, "--landmark-dir", str(volumes), "--landmarks", "3", "--out", str(out)) == 0

    coords, _ = read_landmark_csv(out, 3)
    assert sorted(coords) == ["p1", "p2"]
    # read_landmark_csv returns LPS millimetres; map them back to voxels
    lps_to_voxel = np.linalg.inv(AFFINE[:3, :3] * np.array([-1.0, -1.0, 1.0])[:, None])
    for table in coords.values():
        voxels = (table - AFFINE[:3, 3] * np.array([-1, -1, 1])) @ lps_to_voxel.T
        assert np.allclose(voxels[:, ::-1], POSITIONS, atol=1e-3)


def test_ras_and_lps_differ_only_in_the_sign_of_two_axes(script, volumes, tmp_path):
    out = {}
    for system in ("LPS", "RAS"):
        out[system] = tmp_path / f"{system}.csv"
        run(
            script,
            "--landmark-dir",
            str(volumes),
            "--landmarks",
            "3",
            "--coordinate-system",
            system,
            "--out",
            str(out[system]),
        )
    lps, _ = read_landmark_csv(out["LPS"], 3)
    ras, _ = read_landmark_csv(out["RAS"], 3)  # flipped to LPS as it is read
    assert np.allclose(lps["p1"], ras["p1"])


def test_a_subject_without_a_volume_is_named(script, volumes, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("p1\np9\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="p9"):
        run(
            script,
            "--landmark-dir",
            str(volumes),
            "--landmarks",
            "3",
            "--ids",
            str(ids),
            "--out",
            str(tmp_path / "x.csv"),
        )


def test_a_table_missing_a_column_says_which_and_what_is_needed(tmp_path):
    """It reached the caller as a bare KeyError naming only the column -- not which file was
    wrong, nor what a landmark table is supposed to contain."""
    from vox2seglam.data.landmarks import read_landmark_csv

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError) as err:
        read_landmark_csv(wrong, 16)
    message = str(err.value)
    assert "wrong.csv" in message  # which file
    assert "landmark_id" in message and "x_mm" in message  # what it needs
    assert "this one has a, b" in message  # and what it had instead

    empty = tmp_path / "empty.csv"
    empty.write_text("participant_id,landmark_id,x_mm,y_mm,z_mm\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no landmark rows"):
        read_landmark_csv(empty, 16)
