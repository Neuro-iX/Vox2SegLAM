"""External cohorts: the ground-truth table, truncated volumes, mapping back to native space."""

import gzip

import numpy as np
import pytest
import SimpleITK as sitk

from vox2seglam.data.landmarks import read_landmark_csv
from vox2seglam.data.prepare.register import check_complete, to_native
from vox2seglam.utils.geometry import flip_lps_ras

HEADER = "dataset,participant_id,landmark_id,coordinate_system,x_mm,y_mm,z_mm\n"


def write_csv(tmp_path, rows):
    path = tmp_path / "landmarks.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def test_csv_in_lps_and_ras(tmp_path):
    path = write_csv(tmp_path, ["dsA,sub-1,1,LPS,1,2,3\n", "dsA,sub-1,2,RAS,1,2,3\n"])
    coords, groups = read_landmark_csv(path, 2)
    assert groups == {"sub-1": "dsA"}
    assert np.allclose(coords["sub-1"][0], [1, 2, 3])
    assert np.allclose(coords["sub-1"][1], [-1, -2, 3])


def test_csv_missing_landmark_is_nan(tmp_path):
    path = write_csv(tmp_path, ["dsA,sub-1,1,LPS,1,2,3\n"])
    coords, _ = read_landmark_csv(path, 3)
    assert np.isnan(coords["sub-1"][1:]).all()
    assert not np.isnan(coords["sub-1"][0]).any()


def test_csv_rejects_duplicate_and_out_of_range(tmp_path):
    dup = write_csv(tmp_path, ["dsA,sub-1,1,LPS,1,2,3\n", "dsA,sub-1,1,LPS,4,5,6\n"])
    with pytest.raises(ValueError, match="two rows"):
        read_landmark_csv(dup, 2)
    bad = write_csv(tmp_path, ["dsA,sub-1,5,LPS,1,2,3\n"])
    with pytest.raises(ValueError, match="outside 1..2"):
        read_landmark_csv(bad, 2)


def test_csv_without_optional_columns(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("participant_id,landmark_id,x_mm,y_mm,z_mm\nsub-1,1,1,2,3\n", encoding="utf-8")
    coords, groups = read_landmark_csv(path, 1)
    assert groups == {"sub-1": None}  # no dataset column: no grouping, LPS assumed
    assert np.allclose(coords["sub-1"][0], [1, 2, 3])


def test_flip_is_its_own_inverse():
    point = np.array([1.5, -2.5, 3.5])
    assert np.allclose(flip_lps_ras(flip_lps_ras(point)), point)


def test_check_complete_accepts_whole_and_rejects_truncated(tmp_path):
    whole = tmp_path / "whole.nii.gz"
    whole.write_bytes(gzip.compress(b"x" * 4096))
    check_complete(whole)
    cut = tmp_path / "cut.nii.gz"
    cut.write_bytes(gzip.compress(b"x" * 4096)[:-8])
    with pytest.raises(ValueError, match="could not be read to the end"):
        check_complete(cut)


def test_to_native_round_trips_through_the_identity(tmp_path):
    reference = sitk.Image(8, 9, 10, sitk.sitkFloat32)
    reference.SetSpacing((0.7, 0.7, 0.7))
    reference.SetOrigin((-3.0, 2.0, 1.0))
    coords_zyx = np.array([[0.0, 0.0, 0.0], [4.0, 3.0, 2.0]])
    native = to_native(coords_zyx, reference, sitk.Transform(3, sitk.sitkIdentity))
    expected = [
        reference.TransformContinuousIndexToPhysicalPoint([x, y, z]) for z, y, x in coords_zyx
    ]
    assert np.allclose(native, expected)


def test_prepared_scan_names_are_one_contract(tmp_path):
    """What prepare.py writes is what predict.py finds again."""
    from vox2seglam.data.prepare.register import PreparedScan, prepared_scans

    out = PreparedScan.under(tmp_path, "/elsewhere/sub-01_T1w.nii.gz")
    assert out.stem == "sub-01_T1w"
    assert out.model_input.name == "sub-01_T1w_aligned_brain_n4.nii.gz"
    assert {p.parent for p in (out.aligned, out.transform, out.brain, out.model_input)} == {
        tmp_path
    }

    assert prepared_scans(tmp_path) == []  # nothing prepared yet: found, not guessed
    out.transform.write_text("", encoding="utf-8")  # the marker prepare.py leaves
    assert prepared_scans(tmp_path) == [out]  # same object, both sides


def test_nifti_stem_strips_both_extensions():
    from vox2seglam.data.prepare.register import nifti_stem

    assert nifti_stem("a/b/sub-01_T1w.nii.gz") == "sub-01_T1w"
    assert nifti_stem("sub-01_T1w.nii") == "sub-01_T1w"


def test_csv_rejects_an_unknown_coordinate_system(tmp_path):
    """Reading RAS as LPS mirrors left and right, so only the two spellings are allowed."""
    path = tmp_path / "bad.csv"
    path.write_text(
        "participant_id,landmark_id,coordinate_system,x_mm,y_mm,z_mm\nsub-1,1,RAS ,1,2,3\n",
        encoding="utf-8",
    )
    coords, _ = read_landmark_csv(path, 1)  # trailing space tolerated
    assert np.allclose(coords["sub-1"][0], [-1, -2, 3])
    path.write_text(
        "participant_id,landmark_id,coordinate_system,x_mm,y_mm,z_mm\nsub-1,1,RSA,1,2,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coordinate_system"):
        read_landmark_csv(path, 1)


def test_every_unreadable_gz_becomes_one_actionable_message(tmp_path):
    """Three different failures reach the same clause, and neither the exception type nor a
    guess at the cause tells them apart: BadGzipFile covers both a file that never was gzip
    and a damaged deflate stream, while a truncated one raises EOFError. The message quotes
    what gzip said instead of interpreting it."""
    corrupt = tmp_path / "corrupt.nii.gz"
    raw = bytearray(gzip.compress(b"x" * 8192))
    raw[40] ^= 0xFF  # damage the deflate stream itself
    corrupt.write_bytes(bytes(raw))

    cut = tmp_path / "cut.nii.gz"
    cut.write_bytes(gzip.compress(b"x" * 8192)[:40])

    plain = tmp_path / "plain.nii.gz"
    plain.write_text("this was never compressed", encoding="utf-8")

    for path, said in (
        (corrupt, "Incorrect length"),
        (cut, "ended before"),
        (plain, "Not a gzipped file"),
    ):
        with pytest.raises(ValueError) as err:
            check_complete(path)
        assert "could not be read to the end" in str(err.value)
        assert said in str(err.value)  # gzip's own words, not an interpretation


def test_labels_return_to_the_native_grid_without_inventing_any(tmp_path):
    """A user wants the segmentation on the scan they supplied, not on the reference grid.
    Nearest neighbour, and the exact inverse of a rigid transform, so no label is created
    and none is interpolated into a value that is not a class."""
    from vox2seglam.data.prepare.register import labels_to_native

    labels = np.zeros((40, 50, 40), np.uint8)
    labels[14:26, 18:32, 14:26] = 7
    labels[8:12, 8:12, 8:12] = 3

    reference = sitk.GetImageFromArray(labels)
    reference.SetSpacing((0.7, 0.7, 0.7))
    native = sitk.Image(40, 50, 40, sitk.sitkUInt8)
    native.SetSpacing((0.7, 0.7, 0.7))

    transform = sitk.Euler3DTransform()
    transform.SetTranslation((2.0, -1.0, 3.0))
    result = labels_to_native(labels, reference, native, transform)
    out = sitk.GetArrayFromImage(result)

    assert out.shape == (40, 50, 40)
    assert set(np.unique(out)) <= {0, 3, 7}  # no label invented
    assert (out == 7).sum() == (labels == 7).sum()  # a rigid move preserves counts
    assert (out == 3).sum() == (labels == 3).sum()


def test_an_identity_transform_returns_the_labels_unchanged():
    from vox2seglam.data.prepare.register import labels_to_native

    labels = np.zeros((16, 16, 16), np.uint8)
    labels[4:12, 4:12, 4:12] = 5
    grid = sitk.GetImageFromArray(labels)
    out = sitk.GetArrayFromImage(labels_to_native(labels, grid, grid, sitk.Euler3DTransform()))
    assert np.array_equal(out, labels)
