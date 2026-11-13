"""Bringing a user's annotations onto the reference grid, and the dataset that results."""

import sys

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from vox2seglam.config import load_config
from vox2seglam.data.datasets import build_dataset
from vox2seglam.data.prepare.register import labels_to_reference, landmarks_to_reference

GRID = (16, 32, 16)  # ZYX, every side a multiple of 16: the U-Net halves four times,
# and Config.validate refuses a grid it could not reassemble
AFFINE = np.diag([-0.7, 0.7, 0.7, 1.0])  # LAS


def reference_file(path):
    nib.save(nib.Nifti1Image(np.ones(GRID, np.float32).transpose(2, 1, 0), AFFINE), str(path))
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def rigid():
    t = sitk.Euler3DTransform()
    t.SetRotation(0.05, -0.03, 0.04)
    t.SetTranslation((1.2, -0.8, 2.1))
    return t


POSITIONS = [(4, 5, 6), (8, 10, 3), (11, 14, 9), (6, 3, 12)]  # ZYX voxels


def native_points(reference, transform):
    """Where an annotator, working on the native scan, would read each reference voxel."""
    return np.array(
        [
            transform.TransformPoint(reference.TransformContinuousIndexToPhysicalPoint([x, y, z]))
            for z, y, x in POSITIONS
        ]
    )


def test_landmarks_arrive_on_the_voxel_they_came_from(tmp_path):
    reference, transform = reference_file(tmp_path / "ref.nii.gz"), rigid()
    volume = sitk.GetArrayFromImage(
        landmarks_to_reference(native_points(reference, transform), reference, transform)
    )
    for k, position in enumerate(POSITIONS, start=1):
        assert volume[position] == k
    assert np.count_nonzero(volume) == len(POSITIONS)


def test_landmarks_go_the_right_way_through_the_transform(tmp_path):
    """Applying the transform instead of its inverse lands them a few voxels off."""
    reference, transform = reference_file(tmp_path / "ref.nii.gz"), rigid()
    try:
        wrong = sitk.GetArrayFromImage(
            landmarks_to_reference(
                native_points(reference, transform), reference, transform.GetInverse()
            )
        )
    except ValueError:  # off the grid or onto each other: also not the voxels
        return
    assert not all(wrong[position] == k for k, position in enumerate(POSITIONS, start=1))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.__setitem__((1, 0), np.nan), "landmark 2 has no coordinates"),
        (lambda p: p.__setitem__(2, p[0]), "same voxel"),
        (lambda p: p.__setitem__(0, (500.0, 0.0, 0.0)), "outside the reference grid"),
    ],
)
def test_an_unusable_landmark_is_refused_not_guessed(tmp_path, mutate, message):
    reference = reference_file(tmp_path / "ref.nii.gz")
    points = native_points(reference, sitk.Euler3DTransform())
    mutate(points)
    with pytest.raises(ValueError, match=message):
        landmarks_to_reference(points, reference, sitk.Euler3DTransform())


def test_labels_keep_their_numbers_and_follow_the_transform(tmp_path):
    reference = reference_file(tmp_path / "ref.nii.gz")
    shift = sitk.TranslationTransform(3, (1.4, 0.0, 0.0))  # two voxels along x
    native = np.zeros(GRID, np.int16)
    native[4:8, 5:9, 6:10] = 7
    native[8:10, 5:9, 6:10] = 3
    image = sitk.GetImageFromArray(native)
    image.CopyInformation(reference)
    out = sitk.GetArrayFromImage(labels_to_reference(image, reference, shift))
    assert set(np.unique(out)) == {0, 3, 7}  # none invented
    assert np.array_equal(out[:, :, 4:8], native[:, :, 6:10])  # moved, not altered


@pytest.fixture
def script(monkeypatch):
    from vox2seglam.commands import prepare_dataset as module

    def identity(native, reference):
        return sitk.ReadImage(str(native), sitk.sitkFloat32), sitk.Euler3DTransform()

    monkeypatch.setattr(module, "to_reference_grid", identity)
    return module


@pytest.fixture
def cohort(tmp_path):
    ref_path = tmp_path / "ref.nii.gz"
    reference = reference_file(ref_path)
    (tmp_path / "scans").mkdir()
    (tmp_path / "labels").mkdir()
    rows = ["participant_id,landmark_id,x_mm,y_mm,z_mm"]
    ids = [f"p{i}" for i in range(6)]
    points = native_points(reference, sitk.Euler3DTransform())
    for n, sid in enumerate(ids):
        rng = np.random.default_rng(n)
        nib.save(
            nib.Nifti1Image(rng.random(GRID, dtype=np.float32).transpose(2, 1, 0), AFFINE),
            str(tmp_path / "scans" / f"{sid}_T1w.nii.gz"),
        )
        seg = np.zeros(GRID, np.int16)
        seg[3:9, 4:12, 4:12] = 1 + n % 2
        seg[9:12, 4:12, 4:12] = 5
        nib.save(
            nib.Nifti1Image(seg.transpose(2, 1, 0), AFFINE),
            str(tmp_path / "labels" / f"{sid}.nii.gz"),
        )
        for k, (x, y, z) in enumerate(points, start=1):
            rows.append(f"{sid},{k},{x},{y},{z}")
    (tmp_path / "landmarks.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path, ref_path, ids


def run(script, tmp_path, ref_path, *extra):
    argv = [
        "prepare_dataset.py",
        "--scans",
        str(tmp_path / "scans"),
        "--segmentations",
        str(tmp_path / "labels"),
        "--landmarks",
        str(tmp_path / "landmarks.csv"),
        "--reference",
        str(ref_path),
        "--output",
        str(tmp_path / "out"),
        *extra,
    ]
    old, sys.argv = sys.argv, argv
    try:
        return script.main()
    finally:
        sys.argv = old


def test_the_dataset_is_what_training_reads_back(script, cohort, capsys):
    tmp_path, ref_path, ids = cohort
    assert run(script, tmp_path, ref_path, "--split", "0.5", "0.25", "0.25") == 0
    out = tmp_path / "out"
    for sid in ids:
        assert (out / "MRI_Volumes" / f"{sid}_T1w.nii.gz").is_file()
        assert (out / "transforms" / f"{sid}_to_reference.tfm").is_file()
    split = {
        n: (out / "splits" / f"{n}_ids.txt").read_text(encoding="utf-8").split()
        for n in ("train", "val", "test")
    }
    assert sorted(sum(split.values(), [])) == ids and [len(v) for v in split.values()] == [3, 2, 1]

    # the configuration it wrote loads, and the loader returns the landmarks it was given
    cfg = load_config(
        out / "cohort.yaml",
        overrides={
            "preprocess": {"skull_stripped": False, "bias_field_corrected": False},
            "data": {"num_workers": 0},
        },
    )
    cfg.validate()
    assert (cfg.task.n_landmarks, cfg.task.n_classes) == (len(POSITIONS), 6)  # labels 0..5
    sample = build_dataset(cfg, "val")[0]
    assert np.array_equal(
        np.rint(np.asarray(sample["coords"])).astype(int).reshape(-1, 3), np.array(POSITIONS)
    )


def test_a_subject_that_cannot_be_prepared_is_skipped_and_named(script, cohort, capsys):
    tmp_path, ref_path, ids = cohort
    scan = tmp_path / "scans" / "p3_T1w.nii.gz"
    scan.write_bytes(scan.read_bytes()[:100])
    assert run(script, tmp_path, ref_path, "--split", "0.5", "0.25", "0.25") == 1
    out = capsys.readouterr().out
    assert "p3: skipped" in out and "1 subject(s) skipped" in out
    listed = sum(
        (
            (tmp_path / "out" / "splits" / f"{n}_ids.txt").read_text(encoding="utf-8").split()
            for n in ("train", "val", "test")
        ),
        [],
    )
    assert sorted(listed) == [i for i in ids if i != "p3"]


def test_a_second_run_leaves_finished_subjects_alone(script, cohort, capsys):
    tmp_path, ref_path, ids = cohort
    run(script, tmp_path, ref_path)
    capsys.readouterr()
    run(script, tmp_path, ref_path)
    assert capsys.readouterr().out.count("already done") == len(ids)


def test_a_missing_annotation_names_the_subject(script, cohort):
    tmp_path, ref_path, ids = cohort
    (tmp_path / "labels" / "p3.nii.gz").unlink()
    with pytest.raises(SystemExit, match="no segmentation for p3"):
        run(script, tmp_path, ref_path)
    assert not (tmp_path / "out" / "MRI_Volumes").exists() or not any(
        (tmp_path / "out" / "MRI_Volumes").iterdir()
    )
