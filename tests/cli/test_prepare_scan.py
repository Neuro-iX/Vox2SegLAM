"""Preparing a scan from another cohort: the guard against an empty strip, and when to reuse."""

import ast
import os
from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from conftest import ROOT
from vox2seglam.config import load_config
from vox2seglam.data.prepare import scan as scan_module
from vox2seglam.data.prepare.register import PreparedScan
from vox2seglam.data.prepare.scan import (
    pending,
    prepare_if_needed,
    prepare_scan,
    preparation_status,
)

AFFINE = np.diag([-0.7, 0.7, 0.7, 1.0])


def write(path, voxels=None):
    volume = np.zeros((16, 16, 16), np.float32)
    volume[4:12, 4:12, 4:12] = 1.0 if voxels is None else voxels
    nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), AFFINE), str(path))
    return path


@pytest.fixture
def scan(tmp_path):
    return write(tmp_path / "v.nii.gz")


@pytest.fixture
def reference(tmp_path):
    return write(tmp_path / "reference.nii.gz")


@pytest.fixture
def cfg():
    return load_config("baseline")


@pytest.fixture
def stripper(monkeypatch):
    """A stand-in for SynthStrip that keeps a small brain, and counts how often it ran."""
    calls = []

    def strip(src, dst, cfg, script):
        calls.append(src)
        mask = np.zeros((16, 16, 16), np.float32)
        mask[6:10, 6:10, 6:10] = 1.0
        nib.save(nib.Nifti1Image(mask.transpose(2, 1, 0), AFFINE), str(dst))

    monkeypatch.setattr(scan_module, "synthstrip", strip)
    return calls


@pytest.fixture
def fast(monkeypatch):
    """An identity registration stand-in: the real one costs seconds per call and the reuse
    tests are not about it."""

    def identity(native, reference):
        return sitk.ReadImage(str(native), sitk.sitkFloat32), sitk.Euler3DTransform()

    monkeypatch.setattr(scan_module, "to_reference_grid", identity)


def test_a_strip_that_leaves_nothing_is_refused(scan, reference, cfg, tmp_path, monkeypatch):
    """N4 and the z-score accept an all-zero volume silently."""
    monkeypatch.setattr(
        scan_module,
        "synthstrip",
        lambda src, dst, c, s: nib.save(
            nib.Nifti1Image(np.zeros((16, 16, 16), np.float32), AFFINE), str(dst)
        ),
    )
    with pytest.raises(SystemExit, match="left nothing"):
        prepare_scan(scan, reference, tmp_path / "out", cfg)


def test_a_strip_that_keeps_a_brain_goes_through(scan, reference, cfg, tmp_path, stripper):
    out = prepare_scan(scan, reference, tmp_path / "out", cfg)
    assert out.model_input.name == "v_aligned_brain_n4.nii.gz" and out.complete


def test_only_what_is_read_again_is_kept(scan, reference, cfg, tmp_path, stripper):
    """The registered and the stripped volume are steps towards model_input; keeping all three
    costs three full-size volumes per scan for two nothing reads again."""
    out = prepare_scan(scan, reference, tmp_path / "out", cfg)
    assert out.complete and not out.aligned.exists() and not out.brain.exists()

    kept = prepare_scan(scan, reference, tmp_path / "kept", cfg, keep_intermediates=True)
    assert kept.aligned.is_file() and kept.brain.is_file() and kept.complete


def test_each_step_is_announced_before_it_runs(scan, reference, cfg, tmp_path, stripper):
    """Preparation is a minute of silence per scan; a caller with a terminal needs the steps."""
    seen = []
    prepare_scan(scan, reference, tmp_path / "out", cfg, stage=seen.append)
    assert [s.split()[0] for s in seen] == ["registering", "skull-stripping", "correcting"]


def test_already_stripped_skips_the_stripper(scan, reference, cfg, tmp_path, stripper):
    out = prepare_scan(scan, reference, tmp_path / "out", cfg, already_stripped=True)
    assert out.complete and stripper == []


def test_a_second_call_reuses_the_first_preparation(scan, reference, cfg, tmp_path, stripper, fast):
    first, why = prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    assert why == "first time" and len(stripper) == 1

    again, why = prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    assert why == "reused" and again == first
    assert len(stripper) == 1  # nothing was done a second time


def test_force_redoes_it(scan, reference, cfg, tmp_path, stripper, fast):
    prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    _, why = prepare_if_needed(scan, reference, tmp_path / "out", cfg, force=True)
    assert why == "forced" and len(stripper) == 2


def test_a_changed_input_file_is_not_served_from_the_old_preparation(
    scan, reference, cfg, tmp_path, stripper, fast
):
    prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    write(scan, voxels=2.0)  # a different scan under the same name
    os.utime(scan, ns=(10**18, 10**18))
    _, why = prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    assert "input file has changed" in why and len(stripper) == 2


def test_a_different_reference_is_not_served_from_the_old_preparation(
    scan, reference, cfg, tmp_path, stripper, fast
):
    prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    other = write(tmp_path / "other_reference.nii.gz", voxels=3.0)
    _, why = prepare_if_needed(scan, other, tmp_path / "out", cfg)
    assert "different reference" in why


def test_a_different_stripper_is_not_served_from_the_old_preparation(
    scan, reference, cfg, tmp_path, stripper, fast
):
    """A scan stripped another way is a different input to the model, not the same preparation."""
    prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    container = replace(
        cfg, preprocess=replace(cfg.preprocess, synthstrip_command=("apptainer", "run", "s.sif"))
    )
    _, why = prepare_if_needed(scan, reference, tmp_path / "out", container)
    assert "skull-stripped another way" in why
    _, why = prepare_if_needed(scan, reference, tmp_path / "out", container, already_stripped=True)
    assert "another way" in why


def test_changed_preprocessing_settings_are_not_served_from_the_old_preparation(
    scan, reference, cfg, tmp_path, stripper, fast
):
    prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    softer = replace(cfg, preprocess=replace(cfg.preprocess, n4_iterations=(10, 10)))
    _, why = prepare_if_needed(scan, reference, tmp_path / "out", softer)
    assert "settings have changed" in why


def test_an_interrupted_preparation_is_redone_not_trusted(
    scan, reference, cfg, tmp_path, stripper, fast
):
    """The record is written last, so a run killed before it leaves nothing that looks whole."""
    done, _ = prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    done.record.unlink()
    _, why = prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    assert "did not finish" in why


def test_the_record_says_what_the_preparation_was_made_from(
    scan, reference, cfg, tmp_path, stripper, fast
):
    import json

    done, _ = prepare_if_needed(scan, reference, tmp_path / "out", cfg)
    record = json.loads(done.record.read_text(encoding="utf-8"))
    assert record["source"]["path"] == str(scan.resolve())
    assert record["reference"]["bytes"] == reference.stat().st_size
    assert record["stripper"].startswith("builtin:nipreps-synthstrip")
    assert record["settings"]["border_mm"] == cfg.preprocess.synthstrip_border_mm
    assert "written" in record


def test_pending_names_only_what_still_needs_preparing(
    scan, reference, cfg, tmp_path, stripper, fast
):
    """The stripper's weights are needed only for scans that still have to be stripped."""
    other = write(tmp_path / "w.nii.gz")
    out = tmp_path / "out"
    assert pending([scan, other], reference, out, cfg) == [scan, other]
    prepare_if_needed(scan, reference, out, cfg)
    assert pending([scan, other], reference, out, cfg) == [other]
    prepare_if_needed(other, reference, out, cfg)
    assert pending([scan, other], reference, out, cfg) == []


def test_without_a_reference_on_disk_everything_is_pending(scan, cfg, tmp_path, stripper, fast):
    missing = tmp_path / "not_downloaded_yet.nii.gz"
    assert preparation_status(scan, missing, tmp_path / "out", cfg) == "first time"


def test_the_prepared_scan_files_are_one_contract(tmp_path):
    out = PreparedScan.under(tmp_path, "/elsewhere/sub-01_T1w.nii.gz")
    assert out.record.name == "sub-01_T1w_prepared.json"
    assert not out.complete


def _numpy_views_of_temporaries():
    """Calls like GetArrayViewFromImage(<a call>): the view outlives the image it borrows."""
    found = []
    for path in sorted((ROOT / "vox2seglam").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and node.args
                and getattr(node.func, "attr", getattr(node.func, "id", ""))
                == "GetArrayViewFromImage"
                and isinstance(node.args[0], ast.Call)
            ):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def test_no_numpy_view_is_taken_of_a_temporary_image():
    assert _numpy_views_of_temporaries() == [], (
        "GetArrayViewFromImage borrows the image's memory; given a temporary it reads "
        "released memory. Hold the image in a variable, or use GetArrayFromImage (a copy)."
    )
