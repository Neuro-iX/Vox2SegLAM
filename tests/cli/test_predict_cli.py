"""predict.py as a user runs it: from a raw scan to a result, in a subprocess."""

import hashlib
import json
import subprocess
import sys
import textwrap

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch

from conftest import command, clean_env
from vox2seglam import assets
from vox2seglam.config import load_config, save_config
from vox2seglam.models import Vox2SegLAM

GRID = (32, 48, 32)  # ZYX, a reduced stand-in for 240 x 304 x 240
SCRIPT = "predict"


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """A published archive: weights, their configuration, the reference, and the manifest,
    flat as a record holds them."""
    root = tmp_path_factory.mktemp("archive")
    cfg = load_config("hoa16f_small_stg_s0", overrides={"task": {"patch_size": list(GRID)}})
    torch.manual_seed(0)
    torch.save(
        {"model": Vox2SegLAM.from_config(cfg).state_dict(), "epoch": 1, "config_name": cfg.name},
        root / assets.ASSETS["checkpoint"].published_as(),
    )
    save_config(cfg, root / assets.ASSETS["config"].published_as())

    reference = np.random.default_rng(0).random(GRID, dtype=np.float32)
    reference[6:-6, 8:-8, 6:-6] += 2.0
    nib.save(
        nib.Nifti1Image(reference.transpose(2, 1, 0), np.diag([-0.7, 0.7, 0.7, 1.0])),
        str(root / assets.ASSETS["reference"].published_as()),
    )

    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.name] = {
                "bytes": path.stat().st_size,
                "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            }
    (root / "assets.json").write_text(
        json.dumps({"version": "2.0.0", "files": files}), encoding="utf-8"
    )
    return root


@pytest.fixture(scope="module")
def stripper(tmp_path_factory):
    """Behaves like mri_synthstrip - -i, -o, -b - by keeping a central box as the brain."""
    script = tmp_path_factory.mktemp("stripper") / "fake_synthstrip.py"
    script.write_text(
        textwrap.dedent("""
        import argparse
        import nibabel as nib
        import numpy as np
        ap = argparse.ArgumentParser()
        ap.add_argument("-i"); ap.add_argument("-o"); ap.add_argument("-b")
        a = ap.parse_args()
        img = nib.load(a.i)
        data = np.asanyarray(img.dataobj).astype(np.float32)
        mask = np.zeros(data.shape, bool)
        lo = [s // 4 for s in data.shape]; hi = [s - s // 4 for s in data.shape]
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        data[~mask] = 0
        nib.save(nib.Nifti1Image(data, img.affine), a.o)
    """),
        encoding="utf-8",
    )
    # the route a container has, and the only one this command keeps now that --set and
    # --freesurfer-home belong to prepare.py
    return {"VOX2SEGLAM_SYNTHSTRIP": json.dumps([sys.executable, str(script)])}


@pytest.fixture
def scan(tmp_path):
    rng = np.random.default_rng(1)
    volume = rng.random((28, 40, 28), dtype=np.float32)
    volume[5:-5, 7:-7, 5:-5] += 2.0
    path = tmp_path / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.diag([-1.0, 1.0, 1.0, 1.0])), str(path))
    return path


def run(tmp_path, archive, *args, tty=False, stripper=None):
    env = clean_env(
        tmp_path,
        VOX2SEGLAM_ASSETS=str(tmp_path / "assets"),
        VOX2SEGLAM_RELEASE_URL=archive.as_uri(),
        **(stripper or {}),
    )
    return subprocess.run(
        [*command(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=900,
    )


def test_one_command_goes_from_a_raw_scan_to_a_result(tmp_path, archive, stripper, scan):
    out = tmp_path / "results"
    done = run(
        tmp_path,
        archive,
        "--input",
        str(scan),
        "--output-dir",
        str(out),
        "--download",
        "--device",
        "cpu",
        stripper=stripper,
    )
    assert done.returncode == 0, done.stdout + done.stderr

    # One file carries both readers. Slicer decides what a .json is by reading it and refuses
    # one without a "markups" key, so the landmarks are a markups list in RAS, labelled...
    document = json.loads((out / "sub-01_T1w_landmarks.mrk.json").read_text(encoding="utf-8"))
    assert "markups-schema" in document["@schema"]
    # exactly one list, in the scan's own space: Slicer shows one node, not a pair to choose
    assert len(document["markups"]) == 1
    fiducials = document["markups"][0]
    assert fiducials["type"] == "Fiducial" and fiducials["coordinateSystem"] == "RAS"
    assert fiducials["name"] == "sub-01_T1w"
    points = fiducials["controlPoints"]
    assert [p["id"] for p in points] == [str(i) for i in range(1, 17)]
    assert [p["label"] for p in points] == [f"LM{i}" for i in range(1, 17)]
    assert all(len(p["position"]) == 3 and len(p["orientation"]) == 9 for p in points)
    assert points[0]["description"] == "Left Putamen Anterior Start"

    # ...and what Slicer has no place for sits under one key beside it
    record = document["vox2seglam"]
    assert record["producer"] == "predict.py" and len(record["landmarks"]) == 16
    assert set(record["landmarks"]["1"]) == {
        "reference_voxel_zyx",
        "reference_ras_mm",
        "native_lps_mm",
        "native_ras_mm",
    }
    assert points[0]["position"] == record["landmarks"]["1"]["native_ras_mm"]

    # one landmark file and one label map per scan, nothing else
    assert not (out / "sub-01_T1w_planes.json").exists()

    # ...plus one coordinate table for the run, in the form the project's own commands read:
    # a markups file is for a viewer, and a prediction that cannot be fed back into
    # evaluate.py or prepare_dataset.py is a dead end
    from vox2seglam.data.landmarks import read_landmark_csv

    table = out / "landmarks.csv"
    header = table.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:5] == [
        "participant_id",
        "landmark_id",
        "landmark_name",
        "coordinate_system",
        "x_mm",
    ]
    coordinates, _ = read_landmark_csv(table, 16)
    assert list(coordinates) == ["sub-01_T1w"]
    lps = coordinates["sub-01_T1w"]
    assert lps.shape == (16, 3) and not np.isnan(lps).any()
    # the same points as the markups file, which is in RAS: x and y change sign
    ras = np.array(record["landmarks"]["1"]["native_ras_mm"])
    assert np.allclose(lps[0] * [-1, -1, 1], ras, atol=1e-3)

    # the segmentation lands on the scan the user supplied, geometry included
    seg, original = sitk.ReadImage(str(out / "sub-01_T1w_seg.nii.gz")), sitk.ReadImage(str(scan))
    assert seg.GetSize() == original.GetSize()
    assert np.allclose(seg.GetSpacing(), original.GetSpacing(), atol=1e-5)
    assert np.allclose(seg.GetOrigin(), original.GetOrigin(), atol=1e-4)
    assert np.allclose(seg.GetDirection(), original.GetDirection(), atol=1e-6)


def test_one_truncated_scan_does_not_stop_the_cohort(tmp_path, archive, stripper, scan):
    """A truncated .gz is skipped and named; the other scans are still predicted, and the exit
    status says something was left out."""
    scans = tmp_path / "scans"
    scans.mkdir()
    good = scans / "good.nii.gz"
    good.write_bytes(scan.read_bytes())
    (scans / "bad.nii.gz").write_bytes(scan.read_bytes()[:200])
    out = tmp_path / "results"
    done = run(
        tmp_path,
        archive,
        "--input",
        str(scans),
        "--output-dir",
        str(out),
        "--download",
        "--device",
        "cpu",
        stripper=stripper,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert (out / "good_landmarks.mrk.json").is_file() and not (
        out / "bad_landmarks.mrk.json"
    ).exists()
    assert "bad.nii.gz" in done.stdout and "truncated" in done.stdout
    assert "Traceback" not in done.stderr


def test_the_provenance_of_what_was_fetched_is_recorded(tmp_path, archive, stripper, scan):
    run(
        tmp_path,
        archive,
        "--input",
        str(scan),
        "--output-dir",
        str(tmp_path / "r"),
        "--download",
        "--device",
        "cpu",
        stripper=stripper,
    )
    record = json.loads((tmp_path / "assets" / "assets.json").read_text(encoding="utf-8"))
    assert record["version"] == "2.0.0" and record["source"] == archive.as_uri()
    assert f"weights/{assets.DEFAULT_MODEL}/best.pt" in record["files"]
    assert all(f["md5"] for f in record["files"].values())


def test_a_second_run_prepares_and_fetches_nothing(tmp_path, archive, stripper, scan):
    args = ("--input", str(scan), "--output-dir", str(tmp_path / "r"), "--device", "cpu")
    first = run(tmp_path, archive, *args, "--download", stripper=stripper)
    assert first.returncode == 0, first.stdout + first.stderr
    # a first preparation says so plainly: it used to read "prepared (not prepared yet)",
    # which names a reason where there is none and reads as a contradiction
    assert "prepared" in first.stdout
    assert "already prepared" not in first.stdout and "prepared again" not in first.stdout
    again = run(
        tmp_path, archive, *args, "--no-download", stripper=stripper
    )  # would fail if it fetched
    assert again.returncode == 0, again.stdout + again.stderr
    assert "already prepared" in again.stdout and "fetching" not in again.stdout


def test_a_changed_scan_is_prepared_again_and_says_so(tmp_path, archive, stripper, scan):
    args = (
        "--input",
        str(scan),
        "--output-dir",
        str(tmp_path / "r"),
        "--device",
        "cpu",
        "--download",
    )
    run(tmp_path, archive, *args, stripper=stripper)
    volume = np.asanyarray(nib.load(str(scan)).dataobj) + 1.0
    nib.save(nib.Nifti1Image(volume, np.diag([-1.0, 1.0, 1.0, 1.0])), str(scan))
    again = run(tmp_path, archive, *args, stripper=stripper)
    assert "prepared again (the input file has changed)" in again.stdout


def test_no_terminal_means_instructions_not_a_prompt_or_a_hang(tmp_path, archive, stripper, scan):
    """A batch job has nobody to answer: it must fail at once, naming what is missing."""
    done = run(
        tmp_path,
        archive,
        "--input",
        str(scan),
        "--output-dir",
        str(tmp_path / "r"),
        "--device",
        "cpu",
        stripper=stripper,
    )
    assert done.returncode != 0
    message = done.stdout + done.stderr
    assert "missing asset" in message and "VOX2SEGLAM_ASSETS" in message
    assert str(tmp_path / "assets") in message
    assert not (tmp_path / "assets" / "weights").exists()  # nothing was fetched


def test_either_input_or_prepared_dir_is_required(tmp_path, archive):
    done = run(tmp_path, archive, "--output-dir", str(tmp_path / "r"))
    assert done.returncode != 0 and "--input" in done.stderr


def test_predict_py_fetches_the_configuration_with_the_weights(tmp_path, archive, stripper):
    """The configuration is fetched beside the weights, so no clone is needed to find the recipe;
    a volume already on the grid goes straight in.

    The stripper is the stand-in, as everywhere else here: SynthStrip conforms its input onto a
    256-cubed grid whatever it is given, which on a virtualised CPU runner takes longer than the
    whole rest of the suite. What this test is about is the configuration travelling with the
    weights, and `tests/cli/test_synthstrip.py` covers the stripper itself."""
    volume = tmp_path / "ready.nii.gz"
    data = np.random.default_rng(2).random(GRID, dtype=np.float32)
    nib.save(nib.Nifti1Image(data.transpose(2, 1, 0), np.diag([-0.7, 0.7, 0.7, 1.0])), str(volume))
    env = clean_env(
        tmp_path,
        VOX2SEGLAM_ASSETS=str(tmp_path / "assets"),
        VOX2SEGLAM_RELEASE_URL=archive.as_uri(),
        **stripper,
    )
    done = subprocess.run(
        [
            *command("predict"),
            "--input",
            str(volume),
            "--output-dir",
            str(tmp_path / "out"),
            "--download",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "assets" / "weights" / assets.DEFAULT_MODEL / "config.yaml").is_file()
    assert (tmp_path / "out" / "ready_landmarks.mrk.json").is_file()


def test_postprocessing_is_requested_along_with_the_weights(tmp_path, archive):
    """--postprocess needs postprocess.json beside the checkpoint; an archive without one must
    give that error, not a bare missing file."""
    volume = tmp_path / "ready.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros(GRID[::-1], np.float32), np.diag([-0.7, 0.7, 0.7, 1.0])),
        str(volume),
    )
    env = clean_env(
        tmp_path,
        VOX2SEGLAM_ASSETS=str(tmp_path / "assets"),
        VOX2SEGLAM_RELEASE_URL=archive.as_uri(),
    )
    done = subprocess.run(
        [
            *command("predict"),
            "--input",
            str(volume),
            "--output-dir",
            str(tmp_path / "out"),
            "--postprocess",
            "--download",
        ],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=900,
    )
    assert done.returncode != 0
    assert "does not list" in done.stdout + done.stderr
    assert "postprocess.json" in done.stdout + done.stderr
