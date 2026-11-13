"""Which published model a user gets for --task."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conftest import command, clean_env
from vox2seglam import assets
from vox2seglam.config import Config, save_config

ROOT = Path(__file__).resolve().parents[2]


def test_default_is_the_16_structure_model():
    assert assets.model_asset_names() == ["checkpoint", "config"]
    assert assets.model_asset_names(postprocess=True) == ["checkpoint", "config", "postprocess"]


def test_the_39_structure_model_has_its_own_files():
    checkpoint, config = assets.model_asset_names(39)
    assert checkpoint.startswith("checkpoint:") and config.startswith("config:")
    assert assets.ASSETS[checkpoint].relative_path != assets.ASSETS["checkpoint"].relative_path
    assert assets.ASSETS[checkpoint].relative_path.endswith("best.pt")
    assert assets.ASSETS[config].relative_path.endswith("config.yaml")


def test_an_unpublished_number_of_structures_lists_the_available_ones():
    with pytest.raises(ValueError, match="available: 16, 39"):
        assets.model_asset_names(20)


def test_the_protocol_postprocessing_is_for_16_structures_only():
    with pytest.raises(ValueError, match="subcortical model only"):
        assets.model_asset_names(39, postprocess=True)


def test_predict_refuses_postprocessing_with_39_structures(tmp_path):
    run = subprocess.run(
        [
            *command("predict"),
            "--input",
            str(tmp_path / "none.nii.gz"),
            "--task",
            "full-brain",
            "--postprocess",
        ],
        capture_output=True,
        text=True,
        env=clean_env(tmp_path),
    )
    assert run.returncode != 0
    assert "subcortical model only" in run.stderr and "Traceback" not in run.stderr


def test_a_checkpoint_and_a_model_choice_do_not_mix(tmp_path):
    run = subprocess.run(
        [
            *command("predict"),
            "--input",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "x.pt"),
            "--task",
            "full-brain",
        ],
        capture_output=True,
        text=True,
    )
    assert "cannot be combined with --checkpoint" in run.stderr


def _run(directory: Path, tensors: dict, n_classes: int):
    directory.mkdir()
    torch.save({"model": tensors, "epoch": 3, "config_name": "t"}, directory / "best.pt")
    cfg = Config()
    save_config(
        replace(cfg, task=replace(cfg.task, n_classes=n_classes)), directory / "config.yaml"
    )


def _run_predict(tmp_path, *args):
    import subprocess

    return subprocess.run(
        [
            *command("predict"),
            "--input",
            "scan.nii.gz",
            "--output-dir",
            str(tmp_path),
            "--no-download",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_enforcing_the_protocol_is_one_option(tmp_path):
    """--postprocess and nothing else.

    Five flags used to govern the enforcement -- the frame the planes are built in, whether
    they are snapped to a section, whether the tolerance is dropped, whether the map is built
    from labels or probabilities. All of them were dead without --postprocess, and the choice
    they offered belongs to repeating a published measurement rather than to predicting on
    somebody's scan. One option now does the thing it names, strictly.
    """
    for gone in (
        "--protocol",
        "--plane-frame",
        "--snap-to-sections",
        "--strict-protocol",
        "--postprocess-from-labels",
    ):
        done = _run_predict(tmp_path, gone, "x")
        assert done.returncode != 0 and "unrecognized arguments" in done.stderr, gone


def test_the_fitted_postprocessing_is_found_under_either_name(tmp_path):
    """The fit-postprocess command writes `postprocess.json` beside the weights; the published
    having no directories, names the same file after its model. Looking only for the first
    spelling made `--postprocess` fail with a traceback on the files as they are distributed.
    """
    from vox2seglam.cli import resolve_postprocess

    checkpoint = tmp_path / "vox2seglam-structures16_landmarks16.pt"
    checkpoint.write_bytes(b"")
    published = tmp_path / "vox2seglam-structures16_landmarks16-postprocess.json"
    published.write_text("{}", encoding="utf-8")
    assert resolve_postprocess(checkpoint, "") == published

    published.unlink()
    beside = tmp_path / "postprocess.json"
    beside.write_text("{}", encoding="utf-8")
    assert resolve_postprocess(checkpoint, "") == beside

    beside.unlink()
    with pytest.raises(SystemExit) as err:
        resolve_postprocess(checkpoint, "")
    assert "neither" in str(err.value) and "vox2seglam fit-postprocess" in str(err.value)

    with pytest.raises(SystemExit, match="no such file"):
        resolve_postprocess(checkpoint, str(tmp_path / "nowhere.json"))
