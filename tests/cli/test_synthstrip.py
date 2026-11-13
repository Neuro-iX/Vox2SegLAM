"""Reaching SynthStrip without installing FreeSurfer's 19 GB."""

import sys
from dataclasses import replace

import pytest

from vox2seglam import assets
from vox2seglam.config import load_config
from vox2seglam.data.prepare.synthstrip import synthstrip_argv, synthstrip_script, uses_builtin

CONTAINER = ("apptainer", "run", "--cleanenv", "/opt/synthstrip.sif")


def with_command(command=CONTAINER, freesurfer_home=""):
    cfg = load_config("baseline")
    return replace(
        cfg,
        preprocess=replace(cfg.preprocess, synthstrip_command=command),
        paths=replace(cfg.paths, freesurfer_home=freesurfer_home),
    )


def test_a_configured_command_needs_no_freesurfer(monkeypatch):
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    cfg = with_command()
    assert synthstrip_script(cfg) is None  # nothing to locate on disk
    argv, env = synthstrip_argv(cfg, None)
    assert argv == list(CONTAINER)
    # the container carries its own install: nothing is injected, and a caller who has no
    # FreeSurfer at all is not required to pretend otherwise
    assert "FREESURFER_HOME" not in env


def test_a_freesurfer_install_is_run_with_its_own_interpreter(tmp_path):
    """FreeSurfer's bin/fspython carries the surfa the script shipped with; a current
    `pip install surfa` fails on it, which is why that route is not offered."""
    home = tmp_path / "freesurfer"
    script = home / "python" / "scripts" / "mri_synthstrip"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    fspython = home / "bin" / "fspython"
    fspython.parent.mkdir()
    fspython.write_text("#!/bin/sh\n", encoding="utf-8")

    cfg = with_command(command=(), freesurfer_home=str(home))
    assert synthstrip_script(cfg) == script
    argv, env = synthstrip_argv(cfg, script)
    assert argv == [str(fspython), str(script)]
    assert env["FREESURFER_HOME"] == str(home)


def test_with_nothing_configured_the_builtin_route_is_used(tmp_path, monkeypatch):
    """The route nearly everybody wants: the pip package and SynthStrip's own weights."""
    monkeypatch.setenv("VOX2SEGLAM_ASSETS", str(tmp_path))
    weights = assets.asset_path("synthstrip")
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")

    cfg = with_command(command=())
    assert uses_builtin(cfg)
    assert synthstrip_script(cfg) is None  # nothing to locate on disk
    argv, _ = synthstrip_argv(cfg, None)
    assert argv == [sys.executable, "-m", "nipreps.synthstrip", "--model", str(weights)]


def test_the_builtin_route_without_its_weights_says_where_to_get_them(tmp_path, monkeypatch):
    monkeypatch.setenv("VOX2SEGLAM_ASSETS", str(tmp_path))
    with pytest.raises(RuntimeError) as err:
        synthstrip_argv(with_command(command=()), None)
    message = str(err.value)
    assert assets.ASSETS["synthstrip"].url in message  # the manual way
    assert "vox2seglam fetch-assets" in message  # and the explicit one


def test_an_explicit_route_wins_over_the_builtin_one(tmp_path):
    assert not uses_builtin(with_command())  # a command
    assert not uses_builtin(with_command(command=(), freesurfer_home="/opt/fs"))


def test_a_freesurfer_home_without_the_script_says_which_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="mri_synthstrip"):
        synthstrip_script(with_command(command=(), freesurfer_home=str(tmp_path)))


def test_the_command_is_a_config_field_so_a_recipe_can_carry_it():
    from vox2seglam.config import parse_set

    override = parse_set(['preprocess.synthstrip_command=["apptainer","run","s.sif"]'])
    cfg = load_config("baseline", overrides=override)
    assert cfg.preprocess.synthstrip_command == ("apptainer", "run", "s.sif")
