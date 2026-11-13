"""Shared fixtures. The tests need no dataset: volumes and landmarks are synthetic.

They are grouped by what they keep from breaking:

    cli/          what a user runs -- the commands, the Predictor, the downloads
    protocol/     the rules, the planes, the enforcement and the metrics
    training/     resuming bit for bit, determinism, the landmark templates
    model/        the architecture, the losses, the recipes and their overrides
    io/           coordinates, geometry, the dataset layout, the release metadata

Fixtures live here, at the root of tests/, so every group sees them."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def command(name: str) -> list[str]:
    """The argv prefix that runs one command in a subprocess.

    `python -m vox2seglam.commands.<module>` rather than the installed `vox2seglam` script,
    which would test the install rather than the code in the clone. The name may be the
    subcommand ("fetch-assets"), the module ("fetch_assets") or the old file name.
    """
    from vox2seglam.commands import COMMANDS

    name = name.removesuffix(".py")
    module = COMMANDS[name][0] if name in COMMANDS else name.replace("-", "_")
    return [sys.executable, "-m", f"vox2seglam.commands.{module}"]


def clean_env(home, **extra) -> dict:
    """The environment of a subprocess: the caller's, minus anything that would point it at the
    user's own assets, with HOME and the cache redirected (Windows needs SYSTEMROOT and the
    like, so it is not built from scratch)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VOX2SEGLAM", "XDG_"))}
    env.update(HOME=str(home), USERPROFILE=str(home), **extra)
    return env


@pytest.fixture(autouse=True)
def _never_the_users_own_cache(tmp_path, monkeypatch):
    """Point every test at a cache of its own. One test fetched into the real
    ~/.cache/vox2seglam and overwrote the reference template with its payload."""
    monkeypatch.setenv("VOX2SEGLAM_ASSETS", str(tmp_path / "assets"))


@pytest.fixture(scope="session")
def spec():
    """The HOA protocol (21 planes, 16 rules)."""
    from vox2seglam.postprocess import PlaneShift

    return PlaneShift({}).spec


@pytest.fixture
def landmarks16() -> np.ndarray:
    """16 HOA-like landmarks (ZYX voxels) on a 40^3 grid, left/right pairs mirrored in X."""
    return np.array(
        [
            [20, 28, 26],
            [20, 28, 14],
            [18, 27, 24],
            [18, 27, 16],
            [18, 22, 24],
            [18, 22, 16],
            [17, 20, 24],
            [17, 20, 16],
            [21, 24, 20],
            [21, 23, 20],
            [12, 18, 22],
            [12, 18, 18],
            [22, 10, 28],
            [22, 10, 12],
            [21, 15, 20],
            [8, 16, 20],
        ],
        dtype=float,
    )
