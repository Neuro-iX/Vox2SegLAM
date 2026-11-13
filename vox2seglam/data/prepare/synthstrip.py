"""Skull stripping with SynthStrip, the stripper the published checkpoints were trained with."""

import importlib.metadata
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ... import assets

__all__ = [
    "synthstrip",
    "synthstrip_script",
    "synthstrip_argv",
    "uses_builtin",
    "stripper_identity",
]


def uses_builtin(cfg) -> bool:
    """Whether stripping goes through the built-in route (and so needs SynthStrip's weights)."""
    return (
        not cfg.preprocess.synthstrip_command
        and not cfg.paths.freesurfer_home
        and not env_command()
    )


def _installed(module: str) -> bool:
    # find_spec raises, instead of returning None, when a parent package is missing
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _builtin_command() -> list[str]:
    if not _installed("nipreps.synthstrip"):
        raise RuntimeError(
            "the built-in skull stripper needs the nipreps-synthstrip package, which is a "
            "dependency of vox2seglam: reinstall it with `pip install -e .`, or configure "
            "preprocess.synthstrip_command / paths.freesurfer_home to use another route."
        )
    try:
        weights = assets.resolve("synthstrip")["synthstrip"]
    except assets.AssetUnavailable as err:
        raise assets.AssetUnavailable(
            f"{err}\nor fetch them with `vox2seglam fetch-assets synthstrip` -- once, "
            "from a machine with a network, if the one doing the work has none."
        ) from None
    return [sys.executable, "-m", "nipreps.synthstrip", "--model", str(weights)]


#: a command, as a JSON list, that stands in for the built-in skull stripper. It is the one
#: route a container or a cluster has: predict.py offers no flag for it, because the choice of
#: stripper belongs to a site rather than to a prediction, and prepare.py's --set would mean
#: preparing the cohort in a separate step. The command is called like mri_synthstrip:
#: -i input -o output -b border.
STRIPPER_ENV = "VOX2SEGLAM_SYNTHSTRIP"


def env_command() -> list[str]:
    """The stripper named by STRIPPER_ENV, or [] when it is unset."""
    raw = os.environ.get(STRIPPER_ENV, "").strip()
    if not raw:
        return []
    try:
        command = json.loads(raw)
    except json.JSONDecodeError:
        command = shlex.split(raw)
    if isinstance(command, str):
        command = shlex.split(command)
    if not command:
        raise ValueError(f"{STRIPPER_ENV} is set but empty")
    return [str(part) for part in command]


def synthstrip_argv(cfg, script: Path | None) -> tuple[list[str], dict[str, str]]:
    """The command to run, and the environment it needs: any route, same interface."""
    command = list(cfg.preprocess.synthstrip_command) or env_command()
    if command:
        return command, dict(os.environ)
    if script is not None:
        # FreeSurfer's interpreter carries the surfa the script needs; fall back to ours
        home = script.parents[2]
        fspython = home / "bin" / "fspython"
        python = str(fspython) if fspython.is_file() else sys.executable
        return [python, str(script)], dict(os.environ, FREESURFER_HOME=str(home))
    return _builtin_command(), dict(os.environ)


def synthstrip(src: Path, dst: Path, cfg, script: Path | None) -> None:
    command, env = synthstrip_argv(cfg, script)
    argv = command + [
        "-i",
        str(Path(src).resolve()),
        "-o",
        str(Path(dst).resolve()),
        "-b",
        str(cfg.preprocess.synthstrip_border_mm),
    ]
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    if result.returncode != 0 or not dst.is_file():
        raise RuntimeError(
            f"SynthStrip failed on {Path(src).name} (exit {result.returncode})\n"
            f"  command: {' '.join(argv)}\n"
            f"{result.stderr.strip()[-800:] or '(no error output)'}"
        )


# The SynthStrip script inside a configured FreeSurfer install; None for the other routes.
def synthstrip_script(cfg) -> Path | None:
    if cfg.preprocess.synthstrip_command or not cfg.paths.freesurfer_home:
        return None
    script = Path(cfg.paths.freesurfer_home) / "python" / "scripts" / "mri_synthstrip"
    if not script.is_file():
        raise FileNotFoundError(
            f"{script} not found; check paths.freesurfer_home, or drop it to use the "
            f"built-in skull stripper"
        )
    return script


def stripper_identity(cfg, script: Path | None) -> str:
    """A string that changes when the way of skull-stripping does, for provenance and for
    telling a preparation made with another stripper from a current one."""
    command = list(cfg.preprocess.synthstrip_command)
    if command:
        return "command:" + json.dumps(command)
    if script is not None:
        return f"freesurfer:{script.parents[2]}"
    try:
        version = importlib.metadata.version("nipreps-synthstrip")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return (
        f"builtin:nipreps-synthstrip=={version};weights={assets.ASSETS['synthstrip'].sha256[:16]}"
    )
