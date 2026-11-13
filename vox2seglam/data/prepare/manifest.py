import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .n4 import sitk_version
from .synthstrip import stripper_identity, synthstrip_script


# what a cache directory is produced with; the "parameters" part must match the
# current config for the cache to be reused (see check_manifest)
def stage_detail(stage: str, cfg) -> dict:
    return {
        "skull_strip": {
            "tool": "SynthStrip (FreeSurfer)",
            "parameters": {"border_mm": cfg.preprocess.synthstrip_border_mm},
        },
        "bias_field": {
            "tool": "N4BiasFieldCorrection (SimpleITK)",
            "parameters": {
                "shrink_factor": cfg.preprocess.n4_shrink_factor,
                "iterations": list(cfg.preprocess.n4_iterations),
                "mask": "image > 0",
            },
            "version": sitk_version(),
        },
    }[stage]


# a cache built with other parameters must not be reused silently; a directory without
# prepare.json (built outside this pipeline) has nothing to compare and is accepted
def check_manifest(stage: str, out_dir: Path, cfg) -> None:
    path = out_dir / "prepare.json"
    if not path.is_file():
        return
    try:
        recorded = json.loads(path.read_text(encoding="utf-8")).get("parameters")
    except json.JSONDecodeError as err:
        raise RuntimeError(
            f"{path} is unreadable ({err}); the cache directory was left half-written. "
            f"Delete {out_dir} to rebuild it."
        ) from err
    expected = stage_detail(stage, cfg)["parameters"]
    if recorded != expected:
        raise RuntimeError(
            f"{out_dir} was built with {stage} parameters {recorded}, but the config "
            f"asks for {expected}. Delete that directory to rebuild it, or set the "
            "preprocess parameters back to the recorded ones."
        )


def write_manifest(stage: str, out_dir: Path, ids: Sequence[str], cfg) -> None:
    detail = stage_detail(stage, cfg)
    if stage == "skull_strip":
        # Which route stripped these volumes. Informational, not part of "parameters": the
        # published training caches were made with FreeSurfer's SynthStrip and carry no such
        # field, and the routes agree to within a voxel at the border of the mask - but a
        # cache should still say which one it is.
        detail = {**detail, "stripper": stripper_identity(cfg, synthstrip_script(cfg))}
    path = out_dir / "prepare.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    # merge with whatever subjects a previous (partial) run already recorded here,
    # so re-running on more subjects accumulates coverage instead of overwriting it
    covered = sorted(set(previous.get("subjects", [])) | set(ids))
    # same reason as the volumes themselves: a half-written prepare.json would make
    # check_manifest raise an unreadable JSONDecodeError on every later run
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(
        json.dumps(
            {
                "stage": stage,
                **detail,
                "subjects": covered,
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
