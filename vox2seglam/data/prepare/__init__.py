import os
from collections.abc import Sequence
from pathlib import Path

from .manifest import check_manifest, write_manifest
from .n4 import n4
from .synthstrip import synthstrip, synthstrip_script

__all__ = ["derived_layout", "ensure_prepared"]


# the chain of cache stages this config needs, each reading the previous one's output:
# image_dir -> [+skull_strip_dir_suffix] -> [+bias_field_dir_suffix]
def derived_layout(
    image_dir: str,
    image_suffix: str,
    skull_stripped: bool,
    bias_field_corrected: bool,
    brain_tag: str = "_brain",
    skull_strip_dir_suffix: str = "_BET",
    bias_field_dir_suffix: str = "_N4",
) -> list[tuple[str, str, str]]:
    stages: list[tuple[str, str, str]] = []
    directory, suffix = image_dir, image_suffix

    if skull_stripped:
        directory += skull_strip_dir_suffix
        suffix = suffix.replace(".nii.gz", f"{brain_tag}.nii.gz")
        stages.append(("skull_strip", directory, suffix))

    if bias_field_corrected:
        directory += bias_field_dir_suffix
        stages.append(("bias_field", directory, suffix))

    return stages


# the entry point datasets.py calls before reading any image: runs the cache stages missing
# for these subjects and returns where the prepared volumes ended up. Cheap when already
# cached: a file-existence check plus a comparison with the cache's prepare.json.
def ensure_prepared(cfg, subject_ids: Sequence[str], quiet: bool = False) -> tuple[str, str]:
    source_dir = cfg.paths.resolve_image_dir()
    stages = derived_layout(
        cfg.paths.image_dir,
        cfg.paths.image_suffix,
        cfg.preprocess.skull_stripped,
        cfg.preprocess.bias_field_corrected,
        brain_tag=cfg.preprocess.brain_tag,
        skull_strip_dir_suffix=cfg.preprocess.skull_strip_dir_suffix,
        bias_field_dir_suffix=cfg.preprocess.bias_field_dir_suffix,
    )
    if not stages:
        return source_dir, cfg.paths.image_suffix

    in_dir, in_suffix = Path(source_dir), cfg.paths.image_suffix

    for stage, dir_name, suffix in stages:
        # dir_name is image_dir plus the stage suffixes, so it resolves like image_dir
        out_dir = Path(cfg.paths.resolve_under_data_root(dir_name))
        check_manifest(stage, out_dir, cfg)
        missing = [sid for sid in subject_ids if not (out_dir / f"{sid}{suffix}").is_file()]
        if missing:
            out_dir.mkdir(parents=True, exist_ok=True)
            _run_stage(stage, cfg, in_dir, in_suffix, out_dir, suffix, missing, quiet)
            write_manifest(stage, out_dir, subject_ids, cfg)
        in_dir, in_suffix = out_dir, suffix  # next stage reads this stage's output

    return str(in_dir), in_suffix


def _run_stage(stage, cfg, in_dir, in_suffix, out_dir, out_suffix, ids, quiet):
    runner = {"skull_strip": synthstrip, "bias_field": n4}[stage]
    if not quiet:
        print(f"preparing {len(ids)} volume(s) | {stage} | {out_dir.name}", flush=True)

    script = synthstrip_script(cfg) if stage == "skull_strip" else None  # unused by n4
    for n_done, sid in enumerate(ids, 1):
        src = in_dir / f"{sid}{in_suffix}"
        if not src.is_file():
            raise FileNotFoundError(f"{src} is missing; cannot prepare {sid}")
        # a cached volume is judged done by its existence, so it must never exist half-written (an
        # interrupted run, or two runs on one directory, would leave a truncated .nii.gz that every
        # later run accepts): write beside it, then rename (os.replace is atomic)
        dst = out_dir / f"{sid}{out_suffix}"
        tmp = out_dir / f".{sid}{out_suffix}.partial"
        try:
            runner(src, tmp, cfg, script)
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
        if not quiet and (n_done % 10 == 0 or n_done == len(ids)):
            print(f"  {n_done}/{len(ids)}", flush=True)
