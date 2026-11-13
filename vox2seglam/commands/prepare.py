"""Prepares your T1w volumes for the model: registration to its grid, skull stripping, N4."""

import argparse
from pathlib import Path

from vox2seglam import assets
from vox2seglam.cli import add_asset_flags
from vox2seglam.config import load_config, parse_set
from vox2seglam.data.prepare.scan import pending, prepare_all
from vox2seglam.data.prepare.synthstrip import synthstrip_script, uses_builtin


def main() -> int:
    epilog = """
Per volume: integrity check of the .gz stream, rigid registration to the reference template and
B-spline resampling onto its grid, then skull stripping (SynthStrip) and N4. Writes, in --out-dir:
  <stem>_aligned.nii.gz            the volume on the reference grid
  <stem>_to_reference.tfm          reference -> native transform
  <stem>_aligned_brain.nii.gz      skull-stripped
  <stem>_aligned_brain_n4.nii.gz   bias-corrected: the model input
  <stem>_prepared.json             what it was made from; a volume whose record still matches is
                                   not prepared again

predict.py --input prepares what is missing before predicting; this is for preparing a
whole cohort ahead of time.
"""
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path, help="a NIfTI file or a directory")
    ap.add_argument(
        "--reference",
        type=Path,
        help="reference template on the training grid (240 x 304 x 240, 0.7 mm); "
        "the published one is used if omitted",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--config",
        default="baseline",
        help="recipe whose preprocessing is applied (SynthStrip border, N4)",
    )
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-key config overrides, e.g. "
        'preprocess.synthstrip_command=\'["apptainer","run","s.sif"]\'',
    )
    ap.add_argument(
        "--freesurfer-home",
        default=None,
        help="skull-strip with this FreeSurfer install instead of the built-in "
        "stripper (overrides paths.freesurfer_home)",
    )
    ap.add_argument(
        "--already-stripped",
        action="store_true",
        help="the input is already skull-stripped: skip SynthStrip. Note that "
        "the published checkpoints were trained on SynthStrip output, so "
        "another stripper changes what the model sees",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="prepare again even where a matching preparation exists",
    )
    add_asset_flags(ap)
    a = ap.parse_args()

    overrides = parse_set(a.set)
    if a.freesurfer_home:
        overrides.setdefault("paths", {})["freesurfer_home"] = a.freesurfer_home
    cfg = load_config(a.config, overrides=overrides or None)

    script = None
    if not a.already_stripped:
        try:
            script = synthstrip_script(cfg)
        except (RuntimeError, FileNotFoundError) as err:
            # a misconfigured stripper is a setup problem: report it without a traceback
            raise SystemExit(str(err))

    if not a.input.exists():
        raise SystemExit(f"--input {a.input}: no such file or directory")
    inputs = (
        [a.input]
        if a.input.is_file()
        else sorted(p for p in a.input.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    )
    if not inputs:
        raise SystemExit(f"no NIfTI volumes found in {a.input}")

    # everything missing is asked for at once, and only what is needed
    reference = a.reference or assets.asset_path("reference")
    todo = pending(
        inputs,
        reference,
        a.out_dir,
        cfg,
        script,
        already_stripped=a.already_stripped,
        force=a.force,
    )
    wanted = ([] if a.reference else ["reference"]) + (
        ["synthstrip"] if todo and uses_builtin(cfg) and not a.already_stripped else []
    )
    try:
        found = assets.resolve(*wanted, download=a.download) if wanted else {}
    except assets.AssetUnavailable as err:
        raise SystemExit(str(err))
    reference = a.reference or found.get("reference", reference)

    _, failed = prepare_all(
        inputs,
        reference,
        a.out_dir,
        cfg,
        script,
        already_stripped=a.already_stripped,
        force=a.force,
        show=lambda line: print(line.strip(), flush=True),
    )
    if failed:
        print(f"\n{len(failed)} scan(s) skipped:", flush=True)
        for path, why in failed.items():
            print(f"  {path.name}: {why}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
