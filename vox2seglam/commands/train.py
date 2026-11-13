"""Trains one model: a recipe in, a run directory out."""

import argparse
import os
import sys
from pathlib import Path


# Must be set before CUDA starts: expandable segments cut the reserved GPU memory (31.9 to
# 23.1 GB on the default recipe) without changing results. Not supported on Windows.
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from vox2seglam import assets
from vox2seglam.cli import add_asset_flags, ensure_splits
from vox2seglam.config import load_config, parse_set
from vox2seglam.training.topology import TooFewSubjects
from vox2seglam.training import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trains one model from a recipe: the U-Net, the landmark branch and the\n"
        "refiner together, on the cohort the recipe names. Writes best.pt,\n"
        "last.pt, config.yaml and the TensorBoard logs into paths.exp_dir.\n"
        "A recipe is a name under configs/experiments/ or a path to a YAML file;\n"
        "with none, the built-in defaults are used.",
        epilog="examples:\n"
        "  %(prog)s --config hoa16f_med_jnt_s1\n"
        "  %(prog)s --config configs/examples/own_data.yaml --set train.epochs=350\n"
        "  %(prog)s --config hoa16f_med_jnt_s1 --resume\n\n"
        "A recipe resolves by name from an installed package as well as from a clone; a\n"
        "path resolves against the working directory. Training needs the cohort in\n"
        "place; see docs/training.md for the layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="experiment name or path to a YAML config (default: built-in defaults)",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-key overrides, e.g. train.epochs=350",
    )
    parser.add_argument("--exp-dir", default=None, help="override paths.exp_dir for this run")
    parser.add_argument("--resume", action="store_true", help="continue from <exp_dir>/last.pt")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu")
    add_asset_flags(parser)
    args = parser.parse_args()

    overrides = parse_set(args.set)
    if args.exp_dir:
        overrides.setdefault("paths", {})["exp_dir"] = args.exp_dir
    if args.device:
        overrides.setdefault("train", {})["device"] = args.device

    # what a user can fix - a recipe that does not exist, a setting the architecture cannot
    # honour, a missing file, SynthStrip's weights, too few subjects for the graph - is
    # reported as a message, not as a stack trace
    try:
        cfg = load_config(args.config, overrides=overrides or None)
        ensure_splits(cfg, args.download)
        trainer = Trainer(cfg)
    except (assets.AssetUnavailable, FileNotFoundError, TooFewSubjects, ValueError) as err:
        raise SystemExit(str(err))
    if args.resume:
        checkpoint = Path(cfg.paths.exp_dir) / "last.pt"
        if not checkpoint.exists():
            raise SystemExit(f"--resume given but no checkpoint at {checkpoint}")
        trainer.load_checkpoint(checkpoint)
        print(f"resumed at epoch {trainer.start_epoch}", flush=True)

    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
