"""Fits the plane-shift post-processing of a checkpoint; writes postprocess.json beside it."""

import argparse
from pathlib import Path

import numpy as np

from vox2seglam.cli import add_asset_flags, ensure_splits
from vox2seglam.config import load_config, parse_set
from vox2seglam.data.datasets import read_split
from vox2seglam.inference import Predictor
from vox2seglam.protocol import REGION_TO_CLASS
from vox2seglam.postprocess import SECTION_PLANES, fit_plane_shift
from vox2seglam.training.template_setup import training_landmark_coords
from vox2seglam.utils.io import load_nifti, to_zyx


def training_ground_truth(cfg):
    """(segmentation, landmarks) of every training subject."""
    coords, ids = training_landmark_coords(cfg)
    seg_dir = Path(cfg.paths.resolve_seg_dir())
    for sid, landmarks in zip(ids, coords):
        seg, _ = load_nifti(seg_dir / f"{sid}{cfg.paths.seg_suffix}")
        yield np.rint(to_zyx(seg)).astype(np.int64), landmarks


def main() -> int:
    epilog = """
The defaults are the post-processing of the papers. Shift ranges come from the checkpoint's own
landmarks on the validation split (interquartile range of the signed plane errors,
--percentiles); every plane is exact; the coronal planes that end a structure or a contact zone
(LM3-LM8, LM10) are put on the section of their landmark (--section-planes, empty: they slide
too). --tolerance > 0 gives that many voxels to the planes the reference annotation itself
crosses in at least --min-share of the training subjects. The test split is never read.
"""
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", default=None, help="inferred from the checkpoint if omitted")
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-key config overrides, e.g. paths.data_root=/data",
    )
    ap.add_argument("--device", default=None, help="cuda, cuda:1 or cpu")
    ap.add_argument("--percentiles", type=float, nargs=2, default=(25.0, 75.0))
    ap.add_argument(
        "--section-planes",
        nargs="*",
        default=list(SECTION_PLANES),
        help="planes put on the section of their landmark (empty: none)",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="voxels, on the planes the training ground truth crosses (0: exact)",
    )
    ap.add_argument("--min-share", type=float, default=0.5)
    ap.add_argument("--out", default="postprocess.json", help="written beside the checkpoint")
    add_asset_flags(ap)
    a = ap.parse_args()

    if not a.checkpoint.is_file():
        raise SystemExit(f"--checkpoint {a.checkpoint}: no such file")
    cfg = load_config(
        a.config or a.checkpoint.parent / "config.yaml", overrides=parse_set(a.set) or None
    )
    ensure_splits(cfg, a.download)
    if cfg.task.n_classes != len(REGION_TO_CLASS) + 1:
        raise SystemExit(
            f"the HOA protocol post-processing needs the {len(REGION_TO_CLASS) + 1} "
            f"protocol classes; this checkpoint has {cfg.task.n_classes}"
        )
    predictor = Predictor.from_checkpoint(a.checkpoint, config=cfg, device=a.device)
    pp = fit_plane_shift(
        predictor.landmark_pairs("val"),
        percentiles=tuple(a.percentiles),
        section_planes=a.section_planes,
        tolerance_vox=a.tolerance,
        training_subjects=training_ground_truth(cfg) if a.tolerance > 0 else None,
        min_share=a.min_share,
        provenance={
            "shift_range": {
                "split": Path(cfg.paths.split_val).name,
                "subjects": len(read_split(cfg.paths.resolve_split("val"))),
            },
            **(
                {"tolerance": {"split": Path(cfg.paths.split_train).name}}
                if a.tolerance > 0
                else {}
            ),
        },
    )
    out = a.checkpoint.parent / a.out
    pp.save(out)
    print(
        f"{len(pp.shift_range)} plane ranges, tolerance on "
        f"{sorted(pp.tolerance) or 'no plane'}, {len(a.section_planes)} planes on their "
        f"section -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
