"""Validation metrics of a checkpoint on one split, as the training loop computes them."""

import argparse
import json
from pathlib import Path


import torch

from vox2seglam.cli import add_asset_flags, ensure_splits
from vox2seglam.config import load_config, parse_set
from vox2seglam.data import build_dataloaders
from vox2seglam.evaluate import validate
from vox2seglam.inference import DEFAULT_N_PASSES
from vox2seglam.models import Vox2SegLAM, template_from_state_dict


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Runs a checkpoint over one split and reports the metrics the training\n"
        "loop computes: mean radial error, Dice, the loss terms.",
        epilog="example:\n  %(prog)s --checkpoint experiments/hoa16f_med_jnt_s1/best.pt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=None,
        help="experiment name or path; inferred from config.yaml beside the checkpoint if omitted",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu")
    parser.add_argument(
        "--passes",
        type=int,
        default=DEFAULT_N_PASSES,
        help="full-model passes per volume, same meaning and default as predict.py",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-key config overrides, e.g. paths.data_root=/data (as in train.py)",
    )
    parser.add_argument(
        "--out", default="test_metrics.json", help="file name written beside the checkpoint"
    )
    add_asset_flags(parser)
    args = parser.parse_args()
    overrides = parse_set(args.set) or None

    if not args.checkpoint.is_file():
        raise SystemExit(f"--checkpoint {args.checkpoint}: no such file")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.config is not None:
        cfg = load_config(args.config, overrides=overrides)
    else:
        sidecar = args.checkpoint.parent / "config.yaml"
        cfg = load_config(sidecar if sidecar.exists() else ckpt["config_name"], overrides=overrides)

    ensure_splits(cfg, args.download)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    )

    state = ckpt.get("model", ckpt)
    model = Vox2SegLAM.from_config(cfg, device=device, template=template_from_state_dict(state)).to(
        device
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    loaders = build_dataloaders(cfg, splits=("test",))
    landmark_ids = cfg.task.resolved_landmark_ids()

    print(
        f"checkpoint epoch {ckpt.get('epoch')} | {len(loaders['test'].dataset)} test subjects | "
        f"{args.passes} pass(es) | device {device}",
        flush=True,
    )

    metrics = validate(
        model=model,
        loader=loaders["test"],
        device=device,
        n_v_classes=cfg.task.n_classes,
        landmark_ids=landmark_ids,
        n_passes=args.passes,
    )
    metrics["n_passes"] = args.passes

    print(json.dumps(metrics, indent=2, default=float))

    out_path = args.checkpoint.parent / args.out
    out_path.write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
