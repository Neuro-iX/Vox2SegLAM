"""Predicts on volumes already on the model's grid; predict.py prepares them first."""

import argparse
import json
from pathlib import Path


from vox2seglam import assets
from vox2seglam.cli import (
    TASKS,
    add_asset_flags,
    add_model_flags,
    resolve_postprocess,
    add_precision_flag,
)
from vox2seglam.inference import DEFAULT_N_PASSES, Predictor, markups_document
from vox2seglam.postprocess import PlaneShift
from vox2seglam.protocol import class_names, label_table, landmark_names
from vox2seglam.utils.io import ensure_dir, save_json, save_nifti


def collect_inputs(path: Path) -> list[Path]:
    if not path.exists():
        raise SystemExit(f"--input {path}: no such file or directory")
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    if not files:
        raise SystemExit(f"no NIfTI volumes found in {path}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict landmarks and segmentation on volumes already on the model's grid\n"
        "(LAS, 0.7 mm, 240 x 304 x 240, skull-stripped). For a raw scan, use\n"
        "predict.py.",
        epilog="example:\n  %(prog)s --input volume_brain.nii.gz --output-dir predictions/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    io = parser.add_argument_group("input and output")
    io.add_argument(
        "--input", type=Path, metavar="PATH", help="NIfTI volume, or a directory of them"
    )
    io.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./predictions"),
        metavar="DIR",
        help="where results are written (default: %(default)s)",
    )
    io.add_argument(
        "--save-segmentation",
        action="store_true",
        help="also write the predicted class map as NIfTI",
    )

    model = parser.add_argument_group("model")
    add_model_flags(model)
    add_precision_flag(model)
    model.add_argument(
        "--checkpoint",
        type=Path,
        metavar="FILE",
        help="use this checkpoint instead of a published model",
    )
    model.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="experiment name or path of the configuration (default: the one "
        "stored beside the checkpoint)",
    )
    model.add_argument(
        "--passes",
        type=int,
        default=DEFAULT_N_PASSES,
        metavar="N",
        help="landmark refinement passes; 1 disables iteration",
    )
    model.add_argument(
        "--postprocess",
        nargs="?",
        const="",
        default=None,
        type=str,
        metavar="FILE",
        help="apply the fitted protocol post-processing to the segmentation "
        "(16 structures only; default file: postprocess.json beside the "
        "checkpoint)",
    )
    model.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="computation device, e.g. cuda, cuda:1 or cpu",
    )

    inspect = parser.add_argument_group("inspection")
    inspect.add_argument(
        "--show-contract",
        action="store_true",
        help="print the input the checkpoint expects, then exit",
    )
    inspect.add_argument(
        "--show-assets",
        action="store_true",
        help="list the published files, where they belong and which are present, then exit",
    )

    downloads = parser.add_argument_group("downloads")
    add_asset_flags(downloads)
    args = parser.parse_args()

    if args.show_assets:
        print(assets.describe())
        return 0

    # before anything is offered for download: a mistyped path must not cost 36 MB first

    # before anything is offered for download: a mistyped path must not cost 36 MB first
    volumes = collect_inputs(args.input) if args.input is not None else []

    checkpoint = args.checkpoint
    if checkpoint is not None and not checkpoint.is_file():
        raise SystemExit(f"--checkpoint {checkpoint}: no such file")
    if checkpoint is None:
        # the weights come with the configuration they were trained with and, if asked for,
        # the fitted post-processing - all three live side by side, so one question covers them
        if args.postprocess is not None and args.task != "subcortical":
            raise SystemExit(
                "--postprocess applies to the subcortical model only; omit "
                "it or use --task subcortical"
            )
        try:
            wanted = assets.model_asset_names(TASKS[args.task], postprocess=args.postprocess == "")
            checkpoint = assets.resolve(*wanted, download=args.download)[wanted[0]]
        except ValueError as err:
            raise SystemExit(str(err))
        except assets.AssetUnavailable as err:
            raise SystemExit(str(err))

    predictor = Predictor.from_checkpoint(
        checkpoint,
        config=args.config,
        device=args.device,
        n_passes=args.passes,
        precision=args.precision,
    )

    if args.show_contract:
        print(json.dumps(predictor.expected_input(), indent=2))
        return 0
    if args.input is None:
        raise SystemExit("--input is required unless --show-contract is given")

    postprocess = None
    if args.postprocess is not None:
        path = resolve_postprocess(checkpoint, args.postprocess)
        postprocess = PlaneShift.load(path)

    ensure_dir(args.output_dir)
    print(
        f"checkpoint epoch {getattr(predictor, 'checkpoint_epoch', '?')} | "
        f"{len(volumes)} volume(s) | device {predictor.device}",
        flush=True,
    )

    import nibabel as nib

    structures = class_names(predictor.cfg.task.n_classes)
    if args.save_segmentation:
        labels = label_table(predictor.cfg.task.n_classes)
        if labels is not None:
            (args.output_dir / "structures.tsv").write_text(labels, encoding="utf-8")
    vocabulary = landmark_names(predictor.cfg.task.n_landmarks, predictor.cfg.task.n_classes)
    for path in volumes:
        stem = path.name.replace(".nii.gz", "").replace(".nii", "")
        result = predictor.predict_file(path, return_probabilities=postprocess is not None)
        out = result.as_dict()
        # as predict.py: a class map of integers says nothing on its own
        if result.segmentation is not None and structures is not None:
            out["structures"] = {str(k): v for k, v in structures.items()}
        if postprocess is not None:
            if result.segmentation is None:
                raise SystemExit("--postprocess needs a checkpoint with a segmentation head")
            result.segmentation, shifts = postprocess(
                result.probabilities, result.coordinates, result.segmentation
            )
            out["postprocess"] = {"plane_shifts_vox": shifts}
        # the same file predict.py writes: a 3D Slicer markups list whose
        # "vox2seglam" key carries the record. A plain record, which Slicer refuses to open,
        # is kept only when the volume's affine is unknown and there are no RAS millimetres.
        document = (
            markups_document(out["landmarks"], name=stem, record=out, names=vocabulary) or out
        )
        save_json(args.output_dir / f"{stem}_landmarks.mrk.json", document)

        if args.save_segmentation and result.segmentation is not None:
            affine = nib.load(str(path)).affine
            seg_xyz = result.segmentation.transpose(2, 1, 0)
            save_nifti(seg_xyz, affine, args.output_dir / f"{stem}_seg.nii.gz")

        print(
            f"  {stem}: {len(result.landmark_ids)} landmarks"
            + (
                ""
                if result.segmentation is None
                else f", seg classes {int(result.segmentation.max())}"
            ),
            flush=True,
        )

    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
