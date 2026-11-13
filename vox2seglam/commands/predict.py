"""Predicts landmarks and segmentation on a scan of your own, in your scan's own space."""

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import SimpleITK as sitk

from vox2seglam import assets
from vox2seglam.cli import TASKS, add_asset_flags, add_model_flags, banner, resolve_postprocess
from vox2seglam.config import Config
from vox2seglam.data.prepare.register import prepared_scans
from vox2seglam.data.prepare.scan import pending, prepare_all
from vox2seglam.data.prepare.synthstrip import synthstrip_script, uses_builtin
from vox2seglam import __version__

# Predictor and markups_document are imported inside main(): naming them here would resolve
# them at import time and pay for torch before the first line could be printed.
from vox2seglam.inference.constants import DEFAULT_N_PASSES
from vox2seglam.inference.scan import predict_scan
from vox2seglam.postprocess import PlaneShift
from vox2seglam.protocol import class_names, label_table, landmark_names
from vox2seglam.utils.io import ensure_dir, save_json


def _field(label: str, value: str) -> None:
    """One line of the run header: a left-aligned name, then what was actually used."""
    print(f"  {label:<12}{value}", flush=True)


def _n_parameters(model) -> int:
    """Distinct tensors only: the backbone stores several convolutions twice (one alias per
    view of the same storage), and counting the state_dict naively doubles them."""
    unique = {p.data_ptr(): p for p in model.parameters()}
    return sum(p.numel() for p in unique.values())


def _device_label(device) -> str:
    import torch

    if device.type != "cuda":
        return str(device)
    try:
        return f"{device} ({torch.cuda.get_device_name(device)})"
    except (AssertionError, RuntimeError):
        return str(device)


def _header(
    predictor,
    *,
    checkpoint: Path,
    config,
    reference_file: Path,
    post_file,
    n_landmarks: int,
    segmentation: bool,
    verbose: bool,
) -> None:
    """One line saying what is about to run. Everything else is provenance: it is written into
    the result either way, and only printed on --verbose, where somebody is checking."""
    cfg = predictor.cfg
    task = (
        f"{cfg.task.n_classes - 1} structures, {n_landmarks} landmarks"
        if segmentation
        else f"{n_landmarks} landmarks, no segmentation"
    )
    print(
        f"  {task}"
        + (", protocol enforced" if post_file else "")
        + f" - {predictor.device}, {predictor.precision}",
        flush=True,
    )
    if not verbose:
        return
    contract = predictor.expected_input()
    size = sitk.ReadImage(str(reference_file)).GetSize()  # x, y, z
    spacing = cfg.task.voxel_spacing_mm
    _field(
        "checkpoint",
        f"{checkpoint.name}"
        + (f" (epoch {predictor.checkpoint_epoch})" if predictor.checkpoint_epoch else ""),
    )
    _field(
        "config", Path(str(config)).name if config else f"{cfg.name} (recorded in the checkpoint)"
    )
    _field("parameters", f"{_n_parameters(predictor.model) / 1e6:.2f} M")
    _field(
        "device",
        f"{_device_label(predictor.device)} - {predictor.precision} - "
        f"{predictor.n_passes} refinement pass"
        f"{'es' if predictor.n_passes != 1 else ''}",
    )
    _field(
        "reference",
        f"{reference_file.name} - {size[0]}x{size[1]}x{size[2]} grid at "
        f"{spacing[0]}x{spacing[1]}x{spacing[2]} mm",
    )
    _field(
        "protocol",
        f"plane shifts fitted in {Path(post_file).name}"
        if post_file is not None
        else "not enforced (--postprocess enforces it)",
    )
    _field(
        "expects",
        f"{contract['orientation']}, skull-stripped, bias-corrected, {contract['normalisation']}",
    )


# The project's own coordinate table: what export_annotations.py writes, prepare_dataset.py
# turns into a dataset and evaluate.py scores against. A Slicer markups file is the right
# thing to open in a viewer and the wrong thing to read in a script, so a run also leaves the
# landmarks in the format its own commands already speak.
LANDMARK_COLUMNS = [
    "participant_id",
    "landmark_id",
    "landmark_name",
    "coordinate_system",
    "x_mm",
    "y_mm",
    "z_mm",
    "voxel_x",
    "voxel_y",
    "voxel_z",
]


def landmark_rows(stem, landmarks, names, image):
    """One row per landmark, in the scan's own LPS millimetres and its own voxel index."""
    _, full = names or ({}, {})
    for lid, entry in sorted(landmarks.items(), key=lambda kv: int(kv[0])):
        mm = [float(v) for v in entry["native_lps_mm"]]
        voxel = image.TransformPhysicalPointToContinuousIndex(mm) if image is not None else None
        row = {
            "participant_id": stem,
            "landmark_id": int(lid),
            "landmark_name": full.get(int(lid), ""),
            "coordinate_system": "LPS",
            "x_mm": f"{mm[0]:.4f}",
            "y_mm": f"{mm[1]:.4f}",
            "z_mm": f"{mm[2]:.4f}",
        }
        for axis, key in enumerate(("voxel_x", "voxel_y", "voxel_z")):
            row[key] = f"{voxel[axis]:.4f}" if voxel is not None else ""
        yield row


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Predict anatomical landmarks and a subcortical segmentation on a\n"
        "T1-weighted MRI of your own, in the scan's native space.",
        epilog="examples:\n"
        "  %(prog)s --input scan.nii.gz --output-dir results/\n"
        "  %(prog)s --input scans/ --task full-brain\n"
        "  %(prog)s --input scan_brain.nii.gz --already-stripped\n\n"
        "Whatever is not prepared yet is registered onto the reference grid, skull-stripped\n"
        "and bias-corrected, then predicted and mapped back. That preparation is kept in\n"
        "<output-dir>/prepared and reused on later runs that write there, unless the scan,\n"
        "the reference, the stripper or the settings have changed -- so a run with another\n"
        "--output-dir prepares again unless --prepared-dir points at the first one.\n"
        "Per scan, in --output-dir:\n"
        "  <stem>_landmarks.mrk.json   the landmarks, as a 3D Slicer markups list that\n"
        '                          opens by drag and drop; its "vox2seglam" key adds\n'
        "                          the reference-grid voxel and LPS millimetres of each\n"
        "                          one, and what produced them\n"
        "  <stem>_seg.nii.gz       the class map on the scan's own voxels\n"
        "  <stem>_seg_reference.nii.gz  --save-reference-grid: the same map on the\n"
        "                          reference grid, where the protocol's planes are\n"
        "                          axis-aligned\n"
        "Missing files (weights, reference volume, SynthStrip's weights) are offered for\n"
        "download in one question, on a terminal only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    io = ap.add_argument_group("input and output")
    io.add_argument(
        "--input",
        type=Path,
        metavar="PATH",
        help="T1-weighted NIfTI volume, or a directory of them. Scans are "
        "prepared automatically when needed.",
    )
    io.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./predictions"),
        metavar="DIR",
        help="where results are written (default: %(default)s)",
    )
    io.add_argument(
        "--prepared-dir",
        type=Path,
        metavar="DIR",
        help="where prepared scans are kept and looked for (default: <output-dir>/prepared). "
        "Preparation costs about 50 s and 32 MB a scan, so point several runs at one "
        "directory to do it once -- otherwise a different --output-dir prepares again",
    )

    model = ap.add_argument_group("model")
    add_model_flags(model)
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
        help="configuration of the checkpoint (default: the one stored beside it)",
    )
    model.add_argument(
        "--postprocess",
        nargs="?",
        const="",
        default=None,
        type=str,
        metavar="FILE",
        help="enforce the anatomical protocol on the segmentation: every "
        "rule holds of the file that is written, on your volume's own "
        "voxels and with no slack. 16 structures only; the fitted plane "
        "ranges come from postprocess.json beside the checkpoint unless "
        "a file is given.",
    )
    model.add_argument(
        "--save-reference-grid",
        action="store_true",
        help="also write the class map on the reference grid, where the "
        "protocol's planes are axis-aligned and the sampling is 0.7 mm",
    )
    model.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="computation device, e.g. cuda, cuda:1 or cpu (default: cuda if available)",
    )

    prep = ap.add_argument_group("scan preparation")
    prep.add_argument(
        "--reference",
        type=Path,
        metavar="FILE",
        help="reference template the scans are registered to (default: the published one)",
    )
    prep.add_argument(
        "--already-stripped",
        action="store_true",
        help="the input is already skull-stripped; skip skull stripping",
    )
    prep.add_argument(
        "--force",
        action="store_true",
        help="redo preparation even when an up-to-date result exists",
    )

    downloads = ap.add_argument_group("downloads")
    add_asset_flags(downloads)
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the checkpoint, the reference grid and the preprocessing "
        "contract before running (all of it is written into the result "
        "either way)",
    )
    a = ap.parse_args()
    if a.input is None and a.prepared_dir is None:
        ap.error(
            "give --input (a scan to prepare and predict) or --prepared-dir (scans "
            "already prepared)"
        )

    banner()
    model_names = []
    if a.postprocess is not None and a.task != "subcortical":
        ap.error(
            "--postprocess applies to the subcortical model only; omit it or use --task subcortical"
        )
    if a.checkpoint is not None and a.task != "subcortical":
        ap.error("--task selects a published model; it cannot be combined with --checkpoint")
    if a.checkpoint is not None and not a.checkpoint.is_file():
        ap.error(f"--checkpoint {a.checkpoint}: no such file")
    if a.checkpoint is None:
        try:
            model_names = assets.model_asset_names(TASKS[a.task], postprocess=a.postprocess == "")
        except ValueError as err:
            ap.error(str(err))

    # The built-in skull stripper, always. A config saved beside a checkpoint records its
    # author's machine -- a FreeSurfer path that exists nowhere else -- and must not decide
    # how a stranger's scan is stripped. A cohort that has to be stripped another way goes
    # through prepare.py, which takes --freesurfer-home and --set; this command then reads
    # the result with --prepared-dir.
    route = Config()

    inputs, prepared_dir = [], a.prepared_dir
    if a.input is not None:
        if not a.input.exists():
            raise SystemExit(f"--input {a.input}: no such file or directory")
        inputs = (
            [a.input]
            if a.input.is_file()
            else sorted(p for p in a.input.iterdir() if p.name.endswith((".nii", ".nii.gz")))
        )
        if not inputs:
            raise SystemExit(f"no NIfTI volumes found in {a.input}")
        prepared_dir = prepared_dir or a.output_dir / "prepared"
    script = None
    if inputs and not a.already_stripped:
        try:
            script = synthstrip_script(route)
        except (RuntimeError, FileNotFoundError) as err:
            raise SystemExit(str(err))

    # Everything missing is asked for in one question, and only what this run needs: the skull
    # stripper's weights only if some scan has to be stripped. (Whether a scan is up to date is
    # judged here with default preprocessing settings; the checkpoint's own are applied when it
    # is actually prepared, so the worst case is one unneeded offer, never a stale reuse.)
    reference_path = a.reference or assets.asset_path("reference")
    todo = (
        pending(
            inputs,
            reference_path,
            prepared_dir,
            route,
            script,
            already_stripped=a.already_stripped,
            force=a.force,
        )
        if inputs
        else []
    )
    wanted = list(model_names)
    if a.reference is None:
        wanted.append("reference")
    if todo and uses_builtin(route) and not a.already_stripped:
        wanted.append("synthstrip")
    try:
        found = assets.resolve(*wanted, download=a.download) if wanted else {}
    except assets.AssetUnavailable as err:
        raise SystemExit(str(err))

    checkpoint = a.checkpoint or found[model_names[0]]
    postprocess, post_file = None, None
    if a.postprocess is not None:
        post_file = resolve_postprocess(checkpoint, a.postprocess, found.get("postprocess"))
        # strict by design: the fitted tolerance exists so that enforcement would not correct
        # the HOA2.0 annotations where they cross their own planes -- a choice that belongs to
        # reproducing the paper, not to predicting on somebody's scan
        postprocess = PlaneShift.load(post_file).strict()

    # Loading is a few seconds of torch, CUDA and the weights. Say so before it starts: a
    # command that prints nothing for four seconds looks stuck.
    print("  loading the model...", end="", flush=True)
    clock = time.perf_counter()
    from vox2seglam.inference import Predictor, markups_document

    predictor = Predictor.from_checkpoint(
        checkpoint, config=a.config, device=a.device, n_passes=DEFAULT_N_PASSES
    )
    reference_file = a.reference or found.get("reference", reference_path)
    reference = sitk.ReadImage(str(reference_file))
    print(f" {time.perf_counter() - clock:.1f} s", flush=True)
    has_segmentation = predictor.cfg.task.n_classes > 1
    structures = class_names(predictor.cfg.task.n_classes)
    vocabulary = landmark_names(predictor.cfg.task.n_landmarks, predictor.cfg.task.n_classes)
    _header(
        predictor,
        checkpoint=checkpoint,
        config=a.config,
        reference_file=reference_file,
        post_file=post_file if postprocess is not None else None,
        n_landmarks=len(predictor.landmark_ids),
        segmentation=has_segmentation,
        verbose=a.verbose,
    )

    # the checkpoint's own preprocessing settings, with the built-in stripper
    cfg = replace(
        predictor.cfg,
        paths=replace(predictor.cfg.paths, freesurfer_home=route.paths.freesurfer_home),
        preprocess=replace(
            predictor.cfg.preprocess, synthstrip_command=route.preprocess.synthstrip_command
        ),
    )

    ensure_dir(a.output_dir)
    started = time.perf_counter()
    failed = {}
    if inputs:
        # prepare_all reports one line per scan as it finishes; the index and the time it took
        # are added here, so that a cohort of fifty shows its progress rather than going quiet
        todo_now = len(todo)
        if not todo_now:
            print(
                f"{len(inputs)} scan{'s' if len(inputs) != 1 else ''} already prepared in "
                f"{prepared_dir}",
                flush=True,
            )
        else:
            print(
                f"preparing {todo_now} scan{'s' if todo_now != 1 else ''} -> {prepared_dir}"
                + (f" ({len(inputs) - todo_now} already done)" if todo_now < len(inputs) else ""),
                flush=True,
            )
        # Preparation is the slow part -- a minute or so per scan, in three steps that print
        # nothing of their own. Each step is announced before it starts and timed when it
        # ends, on one rewritten line per scan, so the wait is never blank.
        state = {"i": 0, "open": False, "step": 0.0, "scan": 0.0}

        def stage(path: Path, name: str) -> None:
            now = time.perf_counter()
            if state["open"]:
                print(f" {now - state['step']:.0f} s", flush=True)
            else:
                state["i"] += 1
                state["scan"] = now
                print(f"  [{state['i']}/{todo_now}] {path.name}", flush=True)
            print(f"        {name}...", end="", flush=True)
            state["open"], state["step"] = True, now

        # prepare_all reports each scan once it is done, and says why it was (re)done -- the
        # reason belongs in the output, so it closes the scan's block
        def show(line: str) -> None:
            now = time.perf_counter()
            why = line.split(": ", 1)[-1].strip()
            if state["open"]:
                print(f" {now - state['step']:.0f} s", flush=True)
                print(f"        done in {now - state['scan']:.0f} s - {why}", flush=True)
                state["open"] = False
            elif "skipped" in why:
                print(f"  {line.strip()}", flush=True)

        scans, failed = prepare_all(
            inputs,
            reference_file,
            prepared_dir,
            cfg,
            script,
            already_stripped=a.already_stripped,
            force=a.force,
            show=show,
            stage=stage,
        )
        if not scans:
            raise SystemExit(
                "no scan could be prepared:\n  "
                + "\n  ".join(f"{p.name}: {why}" for p, why in failed.items())
            )
    else:
        scans = prepared_scans(prepared_dir)
        if not scans:
            raise SystemExit(f"no scan prepared by prepare.py in {prepared_dir}")
        print(
            f"{len(scans)} scan{'s' if len(scans) != 1 else ''} already prepared in {prepared_dir}",
            flush=True,
        )

    print(f"\npredicting {len(scans)} scan{'s' if len(scans) != 1 else ''}", flush=True)
    table = []
    for i, scan in enumerate(scans, 1):
        clock = time.perf_counter()
        out = predict_scan(
            predictor,
            scan,
            reference,
            postprocess=postprocess,
            segmentation=has_segmentation,
            on_reference=a.save_reference_grid,
        )
        elapsed = time.perf_counter() - clock

        record = {
            "producer": "predict.py",
            "version": __version__,
            "checkpoint": str(checkpoint),
            "epoch": predictor.checkpoint_epoch,
            "reference": str(reference_file),
            "precision": out.precision,
            "n_passes": predictor.n_passes,
            "coordinate_systems": {
                "reference_voxel_zyx": "voxel index on the reference grid",
                "reference_ras_mm": "millimetres on the reference grid, RAS -- the "
                "space the prepared volume lives in",
                "native_lps_mm": "millimetres in the scan's own space, LPS (ITK)",
                "native_ras_mm": "millimetres in the scan's own space, RAS (Slicer)",
            },
        }
        # the class map is a volume of integers; without this nothing says that 6 is the left
        # putamen, and the 39-structure model has 39 of them
        if has_segmentation and structures is not None:
            record["structures"] = {str(k): v for k, v in structures.items()}
        if out.plane_shifts is not None:
            record["postprocess"] = {"fit": str(post_file), "plane_shifts_vox": out.plane_shifts}
            if out.protocol_check is not None:
                record["postprocess"]["verified"] = out.protocol_check

        # one file: a Slicer markups list (it opens by drag and drop) whose "vox2seglam" key
        # carries the record Slicer has no place for
        target = a.output_dir / f"{scan.stem}_landmarks.mrk.json"
        save_json(
            target, markups_document(out.landmarks, name=scan.stem, record=record, names=vocabulary)
        )

        table.extend(landmark_rows(scan.stem, out.landmarks, vocabulary, out.segmentation))

        print(
            f"  [{i}/{len(scans)}] {scan.stem}: {len(out.landmarks)} landmarks in {elapsed:.1f} s",
            flush=True,
        )
        # one line, and only the outcome: how many planes moved and how many rules were read
        # are in the result file, where a script can have them. What a reader needs here is
        # whether the map that was written satisfies the protocol -- and the detail only when
        # it does not, since that is a defect to report.
        if out.protocol_check is not None:
            left = out.protocol_check["violating_voxels_after"]
            print(
                "      protocol   "
                + (
                    "verified, nothing outside the admissible zones"
                    if not left
                    else f"{out.protocol_check['violating_after_mm3']} mm3 still outside the "
                    f"admissible zones ({', '.join(f'{k} {v}' for k, v in left.items())})"
                ),
                flush=True,
            )
        print(f"      landmarks  {target.name}", flush=True)
        if out.segmentation is not None:
            seg_file = a.output_dir / f"{scan.stem}_seg.nii.gz"
            sitk.WriteImage(out.segmentation, str(seg_file))
            print(f"      labels     {seg_file.name}", flush=True)
        if out.on_reference is not None:
            grid_file = a.output_dir / f"{scan.stem}_seg_reference.nii.gz"
            sitk.WriteImage(out.on_reference, str(grid_file))
            print(f"      reference  {grid_file.name}  (on the reference grid, 0.7 mm)", flush=True)

    # a class map is a volume of integers: every viewer needs the names beside it
    if has_segmentation:
        labels = label_table(predictor.cfg.task.n_classes)
        if labels is not None:
            (a.output_dir / "structures.tsv").write_text(labels, encoding="utf-8")

    # one table for the whole run, as read_landmark_csv reads it: every subject, one row per
    # landmark, so a cohort comes out of the command ready to be scored or trained on
    if table:
        coordinates = a.output_dir / "landmarks.csv"
        with coordinates.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LANDMARK_COLUMNS)
            writer.writeheader()
            writer.writerows(table)
        print(
            f"\ncoordinates  {coordinates.name}  ({len(table)} rows, "
            f"{len(scans)} subject{'s' if len(scans) != 1 else ''})",
            flush=True,
        )

    print(
        f"\n{len(scans)} scan{'s' if len(scans) != 1 else ''} in "
        f"{time.perf_counter() - started:.1f} s; written to {a.output_dir}",
        flush=True,
    )
    if failed:
        print(f"\n{len(failed)} scan(s) skipped:", flush=True)
        for path, why in failed.items():
            print(f"  {path.name}: {why}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
