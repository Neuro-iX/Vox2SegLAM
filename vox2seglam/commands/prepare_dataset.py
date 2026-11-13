"""Turns a cohort of native scans and annotations into a training dataset on the reference grid."""

import argparse
import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from vox2seglam import assets
from vox2seglam.cli import add_asset_flags
from vox2seglam.data.landmarks import read_landmark_csv
from vox2seglam.data.prepare.register import (
    check_complete,
    labels_to_reference,
    landmarks_to_reference,
    nifti_stem,
    to_reference_grid,
)

IMAGE_DIR, SEG_DIR, LANDMARK_DIR = "MRI_Volumes", "Segmentation_Volumes", "Landmarks_Volumes"
IMAGE_SUFFIX, LANDMARK_SUFFIX = "_T1w.nii.gz", "_landmarks.nii.gz"


def subject_id(path: Path) -> str:
    return nifti_stem(path).removesuffix("_T1w")


def n_landmarks_in(table: Path) -> int:
    with open(table, newline="", encoding="utf-8-sig") as fh:
        ids = [int(row["landmark_id"]) for row in csv.DictReader(fh)]
    if not ids:
        raise SystemExit(f"{table}: no landmark rows")
    return max(ids)


def match(ids: list[str], directory: Path, suffix: str, what: str) -> dict[str, Path]:
    found = {i: directory / f"{i}{suffix}" for i in ids}
    missing = [i for i, p in found.items() if not p.is_file()]
    if missing:
        shown = ", ".join(missing[:5]) + (
            f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        )
        raise SystemExit(f"no {what} for {shown} (looked for {{id}}{suffix} in {directory})")
    return found


def correlation(aligned: sitk.Image, reference: sitk.Image) -> float:
    """Normalised cross-correlation inside the reference's own brain: a registration that
    missed shows as a value far below the rest of the cohort's."""
    a = sitk.GetArrayFromImage(aligned).astype(np.float64)
    r = sitk.GetArrayFromImage(reference).astype(np.float64)
    inside = r > 0
    if inside.sum() < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):  # a constant image has no correlation
        return float(np.corrcoef(a[inside], r[inside])[0, 1])


def outliers(scores: dict[str, float]) -> list[str]:
    """Subjects whose match is far below the cohort's (median - 3 scaled MAD), if there are
    enough subjects for that to mean something."""
    values = np.array([v for v in scores.values() if np.isfinite(v)])
    if len(values) < 5:
        return []
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    return [i for i, v in scores.items() if np.isfinite(v) and mad > 0 and v < median - 3 * mad]


def write_splits(ids: list[str], fractions: list[float], out: Path, seed: int) -> dict[str, int]:
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    total = sum(fractions)
    n_train = round(len(ids) * fractions[0] / total)
    n_val = round(len(ids) * fractions[1] / total)
    parts = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    (out / "splits").mkdir(exist_ok=True)
    for name, members in parts.items():
        (out / "splits" / f"{name}_ids.txt").write_text(
            "\n".join(sorted(members)) + "\n", encoding="utf-8"
        )
    return {name: len(members) for name, members in parts.items()}


def write_config(
    out: Path, reference: Path, n_landmarks: int, n_classes: int | None, split: dict | None
) -> Path:
    image = nib.load(str(reference))
    zyx = [int(v) for v in reversed(image.shape[:3])]
    spacing = [float(v) for v in reversed(image.header.get_zooms()[:3])]
    orientation = "".join(nib.aff2axcodes(image.affine))
    rows = [
        f"# Written by prepare_dataset.py on {datetime.now(UTC):%Y-%m-%d}. Train with:",
        f"#     vox2seglam train --config {out.resolve() / 'cohort.yaml'}",
        "",
        f"name: {out.resolve().name}",
        "extends: baseline",
        "",
        "paths:",
        f"  data_root: {out.resolve()}",
        f"  image_dir: {IMAGE_DIR}",
        f"  image_suffix: {IMAGE_SUFFIX}",
        f"  landmark_dir: {LANDMARK_DIR}",
        f"  landmark_suffix: {LANDMARK_SUFFIX}",
    ]
    if n_classes:
        rows += [f"  seg_dir: {SEG_DIR}", "  seg_suffix: .nii.gz"]
    if split:
        rows += [
            f"  splits_dir: {out.resolve() / 'splits'}",
            "  split_train: train_ids.txt",
            "  split_val: val_ids.txt",
            "  split_test: test_ids.txt",
        ]
    else:
        rows += [
            "  # splits_dir: /where/your/split/files/are",
            "  # split_train: train_ids.txt   (also split_val, split_test)",
        ]
    rows += [
        f"  exp_dir: ./experiments/{out.resolve().name}",
        "",
        "task:",
        f"  n_landmarks: {n_landmarks}",
        f"  n_classes: {n_classes if n_classes else 1}"
        + ("" if n_classes else "  # unused without a segmentation head"),
        f"  patch_size: {zyx}",
        f"  voxel_spacing_mm: {spacing}",
        "",
        "preprocess:",
        f"  orientation: {orientation}",
        "  skull_stripped: true       # made and cached by the first training run",
        "  bias_field_corrected: true",
    ]
    if not n_classes:
        rows += [
            "",
            "# no segmentation was given: a landmark-only model",
            "model:",
            "  seg_head:",
            "    enabled: false",
            "    deep_supervision: false",
            "loss:",
            "  adaptive_weighting: false",
            "  init_w_seg: 0.0",
            "  init_w_lm: 1.0",
            "staging:",
            "  gnn_must_beat_template: false",
        ]
    path = out / "cohort.yaml"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bring native scans, segmentations and landmarks onto the reference grid "
        "to train on.",
        epilog="example:\n  %(prog)s --scans scans/ --segmentations labels/ "
        "--landmarks landmarks.csv --output cohort/ --split 0.7 0.15 0.15\n\n"
        "Each scan is registered rigidly to the reference template and resampled onto it;\n"
        "the segmentation goes through the same transform by nearest neighbour (no label\n"
        "is invented) and each landmark to its nearest voxel. Skull stripping and N4 are\n"
        "done by the first training run. Writes, under --output:\n"
        "  MRI_Volumes/            {id}_T1w.nii.gz        the scan on the reference grid\n"
        "  Segmentation_Volumes/   {id}.nii.gz            labels, numbered as in your files\n"
        "  Landmarks_Volumes/      {id}_landmarks.nii.gz  voxel of landmark k holds k\n"
        "  transforms/             {id}_to_reference.tfm  reference -> native\n"
        "  splits/                 train/val/test_ids.txt (with --split)\n"
        "  cohort.yaml             a training configuration for this dataset\n"
        "  prepare_dataset.json    per subject: files and match with the reference\n"
        "Landmarks: columns participant_id, landmark_id, x_mm, y_mm, z_mm and optionally\n"
        "coordinate_system (LPS by default, or RAS), in each scan file's own millimetres.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inp = ap.add_argument_group("input")
    inp.add_argument(
        "--scans",
        type=Path,
        required=True,
        metavar="DIR",
        help="native T1-weighted NIfTI volumes; the subject id is the file name "
        "without .nii.gz (and without a trailing _T1w)",
    )
    inp.add_argument(
        "--segmentations",
        type=Path,
        metavar="DIR",
        help="native label maps, {id}<suffix> in this directory; leave out to "
        "train a landmark-only model",
    )
    inp.add_argument(
        "--segmentation-suffix", default=".nii.gz", metavar="SUFFIX", help="(default: %(default)s)"
    )
    inp.add_argument(
        "--landmarks",
        type=Path,
        required=True,
        metavar="FILE",
        help="coordinate table with participant_id, landmark_id, x_mm, y_mm, z_mm",
    )
    inp.add_argument(
        "--reference",
        type=Path,
        metavar="FILE",
        help="reference template to register to (default: the published one)",
    )
    out = ap.add_argument_group("output")
    out.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="the dataset to write; becomes paths.data_root",
    )
    out.add_argument(
        "--split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        help="also write train/val/test id lists in these proportions",
    )
    out.add_argument("--seed", type=int, default=0, help="seed of the split (default: %(default)s)")
    out.add_argument(
        "--force", action="store_true", help="redo subjects whose outputs already exist"
    )
    add_asset_flags(ap.add_argument_group("downloads"))
    a = ap.parse_args()

    if not a.scans.is_dir():
        raise SystemExit(f"--scans {a.scans}: no such directory")
    scans = sorted(p for p in a.scans.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    if not scans:
        raise SystemExit(f"no NIfTI volumes in {a.scans}")
    by_id = {subject_id(p): p for p in scans}
    if len(by_id) != len(scans):
        raise SystemExit("two scans have the same subject id once a trailing _T1w is dropped")
    ids = sorted(by_id)
    segmentations = (
        match(ids, a.segmentations, a.segmentation_suffix, "segmentation")
        if a.segmentations
        else {}
    )
    k = n_landmarks_in(a.landmarks)
    points, _ = read_landmark_csv(a.landmarks, k)
    absent = [i for i in ids if i not in points]
    if absent:
        raise SystemExit(
            f"{a.landmarks} has no landmarks for {', '.join(absent[:5])}"
            + (f" and {len(absent) - 5} more" if len(absent) > 5 else "")
        )

    reference_file = a.reference
    if reference_file is None:
        try:
            reference_file = assets.resolve("reference", download=a.download)["reference"]
        except assets.AssetUnavailable as err:
            raise SystemExit(str(err))
    reference = sitk.ReadImage(str(reference_file), sitk.sitkFloat32)

    for name in (IMAGE_DIR, SEG_DIR if segmentations else None, LANDMARK_DIR, "transforms"):
        if name:
            (a.output / name).mkdir(parents=True, exist_ok=True)

    summary_file = a.output / "prepare_dataset.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    subjects = summary.setdefault("subjects", {})

    def prepare_subject(n: int, sid: str) -> None:
        targets = {
            "image": a.output / IMAGE_DIR / f"{sid}{IMAGE_SUFFIX}",
            "transform": a.output / "transforms" / f"{sid}_to_reference.tfm",
        }
        if segmentations:
            targets["segmentation"] = a.output / SEG_DIR / f"{sid}.nii.gz"
        targets["landmarks"] = a.output / LANDMARK_DIR / f"{sid}{LANDMARK_SUFFIX}"
        if not a.force and sid in subjects and all(p.is_file() for p in targets.values()):
            print(f"[{n}/{len(ids)}] {sid}: already done", flush=True)
            return
        for p in (by_id[sid], *([segmentations[sid]] if segmentations else [])):
            check_complete(p)
        aligned, transform = to_reference_grid(by_id[sid], reference_file)
        # annotations first, then the files: a subject whose landmarks fall off the grid
        # leaves nothing behind
        seg_image = None
        top_label = 0
        if segmentations:
            seg_image = labels_to_reference(
                sitk.ReadImage(str(segmentations[sid])), reference, transform
            )
            top_label = int(sitk.GetArrayFromImage(seg_image).max())
        lm_image = landmarks_to_reference(points[sid], reference, transform)
        sitk.WriteImage(aligned, str(targets["image"]))
        sitk.WriteTransform(transform, str(targets["transform"]))
        if seg_image is not None:
            sitk.WriteImage(seg_image, str(targets["segmentation"]))
        sitk.WriteImage(lm_image, str(targets["landmarks"]))
        score = correlation(aligned, reference)
        subjects[sid] = {
            "registration_correlation": round(score, 4),
            "max_label": top_label,
            "source": str(by_id[sid].resolve()),
        }
        summary_file.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"[{n}/{len(ids)}] {sid}: done (match {score:.3f})", flush=True)

    # one subject that cannot be prepared (a truncated file, a landmark off the grid) is
    # reported and left out, so it does not stop the rest of the cohort
    failed = {}
    for n, sid in enumerate(ids, start=1):
        try:
            prepare_subject(n, sid)
        except assets.AssetUnavailable:
            raise
        except (ValueError, RuntimeError, OSError) as err:
            reason = str(err).strip()
            failed[sid] = reason.splitlines()[0] if reason else repr(err)
            print(f"[{n}/{len(ids)}] {sid}: skipped - {failed[sid]}", flush=True)
    ids = [sid for sid in ids if sid not in failed]
    if not ids:
        raise SystemExit(
            "no subject could be prepared:\n  "
            + "\n  ".join(f"{sid}: {why}" for sid, why in failed.items())
        )

    scores = {sid: subjects[sid]["registration_correlation"] for sid in ids}
    for sid in outliers(scores):
        print(
            f"warning: {sid} matches the reference much worse than the rest of the cohort "
            f"({scores[sid]:.3f}); check its registration before training on it",
            flush=True,
        )
    n_classes = max(subjects[sid]["max_label"] for sid in ids) + 1 if segmentations else None
    split = write_splits(ids, a.split, a.output, a.seed) if a.split else None
    summary.update(
        {
            "reference": str(Path(reference_file).resolve()),
            "n_landmarks": k,
            "n_classes": n_classes,
            "split": split,
        }
    )
    summary_file.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    config = write_config(a.output, Path(reference_file), k, n_classes, split)
    print(
        f"\n{len(ids)} subjects -> {a.output}"
        + f"; {k} landmarks"
        + (f", {n_classes} classes (background included)" if n_classes else "")
        + (f"; split {split}" if split else "")
    )
    if failed:
        print(f"\n{len(failed)} subject(s) skipped:")
        for sid, why in failed.items():
            print(f"  {sid}: {why}")
    print(f"train with: vox2seglam train --config {config}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
