"""Scores predictions against your own reference, and checks them against the protocol."""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from vox2seglam.data.landmarks import read_landmark_csv
from vox2seglam.evaluate import landmark_summary, radial_errors, segmentation_metrics
from vox2seglam.postprocess import PlaneShift
from vox2seglam.protocol import REGION_TO_CLASS
from vox2seglam.utils.io import save_json

LANDMARK_FILE = "_landmarks.mrk.json"
SEG_FILE = "_seg.nii.gz"


def predictions(directory: Path) -> dict[str, dict]:
    """{subject: {"landmarks": path, "segmentation": path or None}} of a predict.py run."""
    import json

    found = {}
    for path in sorted(directory.glob(f"*{LANDMARK_FILE}")):
        stem = path.name[: -len(LANDMARK_FILE)]
        record = json.loads(path.read_text(encoding="utf-8")).get("vox2seglam")
        if record is None:
            raise SystemExit(f"{path.name} was not written by predict.py (no vox2seglam key)")
        seg = directory / f"{stem}{SEG_FILE}"
        found[stem] = {"record": record, "segmentation": seg if seg.is_file() else None}
    if not found:
        raise SystemExit(
            f"{directory}: no such directory"
            if not directory.is_dir()
            else f"no *{LANDMARK_FILE} in {directory}; is this a predict.py output?"
        )
    return found


def coordinates(record: dict, *keys: str) -> np.ndarray:
    """The (K, 3) coordinates of one prediction, in the first of `keys` it carries.

    predict.py returns a scan's own millimetres and predict_prepared.py the model grid's
    voxels, under different names; a caller asks for the ones it can work with and gets a
    message rather than a KeyError when a file has none of them.
    """
    marks = record["landmarks"]
    first = next(iter(marks.values()), {})
    for key in keys:
        if key in first:
            return np.array([marks[str(i)][key] for i in sorted(map(int, marks))], dtype=float)
    raise SystemExit(
        f"these landmarks carry {sorted(first)}, none of which is "
        f"{' or '.join(keys)}; was the file written by predict.py?"
    )


def match(reference: Path, stem: str, suffix: str) -> Path | None:
    """The reference volume of `stem`, named after it or after the subject id before it."""
    for name in (f"{stem}{suffix}", f"{stem.split('_')[0]}{suffix}"):
        if (reference / name).is_file():
            return reference / name
    return None


def score_segmentation(pred_path: Path, truth_path: Path, classes: dict[int, str]) -> dict:
    pred_img, truth_img = sitk.ReadImage(str(pred_path)), sitk.ReadImage(str(truth_path))
    if pred_img.GetSize() != truth_img.GetSize():
        raise SystemExit(
            f"{pred_path.name} is {pred_img.GetSize()} and "
            f"{truth_path.name} is {truth_img.GetSize()}; they must be the same "
            "volume on the same grid"
        )
    scored = segmentation_metrics(
        sitk.GetArrayFromImage(pred_img),
        sitk.GetArrayFromImage(truth_img),
        classes,
        spacing=pred_img.GetSpacing(),
    )["whole"]
    return {"per_structure": scored["per"], "mean": scored["mean"]}


def compliance(pred_path: Path, record: dict, spec) -> dict:
    """Volume of predicted labels outside their admissible zone, in mm3."""
    from vox2seglam.inference.scan import _to_las_zyx
    from vox2seglam.protocol import protocol_planes, rule_depth

    image = sitk.ReadImage(str(pred_path))
    labels = sitk.GetArrayFromImage(image)
    marks = next(iter(record["landmarks"].values()), {})

    if "native_lps_mm" in marks:
        planes = protocol_planes(spec, _to_las_zyx(coordinates(record, "native_lps_mm")))

        def points_of(index):
            return _to_las_zyx(
                np.array(
                    [
                        image.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
                        for z, y, x in index
                    ],
                    dtype=float,
                )
            )
    else:
        # already on the model's grid: its voxels are the space the protocol is written in
        planes = protocol_planes(spec, coordinates(record, "voxel_zyx"))

        def points_of(index):
            return index.astype(float)

    counts, voxel_mm3 = {}, float(np.prod(image.GetSpacing()))
    for rule in spec.rules:
        index = np.argwhere(labels == rule.region_class)
        counts[rule.region_name] = (
            0 if not len(index) else int((rule_depth(points_of(index), planes, rule) > 1e-6).sum())
        )
    return {
        "violating_mm3": round(sum(counts.values()) * voxel_mm3, 3),
        "violating_voxels": {k: v for k, v in counts.items() if v},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scores a prediction against your own reference: --segmentations for\n"
        "overlap and surface distances, a --landmarks table for localisation\n"
        "error, either or both.",
        epilog="examples:\n"
        "  %(prog)s --predictions results/ --segmentations cohort/Segmentation\n"
        "  %(prog)s --predictions results/ --landmarks annotations.csv\n"
        "  %(prog)s --predictions results/ --protocol --out scores.json\n\n"
        "The landmark table is what export_annotations.py writes: one row per subject\n"
        "and landmark, with landmark_id, x_mm, y_mm and z_mm in the scan's own space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--predictions",
        required=True,
        type=Path,
        metavar="DIR",
        help="a directory predict.py wrote",
    )
    ap.add_argument(
        "--segmentations",
        type=Path,
        metavar="DIR",
        help="reference label maps, on the same grid as the predicted ones",
    )
    ap.add_argument(
        "--seg-suffix",
        default=".nii.gz",
        metavar="SUFFIX",
        help="how a reference label map is named after its subject (default: %(default)s)",
    )
    ap.add_argument(
        "--landmarks",
        type=Path,
        metavar="FILE",
        help="reference landmark coordinates, as export_annotations.py writes them",
    )
    ap.add_argument(
        "--id-column",
        default="participant_id",
        metavar="NAME",
        help="the subject column of that table (default: %(default)s)",
    )
    ap.add_argument(
        "--protocol",
        action="store_true",
        help="also measure protocol compliance, which needs no reference at all. It is read "
        "against the planes of the predicted landmarks with no shift, so it is stricter "
        "than, and not comparable with, the figure predict.py reports on the same map",
    )
    ap.add_argument(
        "--out", type=Path, metavar="FILE", help="write every per-subject figure here as JSON"
    )
    a = ap.parse_args()

    found = predictions(a.predictions)
    spec = PlaneShift(shift_range={}).spec
    classes = {c: name for name, c in REGION_TO_CLASS.items()}
    # a table that is empty, mis-columned or in the wrong system is a file the user can fix;
    # it reached them as a traceback from inside the reader
    try:
        truth, _ = (
            read_landmark_csv(a.landmarks, 16, id_column=a.id_column) if a.landmarks else ({}, {})
        )
    except (ValueError, KeyError, OSError) as err:
        said = str(err)
        raise SystemExit(said if str(a.landmarks) in said else f"--landmarks {a.landmarks}: {said}")

    subjects, errors = {}, []
    for stem, entry in sorted(found.items()):
        one: dict = {}
        if a.segmentations is not None and entry["segmentation"] is not None:
            reference = match(a.segmentations, stem, a.seg_suffix)
            if reference is None:
                print(f"  {stem}: no reference segmentation, skipped", flush=True)
            else:
                one["segmentation"] = score_segmentation(entry["segmentation"], reference, classes)
        if truth:
            key = stem if stem in truth else stem.split("_")[0]
            if key in truth:
                # the table is LPS millimetres; predict_prepared.py gives RAS, so it flips
                marks = next(iter(entry["record"]["landmarks"].values()), {})
                pred = coordinates(entry["record"], "native_lps_mm", "world_ras_mm")
                if "native_lps_mm" not in marks:
                    pred = pred * np.array([-1.0, -1.0, 1.0])
                e = radial_errors(pred, truth[key], spacing=(1.0, 1.0, 1.0))
                one["landmarks"] = landmark_summary(e)
                errors.append(e)
            else:
                print(f"  {stem}: no reference landmarks, skipped", flush=True)
        if a.protocol and entry["segmentation"] is not None:
            one["protocol"] = compliance(entry["segmentation"], entry["record"], spec)
        subjects[stem] = one

    # the cohort figures, computed once: printed, and written out, so that what a reader sees
    # on screen is what a script finds in the file rather than something it has to recompute
    overall: dict = {"n_subjects": len(subjects)}
    print(f"\n{len(subjects)} subject{'s' if len(subjects) != 1 else ''}", flush=True)

    seg = [s["segmentation"]["mean"] for s in subjects.values() if "segmentation" in s]
    if seg:
        overall["segmentation"] = {k: float(np.mean([m[k] for m in seg])) for k in seg[0]}
        print(
            f"  segmentation   DSC {100 * overall['segmentation']['dice']:.2f}"
            f"   ASSD {overall['segmentation']['assd']:.3f} mm"
            f"   HD95 {overall['segmentation']['hd95']:.3f} mm",
            flush=True,
        )
    if errors:
        overall["landmarks"] = landmark_summary(np.concatenate([e.ravel() for e in errors]))
        print(
            f"  landmarks      MRE {overall['landmarks']['mre']:.3f} mm"
            f"   SDR@2 {overall['landmarks']['sdr2']:.1f} %"
            f"   SDR@4 {overall['landmarks']['sdr4']:.1f} %",
            flush=True,
        )
    protocol = [s["protocol"]["violating_mm3"] for s in subjects.values() if "protocol" in s]
    if protocol:
        overall["protocol"] = {
            "violating_mm3": float(np.mean(protocol)),
            "planes": "the unshifted planes of the predicted landmarks",
        }
        print(
            f"  protocol       {overall['protocol']['violating_mm3']:.1f} mm3 outside the "
            f"admissible zones on average, against the unshifted planes of the predicted "
            f"landmarks",
            flush=True,
        )
    if not (seg or errors or protocol):
        print(
            "  nothing to score: give --segmentations, --landmarks or --protocol",
            flush=True,
        )

    if a.out is not None:
        save_json(
            a.out, {"predictions": str(a.predictions), "overall": overall, "subjects": subjects}
        )
        print(f"\nwritten to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
