"""Exports landmark label volumes as one coordinate table, the form the tools read."""

import argparse
import csv
from pathlib import Path

import numpy as np

from vox2seglam.data.landmarks import as_integer_labels, extract_landmarks
from vox2seglam.data.prepare.register import nifti_stem
from vox2seglam.protocol.constants import LANDMARK_NAMES
from vox2seglam.utils.io import load_nifti, to_zyx

COLUMNS = [
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


def coordinates(path: Path, n_landmarks: int) -> tuple[np.ndarray, np.ndarray]:
    """(K, 3) ZYX voxels and the volume's voxel-to-world affine."""
    array, affine = load_nifti(path)
    volume = as_integer_labels(np.rint(to_zyx(array)))
    return extract_landmarks(volume, list(range(1, n_landmarks + 1))), affine


def rows(volumes: dict[str, Path], n_landmarks: int, system: str):
    for subject, path in volumes.items():
        zyx, affine = coordinates(path, n_landmarks)
        xyz = zyx[:, ::-1]
        world = xyz @ affine[:3, :3].T + affine[:3, 3]  # NIfTI affines are RAS+
        if system == "LPS":
            world = world * np.array([-1.0, -1.0, 1.0])
        for k in range(n_landmarks):
            yield {
                "participant_id": subject,
                "landmark_id": k + 1,
                "landmark_name": LANDMARK_NAMES.get(k + 1, ""),
                "coordinate_system": system,
                **{f"{a}_mm": round(float(world[k][i]), 4) for i, a in enumerate("xyz")},
                **{f"voxel_{a}": round(float(xyz[k][i]), 3) for i, a in enumerate("xyz")},
            }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="example:\n  %(prog)s --landmark-dir DATA/Landmarks_Volumes_16 \\\n"
        "      --suffix _landmark_16.nii.gz --landmarks 16 --out landmarks.csv\n\n"
        "One row per subject and landmark, with the columns read_landmark_csv reads\n"
        "(participant_id, landmark_id, x_mm, y_mm, z_mm, coordinate_system) plus the\n"
        "landmark's name and its voxel index. The coordinates are those of the volume\n"
        "the label was read from, so for volumes on the reference grid they are\n"
        "positions in that grid's own space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--landmark-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="label volumes, one per subject, voxel of landmark k holding k",
    )
    ap.add_argument(
        "--suffix",
        default="_landmark_16.nii.gz",
        metavar="SUFFIX",
        help="what follows the subject id in a file name (default: %(default)s)",
    )
    ap.add_argument(
        "--landmarks",
        type=int,
        default=16,
        metavar="K",
        help="number of landmarks per subject (default: %(default)s)",
    )
    ap.add_argument(
        "--ids",
        type=Path,
        nargs="*",
        default=[],
        metavar="FILE",
        help="split files; without them every volume of the directory is taken",
    )
    ap.add_argument(
        "--coordinate-system",
        choices=("LPS", "RAS"),
        default="LPS",
        help="what the millimetres are in (default: %(default)s)",
    )
    ap.add_argument("--out", required=True, type=Path, metavar="FILE")
    a = ap.parse_args()

    found = {
        nifti_stem(p).removesuffix(a.suffix.split(".")[0]): p
        for p in sorted(a.landmark_dir.glob(f"*{a.suffix}"))
    }
    if a.ids:
        wanted = [
            line.strip()
            for f in a.ids
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        missing = [i for i in wanted if i not in found]
        if missing:
            raise SystemExit(
                f"no landmark volume for {', '.join(missing[:5])}"
                + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            )
        found = {i: found[i] for i in wanted}
    if not found:
        raise SystemExit(f"no *{a.suffix} in {a.landmark_dir}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        written = 0
        for row in rows(found, a.landmarks, a.coordinate_system):
            writer.writerow(row)
            written += 1
    print(f"{len(found)} subjects, {written} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
