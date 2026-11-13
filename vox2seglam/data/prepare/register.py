"""Rigid registration of a native scan to the reference grid, and the mapping of landmarks and labels through it."""

import gzip
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

__all__ = [
    "PreparedScan",
    "prepared_scans",
    "check_complete",
    "register_rigid",
    "to_reference_grid",
    "to_native",
    "labels_to_native",
    "labels_to_reference",
    "landmarks_to_reference",
]

TRANSFORM_SUFFIX = "_to_reference.tfm"


def nifti_stem(path: str | Path) -> str:
    """A volume's name without .nii or .nii.gz - what every file below is named after."""
    return Path(path).name.removesuffix(".gz").removesuffix(".nii")


# The files one prepared scan leaves on disk; defined once because the scripts that write
# and read them are different.
@dataclass(frozen=True)
class PreparedScan:
    """The files of one prepared scan, named after the native volume's stem."""

    stem: str
    aligned: Path  # on the reference grid
    transform: Path  # reference -> native
    brain: Path  # skull-stripped
    model_input: Path  # bias-corrected: what the checkpoint runs on
    record: Path  # provenance; written last, so its presence means the others are whole

    @classmethod
    def under(cls, out_dir: str | Path, native_path: str | Path) -> "PreparedScan":
        out_dir, stem = Path(out_dir), nifti_stem(native_path)
        return cls(
            stem,
            out_dir / f"{stem}_aligned.nii.gz",
            out_dir / f"{stem}{TRANSFORM_SUFFIX}",
            out_dir / f"{stem}_aligned_brain.nii.gz",
            out_dir / f"{stem}_aligned_brain_n4.nii.gz",
            out_dir / f"{stem}_prepared.json",
        )

    @property
    def complete(self) -> bool:
        # `aligned` and `brain` are steps on the way to `model_input` and are deleted once it
        # exists (three full-size volumes per scan, for one that is ever read again), so a
        # preparation is whole without them.
        return all(p.is_file() for p in (self.transform, self.model_input, self.record))


def prepared_scans(directory: str | Path) -> list[PreparedScan]:
    """Every prepared scan of a directory, found by its transform file."""
    directory = Path(directory)
    return [
        PreparedScan.under(directory, tfm.name.removesuffix(TRANSFORM_SUFFIX) + ".nii.gz")
        for tfm in sorted(directory.glob(f"*{TRANSFORM_SUFFIX}"))
    ]


def check_complete(path: str | Path) -> None:
    """Fail on a .gz volume that cannot be read through, which a NIfTI reader only notices
    when it reaches the voxels."""
    path = Path(path)
    if path.suffix != ".gz":
        return
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 24):
                pass
    except (EOFError, OSError, zlib.error) as err:  # zlib.error is not an OSError
        # Three things land here and gzip's own message tells them apart better than the type
        # does: BadGzipFile covers both a file that never was gzip and a damaged deflate
        # stream, while a truncated one raises EOFError. So the cause is quoted, not guessed.
        raise ValueError(
            f"{path}: this .gz could not be read to the end ({err}). A .nii.gz has to be a "
            "complete gzip file: check the download, or the file at its source."
        )


def register_rigid(reference: sitk.Image, moving: sitk.Image, seed: int = 42) -> sitk.Transform:
    """Rigid (Euler 3D) transform from the reference grid to the moving image."""
    fixed = sitk.Cast(reference, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20, seed=seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300, gradientMagnitudeTolerance=1e-6
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    return reg.Execute(fixed, moving)


def to_reference_grid(native_path: str | Path, reference_path: str | Path):
    """(volume resampled onto the reference grid, reference -> native transform)."""
    check_complete(native_path)
    reference = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    native = sitk.ReadImage(str(native_path), sitk.sitkFloat32)
    transform = register_rigid(reference, native)
    aligned = sitk.Resample(
        native, reference, transform, sitk.sitkBSpline, 0.0, native.GetPixelID()
    )
    return aligned, transform


def to_native(
    coords_zyx: np.ndarray, reference: sitk.Image, transform: sitk.Transform
) -> np.ndarray:
    """(K, 3) voxel coordinates on the reference grid -> native LPS millimetres."""
    out = []
    for z, y, x in np.asarray(coords_zyx, dtype=float):
        point = reference.TransformContinuousIndexToPhysicalPoint([x, y, z])
        out.append(transform.TransformPoint(point))
    return np.array(out)


# Nearest neighbour, so no label is invented. Resampling onto the native grid needs the
# native -> reference direction, hence the inverse (exact for a rigid transform). The result
# is an image carrying the native origin, spacing and orientation.
def labels_to_native(
    labels_zyx: np.ndarray, reference: sitk.Image, native: sitk.Image, transform: sitk.Transform
) -> sitk.Image:
    """(Z, Y, X) labels on the reference grid -> an image on the native volume's grid."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(labels_zyx).astype(np.uint8))
    image.CopyInformation(reference)
    return sitk.Resample(
        image, native, transform.GetInverse(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )


# Training-side counterparts: annotations made on the native scan travel to the reference
# grid through the registration, in the direction it was estimated.
def labels_to_reference(
    native_labels: sitk.Image, reference: sitk.Image, transform: sitk.Transform
) -> sitk.Image:
    """A native label map on the reference grid, nearest neighbour; label numbers are kept."""
    return sitk.Resample(
        native_labels, reference, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16
    )


def landmarks_to_reference(
    points_lps: np.ndarray, reference: sitk.Image, transform: sitk.Transform
) -> sitk.Image:
    """(K, 3) native LPS millimetres -> a label volume on the reference grid where the voxel
    nearest landmark k holds k.

    Raises if a landmark is missing (nan), outside the grid, or on another landmark's voxel.
    """
    points = np.asarray(points_lps, dtype=float)
    volume = np.zeros(sitk.GetArrayFromImage(reference).shape, np.int16)
    inverse = transform.GetInverse()
    size = reference.GetSize()
    taken: dict[tuple, int] = {}
    for k, point in enumerate(points, start=1):
        if np.isnan(point).any():
            raise ValueError(f"landmark {k} has no coordinates")
        x, y, z = (
            int(round(v))
            for v in reference.TransformPhysicalPointToContinuousIndex(
                inverse.TransformPoint([float(v) for v in point])
            )
        )
        if not (0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]):
            raise ValueError(
                f"landmark {k} falls outside the reference grid after registration "
                f"(voxel {(x, y, z)} of {tuple(size)}); the scan may be badly registered, "
                "or the coordinates may not be in the scan's own millimetres"
            )
        if (z, y, x) in taken:
            raise ValueError(
                f"landmarks {taken[(z, y, x)]} and {k} fall on the same voxel "
                f"{(z, y, x)} of the reference grid"
            )
        taken[(z, y, x)] = k
        volume[z, y, x] = k
    image = sitk.GetImageFromArray(volume)
    image.CopyInformation(reference)
    return image
