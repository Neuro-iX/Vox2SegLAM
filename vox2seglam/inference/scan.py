"""Prediction on a scan from another cohort, returned on the scan's own grid."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ..data.prepare.register import PreparedScan, labels_to_native, to_native
from ..utils.geometry import flip_lps_ras

__all__ = ["ScanPrediction", "predict_scan", "native_source"]


# LPS (x, y, z) millimetres <-> a LAS, (z, y, x)-ordered space. protocol_planes reads its
# coordinates the way the reference grid is indexed -- LAS, z first -- so axis 1 means
# anterior there; handing it anything else would build "coronal" around the wrong axis and
# silently invert an anterior/posterior rule. The map is a signed permutation, so it is its
# own inverse-transpose and a normal is reordered exactly like a point.
def _to_las_zyx(points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return np.stack([points[:, 2], -points[:, 1], points[:, 0]], axis=1)


def _to_lps_xyz(point: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(point, dtype=np.float64)
    return np.array([c, -b, a])


@dataclass
class ScanPrediction:
    landmarks: dict[str, dict]
    segmentation: sitk.Image | None  # on the scan's own grid, carrying its geometry
    plane_shifts: dict[str, float] | None = None
    precision: str = "float32"
    protocol_check: dict | None = None  # the rules, verified on the scan's own grid
    on_reference: sitk.Image | None = None  # the same class map on the reference grid


def native_source(scan: PreparedScan) -> Path:
    """The scan this preparation was made from, as recorded when it was prepared."""
    try:
        source = Path(json.loads(scan.record.read_text(encoding="utf-8"))["source"]["path"])
    except (OSError, KeyError, json.JSONDecodeError) as err:
        raise FileNotFoundError(
            f"{scan.record.name} does not say which scan {scan.stem} was prepared from, so "
            "its segmentation cannot be brought back onto it. Prepare it again with "
            "predict.py --input or prepare.py."
        ) from err
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} no longer exists; the segmentation of {scan.stem} is mapped back onto "
            "that scan, so it has to be where it was when it was prepared."
        )
    return source


#: what enforcing on the scan's own grid needs of a post-processor. `postprocess` is
#: documented as a fitted PlaneShift but is only ever called, so a plain callable keeps
#: working: it simply gets the enforcement on the reference grid and nothing more.
_GEOMETRY = ("spec", "tolerance", "violations")


def _carries_geometry(postprocess) -> bool:
    return all(hasattr(postprocess, name) for name in _GEOMETRY)


def planes_in_scan_frame(
    spec,
    coords_zyx,
    reference: sitk.Image,
    transform: sitk.Transform,
    shifts: dict[str, float] | None = None,
) -> dict[str, tuple]:
    """The protocol's planes built in the scan's own axes, in its millimetres (LPS).

    The counterpart of planes_in_native, which builds them where HOA2.0 was annotated and
    carries them over. Here a "coronal" plane is perpendicular to *this scan's*
    anterior-posterior axis, so the scan's own coronal sections are parallel to it and a
    section holds one structure or the other, never both -- the protocol read in the frame the
    volume arrived in.

    The price is that "coronal" then depends on how the head lay in the scanner: two
    acquisitions of one subject, tilted differently, obey two different protocols. The five
    planes fitted through two landmarks (put_nacc_border, lv_body_floor, brainstem_top) are
    oblique in any frame, so this changes nothing for them.

    The shifts are the ones fitted on the reference grid, in reference voxels. They are
    applied here as the same displacement in millimetres along this frame's normal, which is
    an approximation -- the two normals differ by the registration's rotation.
    """
    from ..protocol.geometry import protocol_planes

    native_mm = to_native(coords_zyx, reference, transform)
    planes = protocol_planes(spec, _to_las_zyx(native_mm))
    scale = float(np.asarray(reference.GetSpacing(), dtype=float).mean())
    out = {}
    for pid, (centre, normal) in planes.items():
        centre = np.asarray(centre) + float((shifts or {}).get(pid, 0.0)) * scale * np.asarray(
            normal
        )
        out[pid] = (_to_lps_xyz(centre), _to_lps_xyz(normal))
    return out


def _las_axes(image: sitk.Image) -> tuple[list[int], np.ndarray]:
    """Which array axis of `image` runs superior, anterior and left, and with which sign.

    protocol_planes reads its coordinates LAS and z first, as the reference grid is indexed.
    A scan's array axes are in whatever order and sign the acquisition left them, and on a
    real cohort they are also oblique -- 19 of the 25 out-of-domain volumes by more than a
    degree, one by twenty. Each LAS direction is therefore matched to the array axis most
    aligned with it, which is exact as long as the obliquity stays under 45 degrees.
    """
    # columns of (direction x spacing) are the physical directions of the array axes, in LPS
    step = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3) @ np.diag(
        np.asarray(image.GetSpacing(), dtype=float)
    )
    step = step / np.linalg.norm(step, axis=0)
    want = np.array(
        [
            [0.0, 0.0, 1.0],  # superior: +z in LPS
            [0.0, -1.0, 0.0],  # anterior: -y in LPS
            [1.0, 0.0, 0.0],
        ]
    )  # left:     +x in LPS
    order, sign = [], []
    for direction in want:
        dots = step.T @ direction
        axis = int(np.argmax(np.abs(dots)))
        order.append(axis)
        sign.append(1.0 if dots[axis] > 0 else -1.0)
    if len(set(order)) != 3:
        raise ValueError(
            f"cannot tell this volume's axes apart from its direction matrix:\n"
            f"{np.asarray(image.GetDirection()).reshape(3, 3)}"
        )
    return order, np.asarray(sign)


def planes_in_scan_sections(
    spec,
    coords_zyx,
    reference: sitk.Image,
    native: sitk.Image,
    transform: sitk.Transform,
    shifts=None,
) -> dict[str, tuple]:
    """The protocol's planes on the scan's own sections, placed *between* two of them.

    This is the protocol read literally: "the posterior ventral diencephalon starts on the
    section after the posterior limit of the mammillary body".

    What keeps a section whole is the plane's *axis*: built here, a single-landmark plane is
    perpendicular to one of the array's axes, and every voxel of a section perpendicular to
    that axis shares one index, so the whole section falls on one side wherever the plane
    lands. Compliance becomes an integer comparison, for every rule reading that plane, with
    no geometry left to interpret. Built in the physical axes instead (planes_in_scan_frame),
    the same plane is a few degrees off the array's axis on an obliquely acquired volume and
    does cut sections.

    The five planes fitted through two landmarks are built in millimetres, not in indices, and
    are left where they are. "A plane that follows the line joining them" is a statement about an
    anatomical direction, and an index space with anisotropic spacing is not a similarity of
    the physical one: fitting lv_body_floor there tilts it by 10 degrees on 0.9 x 0.9 x 1.1 mm
    voxels and by 29 on 0.49 x 0.49 x 1.0. They were never section boundaries either, so
    nothing is lost by leaving them where the anatomy puts them.
    """
    from ..protocol.geometry import protocol_planes

    order, sign = _las_axes(native)
    native_mm = to_native(coords_zyx, reference, transform)
    index = np.array(
        [native.TransformPhysicalPointToContinuousIndex(tuple(q)) for q in native_mm]
    )  # (K, 3) as x, y, z
    # into a LAS, z-first space of array indices: axis 0 superior, 1 anterior, 2 left
    las = np.stack([sign[k] * index[:, order[k]] for k in range(3)], axis=1)
    on_sections = protocol_planes(spec, las)
    # the fitted planes come from the millimetres, where "the line joining them" means what it
    # says; only the single-landmark ones belong to a section
    in_mm = planes_in_scan_frame(spec, coords_zyx, reference, transform, shifts)

    single = {p.plane_id for p in spec.planes if len(p.landmark_ids) == 1}
    scale = float(np.asarray(reference.GetSpacing(), dtype=float).mean())
    out = {}
    for pid, (centre, normal) in on_sections.items():
        centre = np.asarray(centre, dtype=float)
        normal = np.asarray(normal, dtype=float)
        axis = int(np.argmax(np.abs(normal)))
        if not (pid in single and abs(normal[axis]) > 0.999):
            out[pid] = in_mm[pid]  # fitted through two landmarks: millimetres
            continue
        # back to the array's own (x, y, z) index space, then to millimetres. The normal is
        # along one axis, so the point map and the normal map agree up to scale there, which
        # they do not for an oblique one -- hence the millimetre route above.
        idx = np.empty(3)
        nrm = np.empty(3)
        for k in range(3):
            idx[order[k]] = sign[k] * centre[k]
            nrm[order[k]] = sign[k] * normal[k]
        origin = np.asarray(native.TransformContinuousIndexToPhysicalPoint(tuple(idx)))
        tip = np.asarray(native.TransformContinuousIndexToPhysicalPoint(tuple(idx + nrm)))
        direction = tip - origin
        direction = direction / np.linalg.norm(direction)
        # the fitted shift is reference voxels; in millimetres along the final normal it keeps
        # its meaning whatever this volume's own spacing is
        origin = origin + float((shifts or {}).get(pid, 0.0)) * scale * direction
        out[pid] = (origin, direction)
    return out


def _native_points(labels: sitk.Image, index) -> np.ndarray:
    """The physical position (LPS mm) of each voxel of `index`, on `labels`' own grid."""
    return np.array(
        [labels.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in zip(*index)],
        dtype=np.float64,
    )


def _tolerance_mm(postprocess, reference: sitk.Image) -> dict[str, float]:
    """The fitted per-plane tolerances, which are reference voxels, as millimetres."""
    scale = float(np.asarray(reference.GetSpacing(), dtype=float).mean())
    return {k: v * scale for k, v in postprocess.tolerance.items()}


def _report(postprocess, before, after, planes, points, tolerance, spacing) -> dict:
    """The rules re-checked on the grid that is written out.

    "enforced" is then a measurement of the file on disk rather than a claim about an earlier
    step. The voxel count is what the class map holds differently from a plain nearest
    neighbour resampling of the enforced labels: a decision taken here, not a repair.
    """
    voxel_mm3 = float(np.prod(spacing))
    was = postprocess.violations(before, points, planes, tolerance)
    now = postprocess.violations(after, points, planes, tolerance)
    return {
        "checked_on": "the scan's own grid",
        "n_rules": len(now),
        "voxels_differing_from_nearest_neighbour": int((after != before).sum()),
        "violating_before_mm3": round(sum(was.values()) * voxel_mm3, 3),
        "violating_after_mm3": round(sum(now.values()) * voxel_mm3, 3),
        "violating_voxels_before": {k: v for k, v in was.items() if v},
        "violating_voxels_after": {k: v for k, v in now.items() if v},
    }


def _enforce_from_probabilities(
    labels_ref,
    probabilities,
    postprocess,
    coords_zyx,
    reference: sitk.Image,
    native: sitk.Image,
    transform: sitk.Transform,
    shifts,
) -> tuple[sitk.Image, dict]:
    """The class map built on the scan's own grid from the probabilities, not from the labels.

    The other path resamples the *labels* with nearest neighbour and then repairs whatever
    crossed a plane. This one resamples the probabilities -- trilinearly, in the float32 the
    network produced -- and runs the enforcement on the scan's own voxels, so no label is ever
    carried by a nearest neighbour and the class each voxel takes is the most probable one its
    rules allow *there*.

    The shifts still come from the fit on the reference grid: the fitted ranges and tolerances
    are reference voxels, and refitting here would no longer be the published post-processing.

    What this does not buy is fidelity. A 0.7 mm grid holds about three times more samples
    than a 1 mm one, so a structure's volume still moves by a percent or two either way -- that
    is the sampling, not the resampling. Measured against the reference-grid result on a 1 mm
    scan, the two paths trade places from structure to structure.
    """
    from scipy.ndimage import binary_dilation, map_coordinates

    # nearest neighbour decides *where* to look, never what the label is: trilinear sampling
    # can reach one voxel further than it did, so the window is grown before being used
    nn = labels_to_native(labels_ref, reference, native, transform)
    window = binary_dilation(sitk.GetArrayViewFromImage(nn) > 0, iterations=3)
    index = np.nonzero(window)
    if not len(index[0]):
        return nn, {}

    points = _native_points(nn, index)
    inverse = transform.GetInverse()  # exact for a rigid transform
    at = np.array(
        [
            reference.TransformPhysicalPointToContinuousIndex(
                tuple(inverse.TransformPoint(tuple(q)))
            )
            for q in points
        ]
    )
    coordinates = np.vstack([at[:, 2], at[:, 1], at[:, 0]])  # x,y,z -> z,y,x
    # interpolated in the float the network produced, then put on the 0..255 scale _fit and
    # _constrained are written for -- quantising first and interpolating after is what cost
    # 1419 voxels to background in an earlier attempt
    sampled = (
        np.stack(
            [
                map_coordinates(channel, coordinates, order=1, mode="constant", cval=0.0)
                for channel in probabilities
            ]
        )
        * 255.0
    )

    # The planes go in unshifted: the shift is not a property of the reference grid, it is the
    # position that destroys the least of the network's confidence, and here the confidence is
    # the one resampled onto these voxels. Searching it again is what makes this the whole
    # post-processing rather than the enforcement half of it. `scaled` carries the fitted
    # interval, the step, the margin and the tolerances into millimetres so the search reads
    # them in the unit the geometry is in.
    nominal = planes_in_scan_sections(postprocess.spec, coords_zyx, reference, nn, transform)
    in_mm = postprocess.scaled(float(np.asarray(reference.GetSpacing(), dtype=float).mean()))
    labels_here = sitk.GetArrayViewFromImage(nn)[index]
    placed, native_shifts = in_mm._fit(sampled, labels_here, points, None, planes=nominal)
    taken = in_mm._constrained(sampled.astype(np.float32), points, placed, in_mm.spec.rules)
    planes, tolerance = placed, in_mm.tolerance

    array = np.zeros_like(sitk.GetArrayViewFromImage(nn))
    array[index] = taken.astype(array.dtype)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(nn)
    report = _report(
        postprocess, labels_here, array[index], planes, points, tolerance, nn.GetSpacing()
    )
    report["plane_shifts_mm"] = {k: round(float(v), 4) for k, v in native_shifts.items()}
    return image, report


def predict_scan(
    predictor,
    scan: PreparedScan,
    reference: sitk.Image,
    *,
    postprocess=None,
    segmentation: bool = False,
    on_reference: bool = False,
) -> ScanPrediction:
    """Landmarks (and, if asked, the segmentation) of one prepared scan, on its native grid.

    postprocess: a fitted PlaneShift. The rules are then enforced on the scan's own voxels --
    the probabilities resampled onto them, the planes built in the volume's own axes -- so
    that they hold of the map that is written, and the result says what it found there.
    """
    transform = sitk.ReadTransform(str(scan.transform))
    result = predictor.predict_file(scan.model_input, return_probabilities=postprocess is not None)

    labels, shifts, check = result.segmentation, None, None
    if postprocess is not None:
        if labels is None:
            raise ValueError("post-processing needs a checkpoint with a segmentation head")
        labels, shifts = postprocess(result.probabilities, result.coordinates, labels)

    native = to_native(result.coordinates, reference, transform)
    # Also in the reference grid's own millimetres: the prepared volume
    # (*_aligned_brain_n4.nii.gz) carries the reference's geometry, not the scan's, so a point
    # expressed in the scan's space cannot land on it. Both are kept, each labelled.
    landmarks = {}
    for k, lid in enumerate(result.landmark_ids):
        z, y, x = (float(v) for v in result.coordinates[k])
        grid = reference.TransformContinuousIndexToPhysicalPoint([x, y, z])
        landmarks[str(lid)] = {
            "reference_voxel_zyx": [z, y, x],
            "reference_ras_mm": [float(v) for v in flip_lps_ras(np.asarray(grid))],
            "native_lps_mm": [float(v) for v in native[k]],
            "native_ras_mm": [float(v) for v in flip_lps_ras(native[k])],
        }

    mapped = None
    if segmentation:
        if labels is None:
            raise ValueError(
                "this checkpoint has no segmentation head, so there is no class map to save"
            )
        native = sitk.ReadImage(str(native_source(scan)))
        if postprocess is None or not _carries_geometry(postprocess):
            mapped = labels_to_native(labels, reference, native, transform)
        else:
            # the class map is *built* on the scan's own grid from the probabilities, so no
            # label is carried there by a nearest neighbour at all
            mapped, check = _enforce_from_probabilities(
                labels,
                result.probabilities,
                postprocess,
                result.coordinates,
                reference,
                native,
                transform,
                shifts,
            )
    grid = None
    if on_reference and labels is not None:
        # the same class map where the protocol's planes are axis-aligned and the sampling is
        # 0.7 mm: the only space in which "section t is anterior, t+1 posterior" can be read
        grid = sitk.GetImageFromArray(np.ascontiguousarray(labels).astype(np.uint8))
        grid.CopyInformation(reference)
    return ScanPrediction(landmarks, mapped, shifts, result.precision, check, grid)
