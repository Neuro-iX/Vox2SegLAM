"""Bringing a prediction back onto the scan it was made for."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from vox2seglam.data.prepare.register import PreparedScan
from vox2seglam.inference.scan import native_source, predict_scan

REF_SIZE = (24, 28, 24)  # SimpleITK order: X, Y, Z
LANDMARK_ZYX = (12.0, 14.0, 12.0)  # a voxel of the reference grid
LABEL = 5


@pytest.fixture
def geometry(tmp_path):
    reference = sitk.Image(*REF_SIZE, sitk.sitkFloat32)
    reference.SetSpacing((0.7, 0.7, 0.7))

    transform = sitk.Euler3DTransform()  # reference -> native, as registration gives
    transform.SetRotation(0.0, 0.0, np.deg2rad(5.0))
    transform.SetTranslation((2.0, -1.0, 3.0))

    # a native grid of another spacing, positioned so the reference lands inside it
    centre = reference.TransformContinuousIndexToPhysicalPoint([s / 2 for s in REF_SIZE])
    landing = transform.TransformPoint(centre)
    native = sitk.Image(40, 40, 40, sitk.sitkFloat32)
    native.SetSpacing((0.9, 0.9, 0.9))
    native.SetOrigin(tuple(c - 0.9 * 20 for c in landing))

    source = tmp_path / "sub-01_T1w.nii.gz"
    sitk.WriteImage(native, str(source))
    scan = PreparedScan.under(tmp_path / "prepared", source)
    scan.transform.parent.mkdir()
    sitk.WriteTransform(transform, str(scan.transform))
    scan.record.write_text(json.dumps({"source": {"path": str(source)}}), encoding="utf-8")
    return reference, native, scan, source


def blob_around(landmark_zyx, radius=4):
    labels = np.zeros(REF_SIZE[::-1], np.uint8)
    z, y, x = np.indices(labels.shape)
    labels[
        (z - landmark_zyx[0]) ** 2 + (y - landmark_zyx[1]) ** 2 + (x - landmark_zyx[2]) ** 2
        <= radius**2
    ] = LABEL
    return labels


class StubPredictor:
    """Stands in for a trained model: the geometry is under test, not the network."""

    def __init__(self, labels=None, probabilities=None):
        self.labels, self.probabilities = labels, probabilities
        self.asked_for_probabilities = None

    def predict_file(self, path, return_probabilities=False):
        self.asked_for_probabilities = return_probabilities
        return SimpleNamespace(
            coordinates=np.array([LANDMARK_ZYX, [6.0, 8.0, 18.0]], np.float32),
            landmark_ids=[1, 2],
            segmentation=self.labels,
            precision="float32",
            probabilities=self.probabilities if return_probabilities else None,
        )


def centroid_mm(image: sitk.Image, label: int) -> np.ndarray:
    """Physical position (LPS mm) of the centre of mass of a label."""
    voxels = np.argwhere(sitk.GetArrayFromImage(image) == label)  # (n, 3) as z, y, x
    points = [image.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) for z, y, x in voxels]
    return np.mean(points, axis=0)


def test_a_landmark_sits_on_the_structure_whose_label_surrounds_it(geometry):
    """The two outputs agree about where things are, in the scan's own space."""
    reference, native, scan, _ = geometry
    out = predict_scan(StubPredictor(blob_around(LANDMARK_ZYX)), scan, reference, segmentation=True)
    landmark_mm = np.array(out.landmarks["1"]["native_lps_mm"])
    assert np.linalg.norm(centroid_mm(out.segmentation, LABEL) - landmark_mm) < 0.5


def test_the_segmentation_comes_back_on_the_scans_own_grid_with_its_geometry(geometry):
    """An array without origin, spacing and orientation cannot be overlaid on its scan."""
    reference, native, scan, _ = geometry
    seg = predict_scan(
        StubPredictor(blob_around(LANDMARK_ZYX)), scan, reference, segmentation=True
    ).segmentation
    assert seg.GetSize() == native.GetSize()
    # NIfTI stores geometry as float32, so the scan read back from disk is 0.9 to 1e-7
    assert np.allclose(seg.GetSpacing(), native.GetSpacing(), atol=1e-6)
    assert np.allclose(seg.GetOrigin(), native.GetOrigin(), atol=1e-4)
    assert np.allclose(seg.GetDirection(), native.GetDirection(), atol=1e-6)


def test_no_label_is_invented_on_the_way_back(geometry):
    reference, _, scan, _ = geometry
    labels = blob_around(LANDMARK_ZYX)
    labels[2:5, 2:5, 2:5] = 9
    seg = predict_scan(StubPredictor(labels), scan, reference, segmentation=True).segmentation
    assert set(np.unique(sitk.GetArrayFromImage(seg))) <= {0, LABEL, 9}


def test_landmarks_come_in_both_conventions_and_they_mirror_each_other(geometry):
    reference, _, scan, _ = geometry
    landmark = predict_scan(StubPredictor(), scan, reference).landmarks["1"]
    lps, ras = np.array(landmark["native_lps_mm"]), np.array(landmark["native_ras_mm"])
    assert np.allclose(ras, lps * [-1, -1, 1])
    assert landmark["reference_voxel_zyx"] == list(LANDMARK_ZYX)


def test_landmarks_alone_need_neither_the_segmentation_nor_the_source_scan(geometry):
    reference, _, scan, source = geometry
    source.unlink()  # the original is gone
    out = predict_scan(StubPredictor(None), scan, reference)  # no head, none asked for
    assert out.segmentation is None and set(out.landmarks) == {"1", "2"}


def test_asking_for_a_segmentation_the_model_does_not_make_is_refused(geometry):
    reference, _, scan, _ = geometry
    with pytest.raises(ValueError, match="no segmentation head"):
        predict_scan(StubPredictor(None), scan, reference, segmentation=True)


def test_postprocessing_is_applied_before_the_map_goes_back(geometry):
    """It is fitted on the reference grid and must be applied there, as predict.py does."""
    reference, _, scan, _ = geometry
    replaced = np.zeros(REF_SIZE[::-1], np.uint8)
    replaced[10:14, 12:16, 10:14] = 7
    seen = {}

    def postprocess(probabilities, coordinates, labels):
        seen["probabilities"], seen["labels_shape"] = probabilities, labels.shape
        return replaced, {"coronal_plane": 0.5}

    predictor = StubPredictor(blob_around(LANDMARK_ZYX), probabilities="p")
    out = predict_scan(predictor, scan, reference, postprocess=postprocess, segmentation=True)
    assert predictor.asked_for_probabilities is True  # it needs them
    assert seen["probabilities"] == "p" and seen["labels_shape"] == REF_SIZE[::-1]
    assert out.plane_shifts == {"coronal_plane": 0.5}
    assert set(np.unique(sitk.GetArrayFromImage(out.segmentation))) <= {0, 7}


def test_probabilities_are_not_requested_unless_post_processing_needs_them(geometry):
    reference, _, scan, _ = geometry
    predictor = StubPredictor(blob_around(LANDMARK_ZYX))
    predict_scan(predictor, scan, reference, segmentation=True)
    assert predictor.asked_for_probabilities is False  # 1.2 GB it did not have to build


def test_the_source_scan_must_still_be_where_it_was(geometry):
    _, _, scan, source = geometry
    assert native_source(scan) == source
    source.unlink()
    with pytest.raises(FileNotFoundError, match="no longer exists"):
        native_source(scan)


def test_a_record_that_does_not_name_its_source_says_what_to_do(geometry):
    _, _, scan, _ = geometry
    scan.record.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Prepare it again"):
        native_source(scan)


def test_the_rules_are_re_checked_on_the_scans_own_grid(geometry):
    """The file that is written is the one the rules hold of.

    They are enforced on the reference grid first, but the class map is built on the scan's
    own voxels from the probabilities resampled onto them -- so the enforcement happens there
    too, and the result reports a measurement of the output rather than a claim about an
    earlier step.
    """
    reference, _, scan, _ = geometry
    from vox2seglam.postprocess import PlaneShift

    labels = np.zeros(REF_SIZE[::-1], np.uint8)
    labels[10:16, 10:16, 10:16] = 6  # a block for a rule to bite on
    probabilities = np.zeros((17, *REF_SIZE[::-1]), np.float32)
    probabilities[0] = 1.0  # background everywhere...
    probabilities[6][labels == 6] = 1.0  # ...except where the block is
    probabilities[0][labels == 6] = 0.0

    class Sixteen(StubPredictor):
        """The protocol reads 16 landmarks; the toy predictor above returns two."""

        def predict_file(self, path, return_probabilities=False):
            result = super().predict_file(path, return_probabilities)
            spread = np.linspace(8.0, 18.0, 16)
            result.coordinates = np.stack([spread, spread[::-1], spread], axis=1)
            result.landmark_ids = list(range(1, 17))
            result.probabilities = probabilities
            return result

    class Stub(PlaneShift):
        """A real PlaneShift -- spec, tolerance and the enforcement -- fed fixed labels."""

        def __call__(self, probabilities, landmarks, segmentation=None):
            return labels, {p.plane_id: 0.0 for p in self.spec.planes}

    out = predict_scan(
        Sixteen(labels, probabilities=probabilities),
        scan,
        reference,
        postprocess=Stub(shift_range={}),
        segmentation=True,
    )
    check = out.protocol_check
    assert check is not None and check["checked_on"] == "the scan's own grid"
    assert check["n_rules"] == 16
    # the point of the pass: nothing is left on a forbidden side
    assert check["violating_voxels_after"] == {} and check["violating_after_mm3"] == 0.0
    assert check["voxels_differing_from_nearest_neighbour"] >= 0


def test_the_reference_grid_output_carries_the_reference_geometry(geometry):
    """--save-reference-grid: the one space where the protocol's planes are axis-aligned."""
    reference, _, scan, _ = geometry
    out = predict_scan(
        StubPredictor(blob_around(LANDMARK_ZYX)),
        scan,
        reference,
        segmentation=True,
        on_reference=True,
    )
    grid = out.on_reference
    assert grid is not None and grid.GetSize() == reference.GetSize()
    assert np.allclose(grid.GetSpacing(), reference.GetSpacing())
    assert np.allclose(grid.GetOrigin(), reference.GetOrigin())
    assert np.allclose(grid.GetDirection(), reference.GetDirection())
    # it is the class map as the network made it, not a resampling of the native one
    assert np.array_equal(sitk.GetArrayFromImage(grid), blob_around(LANDMARK_ZYX))

    assert (
        predict_scan(
            StubPredictor(blob_around(LANDMARK_ZYX)), scan, reference, segmentation=True
        ).on_reference
        is None
    )
