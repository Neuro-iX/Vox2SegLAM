"""The geometry the enforcement runs on, and the landmark file Slicer opens."""

import numpy as np
import SimpleITK as sitk

from vox2seglam.inference.scan import _las_axes, planes_in_scan_sections
from vox2seglam.inference.slicer import markups_document
from vox2seglam.postprocess import PlaneShift


def _grids(native_spacing=(0.9, 0.9, 1.1), tilt_deg=0.0):
    reference = sitk.Image(24, 28, 24, sitk.sitkFloat32)
    reference.SetSpacing((0.7, 0.7, 0.7))
    reference.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    native = sitk.Image(40, 40, 40, sitk.sitkFloat32)
    native.SetSpacing(native_spacing)
    t = np.deg2rad(tilt_deg)
    native.SetDirection((np.cos(t), -np.sin(t), 0.0, np.sin(t), np.cos(t), 0.0, 0.0, 0.0, 1.0))
    return reference, native


def _coords():
    coords = np.stack([np.linspace(8.0, 18.0, 16)] * 3, axis=1)
    coords[:, 2] = np.linspace(6.0, 20.0, 16)  # so no two-landmark line is degenerate
    return coords


def test_the_axes_are_read_correctly_however_the_volume_was_acquired():
    """_las_axes says which array axis runs superior, anterior and left.

    A scan's axes come in whatever order and sign the acquisition left them, and obliquely:
    19 of the 25 out-of-domain volumes by more than a degree, one by twenty. The assignment is
    exact while the obliquity stays under 45 degrees.
    """

    def axes_of(direction):
        image = sitk.Image(4, 4, 4, sitk.sitkFloat32)
        image.SetDirection(direction)
        return _las_axes(image)

    order, sign = axes_of((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # LPS, identity
    assert order == [2, 1, 0]  # superior -> z, anterior -> y, left -> x
    assert list(sign) == [1.0, -1.0, 1.0]  # anterior is -y in LPS

    order, _ = axes_of((0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0))  # permuted, flipped
    assert sorted(order) == [0, 1, 2] and len(set(order)) == 3

    t = np.deg2rad(30.0)  # well inside the 45-degree limit
    order, _ = axes_of((np.cos(t), -np.sin(t), 0.0, np.sin(t), np.cos(t), 0.0, 0.0, 0.0, 1.0))
    assert order == [2, 1, 0], "a 30 degree tilt must not change which axis is which"


def test_a_section_plane_is_perpendicular_to_an_axis_of_the_volume():
    """Why no section of the volume is ever split, for any rule reading such a plane.

    Along a grid axis every voxel of a section shares one index, so a plane perpendicular to
    that axis puts the whole section on one side wherever it falls. Built in the physical axes
    instead, the same plane is a few degrees off on an obliquely acquired volume and does cut
    sections.
    """
    spec = PlaneShift(shift_range={}).spec
    reference, native = _grids(tilt_deg=11.0)
    planes = planes_in_scan_sections(spec, _coords(), reference, native, sitk.Euler3DTransform())
    step = np.asarray(native.GetDirection()).reshape(3, 3) @ np.diag(native.GetSpacing())
    step = step / np.linalg.norm(step, axis=0)
    single = {p.plane_id for p in spec.planes if len(p.landmark_ids) == 1}

    aligned = {pid for pid, (_, n) in planes.items() if (np.abs(step.T @ n) > 0.999).any()}
    assert aligned, "no plane came out perpendicular to an axis of the volume"
    assert aligned <= single, "a plane fitted through two landmarks cannot be axis-aligned"


def test_a_fitted_plane_follows_the_anatomy_whatever_the_voxels_measure():
    """The five two-landmark planes are built in millimetres, not in voxel indices.

    "A plane that follows the line joining them" is a statement about an anatomical direction,
    and an index space with anisotropic spacing is not a similarity of the physical one:
    fitting there tilted lv_body_floor by 10 degrees on 0.9 x 0.9 x 1.1 mm voxels and by 29 on
    0.49 x 0.49 x 1.0.
    """
    spec = PlaneShift(shift_range={}).spec
    fitted = [p.plane_id for p in spec.planes if len(p.landmark_ids) == 2]
    coords, transform = _coords(), sitk.Euler3DTransform()

    isotropic = planes_in_scan_sections(spec, coords, *_grids((1.0, 1.0, 1.0)), transform)
    for spacing in ((0.4883, 0.4883, 1.0), (0.9, 0.9, 1.1)):
        other = planes_in_scan_sections(spec, coords, *_grids(spacing), transform)
        for pid in fitted:
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        abs(float(np.asarray(other[pid][1]) @ np.asarray(isotropic[pid][1]))), 0, 1
                    )
                )
            )
            assert angle < 0.1, (pid, spacing, angle)


def test_a_plane_perpendicular_to_an_axis_cannot_split_a_section():
    """The property the rest rests on, in the small: offset is irrelevant, the axis is not."""
    shape = (9, 11, 7)
    index = np.indices(shape).astype(float)

    def split(above, axis):
        return sum(
            1
            for s in range(shape[axis])
            if np.take(above, s, axis=axis).any() and not np.take(above, s, axis=axis).all()
        )

    for axis in range(3):
        for offset in (2.0, 2.25, 2.5, 3.75):  # on a centre, inside, on a boundary
            assert split(index[axis] > offset, axis) == 0, (axis, offset)

    tilt = np.deg2rad(5.0)
    oblique = index[1] * np.cos(tilt) + index[2] * np.sin(tilt) > 5.5
    assert split(oblique, 1) > 0, "a tilted plane must cut at least one section"


def test_both_entry_points_write_the_same_kind_of_landmark_file():
    """predict.py returns the scan's own millimetres, predict_prepared.py the grid's.

    They name those coordinates differently -- native_ras_mm and world_ras_mm -- and for a
    while only one of the two wrote a file Slicer would open.
    """
    external = {"1": {"native_ras_mm": [1.0, 2.0, 3.0], "native_lps_mm": [-1.0, -2.0, 3.0]}}
    internal = {"1": {"world_ras_mm": [1.0, 2.0, 3.0], "voxel_zyx": [10.0, 11.0, 12.0]}}
    for landmarks in (external, internal):
        doc = markups_document(landmarks, name="s")
        assert doc is not None and doc["markups"][0]["coordinateSystem"] == "RAS"
        assert doc["markups"][0]["controlPoints"][0]["position"] == [1.0, 2.0, 3.0]

    doc = markups_document(external, name="s", position_key="native_lps_mm")
    assert doc["markups"][0]["controlPoints"][0]["position"] == [-1.0, -2.0, 3.0]

    assert markups_document({"1": {"voxel_zyx": [1.0, 2.0, 3.0]}}, name="s") is None


def test_the_landmark_file_is_a_markups_list_carrying_its_record():
    """Slicer decides what a .json is by reading it and refuses one without a markups key."""
    from vox2seglam.protocol import landmark_names

    landmarks = {str(i): {"native_ras_mm": [float(i), 0.0, 0.0]} for i in range(1, 17)}
    doc = markups_document(
        landmarks, name="sub-01", record={"producer": "predict.py"}, names=landmark_names(16, 17)
    )
    assert "markups-schema" in doc["@schema"] and len(doc["markups"]) == 1
    points = doc["markups"][0]["controlPoints"]
    assert [p["label"] for p in points] == [f"LM{i}" for i in range(1, 17)]
    assert points[0]["description"] == "Left Putamen Anterior Start"
    assert all(len(p["orientation"]) == 9 for p in points)
    assert doc["vox2seglam"]["producer"] == "predict.py"
    assert doc["vox2seglam"]["landmarks"] == landmarks


def test_another_protocol_s_landmarks_are_not_given_these_names():
    """The AFIDs model predicts 32 landmarks whose ids 1..16 collide with the HOA protocol's
    and mean something else. Labelling them id by id from the protocol's vocabulary described
    the wrong anatomy on half the points of every AFIDs prediction, in the file a viewer opens.
    """
    from vox2seglam.protocol import landmark_names

    assert landmark_names(32, 40) is None
    assert landmark_names(16, 17) is not None

    afids = {str(i): {"native_ras_mm": [float(i), 1.0, 2.0]} for i in range(1, 33)}
    bare = markups_document(afids, name="s", names=landmark_names(32, 40))
    assert [p["description"] for p in bare["markups"][0]["controlPoints"]] == [""] * 32
    assert [p["label"] for p in bare["markups"][0]["controlPoints"][:2]] == ["LM1", "LM2"]

    hoa = {str(i): {"native_ras_mm": [float(i), 1.0, 2.0]} for i in range(1, 17)}
    named = markups_document(hoa, name="s", names=landmark_names(16, 17))
    points = named["markups"][0]["controlPoints"]
    assert points[0]["description"] == "Left Putamen Anterior Start"
    assert all(p["description"] for p in points)
