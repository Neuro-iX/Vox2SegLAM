"""The plane-shift post-processing enforces the protocol, up to the per-plane tolerances."""

import numpy as np
import pytest

from vox2seglam.postprocess import (
    SECTION_PLANES,
    PlaneShift,
    gt_tolerances,
    plane_errors,
    shift_ranges,
)
from vox2seglam.protocol import protocol_planes, rule_depth

SHAPE = (40, 40, 40)


def random_probabilities(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(17, *SHAPE)).astype(np.float32)
    prob = np.exp(logits) / np.exp(logits).sum(0)
    return prob


def violations(seg, spec, planes, tolerance=None) -> int:
    n = 0
    for rule in spec.rules:
        vox = np.argwhere(seg == rule.region_class).astype(float)
        if len(vox):
            n += int((rule_depth(vox, planes, rule, tolerance) > 1e-9).sum())
    return n


@pytest.mark.parametrize("tolerance", [{}, {"midline_ac": 1.0, "put_nacc_border_L": 1.0}])
def test_output_complies_with_the_shifted_planes(spec, landmarks16, tolerance):
    prob = random_probabilities()
    raw = prob.argmax(0)
    ranges = {p.plane_id: (-1.0, 1.0) for p in spec.planes}
    seg, shifts = PlaneShift(ranges, tolerance)(prob, landmarks16, raw)
    assert violations(raw, spec, protocol_planes(spec, landmarks16)) > 0
    planes = protocol_planes(spec, landmarks16)
    planes = {k: (c + shifts.get(k, 0.0) * n, n) for k, (c, n) in planes.items()}
    assert violations(seg, spec, planes, tolerance) == 0
    assert all(-1.0 <= s <= 1.0 for s in shifts.values())


def test_no_range_means_no_shift(spec, landmarks16):
    _, shifts = PlaneShift({})(random_probabilities(1), landmarks16)
    assert set(shifts.values()) == {0.0}


def test_save_and_load(tmp_path):
    pp = PlaneShift({"midline_ac": (-0.5, 0.7)}, {"midline_ac": 1.0}, fitted_on={"split": "val"})
    pp.save(tmp_path / "pp.json")
    back = PlaneShift.load(tmp_path / "pp.json")
    assert back.to_dict() == pp.to_dict()


def test_shift_ranges_are_percentiles_of_the_plane_errors(spec, landmarks16):
    pairs = [(landmarks16 + [0, 0, d], landmarks16) for d in np.linspace(-1, 1, 11)]
    errors = plane_errors(spec, pairs)
    low, high = shift_ranges(errors, (10, 90))["midline_ac"]
    assert np.isclose(low, -0.8) and np.isclose(high, 0.8)


def test_section_planes_are_coronal_and_land_on_a_section(spec, landmarks16):
    coronal = {p.plane_id for p in spec.planes if p.plane_type == 1 and len(p.landmark_ids) == 1}
    assert set(SECTION_PLANES) <= coronal
    ranges = {p.plane_id: (-1.0, 1.0) for p in spec.planes}
    off_grid = landmarks16 + 0.3  # landmarks between voxel centres
    seg, shifts = PlaneShift(ranges, section_planes=SECTION_PLANES)(
        random_probabilities(2), off_grid
    )
    planes = protocol_planes(spec, off_grid)
    planes = {k: (c + shifts.get(k, 0.0) * n, n) for k, (c, n) in planes.items()}
    for pid in SECTION_PLANES:  # the shift is from the predicted plane
        centroid, normal = planes[pid]
        axis = int(np.argmax(np.abs(normal)))
        assert np.isclose(centroid[axis], np.round(centroid[axis]))
    assert violations(seg, spec, planes) == 0


def test_tolerance_goes_to_planes_the_ground_truth_crosses(spec, landmarks16):
    seg = np.zeros(SHAPE, np.int64)
    x0 = int(landmarks16[9, 2])  # mid-sagittal plane through LM10
    seg[18:22, 18:22, x0 - 1 : x0 + 4] = 9  # TH_L, one slice past the midline
    tol, counts, n = gt_tolerances(spec, [(seg, landmarks16)] * 3 + [(seg * 0, landmarks16)])
    assert counts == {"midline_ac": 3} and n == 4 and tol == {"midline_ac": 1.0}


def test_confident_forbidden_voxel_takes_the_clause_target(spec, landmarks16):
    """A voxel the network gives 100 % to VDC_A_L, behind the left mammillary plane, must become
    VDC_P_L (the PQL target of that clause), not background by an argmax tie at 0."""
    planes = protocol_planes(spec, landmarks16)
    cls = {r.region_name: r.region_class for r in spec.rules}
    c, n = planes["mammillary_L"]
    left = planes["midline_ac"][0][2] + 3.0  # left of the midline
    behind = c + 2.0 * n  # 2 voxels posterior to the plane
    vox = np.array([[behind[0], behind[1], left]])
    rule = next(r for r in spec.rules if r.region_name == "VDC_P_L")
    assert rule_depth(vox, planes, rule)[0] <= 0  # VDC_P_L is allowed there
    pp = PlaneShift({})
    prob = np.zeros((17, 1), np.float32)
    prob[cls["VDC_A_L"]] = 1.0
    out = pp._constrained(prob, vox, planes, spec.rules)
    assert out[0] == cls["VDC_P_L"]


def test_fit_plane_shift_matches_the_step_by_step_fit(landmarks16):
    """One call must produce what plane_errors + shift_ranges + PlaneShift did by hand; both
    scripts go through it, so it carries the published fitting procedure."""
    from vox2seglam.postprocess import fit_plane_shift, plane_errors, shift_ranges

    rng = np.random.default_rng(0)
    pairs = [(landmarks16 + rng.normal(0, 0.4, landmarks16.shape), landmarks16) for _ in range(12)]

    by_hand = shift_ranges(plane_errors(PlaneShift({}).spec, iter(pairs)), (25.0, 75.0))
    fitted = fit_plane_shift(
        iter(pairs),
        percentiles=(25.0, 75.0),
        provenance={"shift_range": {"split": "val_ids.txt", "subjects": len(pairs)}},
    )

    assert fitted.shift_range == by_hand
    assert len(fitted.section_planes) == 7  # the coronal section planes of the paper
    assert fitted.tolerance == {}  # exact protocol unless asked otherwise
    recorded = fitted.to_dict()["fitted_on"]["shift_range"]
    assert recorded == {"split": "val_ids.txt", "subjects": 12, "percentiles": [25.0, 75.0]}


def test_fit_plane_shift_refuses_an_empty_validation_split():
    """Fitting on nothing must raise, not write a postprocess.json with no range."""
    import pytest

    from vox2seglam.postprocess import fit_plane_shift

    with pytest.raises(ValueError, match="nothing to fit"):
        fit_plane_shift(iter([]))


def test_scaled_reads_the_same_interval_in_another_unit():
    """The native search works in millimetres; the fit is in reference voxels.

    `shift_range`, `step`, `margin` and the tolerances are all lengths on the grid the ranges
    were fitted on. Searching on a scan's own voxels means expressing the geometry in
    millimetres, so they have to come along -- and the search must then find the same shift,
    scaled. Nothing is refitted.
    """
    import numpy as np

    from vox2seglam.postprocess import PlaneShift

    fit = PlaneShift(shift_range={"a": (-2.0, 3.0)}, tolerance={"a": 1.0}, step=0.1, margin=2.0)
    mm = fit.scaled(0.7)
    assert np.allclose(mm.shift_range["a"], (-1.4, 2.1))
    assert np.isclose(mm.tolerance["a"], 0.7)
    assert np.isclose(mm.step, 0.07) and np.isclose(mm.margin, 1.4)
    assert mm.fitted_on["scaled_by"] == 0.7
    # the protocol itself is untouched: same planes, same rules
    assert [p.plane_id for p in mm.spec.planes] == [p.plane_id for p in fit.spec.planes]
    assert fit.shift_range["a"] == (-2.0, 3.0), "the original must not be modified"


def test_strict_drops_only_the_tolerances():
    from vox2seglam.postprocess import PlaneShift

    fit = PlaneShift(shift_range={"a": (-2.0, 3.0)}, tolerance={"a": 1.0}, step=0.1)
    bare = fit.strict()
    assert bare.tolerance == {} and bare.shift_range == fit.shift_range
    assert bare.step == fit.step and bare.fitted_on["tolerance_dropped"] is True
    assert fit.tolerance == {"a": 1.0}, "the original must not be modified"
