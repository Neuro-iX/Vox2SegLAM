"""The Predictor, the only entry point an external user has."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from vox2seglam.config import load_config
from vox2seglam.inference import Predictor
from vox2seglam.models import Vox2SegLAM

GRID = (32, 48, 32)


@pytest.fixture(scope="module")
def predictor():
    cfg = load_config("hoa16f_small_stg_s0", overrides={"task": {"patch_size": list(GRID)}})
    model = Vox2SegLAM.from_config(cfg)
    return Predictor(model=model, cfg=cfg)


@pytest.fixture
def volume():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(GRID, generator=g).numpy().astype(np.float32)
    x[4:-4, 6:-6, 4:-4] += 1.0  # a bright "brain" inside a dark frame
    return x


def test_predicts_a_volume(predictor, volume):
    out = predictor.predict_volume(volume)
    assert out.coordinates.shape == (16, 3)
    assert out.coordinates.dtype == np.float32
    assert out.landmark_ids == list(range(1, 17))
    assert out.segmentation.shape == GRID
    # every landmark is indexable in the volume it came from
    assert ((out.coordinates >= 0) & (out.coordinates <= np.array(GRID) - 1)).all()


def test_the_same_volume_gives_the_same_answer(predictor, volume):
    """eval() mode, no dropout, no augmentation: inference must be deterministic."""
    a, b = predictor.predict_volume(volume), predictor.predict_volume(volume)
    assert np.array_equal(a.coordinates, b.coordinates)
    assert np.array_equal(a.segmentation, b.segmentation)


def test_a_gpu_out_of_memory_retries_in_half_precision(predictor, volume, monkeypatch):
    real, calls = predictor._forward, []

    def forward(x):
        calls.append(predictor.precision)
        if predictor.precision == "float32":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(x)

    monkeypatch.setattr(predictor, "_forward", forward)
    with pytest.warns(RuntimeWarning, match="continuing in"):
        predictor.predict_volume(volume)
    assert calls[0] == "float32" and calls[1] in ("bfloat16", "float16")
    assert predictor.precision == calls[1]  # later volumes do not fail again first


def test_a_gpu_out_of_memory_says_what_to_do(predictor, volume, monkeypatch):
    def refuse(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(predictor, "model", refuse)
    with pytest.raises(RuntimeError, match="--device cpu"):
        predictor.predict_volume(volume)


def test_probabilities_are_asked_for_not_assumed(predictor, volume):
    assert predictor.predict_volume(volume).probabilities is None
    prob = predictor.predict_volume(volume, return_probabilities=True).probabilities
    assert prob.shape == (17, *GRID)
    assert np.allclose(prob.sum(axis=0), 1.0, atol=1e-5)  # a distribution per voxel


def test_a_wrong_grid_is_refused_with_both_shapes(predictor):
    with pytest.raises(ValueError) as err:
        predictor.predict_volume(np.zeros((16, 16, 16), np.float32))
    assert "16, 16, 16" in str(err.value) and "32, 48, 32" in str(err.value)


def test_a_shape_that_is_neither_a_volume_nor_a_batch_is_refused(predictor):
    with pytest.raises(ValueError, match="3D volume"):
        predictor.predict_volume(np.zeros((4, 4), np.float32))


def test_n_passes_must_be_at_least_one(predictor):
    with pytest.raises(ValueError, match="n_passes"):
        Predictor(model=predictor.model, cfg=predictor.cfg, n_passes=0)


def test_a_second_pass_reinitialises_from_the_first(predictor, volume):
    """n_passes > 1 re-runs the whole model, so the result must differ from a single pass."""
    once = predictor.predict_volume(volume).coordinates
    twice = (
        Predictor(model=predictor.model, cfg=predictor.cfg, n_passes=2)
        .predict_volume(volume)
        .coordinates
    )
    assert not np.array_equal(once, twice)


def test_expected_input_states_the_preprocessing_contract(predictor):
    contract = predictor.expected_input()
    assert contract["shape_zyx"] == GRID
    assert contract["orientation"] == "LAS"
    assert "z-score" in contract["normalisation"]


def test_coordinate_systems_are_named_and_consistent(predictor, volume):
    out = predictor.predict_volume(volume)
    assert np.array_equal(out.coordinates_xyz(), out.coordinates[:, ::-1])
    assert out.coordinates_world_mm() is None  # no affine without a file

    out.affine = np.diag([0.7, 0.7, 0.7, 1.0])  # as predict_file would set it
    world = out.coordinates_world_mm()
    assert np.allclose(world, out.coordinates_xyz() * 0.7)

    keyed = out.as_dict()["landmarks"]
    assert set(keyed) == {str(i) for i in range(1, 17)}
    assert set(keyed["1"]) == {"voxel_zyx", "voxel_xyz", "world_ras_mm"}
    assert keyed["1"]["voxel_zyx"] == [float(v) for v in out.coordinates[0]]


def write_nifti(path, array_zyx, affine):
    import nibabel as nib

    # the Predictor reads XYZ from disk and flips to ZYX itself (utils.io.to_zyx)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(array_zyx.transpose(2, 1, 0)), affine), str(path))
    return path


def test_predict_file_reads_the_affine_and_keeps_it(predictor, volume, tmp_path):
    affine = np.diag([-0.7, 0.7, 0.7, 1.0])  # LAS, the training orientation
    path = write_nifti(tmp_path / "sub-01_T1w.nii.gz", volume, affine)

    out = predictor.predict_file(path)
    assert np.allclose(out.affine, affine)
    assert out.coordinates_world_mm() is not None  # now that an affine is known
    # the same volume through the file path and the array path agree
    assert np.allclose(out.coordinates, predictor.predict_volume(volume).coordinates)


def test_predict_file_refuses_the_wrong_orientation(predictor, volume, tmp_path):
    """The model does no reorienting: an RAS volume would be mirrored left to right in silence,
    so the check refuses it."""
    path = write_nifti(tmp_path / "ras.nii.gz", volume, np.eye(4))  # RAS, not LAS
    with pytest.raises(ValueError) as err:
        predictor.predict_file(path)
    message = str(err.value)
    assert "RAS" in message and "LAS" in message and "reorient" in message


def test_the_orientation_check_can_be_disabled(predictor, volume, tmp_path):
    """preprocess.orientation: null is the documented escape hatch."""
    cfg = replace(predictor.cfg, preprocess=replace(predictor.cfg.preprocess, orientation=None))
    relaxed = Predictor(model=predictor.model, cfg=cfg)
    path = write_nifti(tmp_path / "ras.nii.gz", volume, np.eye(4))
    assert relaxed.predict_file(path).coordinates.shape == (16, 3)


def test_the_same_prediction_twice_gives_the_same_numbers(predictor, volume):
    """Not np.allclose: equal. The graph steps reduce over the landmark neighbourhoods, and on
    a GPU that reduction has no fixed order, which moved the coordinates by about 1e-4 voxel
    between two runs of the same command -- enough to shift a boundary voxel and change a Dice
    in its fifth decimal. Predictor asks torch for deterministic algorithms for that reason.
    """
    import torch

    assert torch.are_deterministic_algorithms_enabled()
    first, second = predictor.predict_volume(volume), predictor.predict_volume(volume)
    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.segmentation, second.segmentation)
