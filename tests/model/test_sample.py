"""Sample preprocessing: the unaugmented-sample case is counted, not silent."""

import warnings

import numpy as np
import pytest

import vox2seglam.data.sample as sample
from vox2seglam.config import load_config
from vox2seglam.data import build_train_augmenter


@pytest.fixture
def volume():
    image = np.zeros((8, 8, 8), np.float32)
    image[2:6, 2:6, 2:6] = 1.0
    landmarks = np.zeros((8, 8, 8), np.int16)
    landmarks[3, 3, 3] = 1
    return image.transpose(2, 1, 0), landmarks.transpose(2, 1, 0)


@pytest.fixture
def augmenter():
    aug = build_train_augmenter(load_config("baseline").aug)
    aug.set_random_state(0)
    return aug


@pytest.fixture(autouse=True)
def reset_counter():
    sample.UNAUGMENTED_SAMPLES = 0
    yield
    sample.UNAUGMENTED_SAMPLES = 0


def run(volume, augmenter, max_attempts):
    image_xyz, lm_xyz = volume
    return sample.preprocess_sample_train(
        image_xyz=image_xyz,
        seg_xyz=None,
        lm_xyz=lm_xyz,
        landmark_ids=[1],
        augmenter=augmenter,
        max_attempts=max_attempts,
    )


def test_exhausted_draws_are_counted_and_warned_once(volume, augmenter):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(volume, augmenter, max_attempts=0)  # no draw can succeed
        assert sample.UNAUGMENTED_SAMPLES == 1
        assert len(caught) == 1
        assert "unaugmented" in str(caught[0].message)
        run(volume, augmenter, max_attempts=0)
        assert sample.UNAUGMENTED_SAMPLES == 2  # counted again...
        assert len(caught) == 1  # ...but the process says it once


def test_a_successful_draw_counts_nothing(volume, augmenter):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run(volume, augmenter, max_attempts=8)
    assert sample.UNAUGMENTED_SAMPLES == 0 and not caught
    assert out["coords"].shape == (1, 3)


def test_the_landmark_stays_inside_the_frame(volume, augmenter):
    """What the retry loop exists for: an accepted sample has in-frame coordinates."""
    out = run(volume, augmenter, max_attempts=8)
    coords = np.asarray(out["coords"])
    assert ((coords >= 0) & (coords <= 7)).all()
