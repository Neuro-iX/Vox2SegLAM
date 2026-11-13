import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..utils.io import to_zyx
from .normalize import normalize_zscore_brain_masked, normalize_zscore_volume
from .landmarks import apply_affine_to_coords_zyx, as_integer_labels, extract_landmarks

__all__ = ["preprocess_sample_train", "preprocess_sample_eval", "UNAUGMENTED_SAMPLES"]


def _to_tensors(image_xyz, seg_xyz, lm_xyz):
    image = torch.from_numpy(to_zyx(image_xyz).astype(np.float32))[None]
    seg = None if seg_xyz is None else torch.from_numpy(to_zyx(seg_xyz).astype(np.int64))[None]
    return image, seg, as_integer_labels(to_zyx(lm_xyz))


# what MONAI's dict transforms must move together (image + seg, or image + brain mask)
# Training samples this process kept unaugmented because every affine draw pushed a landmark
# out of frame (see _sample); per process, since each dataloader worker has its own copy.
UNAUGMENTED_SAMPLES = 0


def _movable(image, seg, **extra):
    d = {"image": image} if seg is None else {"image": image, "seg": seg}
    d.update(extra)
    return d


def _sample(image, seg, coords_np):
    out = {"image": image.float(), "coords": torch.from_numpy(coords_np).float()}
    if seg is not None:
        out["seg"] = seg.long()
    return out


def _normalise(image, brain_masked, eps, mask):
    if brain_masked:
        return normalize_zscore_brain_masked(image, eps, mask=mask)
    return normalize_zscore_volume(image, eps)


def preprocess_sample_train(
    image_xyz: np.ndarray,
    seg_xyz: np.ndarray,
    lm_xyz: np.ndarray,
    landmark_ids: Sequence[int],
    augmenter=None,
    zscore_brain_masked: bool = False,
    max_attempts: int = 8,
    zscore_eps: float = 1e-8,
) -> dict[str, Any]:
    image_t, seg_t, lm_zyx = _to_tensors(image_xyz, seg_xyz, lm_xyz)
    coords_np = extract_landmarks(lm_zyx, landmark_ids)

    if augmenter is not None and not hasattr(augmenter, "last_affine_zyx"):
        raise TypeError("augmenter must be a TrainAugmenter (see build_train_augmenter)")

    orig_image_t, orig_seg_t, coords_orig = image_t, seg_t, coords_np
    brain = None
    accepted = False

    # geometric augmentation moves the image, and the landmark coordinates are moved to match
    # analytically (landmarks.py). If a draw pushes a landmark out of frame, retry up to
    # max_attempts; if none succeeds, keep the untransformed sample rather than a wrong landmark.
    for _ in range(max_attempts if augmenter is not None else 0):
        out = augmenter.geometric(_movable(orig_image_t, orig_seg_t))
        cand_image_t, cand_seg_t = out["image"], out.get("seg")
        cand_A = augmenter.last_affine_zyx()

        if cand_A is None:
            image_t, seg_t, accepted = cand_image_t, cand_seg_t, True
            break

        shape = cand_image_t.shape[1:]
        cand = apply_affine_to_coords_zyx(coords_orig, cand_A, shape)
        inside = ((cand >= 0) & (cand <= np.asarray(shape) - 1)).all()
        if inside:
            image_t, seg_t, coords_np, accepted = cand_image_t, cand_seg_t, cand, True
            break

    if augmenter is not None and not accepted:
        image_t, seg_t, coords_np = orig_image_t, orig_seg_t, coords_orig
        # no augmentation at all for this sample, not even the intensity stage; a subject whose
        # landmarks sit near the frame can stay unaugmented for a whole run. Keeping it is intended
        # (a wrong landmark would be worse), but it is counted and reported once per process
        global UNAUGMENTED_SAMPLES
        UNAUGMENTED_SAMPLES += 1
        if UNAUGMENTED_SAMPLES == 1:
            warnings.warn(
                f"all {max_attempts} affine draws pushed a landmark out of the frame; this "
                "sample is kept unaugmented -- no geometric *and* no intensity "
                "augmentation. Raise aug.affine_max_attempts or soften aug.scale_range / "
                "aug.rotate_range if this is frequent (count: data.sample"
                ".UNAUGMENTED_SAMPLES, per worker process).",
                RuntimeWarning,
                stacklevel=2,
            )

    brain = image_t > 0
    image_t = _normalise(image_t, zscore_brain_masked, zscore_eps, brain)

    # intensity augmentation runs after normalisation, inside the brain mask, with the background
    # restored afterwards so noise does not light it up
    if augmenter is not None and accepted:
        background = image_t.min()
        out = augmenter.intensity(_movable(image_t * brain, seg_t, brain=brain))
        image_t, seg_t = out["image"], out.get("seg")
        image_t = torch.where(brain, image_t, background)

    return _sample(image_t, seg_t, coords_np)


def preprocess_sample_eval(
    image_xyz: np.ndarray,
    seg_xyz: np.ndarray,
    lm_xyz: np.ndarray,
    landmark_ids: Sequence[int],
    zscore_brain_masked: bool = False,
    zscore_eps: float = 1e-8,
) -> dict[str, Any]:
    image_t, seg_t, lm_zyx = _to_tensors(image_xyz, seg_xyz, lm_xyz)
    return _sample(
        _normalise(image_t, zscore_brain_masked, zscore_eps, None),
        seg_t,
        extract_landmarks(lm_zyx, landmark_ids),
    )
