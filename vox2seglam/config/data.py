import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataCfg:
    batch_size: int = 1  # whole volumes per step
    eval_batch_size: int = 1

    num_workers: int = min(8, os.cpu_count() or 1)  # dataloader subprocesses, capped at 8
    eval_num_workers: int | None = None  # None = derive from num_workers, see below

    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = (
        True  # deterministic training workers are re-created each epoch anyway
    )
    prefetch_factor: int | None = 2  # batches loaded ahead, per worker

    def resolved_eval_num_workers(self) -> int:
        if self.eval_num_workers is not None:
            return int(self.eval_num_workers)
        return max(0, int(self.num_workers) // 2)


@dataclass(frozen=True)
class PreprocessCfg:
    zscore_brain_masked: bool = True  # statistics over brain voxels only
    zscore_eps: float = 1e-8
    # True: data/prepare runs the step on the raw images (cached) before training, and the
    # Predictor expects inputs that went through it
    skull_stripped: bool = True  # SynthStrip
    bias_field_corrected: bool = True  # N4
    # orientation of the volumes; the Predictor rejects any other instead of mirroring it
    # (None disables the check)
    orientation: str | None = "LAS"

    # cache layout of data/prepare: directory suffix per stage, filename tag once stripped
    brain_tag: str = "_brain"
    skull_strip_dir_suffix: str = "_BET"
    bias_field_dir_suffix: str = "_N4"

    synthstrip_border_mm: int = 1  # margin kept around the detected brain
    # Command that behaves like mri_synthstrip (given -i, -o, -b with absolute paths), e.g. the
    # official container; empty means the built-in stripper or paths.freesurfer_home.
    #   ["apptainer", "run", "--cleanenv", "/path/to/synthstrip.sif"]
    #   ["docker", "run", "--rm", "-v", "/data:/data", "freesurfer/synthstrip"]
    # apptainer needs --cleanenv, or the caller's conda environment shadows the image's Python.
    synthstrip_command: tuple[str, ...] = ()

    # N4: downsample factor while estimating the field (applied at full resolution), and
    # iterations per resolution level, coarse to fine
    n4_shrink_factor: int = 4
    n4_iterations: tuple[int, ...] = (50, 50, 50, 50)


@dataclass(frozen=True)
class AugCfg:
    affine_prob: float = 0.8  # probability of applying rotation/translation/scale
    rotate_range: float = 0.17  # max rotation per axis, radians (~10 degrees)
    translate_range: float = 7.0  # max translation per axis, voxels
    scale_range: tuple[float, float] = (-0.25, 0.25)  # scale factor offset from 1.0
    affine_max_attempts: int = 8  # resample tries before falling back to identity

    # alternative to lowres_* in the same OneOfResolutiond pick (augmentation/builder.py)
    gaussian_smooth_prob: float = 0.20
    gaussian_smooth_sigma_min: float = 0.3  # min blur sigma, voxels
    gaussian_smooth_sigma_max: float = 0.5  # max blur sigma, voxels

    # random-convolution contrast (CABLD, Salari et al., ICCV 2025; augmentation/contrast.py)
    rc_prob: float = 0.35
    # brain-intensity bracket it normalises against, estimated from a strided subsample of at
    # most rc_probe_voxels; exact if the subsample has fewer than rc_min_brain_voxels
    rc_percentile_lo: float = 0.005
    rc_percentile_hi: float = 0.995
    rc_probe_voxels: int = 1 << 16
    rc_min_brain_voxels: int = 100

    brightness_prob: float = 0.15  # one random factor on every intensity
    brightness_factor: float = 0.25  # factor drawn from [1 - f, 1 + f]
    contrast_prob: float = 0.15  # scales the spread, keeps the mean
    contrast_factor: float = 0.25

    lowres_prob: float = 0.25  # simulates a lower-resolution acquisition
    # RandGibbsNoised k-space truncation, alpha in [0, 1]: 0 is the identity
    lowres_alpha_range: tuple[float, float] = (0.0, 0.4)

    noise_prob: float = 0.15
    # std, not variance: inputs are z-scored, and nnU-Net caps the noise variance at 0.1
    noise_std: float = math.sqrt(0.1)
