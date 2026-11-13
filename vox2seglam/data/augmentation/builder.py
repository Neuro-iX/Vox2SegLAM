from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGibbsNoised,
    Randomizable,
    RandScaleIntensityd,
    RandScaleIntensityFixedMeand,
)
from monai.transforms.transform import MapTransform
from monai.utils import MAX_SEED

from .augmenter import TrainAugmenter
from .contrast import RandRCContrastd

__all__ = ["build_train_augmenter", "OneOfResolutiond"]


# picks at most one of several mutually exclusive transforms per call (e.g. k-space
# low-res OR gaussian smooth, never both - they represent the same physical effect at
# different "intensities", stacking them wouldn't make sense). probs may sum to < 1,
# leaving a gap where no branch fires at all.
class OneOfResolutiond(Randomizable, MapTransform):
    def __init__(self, transforms, probs, keys=("image",)) -> None:
        Randomizable.__init__(self)
        MapTransform.__init__(self, keys, allow_missing_keys=False)
        total = float(sum(probs))
        if total > 1.0:
            raise ValueError(
                f"OneOfResolutiond: the branch probabilities sum to {total:.3f}; "
                "exclusive branches cannot exceed 1"
            )
        self.transforms = list(transforms)
        self.probs = [float(p) for p in probs]
        self._choice = -1

    # draws one uniform sample and walks the cumulative probabilities to pick a branch;
    # _choice stays -1 (no-op) if the draw lands past the last cumulative sum
    def randomize(self, data=None) -> None:
        u = float(self.R.random())
        acc, self._choice = 0.0, -1
        for i, p in enumerate(self.probs):
            acc += p
            if u < acc:
                self._choice = i
                break

    def set_random_state(self, seed=None, state=None) -> "OneOfResolutiond":
        super().set_random_state(seed=seed, state=state)
        for transform in self.transforms:
            if isinstance(transform, Randomizable):
                transform.set_random_state(seed=int(self.R.randint(MAX_SEED, dtype="uint32")))
        return self

    def __call__(self, data):
        self.randomize()
        if self._choice < 0:
            return data
        return self.transforms[self._choice](data)


# assembles the two Composes from AugCfg: geometric runs first (moves image+seg
# together; landmark coords are moved to match afterwards, see sample.py), then
# intensity runs image-only (contrast/noise/resolution - never touches coordinates)
def build_train_augmenter(cfg) -> TrainAugmenter:
    geometric = Compose(
        [
            # "seg" is absent when the seg head is disabled (landmark-only recipes)
            EnsureTyped(keys=("image", "seg"), track_meta=True, allow_missing_keys=True),
            RandAffined(
                keys=("image", "seg"),
                allow_missing_keys=True,
                prob=cfg.affine_prob,
                rotate_range=(cfg.rotate_range,) * 3,
                translate_range=(cfg.translate_range,) * 3,
                scale_range=(tuple(cfg.scale_range),) * 3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ]
    )
    intensity = Compose(
        [
            RandRCContrastd(
                keys=("image",),
                prob=cfg.rc_prob,
                probe_voxels=cfg.rc_probe_voxels,
                min_brain_voxels=cfg.rc_min_brain_voxels,
                pct=(cfg.rc_percentile_lo, cfg.rc_percentile_hi),
            ),
            RandScaleIntensityd(
                keys=("image",),
                prob=cfg.brightness_prob,
                factors=cfg.brightness_factor,
            ),
            RandScaleIntensityFixedMeand(
                keys=("image",),
                prob=cfg.contrast_prob,
                factors=cfg.contrast_factor,
                preserve_range=True,
            ),
            # prob=1.0 on both branches: OneOfResolutiond itself decides *whether* and
            # *which* branch runs (via lowres_prob/gaussian_smooth_prob below); once
            # chosen, that branch always applies
            OneOfResolutiond(
                transforms=[
                    RandGibbsNoised(
                        keys=("image",),
                        prob=1.0,
                        alpha=tuple(cfg.lowres_alpha_range),
                    ),
                    RandGaussianSmoothd(
                        keys=("image",),
                        prob=1.0,
                        sigma_x=(cfg.gaussian_smooth_sigma_min, cfg.gaussian_smooth_sigma_max),
                        sigma_y=(cfg.gaussian_smooth_sigma_min, cfg.gaussian_smooth_sigma_max),
                        sigma_z=(cfg.gaussian_smooth_sigma_min, cfg.gaussian_smooth_sigma_max),
                    ),
                ],
                probs=[cfg.lowres_prob, cfg.gaussian_smooth_prob],
            ),
            RandGaussianNoised(
                keys=("image",),
                prob=cfg.noise_prob,
                mean=0.0,
                std=cfg.noise_std,
            ),
        ]
    )
    return TrainAugmenter(geometric, intensity)
