import numpy as np
from monai.transforms import Compose, RandAffined, Randomizable
from monai.utils import MAX_SEED

__all__ = ["TrainAugmenter"]


# enforces that the geometric Compose holds exactly one transform of a given type,
# since TrainAugmenter below reaches into it by type to read back what random affine
# parameters were actually applied
def _sole(compose: Compose, cls: type):
    found = [t for t in compose.transforms if isinstance(t, cls)]
    if len(found) != 1:
        raise ValueError(
            f"the geometric stage must hold exactly one {cls.__name__}, found {len(found)}"
        )
    return found[0]


# wraps the two augmentation stages (geometric: affects coords, intensity: image-only)
# behind one callable, and exposes what random affine transform actually fired so
# sample.py can move the ground-truth landmark coords the same way
class TrainAugmenter(Randomizable):
    def __init__(self, geometric: Compose, intensity: Compose) -> None:
        self.geometric = geometric
        self.intensity = intensity
        self._affine = _sole(geometric, RandAffined)

    # None if RandAffined didn't fire this call (its own `prob` roll), not an error
    def last_affine_zyx(self) -> np.ndarray | None:
        if not self._affine._do_transform:
            return None
        matrix = self._affine.rand_affine.rand_affine_grid.get_transformation_matrix()
        if matrix is None:
            raise RuntimeError("RandAffined fired but reports no transformation matrix")
        return np.asarray(matrix, dtype=np.float32)

    def __call__(self, data: dict) -> dict:
        return self.intensity(self.geometric(data))

    def randomize(self, data=None) -> None:
        self.geometric.randomize(data)
        self.intensity.randomize(data)

    # re-seeds every sub-transform from this augmenter's own RNG, so one seed here
    # deterministically fixes the whole chain (used for worker/dataset seeding)
    def set_random_state(self, seed=None, state=None) -> "TrainAugmenter":
        super().set_random_state(seed=seed, state=state)
        for transform in (*self.geometric.transforms, *self.intensity.transforms):
            if isinstance(transform, Randomizable):
                transform.set_random_state(seed=int(self.R.randint(MAX_SEED, dtype="uint32")))
        return self
