"""The example for training on one's own data must keep loading and validating."""

from vox2seglam.config import load_config
from vox2seglam.config.paths import PACKAGE_DIR

EXAMPLE = PACKAGE_DIR / "configs" / "examples" / "own_data.yaml"


def test_the_own_data_example_loads_and_is_consistent():
    cfg = load_config(EXAMPLE)
    cfg.validate()
    # the shape of the HOA2.0 cohort, which is what a reader copying this file starts from;
    # both are the first thing to change for a cohort of another size
    assert (cfg.task.n_landmarks, cfg.task.n_classes) == (16, 17)
    assert cfg.paths.exp_dir == "./experiments/my_cohort"
