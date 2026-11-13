"""Every shipped recipe loads, and the code defaults are the published baseline."""

import pytest

from vox2seglam.config import load_config
from vox2seglam.config.loading import SHIPPED_EXPERIMENTS
from vox2seglam.config.paths import PathsCfg
from vox2seglam.config.root import Config

RECIPES = sorted(SHIPPED_EXPERIMENTS.rglob("*.yaml"))
# an empty glob would turn the parametrisation below into one skipped test and leave the
# suite green
assert RECIPES, f"no recipe found under {SHIPPED_EXPERIMENTS}"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_recipe_loads_and_validates(recipe):
    load_config(recipe).validate()


def test_defaults_are_the_published_model():
    cfg = Config()
    assert cfg.preprocess.orientation == "LAS"
    assert cfg.repro.deterministic and not cfg.repro.cudnn_benchmark
    lam = cfg.model.lam
    # end to end: the arm the paper reports as its method, and how the published checkpoints
    # were trained. The staged curriculum is a recipe, not the code default.
    assert not (lam.detach_backbone or lam.detach_between_steps or lam.detach_graph_for_refiner)
    assert lam.graph_topology == "mst" and lam.refine_patch_size == 17
    assert cfg.model.seg_head.deep_supervision_levels == (1, 2)


def test_automatic_template_name():
    assert PathsCfg().template_name() == "train_landmark_16"
    paths = PathsCfg(split_train="afids_nnlm_train_n11_ids.txt", landmark_suffix="_afids_32.nii.gz")
    assert paths.template_name() == "afids_nnlm_train_n11_afids_32"


def test_landmark_subset_needs_an_explicit_template():
    with pytest.raises(ValueError, match="template_dir"):
        load_config("baseline", overrides={"task": {"landmark_ids": [1, 2, 3]}}).validate()


def test_voxel_spacing_is_declared_and_checked():
    """Metrics in mm read the grid from the config, and refuse an ambiguous one."""
    from dataclasses import replace

    from vox2seglam.config.task import TaskCfg

    task = TaskCfg()
    assert task.voxel_spacing_mm == (0.7, 0.7, 0.7)
    assert task.isotropic_spacing_mm() == 0.7
    with pytest.raises(ValueError, match="anisotropic"):
        replace(task, voxel_spacing_mm=(1.0, 1.0, 2.0)).isotropic_spacing_mm()


def test_a_misspelt_recipe_name_suggests_the_real_ones():
    with pytest.raises(FileNotFoundError) as err:
        load_config("hoa16_med")
    message = str(err.value)
    assert "Did you mean" in message and "hoa16f_med_jnt_s1" in message
    # the shipped recipes are searched once, however many roots happen to coincide
    assert message.count(str(SHIPPED_EXPERIMENTS)) == 1


def test_a_recipe_resolves_from_the_installed_package(monkeypatch, tmp_path):
    """The recipes are package data, so a name resolves wherever the command is run from --
    this used to need a clone, which made `vox2seglam train --config <name>` unusable once
    installed."""
    monkeypatch.chdir(tmp_path)  # no configs/experiments/ in the cwd
    assert load_config("hoa16f_med_jnt_s1").name == "hoa16f_med_jnt_s1"


def test_a_recipe_of_your_own_resolves_by_name(monkeypatch, tmp_path):
    """configs/experiments/ in the working directory is searched too, so a variant needs no
    path -- and a name that would mean two different files is refused rather than guessed."""
    local = tmp_path / "configs" / "experiments"
    local.mkdir(parents=True)
    (local / "mine.yaml").write_text(
        "name: mine\nextends: baseline\ntrain:\n  epochs: 7\n", encoding="utf-8"
    )
    (local / "hoa16f_med_jnt_s1.yaml").write_text(
        "name: clash\nextends: baseline\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    cfg = load_config("mine")
    assert cfg.name == "mine" and cfg.train.epochs == 7

    with pytest.raises(ValueError, match="exists both at"):
        load_config("hoa16f_med_jnt_s1")  # silently preferring one would train the wrong thing


def test_the_code_defaults_are_the_baseline_recipe_bar_the_gradient_flow():
    """What the message above promises: every setting lives in the installed package, and
    baseline.yaml only names it, picks an output directory and sets the staged gradient flow.

    Those three flags are the one deliberate difference. The code default is the end-to-end
    arm; baseline.yaml states the staged one so that the 125 recipes extending it keep the
    meaning they had when they were run, whatever the default becomes.
    """
    from dataclasses import replace

    code = replace(Config(), name="baseline")
    recipe = load_config("baseline")
    lam = recipe.model.lam
    assert lam.detach_backbone and lam.detach_between_steps and lam.detach_graph_for_refiner

    staged = replace(
        code.model,
        lam=replace(
            code.model.lam,
            detach_backbone=True,
            detach_between_steps=True,
            detach_graph_for_refiner=True,
        ),
    )
    assert (
        replace(code, model=staged, paths=replace(code.paths, exp_dir=recipe.paths.exp_dir))
        == recipe
    )


def test_a_patch_the_backbone_cannot_halve_is_refused():
    """A U-Net of five levels halves the volume four times and puts it back by concatenating
    with the skip of the matching size. An odd side loses a voxel to rounding on the way down,
    and the two tensors no longer meet -- which used to surface as a torch.cat shape error
    deep inside the backbone, long after the run had started.
    """
    with pytest.raises(ValueError, match="divisible by 16"):
        load_config("baseline", overrides={"task": {"patch_size": [32, 40, 32]}}).validate()

    # the reference grid divides by 16, and so must a cohort of another size
    load_config("baseline", overrides={"task": {"patch_size": [32, 48, 32]}}).validate()

    # fewer encoder stages, a smaller factor: the message names the one that applies
    shallow = {
        "model": {
            "backbone": {"encoder_channels": [8, 16, 32], "decoder_channels": [16, 8]},
            "lam": {"aggregate_indices": [[1, 2], [1, 2, 3]], "refine_skip_indices": [0, 4]},
        },
        "task": {"patch_size": [32, 40, 32]},
    }
    load_config("baseline", overrides=shallow).validate()
