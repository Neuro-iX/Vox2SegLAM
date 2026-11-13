"""--set KEY=VALUE: the mechanism every experiment is overridden through."""

import pytest

from vox2seglam.config import load_config, parse_set


def test_parses_literals_booleans_and_none():
    assert parse_set(["train.epochs=5"]) == {"train": {"epochs": 5}}
    assert parse_set(["aug.rc_prob=0.35"]) == {"aug": {"rc_prob": 0.35}}
    assert parse_set(["train.amp=TRUE"]) == {"train": {"amp": True}}
    assert parse_set(["train.amp=false"]) == {"train": {"amp": False}}
    assert parse_set(["preprocess.orientation=none"]) == {"preprocess": {"orientation": None}}
    assert parse_set(["task.patch_size=[32, 48, 32]"]) == {"task": {"patch_size": [32, 48, 32]}}
    # anything that is not a literal stays the string it was
    assert parse_set(["paths.data_root=/mnt/data"]) == {"paths": {"data_root": "/mnt/data"}}
    assert parse_set(["model.lam.graph_lambda=auto"]) == {
        "model": {"lam": {"graph_lambda": "auto"}}
    }


def test_nests_dotted_keys_and_rejects_a_missing_value():
    assert parse_set(["a.b.c=1"]) == {"a": {"b": {"c": 1}}}
    with pytest.raises(SystemExit, match="key=value"):
        parse_set(["train.epochs"])


@pytest.mark.parametrize(
    "override",
    [
        "train.epochs=5",
        "train.amp=true",
        "aug.rc_prob=0.35",
        "aug.rc_prob=1",
        "task.patch_size=[32, 48, 32]",
        "task.voxel_spacing_mm=[1.0, 1.0, 1.0]",
        "paths.data_root=/d",
        "data.eval_num_workers=4",
        "data.eval_num_workers=none",
        "preprocess.orientation=none",
        "model.lam.graph_lambda=0.1",
        "model.lam.graph_lambda=auto",
        "repro.seed=7",
    ],
)
def test_valid_overrides_are_accepted(override):
    load_config("baseline", overrides=parse_set([override]))


@pytest.mark.parametrize(
    "override,why",
    [
        ("aug.rc_prob=0,35", "a decimal comma reads as the tuple (0, 35)"),
        ("train.epochs=350x", "a trailing character leaves it a string"),
        ("train.epochs=2.5", "a float where an integer is declared"),
        ("train.amp=1", "an integer where a boolean is declared"),
        ("task.voxel_spacing_mm=0.7", "a scalar where a sequence is declared"),
        ("repro.seed=[1, 2]", "a sequence where a scalar is declared"),
        ("paths.data_root=5", "a number where a string is declared"),
        ("train.epochs=none", "None on a field that is not optional"),
        ("model.lam.graph_lambda=true", "a boolean on a str | float field"),
        ("model.lam=5", "a whole config section replaced by a value"),
        (
            "loss.max_w=hello",
            "a string on a dict field -- the annotation is resolved, not "
            "read as text, where 'str' appears inside dict[str, float]",
        ),
        ("loss.max_w=2.5", "likewise a float"),
    ],
)
def test_type_mistakes_are_refused(override, why):
    with pytest.raises(TypeError):
        load_config("baseline", overrides=parse_set([override]))


def test_unknown_key_lists_the_valid_ones():
    with pytest.raises(KeyError, match="unknown config key"):
        load_config("baseline", overrides={"train": {"epocs": 5}})
