"""Which loss terms keep a fixed weight rather than a learned one."""

import pytest

from vox2seglam.training.trainer import _fixed_term_names

ALL_DETACHED = dict(detach_backbone=True, detach_between_steps=True, detach_graph_for_refiner=True)
BASELINE = dict(**ALL_DETACHED, has_seg=True, n_steps=2, has_refine=True)


def test_the_baseline_fixes_every_term_that_owns_its_parameters():
    """Everything detached: seg owns the U-Net, each graph step owns its block, and the two
    refiner terms share the refiner head - so they alone are learned."""
    assert _fixed_term_names(**BASELINE) == {"seg", "gnn1", "gnn2"}


def test_a_shared_backbone_leaves_nothing_fixed():
    assert _fixed_term_names(**{**BASELINE, "detach_backbone": False}) == set()


def test_a_shared_backbone_without_a_segmentation_head_also_leaves_nothing_fixed():
    """The Vox2LAM arrangement: no segmentation head, and the landmark losses
    train the U-Net. Every landmark term reaches it, so every one of them has something to
    trade against."""
    assert _fixed_term_names(**{**BASELINE, "has_seg": False, "detach_backbone": False}) == set()


def test_without_a_head_and_with_a_detached_backbone_the_steps_are_on_their_own():
    assert _fixed_term_names(**{**BASELINE, "has_seg": False}) == {"gnn1", "gnn2"}


def test_steps_that_see_each_other_are_learned_together():
    got = _fixed_term_names(**{**BASELINE, "detach_between_steps": False})
    assert got == {"seg"}  # gnn1 and gnn2 now share parameters


def test_a_refiner_that_reaches_the_graph_joins_its_last_step():
    got = _fixed_term_names(**{**BASELINE, "detach_graph_for_refiner": False})
    assert got == {"seg", "gnn1"}  # gnn2 is now shared with the refiner


def test_the_two_refiner_terms_are_never_fixed_while_the_refiner_exists():
    for extra in ({}, {"detach_between_steps": False}, {"has_seg": False}):
        got = _fixed_term_names(**{**BASELINE, **extra})
        assert "refine_cls" not in got and "refine_reg" not in got


def test_without_a_refiner_only_the_graph_and_seg_remain():
    assert _fixed_term_names(**{**BASELINE, "has_refine": False}) == {"seg", "gnn1", "gnn2"}
    assert _fixed_term_names(
        **{**BASELINE, "has_refine": False, "detach_between_steps": False}
    ) == {"seg"}


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_a_single_step_is_handled_like_any_other(steps):
    got = _fixed_term_names(**{**BASELINE, "n_steps": steps})
    assert got == {"seg", *(f"gnn{i + 1}" for i in range(steps))}


def test_every_term_is_either_fixed_or_shared_and_never_invented():
    """The result is a subset of the terms that exist, whatever the flags."""
    from itertools import product

    for backbone, between, refiner, seg, refine in product([True, False], repeat=5):
        terms = (
            ({"seg"} if seg else set())
            | {"gnn1", "gnn2"}
            | ({"refine_cls", "refine_reg"} if refine else set())
        )
        got = _fixed_term_names(
            detach_backbone=backbone,
            detach_between_steps=between,
            detach_graph_for_refiner=refiner,
            has_seg=seg,
            n_steps=2,
            has_refine=refine,
        )
        assert got <= terms
