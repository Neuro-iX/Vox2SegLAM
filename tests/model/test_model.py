"""Model outputs, parameter counts and the stop-gradients between the stages."""

import pytest
import torch

from vox2seglam.config import load_config
from vox2seglam.models import Vox2SegLAM

GRID = [32, 48, 32]


def build(recipe: str) -> Vox2SegLAM:
    torch.manual_seed(0)
    cfg = load_config(recipe, overrides={"task": {"patch_size": GRID}})
    return Vox2SegLAM.from_config(cfg, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "recipe, n_params", [("hoa16f_small_stg_s0", 2_719_985), ("hoa16f_med_jnt_s1", 9_026_049)]
)
def test_parameter_counts(recipe, n_params):
    """The two shipped sizes. A width that drifts changes the count before it changes a
    metric, so this catches an edit to the architecture that nothing else would."""
    assert sum(p.numel() for p in build(recipe).parameters()) == n_params


def test_outputs():
    model = build("hoa16f_small_stg_s0")
    out = model(torch.randn(1, 1, *GRID), coords_init_zyx=torch.full((1, 16, 3), 15.0))
    assert out["coords"].shape == (1, 16, 3)
    assert [s.shape for s in out["seg"]] == [(1, 17, *GRID)] * 3  # D7, D8, full resolution
    assert len(out["coords_steps"]) == 2


def test_each_stage_learns_from_its_own_loss_only():
    model = build("hoa16f_small_stg_s0")
    out = model(torch.randn(1, 1, *GRID), coords_init_zyx=torch.full((1, 16, 3), 15.0))
    out["coords"].sum().backward(retain_graph=True)  # the refiner's output
    assert all(p.grad is None for p in model.unet.parameters())
    assert all(p.grad is None for p in model.landmark_graph.parameters())
    assert any(p.grad is not None for p in model.refiner.parameters())
    model.zero_grad(set_to_none=True)
    out["seg"][-1].sum().backward()  # the segmentation
    assert all(p.grad is None for p in model.landmark_graph.parameters())
    assert any(p.grad is not None for p in model.unet.parameters())


def test_sampler_equals_grid_sample():
    from torch.nn.functional import grid_sample
    from vox2seglam.models.sampling import _sample, normalize_coords

    torch.manual_seed(0)
    full = (24, 28, 20)
    for shape in (full, (12, 14, 10)):
        f = torch.randn(2, 3, *shape, dtype=torch.float64)
        coords = torch.rand(2, 100, 3, dtype=torch.float64) * 30 - 3  # some outside
        g = normalize_coords(coords, full).double()
        ref = grid_sample(f, g, mode="bilinear", padding_mode="border", align_corners=False)
        assert torch.allclose(_sample(f, g), ref[:, :, :, 0, 0].transpose(2, 1), atol=1e-12)


def test_auxiliary_heads_are_training_only():
    """The deep-supervision heads cost ~1.2 GB each on the published grid and are read only
    by the training loss, so eval() must skip them - without moving the full-resolution
    output, which is what inference and every metric read."""
    import torch

    from vox2seglam.config import load_config
    from vox2seglam.models import Vox2SegLAM

    cfg = load_config("baseline", overrides={"task": {"patch_size": [32, 48, 32]}})
    model = Vox2SegLAM.from_config(cfg)
    x = torch.randn(1, 1, 32, 48, 32, generator=torch.Generator().manual_seed(0))

    model.train()
    with torch.no_grad():
        train_seg = model(x)["seg"]
    model.eval()
    with torch.no_grad():
        eval_seg = model(x)["seg"]

    assert len(train_seg) == 3 and len(eval_seg) == 1  # 2 auxiliary + full resolution
    assert torch.equal(train_seg[-1], eval_seg[-1])  # the output readers use
