"""Strict deterministic mode: two identical training steps give identical weights."""

import pytest
import torch

from vox2seglam.config import load_config
from vox2seglam.losses import AdaptiveLossWeighting, compute_losses, loss_term_names
from vox2seglam.losses import weighted_term_names
from vox2seglam.models import Vox2SegLAM
from vox2seglam.training.seeding import set_seed

GRID = [32, 48, 32]


def gradients(device: str, joint: bool = False) -> torch.Tensor:
    """joint: the landmark losses also train the U-Net (Vox2LAM), through the sampler."""
    lam = {"detach_backbone": False} if joint else {}
    cfg = load_config(
        "hoa16f_small_stg_s0", overrides={"task": {"patch_size": GRID}, "model": {"lam": lam}}
    )
    set_seed(cfg)
    model = Vox2SegLAM.from_config(cfg, device=torch.device(device)).to(device)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 1, *GRID, generator=g).to(device)
    batch = {
        "image": x,
        "seg": torch.randint(0, 17, (1, 1, *GRID), generator=g).to(device),
        "coords": (torch.rand(1, 16, 3, generator=g) * 24 + 4).to(device),
    }
    terms = weighted_term_names(loss_term_names(steps=2, has_seg=True, has_refine=True))
    weighter = AdaptiveLossWeighting({t: 1.0 for t in terms}).to(device)
    pred = model(x, coords_init_zyx=batch["coords"] + 2.0)
    _, total = compute_losses(
        pred=pred,
        batch=batch,
        cfg=cfg.loss,
        num_classes=17,
        K=16,
        deep_supervision=True,
        weighter=weighter,
        detach_between_steps=True,
    )
    total.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).cpu()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
        ),
    ],
)
@pytest.mark.parametrize("joint", [False, True])
def test_bit_identical_runs(device, joint):
    assert torch.equal(gradients(device, joint), gradients(device, joint))
