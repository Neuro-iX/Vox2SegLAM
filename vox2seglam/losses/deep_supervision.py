import torch

__all__ = ["deep_supervision_weights"]


# geometric falloff, halving per level from full resolution (last) down to the
# deepest/lowest-res head (first), then renormalised to sum to 1 - the standard
# nnU-Net-style deep-supervision weighting, so the full-res prediction dominates the
# loss while shallower heads still get a (smaller) gradient signal of their own
def deep_supervision_weights(n_levels: int, device: torch.device) -> torch.Tensor:
    if n_levels <= 0:
        raise ValueError("n_levels must be >= 1")

    w = torch.tensor(
        [0.5 ** (n_levels - 1 - i) for i in range(n_levels)],
        device=device,
        dtype=torch.float32,
    )
    return w / w.sum()
