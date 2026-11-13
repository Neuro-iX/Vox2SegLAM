from dataclasses import dataclass, field


@dataclass(frozen=True)
class LossCfg:
    dice_smooth: float = 1e-5  # soft-Dice loss smoothing (training only; evaluation uses hard Dice)

    # per GNN message-passing step
    graph_loss_per_step: tuple[str, ...] = ("mse", "wing")  # "mse" or "wing", one entry per step
    wing_epsilon: float = 1.5  # wing curvature, voxels
    wing_width: float = 6.0  # wing switch point between log and linear regime, voxels

    # refiner head: local classification (heatmap) + regression (offset) loss
    refine_sigma_cells: float = (
        1.0  # std-dev of the soft Gaussian classification target, grid cells
    )
    refine_w_cls: float = 1.0  # weight on the refiner's classification (KL) term (*)
    refine_w_reg: float = 1.0  # weight on the refiner's regression (wing) term (*)
    refine_wing_width: float = 6.0  # wing width for the offset residual, refiner grid cells
    refine_wing_epsilon: float = 0.5  # wing epsilon for the offset residual, refiner grid cells
    refine_eps: float = 1e-8  # numerical floor inside the KL term

    # (*) static multipliers: applied only when adaptive_weighting is False. With
    # adaptive weighting every non-seg term starts at init_w_lm and is then learned
    adaptive_weighting: bool = (
        True  # learn per-term loss weights (uncertainty-based) instead of fixed
    )
    max_w: dict[str, float] = field(
        default_factory=dict
    )  # optional cap on a learned weight, per term name

    init_w_seg: float = 1.0  # segmentation term weight: fixed value, or initial value if adaptive
    init_w_lm: float = 1.0  # default weight for landmark terms not otherwise named above
