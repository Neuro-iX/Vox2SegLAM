from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackboneCfg:
    encoder_channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    decoder_channels: tuple[int, ...] = (128, 64, 32, 16)
    norm: str = "instance"  # "batch" / "instance" / "layer", applied in every ResidualBlock
    p_dropout: float | None = (
        None  # Dropout3d prob in each ResidualBlock's first conv; None/<=0 disables it
    )
    grad_checkpoint: bool = (
        True  # recompute activations on backward instead of storing them (saves VRAM)
    )


@dataclass(frozen=True)
class SegHeadCfg:
    enabled: bool = True  # if False, no seg head/final layer is built at all (landmark-only model)
    deep_supervision: bool = (
        True  # add extra low-res seg heads alongside the full-res one, for training signal
    )
    deep_supervision_levels: tuple[int, ...] = (
        1,
        2,
    )  # decoder stage indices (0=deepest..len-1=shallowest) that get an aux head
    upsample_mode: str = "trilinear"  # interpolation used to resize the aux seg heads to full resolution (the final head already is)


@dataclass(frozen=True)
class LamCfg:
    # graph network - message-passing refinement of landmark coordinates
    graph_channels: tuple[int, ...] = (
        160,
        160,
    )  # GNN latent width per step; len must equal `steps`
    steps: int = 2  # number of coarse-to-fine graph refinement iterations
    # per step: indices into the combined [encoder_channels + decoder_channels] skip
    # list (0..8: 0-4 encoder stages, 5-8 decoder stages) - which backbone features
    # are sampled at the current coords and fed into that step's GNN block
    aggregate_indices: tuple[tuple[int, ...], ...] = ((3, 4, 5), (1, 2, 3, 5, 6, 7))
    graph_norm: str = "layer"  # norm inside each graph residual block ("none"/"layer"/"batch")
    n_residual_blocks: int = 3  # number of graph residual blocks chained per step
    n_hidden_layers: int = 2  # hidden GraphConv+norm layers inside each residual block
    clamp_coords: bool = (
        True  # clamp landmark zyx coords to the patch bounds after init and each step
    )
    # edge topology of the landmark graph: "fully_connected" (all pairs), "mst" (maximum
    # spanning tree of the landmark precision graph: graphical lasso + Kruskal) or
    # "connected" (shortest connected prefix of the same ranking). mst / connected read
    # <template>/edge_index_<topology>.pt and, when it is absent, build it from the training
    # landmarks at the start of training (vox2seglam/training/template_setup.py); the
    # checkpoint then carries the edges.
    graph_topology: str = "mst"
    graph_lambda: str | float = (
        "auto"  # graphical-lasso penalty for mst / connected: "auto" = cross-validated
    )
    # How much gradient crosses between the stages. False, the default, trains the cascade end
    # to end ("jnt"), which is how the published checkpoints were trained; true blocks the
    # pathway so each stage learns from its own loss only ("stg"). baseline.yaml sets all
    # three to true and every shipped recipe inherits from it, so this default moves nothing.
    detach_backbone: bool = (
        False  # stop-gradient on backbone features before they reach the graph net
    )
    detach_between_steps: bool = (
        False  # stop-gradient on latent + coords between graph refinement steps
    )

    # template initialisation - where landmarks start before the graph refines them
    perturb_template: bool = True  # randomly perturb the template-mean init coords during training
    template_perturbation_scale: float = (
        2.5  # multiplies the per-landmark std used for that perturbation
    )
    template_std_filename: str = (
        "std_per_landmark_zyx.npy"  # per-landmark coordinate std, in template_dir
    )

    # local refinement head - sub-voxel correction via a small patch around each landmark
    refine: bool = True  # if False, the model returns only the coarse graph coordinates
    refine_skip_indices: tuple[int, ...] = (
        0,
        8,
    )  # same 0..8 skip-list indices, sampled for the refiner's patch
    refine_patch_size: int = (
        17  # side length of the local cubic patch, in grid cells (spacing = refine_pitch)
    )
    # channel width of the refiner's conv tower: 4 x the U-Net base width (16 -> 64); a
    # sweep of 32..256 channels changed the validation MRE by <= 0.012 mm at every size
    refine_hidden: int = 64
    refine_n_conv_layers: int = 3  # Conv3d+InstanceNorm+FiLM blocks in the refiner tower
    refine_embed_dim: int = (
        64  # per-landmark identity embedding size, fed into the FiLM conditioning vector
    )
    refine_cond_dim: int = (
        128  # width of the conditioning vector (landmark embedding + GNN latent) fed to FiLM
    )
    refine_pitch: float = 1.0  # spacing between adjacent patch grid cells, in backbone-input voxels
    detach_graph_for_refiner: bool = False  # stop-gradient on graph coords/latent before they reach the refiner (see detach_backbone)

    def edge_source(self) -> tuple[str, str]:
        """("fully_connected", "") or ("template", <edge file in the template directory>)."""
        if self.graph_topology == "fully_connected":
            return "fully_connected", ""
        if self.graph_topology in ("mst", "connected"):
            return "template", f"edge_index_{self.graph_topology}.pt"
        raise ValueError(
            f"model.lam.graph_topology={self.graph_topology!r}; expected "
            "'fully_connected', 'mst' or 'connected'"
        )


@dataclass(frozen=True)
class ModelCfg:
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    seg_head: SegHeadCfg = field(default_factory=SegHeadCfg)
    lam: LamCfg = field(default_factory=LamCfg)
