from dataclasses import dataclass, field

from .data import AugCfg, DataCfg, PreprocessCfg
from .loss import LossCfg
from .model import ModelCfg
from .paths import PathsCfg
from .task import TaskCfg
from .training import OptimCfg, ReproCfg, SchedulerCfg, StagingCfg, TrainCfg


@dataclass(frozen=True)
class Config:
    # one field per vox2seglam/config/*.py module - the full schema for one run
    paths: PathsCfg = field(default_factory=PathsCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    data: DataCfg = field(default_factory=DataCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    aug: AugCfg = field(default_factory=AugCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    staging: StagingCfg = field(default_factory=StagingCfg)
    repro: ReproCfg = field(default_factory=ReproCfg)

    name: str = "base"  # set from the experiment's `name:` key (or its filename) by loading.py

    # cross-section checks a single default cannot express; run only at the root of load_config(),
    # after all extends and overrides
    def validate(self) -> "Config":
        lam, bb = self.model.lam, self.model.backbone

        # Each encoder stage after the first halves the volume, and the decoder puts it back
        # by concatenating with the skip of the matching size. A patch that is not divisible
        # by that factor loses a voxel to rounding on the way down and the two tensors no
        # longer meet, which surfaces as a torch.cat shape error deep inside the backbone.
        # The reference grid (240 x 304 x 240) divides by 16; a cohort of another size has to.
        factor = 2 ** (len(bb.encoder_channels) - 1)
        bad = [n for n in self.task.patch_size if n % factor]
        if bad:
            raise ValueError(
                f"task.patch_size={tuple(self.task.patch_size)} is not divisible by {factor} "
                f"({', '.join(str(n) for n in bad)} is not), which a U-Net of "
                f"{len(bb.encoder_channels)} levels needs. Pad or crop the volumes to a "
                f"multiple of {factor}, or use fewer encoder stages."
            )

        # graph topology: a known value, and a penalty that is "auto" or a positive number
        lam.edge_source()
        if lam.graph_lambda != "auto" and not (
            isinstance(lam.graph_lambda, (int, float)) and lam.graph_lambda > 0
        ):
            raise ValueError(
                f"model.lam.graph_lambda={lam.graph_lambda!r}; expected 'auto' or a positive number"
            )

        # graph_channels / aggregate_indices: one entry per GNN step (lam.steps)
        if len(lam.graph_channels) != lam.steps:
            raise ValueError(
                f"model.lam.graph_channels has {len(lam.graph_channels)} entries; "
                f"expected one per step = {lam.steps}"
            )
        if len(lam.aggregate_indices) != lam.steps:
            raise ValueError(
                f"model.lam.aggregate_indices has {len(lam.aggregate_indices)} entries; "
                f"expected one per step = {lam.steps}"
            )

        # aggregate_indices / refine_skip_indices index into the combined
        # [encoder_channels + decoder_channels] skip list - bounds-check against it
        n_skips = len(bb.encoder_channels) + len(bb.decoder_channels)
        for step, idx in enumerate(lam.aggregate_indices):
            if not idx:
                raise ValueError(f"model.lam.aggregate_indices[{step}] is empty")
            if min(idx) < 0 or max(idx) >= n_skips:
                raise ValueError(
                    f"model.lam.aggregate_indices[{step}]={idx} out of range "
                    f"for {n_skips} skip features (0..{n_skips - 1})"
                )
        for i in lam.refine_skip_indices:
            if not 0 <= i < n_skips:
                raise ValueError(
                    f"model.lam.refine_skip_indices contains {i}, out of range "
                    f"for {n_skips} skip features"
                )

        # graph_loss_per_step: one kind per GNN step, each "mse" or "wing"; "mse" is only
        # meaningful once coords are perturbed around the template (else the error is ~0)
        if len(self.loss.graph_loss_per_step) != lam.steps:
            raise ValueError(
                f"loss.graph_loss_per_step has {len(self.loss.graph_loss_per_step)} "
                f"entries; expected one per graph step = {lam.steps}"
            )

        for step, kind in enumerate(self.loss.graph_loss_per_step):
            if kind not in ("mse", "wing"):
                raise ValueError(
                    f"loss.graph_loss_per_step[{step}]={kind!r}; expected 'mse' or 'wing'"
                )
            if kind == "mse" and not lam.perturb_template:
                raise ValueError(
                    f"loss.graph_loss_per_step[{step}]='mse' requires "
                    "model.lam.perturb_template=true to normalise the squared "
                    "error; use 'wing' for this step otherwise"
                )

        if self.train.amp_dtype not in ("bfloat16", "float16"):
            raise ValueError(
                f"train.amp_dtype={self.train.amp_dtype!r}; expected 'bfloat16' or 'float16'"
            )

        # augmentation ranges must be well-formed (low <= high, non-negative where it
        # matters) - a typo'd YAML range would otherwise fail silently or invert
        lo, hi = self.aug.lowres_alpha_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"aug.lowres_alpha_range={self.aug.lowres_alpha_range!r}; "
                "expected 0 <= low <= high <= 1"
            )

        # the automatic template is named after the training split and the landmark files
        # only, so a landmark subset needs its own, explicit template directory
        if not self.paths.template_dir and self.task.landmark_ids is not None:
            raise ValueError(
                "task.landmark_ids selects a landmark subset: set paths.template_dir "
                "explicitly (the automatic template covers all landmarks)"
            )

        if self.train.freeze_unet and not self.train.init_unet_from:
            raise ValueError(
                "train.freeze_unet needs train.init_unet_from: a frozen "
                "backbone must come from a trained checkpoint"
            )

        # a backbone with no seg head and detach_backbone=true receives no gradient at
        # all and would stay at its random initialisation
        if (
            not self.model.seg_head.enabled
            and self.model.lam.detach_backbone
            and not self.train.freeze_unet
        ):
            raise ValueError(
                "model.seg_head.enabled=false with model.lam.detach_backbone="
                "true leaves the U-Net untrained; set detach_backbone=false "
                "(train it from the landmark losses) or freeze a pretrained one "
                "(train.init_unet_from + train.freeze_unet)"
            )

        # checkpointing re-runs the forward during backward, which would update
        # BatchNorm running statistics twice per step
        if bb.norm == "batch" and bb.grad_checkpoint:
            raise ValueError(
                "model.backbone.norm='batch' is incompatible with grad_checkpoint=true "
                "(running statistics would be updated twice per step)"
            )
        return self


BASE = Config()
