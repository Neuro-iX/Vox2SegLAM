from dataclasses import dataclass


@dataclass(frozen=True)
class OptimCfg:
    name: str = "adamw"  # "adamw" / "adam" / "sgd" (optim.py build_optimizers)
    betas: tuple[float, float] = (0.9, 0.999)  # ignored for sgd
    eps: float = 1e-8  # ignored for sgd

    # one optimizer per model stage: "seg" (backbone+seg head) uses lr_voxel, every
    # other stage (graph steps, refiner) uses lr_lam; the adaptive weighter is separate
    lr_voxel: float = 1e-3
    lr_lam: float = 5e-4
    lr_weighter: float = 1e-3  # only used when loss.adaptive_weighting is True

    weight_decay_voxel: float = 1e-4
    weight_decay_lam: float = 1e-3
    weight_decay_weighter: float = 0.0


@dataclass(frozen=True)
class SchedulerCfg:
    name: str = "plateau"  # "plateau" (ReduceLROnPlateau per stage) or "none"/""
    factor: float = 0.5  # lr multiplier on a plateau
    min_lr: float = 1e-8
    eps: float = 0.0  # ReduceLROnPlateau's `eps`: minimum lr change to count as an update
    patience_voxel: int = 10  # epochs without improvement before the "seg"/weighter lr drops
    patience_lam: int = 20  # epochs without improvement before any landmark-branch lr drops


@dataclass(frozen=True)
class StagingCfg:
    enabled: bool = True  # curriculum: start stages one at a time vs. all at once (stages.py)
    # a stage is offered a start at each lr cut of the previous one; if its gate below
    # fails it waits for the previous stage's next lr cut
    # gnnK (K >= 2) starts only once the validation MRE after gnn(K-1) beats the
    # MRE of the starting template
    gnn_must_beat_template: bool = True
    # the "refine" stage has no gate: it starts at the first lr cut of the last graph step


@dataclass(frozen=True)
class TrainCfg:
    epochs: int = 600
    device: str = "cuda"

    amp: bool = True  # mixed-precision training (torch.amp)
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float16"; validated in root.py

    grad_clip: float = 1.0  # max gradient norm, applied to each stage's optimizer separately
    grad_value_clip: float = (
        1e4  # max per-element gradient magnitude (clip_grad_value_); <=0 disables
    )
    grad_accum: int = 2  # optimizer steps every N batches, simulating a larger batch_size
    detect_anomaly: bool = False  # torch.autograd.set_detect_anomaly; slow, debugging only

    eval_every: int = 1  # run validation every N epochs

    # checkpoint "best" selection switches to landmark-MRE-only once the seg branch's lr
    # has plateaued down to this value (see training/checkpoint.py selection_phase)
    selection_lm_only_lr: float = 1e-6
    init_unet_from: str | None = (
        None  # checkpoint whose backbone weights seed this run (relative = repo root)
    )
    freeze_unet: bool = False  # keep those backbone weights fixed (requires init_unet_from)


@dataclass(frozen=True)
class ReproCfg:
    seed: int = 0
    deterministic: bool = True  # bit-exact runs (strict torch deterministic algorithms); slower
    cudnn_benchmark: bool = (
        False  # only read when deterministic is False: cuDNN auto-tuning (faster, not bit-exact)
    )
    seed_workers: bool = True  # seed each dataloader worker process deterministically from `seed`
