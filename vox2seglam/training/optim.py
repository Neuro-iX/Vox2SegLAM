from collections.abc import Collection
from typing import Any

import torch

__all__ = ["build_optimizers", "apply_gradients", "scheduler_metric", "at_rate_floor"]


# one optimizer per stage_parameters() stage ("seg" gets lr_voxel, the rest lr_lam), one for
# the adaptive weighter if present, and a ReduceLROnPlateau per optimizer when configured
def build_optimizers(model, weighter, cfg) -> tuple[dict[str, Any], dict[str, Any]]:
    name = cfg.optim.name.lower()
    if name == "adamw":
        factory = torch.optim.AdamW
    elif name == "adam":
        factory = torch.optim.Adam
    elif name == "sgd":
        factory = lambda params, **kw: torch.optim.SGD(  # noqa: E731
            params, lr=kw["lr"], weight_decay=kw["weight_decay"]
        )
    else:
        raise ValueError(f"unknown optim.name {cfg.optim.name!r}")

    common = {} if name == "sgd" else {"betas": tuple(cfg.optim.betas), "eps": cfg.optim.eps}

    stages = model.stage_parameters()
    lrs = {name: (cfg.optim.lr_voxel if name == "seg" else cfg.optim.lr_lam) for name in stages}
    decays = {
        name: (cfg.optim.weight_decay_voxel if name == "seg" else cfg.optim.weight_decay_lam)
        for name in stages
    }

    optimizers = {
        name: factory(params, lr=lrs[name], weight_decay=decays[name], **common)
        for name, params in stages.items()
    }
    if weighter is not None:
        optimizers["weighter"] = factory(
            weighter.parameters(),
            lr=cfg.optim.lr_weighter,
            weight_decay=cfg.optim.weight_decay_weighter,
            **common,
        )

    schedulers: dict[str, Any] = {}
    if cfg.scheduler.name.lower() == "plateau":
        for key, opt in optimizers.items():
            mode, patience = _schedule_of(key, cfg)
            schedulers[key] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=mode,
                factor=cfg.scheduler.factor,
                patience=patience,
                min_lr=cfg.scheduler.min_lr,
                eps=cfg.scheduler.eps,
            )
    elif cfg.scheduler.name.lower() not in ("none", ""):
        raise ValueError(f"unknown scheduler.name {cfg.scheduler.name!r}")

    return optimizers, schedulers


# "seg" tracks Dice ("max"); the weighter tracks Dice if there is a seg head, else MRE;
# every other stage (gnnN, refine) tracks landmark MRE ("min")
def _schedule_of(key: str, cfg) -> tuple[str, int]:
    if key == "seg":
        return "max", cfg.scheduler.patience_voxel
    if key == "weighter":
        mode = "max" if cfg.model.seg_head.enabled else "min"
        return mode, cfg.scheduler.patience_voxel
    return "min", cfg.scheduler.patience_lam


# value fed to scheduler.step(): "gnnN" prefers that step's own MRE (step_means), falling
# back to the overall MRE
def scheduler_metric(key: str, val: dict[str, float], *, has_seg: bool) -> float:
    if key == "seg" or (key == "weighter" and has_seg):
        return float(val["dice_mean_no_bg"])
    if key.startswith("gnn"):
        step = key[3:]
        return float(val.get(f"lm_mre_vox_mean_step{step}", val["lm_mre_vox_mean"]))
    return float(val["lm_mre_vox_mean"])


# True once every param group has hit its lr floor (`factor`/`min_lr`); the curriculum
# (training/stages.py) retires such a stage
def at_rate_floor(scheduler: Any, optimizer: Any) -> bool:
    min_lrs = getattr(scheduler, "min_lrs", None)
    factor = getattr(scheduler, "factor", None)
    if min_lrs is None or factor is None:
        return False
    eps = float(getattr(scheduler, "eps", 0.0))
    for group, min_lr in zip(optimizer.param_groups, min_lrs):
        lr = float(group["lr"])
        if lr - max(lr * float(factor), float(min_lr)) > eps:
            return False
    return True


# AMP-aware step: unscale, clip each stage on its own (by value, then norm to grad_clip; the
# reported grad_norm_<stage> is before the clip), step only the stages in step_keys, and let
# scaler.update() skip inf/nan steps. Per-stage clipping keeps a large U-Net gradient from
# shrinking the graph's or the refiner's step. Every optimizer's grads are zeroed, stepped or
# not, so grad_accum starts clean.
def apply_gradients(
    optimizers: dict[str, Any],
    scaler: Any,
    grad_clip: float,
    grad_value_clip: float,
    step_keys: Collection[str] | None = None,
) -> dict[str, float]:
    # skip an optimizer none of whose parameters got a gradient (e.g. the weighter while the only
    # active term has a fixed weight); GradScaler.step refuses it
    stepping = {
        key: opt
        for key, opt in optimizers.items()
        if (step_keys is None or key in step_keys)
        and any(p.grad is not None for p in _params_of(opt))
    }

    for opt in stepping.values():
        scaler.unscale_(opt)

    norms: dict[str, float] = {}
    for key, opt in stepping.items():
        params = _params_of(opt)
        if grad_value_clip > 0:
            torch.nn.utils.clip_grad_value_(params, grad_value_clip)
        norms[f"grad_norm_{key}"] = float(
            torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip else float("inf"))
        )

    for opt in stepping.values():
        scaler.step(opt)
    scaler.update()

    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    return norms


def _params_of(optimizer) -> list[Any]:
    return [p for group in optimizer.param_groups for p in group["params"]]
