import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "save",
    "load",
    "new_selection",
    "selection_phase",
    "update_selection",
    "PHASE_CRITERION",
]


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


# Everything needed to resume: model, weighter, optimizer and scheduler state, AMP scaler,
# selection bookkeeping, curriculum state and the main-process RNG. The resume is bit-exact,
# with or without dataloader workers: every epoch re-seeds the loader from (seed, epoch)
# (data/loaders.py reseed_for_epoch); tests/test_resume.py checks it at num_workers 0 and 2.
# The config is not duplicated here: it lives in <exp_dir>/config.yaml, and only its name is
# kept.
def save(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    weighter: torch.nn.Module | None,
    optimizers: dict[str, Any],
    schedulers: dict[str, Any],
    scaler: Any,
    selection: dict[str, Any],
    curriculum: Any,
    config_name: str,
) -> None:
    # written to a temporary file and renamed, so a job stopped mid-write keeps the previous
    # checkpoint
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "weighter": weighter.state_dict() if weighter else None,
            "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
            "schedulers": {k: s.state_dict() for k, s in schedulers.items()},
            "scaler": scaler.state_dict(),
            "selection": dict(selection),
            "curriculum": curriculum.state_dict(),
            "config_name": config_name,
            "rng": _rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)


def load(
    path: Path,
    *,
    device: torch.device,
    model: torch.nn.Module,
    weighter: torch.nn.Module | None,
    optimizers: dict[str, Any],
    schedulers: dict[str, Any],
    scaler: Any,
    selection: dict[str, Any],
    curriculum: Any,
) -> tuple[int, int]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    # the weighter's learned weights only fit the same loss terms as when saved (enabling the
    # protocol term changes their shape); fall back to a fresh weighter rather than crash
    if weighter is not None and ck.get("weighter"):
        try:
            weighter.load_state_dict(ck["weighter"])
        except RuntimeError as exc:
            warnings.warn(
                f"{path}: the adaptive weighter's terms do not match this "
                f"configuration ({exc}); its weights start from "
                "loss.init_w_* instead of resuming.",
                RuntimeWarning,
                stacklevel=2,
            )
    for key, state in ck.get("optimizers", {}).items():
        if key in optimizers:
            optimizers[key].load_state_dict(state)
    for key, state in ck.get("schedulers", {}).items():
        if key in schedulers:
            schedulers[key].load_state_dict(state)
    if ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    selection.update(ck.get("selection", {}))
    if ck.get("curriculum"):
        curriculum.load_state_dict(ck["curriculum"])
    if ck.get("rng"):
        _set_rng_state(ck["rng"])
    return int(ck["epoch"]) + 1, int(ck.get("global_step", 0))


PHASE_CRITERION = {
    1: "dice",
    2: "dice+mre",
    3: "mre",
}  # human-readable label per phase, for reporting.py


def new_selection() -> dict[str, Any]:
    return {"best_dice": -float("inf"), "best_mre": float("inf"), "phase": 1}


# Which metric decides "best checkpoint" changes with training: phase 1 (seg still learning
# fast) Dice alone; phase 2 (seg lr halved) both Dice and MRE must improve; phase 3 (seg lr
# at lm_only_lr, last curriculum stage started) MRE alone. Without a seg head, phase 3 from
# the start.
def selection_phase(
    lr_voxel: float,
    lr_voxel_initial: float,
    lm_only_lr: float,
    *,
    has_seg: bool = True,
    last_stage_started: bool = True,
) -> int:
    if not has_seg:
        return 3
    if lr_voxel <= lm_only_lr and last_stage_started:
        return 3
    if lr_voxel <= lr_voxel_initial / 2:
        return 2
    return 1


# applies selection_phase()'s criterion to this epoch's metrics; "phase" is always
# recorded, best_dice/best_mre advance only when this epoch is the new best
def update_selection(
    selection: dict[str, Any],
    val: dict[str, float],
    *,
    lr_voxel: float,
    lr_voxel_initial: float,
    lm_only_lr: float,
    has_seg: bool = True,
    last_stage_started: bool = True,
) -> bool:
    dice = float(val["dice_mean_no_bg"])
    mre = float(val["lm_mre_vox_mean"])
    phase = selection_phase(
        lr_voxel,
        lr_voxel_initial,
        lm_only_lr,
        has_seg=has_seg,
        last_stage_started=last_stage_started,
    )

    # entering phase 3: the best MRE so far comes from an epoch whose U-Net had not converged,
    # and phase-3 epochs would have to beat it; so phase 3 restarts the comparison and best.pt
    # is chosen among converged-U-Net epochs only
    if phase == 3 and selection.get("phase", 1) < 3 and selection["best_mre"] < float("inf"):
        selection["best_mre"] = float("inf")
        selection["phase3_reset"] = True

    if phase == 1:
        is_best = dice > selection["best_dice"]
    elif phase == 2:
        is_best = dice > selection["best_dice"] and mre < selection["best_mre"]
    else:
        is_best = mre < selection["best_mre"]

    if is_best:
        selection["best_dice"] = dice
        selection["best_mre"] = mre
    selection["phase"] = phase
    return is_best
