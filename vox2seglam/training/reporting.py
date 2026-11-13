from typing import Any

from .checkpoint import PHASE_CRITERION

__all__ = ["log_epoch", "run_header", "epoch_line", "stage_line", "retire_line", "run_footer"]

# shortens a few long raw metric-dict keys to TensorBoard-friendly names
_ALIAS = {
    "lm_mre_vox_mean": "mre",
    "dice_mean_no_bg": "dice",
    "dice_mean_bg": "dice_with_bg",
    "lm_mre_vox_mean_gnn_last": "mre_graph",
    "lm_mre_vox_mean_init": "mre_template",
    "lm_mre_vox_mean_final_step": "mre_final",
    "lm_mre_local_head_gain": "mre_refiner_gain",
}

_LOSS_TERMS = {"total", "seg", "lm", "lm_gnn", "dice", "ce", "wing"}


def _is_loss_term(name: str) -> bool:
    return name in _LOSS_TERMS or name.startswith("lm_")


# routes a metric to the TensorBoard group it is diagnosed from. "dice" is a loss term in
# train (-> loss/) but the eval metric in val (-> seg/): same name, different meaning
def _group(name: str, phase: str) -> str:
    if name.startswith("grad_norm"):
        return "optim"
    if name.startswith("refine_"):
        return "refine"
    if name.startswith("w_"):
        return "weight"
    if phase == "train" and _is_loss_term(name):
        return "loss"
    if name.startswith("dice"):
        return "seg"
    if "mre" in name:
        return "landmark"
    return "loss"


# drops the group's prefix (and verbose "lm_mre_vox_*" patterns) to keep scalar names short
def _short(name: str, group: str) -> str:
    name = _ALIAS.get(name, name)
    name = name.replace("lm_mre_vox_mean_", "mre_").replace("lm_mre_vox_", "mre_")
    prefix = f"{group}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


# final tag "<group>/<short name>_<phase>[/<suffix>]"; the suffix carries a per-class or
# per-landmark index
def _tag(name: str, phase: str) -> str:
    head, _, tail = name.partition("/")
    group = _group(head, phase)
    tag = f"{group}/{_short(head, group)}_{phase}"
    return f"{tag}/{tail}" if tail else tag


# every float train/val metric under its routed tag, plus the optim/ group (grad norms, lrs,
# AMP scale) verbatim
def log_epoch(
    writer,
    epoch: int,
    train: dict[str, Any],
    val: dict[str, Any],
    optim: dict[str, float],
) -> None:
    for phase, values in (("train", train), ("val", val)):
        for name, value in values.items():
            if isinstance(value, float):
                writer.add_scalar(_tag(name, phase), value, epoch)
    for name, value in optim.items():
        writer.add_scalar(f"optim/{name}", value, epoch)


def run_header(device: str, first_epoch: int, last_epoch: int) -> str:
    return f"{device} | epochs {first_epoch}-{last_epoch}"


# console line when Curriculum.maybe_start_next() starts a stage, with the lr cut that
# triggered it
def stage_line(started: str, curriculum: Any, epoch: int) -> str:
    why = getattr(curriculum, "started_because", None)
    reason = {
        "plateau": f"{curriculum.order[curriculum.n_started - 2]} lr cut",
    }.get(why, why or "")
    pending = curriculum.pending
    tail = f" | pending {'+'.join(pending)}" if pending else " | all stages training"
    return f"  ->  epoch {epoch}: stage {started} starts training ({reason}){tail}"


def retire_line(name: str, curriculum: Any, epoch: int) -> str:
    still = curriculum.active
    tail = f" | still training {'+'.join(still)}" if still else " | nothing left training"
    return f"  ->  epoch {epoch}: stage {name} retires at its rate floor, frozen{tail}"


# one-line-per-epoch console summary: headline dice/mre, both lrs, the active selection
# criterion, and whether this epoch is a new best
def epoch_line(
    epoch: int,
    last_epoch: int,
    val: dict[str, float],
    selection: dict[str, Any],
    lr_unet: float,
    lr_lam: float,
    is_best: bool,
) -> str:
    width = len(str(last_epoch))
    return (
        f"epoch {epoch:>{width}}/{last_epoch}"
        f"   dice {val['dice_mean_no_bg']:.4f}"
        f"   mre {val['lm_mre_vox_mean']:.3f}"
        f"   lr_unet {lr_unet:.1e}"
        f"   lr_lam {lr_lam:.1e}"
        f"   select {PHASE_CRITERION[selection['phase']]}" + ("   best" if is_best else "")
    )


def run_footer(selection: dict[str, Any], last_epoch: int) -> str:
    return (
        f"done after {last_epoch} epochs"
        f" | best.pt: epoch {selection.get('best_epoch', '?')}"
        f", dice {selection['best_dice']:.4f}"
        f", mre {selection['best_mre']:.3f}"
        f" | selected on {PHASE_CRITERION[selection.get('best_phase', selection['phase'])]}"
    )
