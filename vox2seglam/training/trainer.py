import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from ..config import Config, save_config
from ..data import build_dataloaders
from ..data.loaders import reseed_for_epoch
from ..models import Vox2SegLAM
from ..utils.device import to_device
from ..utils.io import ensure_dir, save_json
from ..evaluate import validate
from . import checkpoint
from .checkpoint import new_selection, update_selection
from .init_coords import build_train_init_coords, load_template_std
from .template_setup import ensure_template, place_template
from ..losses import AdaptiveLossWeighting, compute_losses, loss_term_names, weighted_term_names
from .stages import Curriculum
from .optim import apply_gradients, at_rate_floor, build_optimizers, scheduler_metric
from .reporting import epoch_line, log_epoch, retire_line, run_footer, run_header, stage_line
from .seeding import set_seed

__all__ = ["Trainer"]

_AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


# Loss terms that keep a fixed weight instead of a learned one: uncertainty weighting only
# balances terms competing for the same parameters, so terms are grouped by shared parameters
# (a missing stop-gradient in model.lam.detach_*) and any term alone in its group is fixed.
# refine_cls and refine_reg always share the refiner head.
def _fixed_term_names(
    *,
    detach_backbone: bool,
    detach_between_steps: bool,
    detach_graph_for_refiner: bool,
    has_seg: bool,
    n_steps: int,
    has_refine: bool,
) -> set[str]:
    gnn = [f"gnn{i + 1}" for i in range(int(n_steps))]
    refine = ["refine_cls", "refine_reg"] if has_refine else []
    seg = ["seg"] if has_seg else []

    # one group per set of shared parameters; a term in no group is on its own
    groups: list[set[str]] = []
    if not detach_backbone:
        # the backbone is shared by whoever reaches it, also with no segmentation head
        # (stage_parameters folds the U-Net into gnn1)
        groups.append(set(seg + gnn + refine[:1]))
    if not detach_between_steps:
        groups.append(set(gnn))
    if not detach_graph_for_refiner and gnn and refine:
        groups.append({gnn[-1], refine[0]})
    if refine:
        groups.append(set(refine))

    # merge groups that overlap, until none do
    merged: list[set[str]] = []
    for group in groups:
        if len(group) < 2:
            continue
        overlapping = [m for m in merged if m & group]
        for m in overlapping:
            merged.remove(m)
        merged.append(set(group).union(*overlapping))

    shared = set().union(*merged) if merged else set()
    return set(seg + gnn + refine) - shared


def _bf16_warning(cfg: Config, device: torch.device) -> str | None:
    """A message when the configured bfloat16 autocast cannot run natively on this GPU."""
    if device.type != "cuda" or not cfg.train.amp or cfg.train.amp_dtype != "bfloat16":
        return None
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return None
    return (
        "train.amp_dtype is bfloat16, which this GPU does not support natively (compute "
        "capability 8.0, Ampere or newer, is needed): training would run in float32 or "
        "emulation, slower and with more memory. Set train.amp_dtype=float16 instead."
    )


class Trainer:
    def __init__(self, cfg: Config, device: torch.device | None = None) -> None:
        self.cfg = cfg.validate()
        set_seed(self.cfg)

        requested = str(cfg.train.device)
        if device is not None:
            self.device = device
        elif requested.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested)
        else:
            if requested.startswith("cuda"):
                warnings.warn(
                    f"train.device={requested!r} but CUDA is unavailable; falling back to CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.device = torch.device("cpu")
        self.exp_dir = Path(cfg.paths.exp_dir)
        ensure_dir(self.exp_dir)
        if message := _bf16_warning(self.cfg, self.device):
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        # fits whatever template file the model will read and lacks, on the training landmarks
        self.cfg = place_template(self.cfg)
        ensure_template(self.cfg)
        self.model = Vox2SegLAM.from_config(self.cfg, device=self.device).to(self.device)
        if self.cfg.train.init_unet_from:
            from ..config.paths import resolve_repo_path

            self._init_unet_from(resolve_repo_path(self.cfg.train.init_unet_from))
        if self.cfg.train.freeze_unet:
            # no gradient ever reaches these weights, so no optimizer moves them
            for p in self.model.unet.parameters():
                p.requires_grad_(False)
            print("U-Net frozen (train.freeze_unet)", flush=True)
        self.loaders = build_dataloaders(self.cfg, splits=("train", "val"))
        if "train" not in self.loaders:
            raise RuntimeError("the train split is empty; nothing to train on")

        self.loss_cfg = self.cfg.loss
        self.loss_extras = dict(
            num_classes=self.cfg.task.n_classes,
            K=self.cfg.task.n_landmarks,
            deep_supervision=self.cfg.model.seg_head.deep_supervision,
            detach_between_steps=self.cfg.model.lam.detach_between_steps,
        )
        self.template_std = load_template_std(self.cfg, self.device)

        self.loss_terms = loss_term_names(
            steps=self.cfg.model.lam.steps,
            has_seg=bool(self.cfg.model.seg_head.enabled),
            has_refine=bool(self.cfg.model.lam.refine),
        )
        lam_cfg = self.cfg.model.lam
        fixed_terms = _fixed_term_names(
            detach_backbone=bool(lam_cfg.detach_backbone),
            detach_between_steps=bool(lam_cfg.detach_between_steps),
            detach_graph_for_refiner=bool(lam_cfg.detach_graph_for_refiner),
            has_seg=bool(self.cfg.model.seg_head.enabled),
            n_steps=int(lam_cfg.steps),
            has_refine=bool(lam_cfg.refine),
        )
        self.weighter = (
            AdaptiveLossWeighting(
                {
                    name: (self.loss_cfg.init_w_seg if name == "seg" else self.loss_cfg.init_w_lm)
                    for name in weighted_term_names(self.loss_terms)
                },
                max_weight=self.loss_cfg.max_w,
                fixed={
                    name: (self.loss_cfg.init_w_seg if name == "seg" else self.loss_cfg.init_w_lm)
                    for name in weighted_term_names(self.loss_terms)
                    if name in fixed_terms
                },
            ).to(self.device)
            if self.loss_cfg.adaptive_weighting
            else None
        )

        self.optimizers, self.schedulers = build_optimizers(self.model, self.weighter, self.cfg)

        self.curriculum = Curriculum(
            self.loss_terms,
            enabled=bool(self.cfg.staging.enabled),
        )
        # The device string stays "cuda": autocast("cpu") would *enable* bfloat16 on the CPU
        # and change the numbers. Only `enabled` is gated, so a CPU run is plain fp32 -- which
        # is what torch does anyway, except that it says so once per call. Same at the autocast
        # below, and the predictor already works this way.
        self.scaler = GradScaler("cuda", enabled=bool(cfg.train.amp) and self.device.type == "cuda")
        self.amp_dtype = _AMP_DTYPES[cfg.train.amp_dtype]

        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "tb"))
        self.landmark_ids = self.cfg.task.resolved_landmark_ids()

        self.start_epoch = 1
        self.global_step = 0
        self.selection = new_selection()
        self.lr_voxel_initial = float(self.cfg.optim.lr_voxel)

    def _stepping_keys(self) -> set[str]:
        return set(self.curriculum.active) | {"weighter"}

    def _stage_state(self) -> dict[str, float]:
        return {
            f"stage_{name}": float(self.curriculum.status(name)) for name in self.curriculum.order
        }

    def _weighter_state(self) -> dict[str, float]:
        if self.weighter is None:
            return {}
        return {f"w_{name}": w for name, w in self.weighter.weights().items()}

    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        cfg = self.cfg
        accum = max(1, int(cfg.train.grad_accum))
        running: dict[str, float] = {}
        running_vec: dict[str, torch.Tensor] = {}
        n_batches = 0
        n_steps = 0

        # The last accumulation window is short when the epoch's batch count is not a multiple of
        # grad_accum; it still takes a step (scaled by 1/accum like the others, so it weighs less)
        # instead of being dropped by the next epoch's zero_grad.
        n_total_batches = len(self.loaders["train"])

        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

        # only the terms of active (started, not retired) stages are optimised; the
        # others are still computed and logged by compute_losses
        active_terms = set(self.curriculum.active)

        for it, raw in enumerate(self.loaders["train"]):
            batch = to_device(raw, self.device)
            x = batch["image"]
            if x.ndim == 4:
                x = x.unsqueeze(0)

            coords_init = build_train_init_coords(
                mean_coords_zyx=self.model.landmark_graph.mean_coords_zyx,
                batch_size=x.shape[0],
                device=self.device,
                std_coords=self.template_std,
                clamp_coords=cfg.model.lam.clamp_coords,
                patch_size=tuple(cfg.task.patch_size),
            )

            with autocast(
                "cuda",
                dtype=self.amp_dtype,
                enabled=bool(cfg.train.amp) and self.device.type == "cuda",
            ):
                pred = self.model(x, coords_init_zyx=coords_init)

                parts, total = compute_losses(
                    pred=pred,
                    batch=batch,
                    cfg=self.loss_cfg,
                    clamp_coords=cfg.model.lam.clamp_coords,
                    weighter=self.weighter,
                    coord_std_vox=self.template_std,
                    active_terms=active_terms,
                    **self.loss_extras,
                )

            self.scaler.scale(total / accum).backward()

            if (it + 1) % accum == 0 or (it + 1) == n_total_batches:
                norms = apply_gradients(
                    self.optimizers,
                    self.scaler,
                    grad_clip=cfg.train.grad_clip,
                    grad_value_clip=cfg.train.grad_value_clip,
                    step_keys=self._stepping_keys(),
                )
                for k, v in norms.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1
                self.global_step += 1

            # compute_losses already includes "total" in parts; stacking it again doubled the logged
            # train loss (the backward used `total` itself)
            scalar_keys = [k for k, v in parts.items() if v.ndim == 0]
            scalars = torch.stack([parts[k].detach() for k in scalar_keys]).cpu()
            for k, val in zip(scalar_keys, scalars.tolist()):
                running[k] = running.get(k, 0.0) + val

            for k, v in parts.items():
                if v.ndim != 0:
                    d = v.detach().float().cpu()
                    running_vec[k] = running_vec[k] + d if k in running_vec else d
            n_batches += 1

        n = max(1, n_batches)
        out = {
            k: v / (n_steps if k.startswith("grad_norm_") and n_steps else n)
            for k, v in running.items()
        }
        for k, vec in running_vec.items():
            for i, val in enumerate((vec / n).tolist()):
                out[f"{k}/lm{i + 1}"] = val
        return out

    def validate(self) -> dict[str, float]:
        return validate(
            model=self.model,
            loader=self.loaders["val"],
            device=self.device,
            n_v_classes=self.cfg.task.n_classes,
            landmark_ids=self.landmark_ids,
        )

    def _init_unet_from(self, path: str) -> None:
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"init_unet_from: no checkpoint at {src}")
        blob = torch.load(src, map_location="cpu", weights_only=False)
        state = blob.get("model", blob)

        target = self.model.unet.state_dict()
        taken, skipped = {}, []
        for k, v in state.items():
            if not k.startswith("unet."):
                continue
            kk = k[len("unet.") :]
            if kk not in target:
                skipped.append(kk)
                continue
            if tuple(target[kk].shape) != tuple(v.shape):
                skipped.append(f"{kk} (shape {tuple(v.shape)} != {tuple(target[kk].shape)})")
                continue
            taken[kk] = v

        missing = [k for k in target if k not in taken]
        self.model.unet.load_state_dict(taken, strict=False)
        print(
            f"init_unet_from {src}: {len(taken)}/{len(target)} tensors loaded, "
            f"{len(missing)} left at their initialisation",
            flush=True,
        )
        if skipped:
            print(
                f"  not loaded (shape or name mismatch): {skipped[:8]}"
                f"{' ...' if len(skipped) > 8 else ''}",
                flush=True,
            )
        if not taken:
            raise RuntimeError(
                f"init_unet_from: no backbone tensor loaded from {src}. The checkpoint "
                "has no 'unet.' prefix -- check that it comes from this pipeline."
            )

    # curriculum start gates: {stage: may start} and, for the log, why; gnnK needs the
    # validation MRE after gnn(K-1) below the template's
    def _stage_gates(self, val: dict[str, float]) -> tuple[dict[str, bool], dict[str, str]]:
        st = self.cfg.staging
        ready: dict[str, bool] = {}
        why: dict[str, str] = {}
        if st.gnn_must_beat_template:
            init = val.get("lm_mre_vox_mean_init", float("nan"))
            for k in range(2, int(self.cfg.model.lam.steps) + 1):
                prev = val.get(f"lm_mre_vox_mean_step{k - 1}", float("nan"))
                ready[f"gnn{k}"] = prev < init  # False if either is nan
                why[f"gnn{k}"] = f"MRE after gnn{k - 1} {prev:.3f} >= template {init:.3f}"
        return ready, why

    def _update_selection(self, val: dict[str, float], lr_voxel: float) -> bool:
        return update_selection(
            self.selection,
            val,
            lr_voxel=lr_voxel,
            lr_voxel_initial=self.lr_voxel_initial,
            lm_only_lr=self.cfg.train.selection_lm_only_lr,
            has_seg=bool(self.cfg.model.seg_head.enabled),
            last_stage_started=self.curriculum.status(self.curriculum.order[-1]) >= 1,
        )

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        checkpoint.save(
            path,
            epoch=epoch,
            global_step=self.global_step,
            model=self.model,
            weighter=self.weighter,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            scaler=self.scaler,
            selection=self.selection,
            curriculum=self.curriculum,
            config_name=self.cfg.name,
        )

    def load_checkpoint(self, path: Path) -> None:
        self.start_epoch, self.global_step = checkpoint.load(
            path,
            device=self.device,
            model=self.model,
            weighter=self.weighter,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            scaler=self.scaler,
            selection=self.selection,
            curriculum=self.curriculum,
        )

    def fit(self) -> dict[str, Any]:
        cfg = self.cfg
        if cfg.train.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        # config.yaml is what Predictor rebuilds best.pt from: written on a fresh run only; a resume
        # under another config keeps it and records the new one beside it
        sidecar = self.exp_dir / "config.yaml"
        if self.start_epoch == 1 or not sidecar.exists():
            save_config(self.cfg, sidecar)
        else:
            resumed = self.exp_dir / f"config.resume_epoch{self.start_epoch}.yaml"
            save_config(self.cfg, resumed)
            if resumed.read_text(encoding="utf-8") == sidecar.read_text(encoding="utf-8"):
                resumed.unlink()
            else:
                warnings.warn(
                    f"resuming with a config that differs from {sidecar}; "
                    f"the new one is saved as {resumed.name}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        print(run_header(str(self.device), self.start_epoch, int(cfg.train.epochs)), flush=True)

        last_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, int(cfg.train.epochs) + 1):
            reseed_for_epoch(self.loaders["train"], int(cfg.repro.seed), epoch)
            started = time.perf_counter()
            losses = self.train_one_epoch()
            epoch_seconds = time.perf_counter() - started
            last_epoch = epoch

            if epoch % max(1, int(cfg.train.eval_every)) != 0:
                self.save_checkpoint(self.exp_dir / "last.pt", epoch)
                continue

            val = self.validate()

            has_seg = bool(self.cfg.model.seg_head.enabled)
            stepping = self._stepping_keys()
            for key, sched in self.schedulers.items():
                if key in stepping:
                    sched.step(scheduler_metric(key, val, has_seg=has_seg))

            lrs = {k: o.param_groups[0]["lr"] for k, o in self.optimizers.items()}
            self.curriculum.note_lrs(lrs)

            exhausted = {
                key
                for key in self.curriculum.active
                if key in self.schedulers
                and at_rate_floor(self.schedulers[key], self.optimizers[key])
            }
            for name in self.curriculum.retire_exhausted(epoch=epoch, exhausted=exhausted):
                print(retire_line(name, self.curriculum, epoch), flush=True)

            ready, why = self._stage_gates(val)
            started = self.curriculum.maybe_start_next(epoch=epoch, lrs=lrs, ready=ready)
            if started is not None:
                print(stage_line(started, self.curriculum, epoch), flush=True)
            elif self.curriculum.deferred is not None:
                name = self.curriculum.deferred
                print(
                    f"  ->  epoch {epoch}: stage {name} not started ({why[name]}); "
                    f"waiting for the next {self.curriculum.newest} lr cut",
                    flush=True,
                )

            lr_unet = lrs[self.curriculum.order[0]]
            lr_lam = lrs.get(self.curriculum.newest, lr_unet)
            is_best = self._update_selection(val, lr_unet)
            if is_best:
                self.selection["best_epoch"] = epoch
                self.selection["best_phase"] = self.selection["phase"]
            self.save_checkpoint(self.exp_dir / "last.pt", epoch)
            if is_best:
                self.save_checkpoint(self.exp_dir / "best.pt", epoch)

            log_epoch(
                self.writer,
                epoch,
                losses,
                val,
                {
                    "lr_unet": lr_unet,
                    "lr_lam": lr_lam,
                    "amp_scale": float(self.scaler.get_scale()),
                    "selection_phase": float(self.selection["phase"]),
                    "epoch_seconds": epoch_seconds,
                    **{f"lr_{k}": v for k, v in lrs.items()},
                    **self._stage_state(),
                    **self._weighter_state(),
                },
            )

            print(
                epoch_line(
                    epoch, int(cfg.train.epochs), val, self.selection, lr_unet, lr_lam, is_best
                ),
                flush=True,
            )

            # also stops when every started stage retired and the next is still gated
            if not self.curriculum.active:
                print(
                    f"  ->  epoch {epoch}: no stage left training, stopping "
                    f"{int(cfg.train.epochs) - epoch} epochs early",
                    flush=True,
                )
                break

        print(run_footer(self.selection, last_epoch), flush=True)
        self.writer.close()
        save_json(
            self.exp_dir / "final.json",
            {
                "config_name": cfg.name,
                "epochs_planned": int(cfg.train.epochs),
                "epochs_run": last_epoch,
                "selection": dict(self.selection),
            },
        )
        return dict(self.selection)
