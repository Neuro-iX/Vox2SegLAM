from collections.abc import Collection
from typing import Any

__all__ = ["Curriculum"]


# names are stage_parameters() keys ("seg", "gnn1", "gnn2", ..., "refine"). When enabled,
# stages start one at a time in `order`: a stage joins once the newest stage's lr scheduler
# has fired, and retires once its own scheduler bottoms out (optim.at_rate_floor).
# enabled=False starts every stage at once, as ordinary joint training.
class Curriculum:
    def __init__(
        self,
        order: list[str],
        *,
        enabled: bool = True,
    ) -> None:
        if not order:
            raise ValueError("a curriculum needs at least one stage")
        self.order = list(order)
        self.enabled = bool(enabled)

        self.n_started = 1 if self.enabled else len(self.order)
        self.retired: set[str] = set()
        self.lr_at_start: dict[str, float] = {}
        self.epoch_started: dict[str, int] = {name: 1 for name in self.order[: self.n_started]}
        self.epoch_retired: dict[str, int] = {}
        self.started_because: str | None = None
        self.deferred: str | None = None  # stage refused at the last lr cut, if any

    @property
    def started(self) -> list[str]:
        return self.order[: self.n_started]

    # currently training: started, and not yet retired
    @property
    def active(self) -> list[str]:
        return [n for n in self.started if n not in self.retired]

    @property
    def pending(self) -> list[str]:
        return self.order[self.n_started :]

    # the most recently started stage - the one maybe_start_next watches for a plateau
    @property
    def newest(self) -> str:
        return self.order[self.n_started - 1]

    @property
    def finished(self) -> bool:
        return not self.pending and not self.active

    # 0 = not started yet, 1 = active, 2 = retired (used by reporting.py's stage/retire lines)
    def status(self, name: str) -> int:
        if name in self.retired:
            return 2
        return 1 if name in self.started else 0

    # records each active stage's lr when first seen, as the reference for later comparison
    def note_lrs(self, lrs: dict[str, float]) -> None:
        for name in self.active:
            if name not in self.lr_at_start and name in lrs:
                self.lr_at_start[name] = float(lrs[name])

    # watches only the newest stage: each time its ReduceLROnPlateau halves its lr, the next
    # stage in `order` is offered a start. The caller's `ready` gate decides (e.g. gnn2 only once
    # gnn1 beats the template; a stage absent from it, like the refiner, is always ready); a stage
    # not ready waits for the next cut and is recorded in `deferred` for the log
    def maybe_start_next(
        self,
        *,
        epoch: int,
        lrs: dict[str, float],
        ready: dict[str, bool] | None = None,
    ) -> str | None:
        self.deferred = None
        if not self.pending:
            return None

        watched = self.newest
        lr_now = lrs.get(watched)
        lr_ref = self.lr_at_start.get(watched)
        cut = lr_now is not None and lr_ref is not None and lr_now < lr_ref * (1.0 - 1e-9)
        if not cut:
            return None

        next_name = self.order[self.n_started]
        if ready is not None and not ready.get(next_name, True):
            self.lr_at_start[watched] = float(lr_now)  # wait for the next cut
            self.deferred = next_name
            return None

        self.n_started += 1
        name = self.newest
        self.epoch_started[name] = epoch
        self.lr_at_start.pop(name, None)
        self.started_because = "plateau"
        return name

    # marks active stages the caller found exhausted (at_rate_floor) as retired; returns them
    def retire_exhausted(self, *, epoch: int, exhausted: Collection[str]) -> list[str]:
        just = [n for n in self.active if n in exhausted]
        for name in just:
            self.retired.add(name)
            self.epoch_retired[name] = epoch
        return just

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "n_started": int(self.n_started),
            "retired": sorted(self.retired),
            "lr_at_start": dict(self.lr_at_start),
            "epoch_started": dict(self.epoch_started),
            "epoch_retired": dict(self.epoch_retired),
        }

    # tolerant of a changed `order` between save and resume: n_started is clamped, and stage
    # names no longer in `order` are dropped
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.n_started = max(1, min(len(self.order), int(state.get("n_started", 1))))
        if not self.enabled:
            self.n_started = len(self.order)
        self.retired = {n for n in state.get("retired", []) if n in self.started}
        self.epoch_retired = {
            k: int(v) for k, v in state.get("epoch_retired", {}).items() if k in self.retired
        }
        self.lr_at_start = {
            k: float(v) for k, v in state.get("lr_at_start", {}).items() if k in self.order
        }
        self.epoch_started = {
            k: int(v) for k, v in state.get("epoch_started", {}).items() if k in self.order
        }
        for name in self.started:
            self.epoch_started.setdefault(name, 1)
