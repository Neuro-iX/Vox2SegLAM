import math
from collections.abc import Mapping

import torch
import torch.nn as nn

__all__ = ["AdaptiveLossWeighting"]


# homoscedastic-uncertainty weighting (Kendall et al., 2018): each weight is exp(-log_var)
# with log_var learned; the "+ log_var" regulariser in forward() prevents driving every
# weight to zero
class AdaptiveLossWeighting(nn.Module):
    def __init__(
        self,
        init_weights: Mapping[str, float],
        max_weight: Mapping[str, float] | None = None,
        fixed: Mapping[str, float] | None = None,
    ):
        super().__init__()
        if not init_weights:
            raise ValueError("AdaptiveLossWeighting needs at least one term")

        fixed_names = set(fixed) if fixed else set()
        unknown_fixed = fixed_names - set(init_weights)
        if unknown_fixed:
            raise KeyError(
                f"fixed names terms that are not weighted: {sorted(unknown_fixed)}; "
                f"declared terms are {sorted(init_weights)}"
            )

        # lv = -log(w), so exp(-lv) == w at initialisation; `fixed` terms get a plain float
        eps = 1e-12
        log_var = {}
        self.fixed_weights: dict[str, float] = {}
        for name, w in init_weights.items():
            w = float(w)
            if w <= 0.0:
                raise ValueError(f"init weight for {name!r} must be > 0, got {w}")
            if name in fixed_names:
                self.fixed_weights[name] = w
            else:
                log_var[name] = nn.Parameter(torch.tensor(-math.log(w + eps), dtype=torch.float32))
        self.log_var = nn.ParameterDict(log_var)

        # max_weight caps an adaptive term's weight (a floor on lv, since weight = exp(-lv)); a cap on
        # a fixed or unknown term is rejected
        caps = {str(k): float(v) for k, v in (max_weight or {}).items()}
        unknown = sorted(set(caps) - set(log_var))
        if unknown:
            raise KeyError(
                f"max_weight names terms that are not ADAPTIVE: {unknown}; "
                f"adaptive terms are {sorted(log_var)}"
                + (
                    f" ({sorted(set(unknown) & fixed_names)} are fixed)"
                    if set(unknown) & fixed_names
                    else ""
                )
            )
        for name, w in caps.items():
            if w <= 0.0:
                raise ValueError(f"max_weight for {name!r} must be > 0, got {w}")
            init = float(torch.exp(-log_var[name].detach()))
            if init > w:
                raise ValueError(
                    f"max_weight[{name!r}] = {w} is below its initial weight "
                    f"{init:.4g}: the term would start already capped"
                )
        self._lv_floor = {n: -math.log(w) for n, w in caps.items()}

    # a configured term absent from `losses` is skipped (e.g. seg without a head); an
    # unrecognised name is an error, since it would silently go unweighted
    def forward(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        expected = set(self.log_var) | set(self.fixed_weights)
        got = set(losses.keys())
        if not got:
            raise KeyError("no loss terms given; the objective would be empty")
        if not got <= expected:
            raise KeyError(
                f"loss terms {sorted(got - expected)} are not weighted; "
                f"declared terms are {sorted(expected)}"
            )

        total = None
        for name, lv in self.log_var.items():
            if name not in losses:
                continue
            lv = self._effective(name, lv)
            term = torch.exp(-lv) * losses[name] + lv
            total = term if total is None else total + term
        for name, w in self.fixed_weights.items():
            if name not in losses:
                continue
            term = w * losses[name]
            total = term if total is None else total + term
        return total

    def _effective(self, name: str, lv: torch.Tensor) -> torch.Tensor:
        floor = self._lv_floor.get(name)
        return lv if floor is None else lv.clamp(min=floor)

    # current weight per term (fixed unchanged, adaptive from log_var); reported as "w_<name>"
    def weights(self) -> dict[str, float]:
        return {
            **{
                name: float(torch.exp(-self._effective(name, lv.detach())))
                for name, lv in self.log_var.items()
            },
            **self.fixed_weights,
        }
