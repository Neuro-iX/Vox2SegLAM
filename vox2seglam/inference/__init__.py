"""Prediction: the cascade run on a prepared volume, and its results.

`Predictor` and `markups_document` are imported lazily (PEP 562) so that a script can parse
its arguments, print its first line and fail on a bad path without waiting for torch.
`DEFAULT_N_PASSES` is a plain constant and stays eager: argparse needs it to build --passes.
"""

import importlib
from typing import TYPE_CHECKING

from .constants import DEFAULT_N_PASSES

_SOURCE = {"Predictor": "predictor", "Prediction": "predictor", "markups_document": "slicer"}

__all__ = ["Predictor", "Prediction", "DEFAULT_N_PASSES", "markups_document"]

if TYPE_CHECKING:  # for type checkers and editors only
    from .predictor import Prediction, Predictor
    from .slicer import markups_document


def __getattr__(name: str):
    module = _SOURCE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SOURCE))
