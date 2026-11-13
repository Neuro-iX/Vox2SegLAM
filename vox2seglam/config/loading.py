import ast
import dataclasses as dc
import difflib
import functools
import json
import typing
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from .paths import PACKAGE_DIR
from .root import BASE, Config

#: the recipes installed with the package; a working directory may shadow them
SHIPPED_EXPERIMENTS = PACKAGE_DIR / "configs" / "experiments"


def parse_set(items: list[str]) -> dict[str, Any]:
    """["train.epochs=5", "paths.data_root=/d"] -> {"train": {"epochs": 5}, ...}.

    Values are Python literals (true/false/none in any case); what does not parse stays a string.
    """
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        low = raw.strip().lower()
        if low in ("true", "false"):
            value: Any = low == "true"
        elif low in ("none", "null"):
            value = None
        else:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
        node = overrides
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


# Applies nested overrides to a config dataclass, checking every key; returns a new instance.
def _apply_overrides(node: Any, overrides: dict[str, Any], path: str = "") -> Any:
    if not dc.is_dataclass(node):
        raise TypeError(f"{path or 'config'} is not a config section")

    valid = {f.name: f for f in dc.fields(node)}
    hints = _resolved_hints(type(node))
    changes: dict[str, Any] = {}

    for key, value in overrides.items():
        where = f"{path}.{key}" if path else key
        if key not in valid:
            raise KeyError(f"unknown config key {where!r}. Valid keys here: {sorted(valid)}")
        current = getattr(node, key)
        if isinstance(value, dict) and dc.is_dataclass(current):
            changes[key] = _apply_overrides(current, value, where)
            continue
        if dc.is_dataclass(current):
            raise TypeError(
                f"{where} is a config section, not a value: got {value!r}. "
                f"Set one of its fields instead ({where}.<field>=...)."
            )
        # before the list -> tuple conversion, so a sequence given to a scalar field is caught
        _check_value(where, value, hints.get(key, valid[key].type))
        changes[key] = (
            tuple(tuple(v) if isinstance(v, list) else v for v in value)
            if isinstance(value, list)
            else value
        )

    return dc.replace(node, **changes)


@functools.lru_cache(maxsize=None)
def _resolved_hints(cls: type) -> dict[str, Any]:
    """{field: resolved annotation} of a config section."""
    return typing.get_type_hints(cls)


# Concrete types an annotation accepts: unions flattened, generics reduced to their origin
# (tuple[int, ...] -> tuple). Built from resolved annotations, not their text, which let
# `dict[str, float]` accept a string because "str" appears in it.
def _accepted_types(declared: Any) -> tuple[type, ...]:
    origin = get_origin(declared)
    if origin in (Union, UnionType):
        return tuple(t for arg in get_args(declared) for t in _accepted_types(arg))
    target = origin or declared
    return (target,) if isinstance(target, type) else ()


# A --set value goes through literal_eval, so a typo changes its type instead of failing
# ("0,35" becomes the tuple (0, 35)); checking it against the field's type stops the run
# at the command line.
def _check_value(where: str, value: Any, declared: Any) -> None:
    accepted = _accepted_types(declared)
    spelled = getattr(declared, "__name__", None) or str(declared)
    if not accepted:  # unresolvable annotation: do not guess
        return

    def check(ok: bool, hint: str = "") -> None:
        if not ok:
            raise TypeError(
                f"{where}={value!r} is a {type(value).__name__}, but {where} is declared "
                f"{spelled}{hint}"
            )

    if value is None:
        check(type(None) in accepted, " and is not optional")
    elif isinstance(value, (tuple, list)):
        # YAML and --set spell a tuple as a list
        check(tuple in accepted, " -- a decimal comma reads as a tuple ('0,35' -> (0, 35))")
    elif tuple in accepted and len(accepted) == 1:
        check(False, " -- pass a sequence as [a, b, ...]")
    elif isinstance(value, bool):  # before int: bool is a subclass of int
        check(bool in accepted)
    elif isinstance(value, int):  # an int stands in for a float; nothing else widens
        check(bool({int, float}.intersection(accepted)))
    elif isinstance(value, float):
        check(float in accepted)
    elif isinstance(value, str):
        check(str in accepted)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise ImportError(
            "PyYAML is required to load YAML experiment configs "
            "(pip install pyyaml), or build a Config in Python instead."
        ) from exc
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return loaded


# A bare name ("baseline"), a grouped one ("augmentation/no_rc") or a path -> a file. The
# recipes ship with the package, so a name resolves from an install as well as from a clone;
# a configs/experiments/ in the working directory is searched first, which is how a user
# overrides a shipped recipe without editing one. A bare name must be unique.
def _resolve_experiment_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"no config file at {str(candidate)!r}. Pass a bare experiment name "
            "to search configs/experiments/ instead of a path that does not exist."
        )

    search_roots = [
        Path.cwd() / "configs" / "experiments",
        SHIPPED_EXPERIMENTS,
    ]
    wanted = str(name_or_path).strip("/")
    found: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        hits = sorted(
            p
            for p in root.rglob("*")
            if p.suffix in (".yaml", ".yml")
            and (rel := p.relative_to(root).with_suffix("").as_posix())
            and (rel == wanted or rel.endswith("/" + wanted))
        )
        if len(hits) > 1:
            listed = "\n  ".join(str(p.relative_to(root)) for p in hits)
            raise ValueError(
                f"experiment name {str(name_or_path)!r} is ambiguous, it matches:\n  "
                f"{listed}\nPass it with its folder, e.g. 'folder/name'."
            )
        found += hits

    # the cwd is searched first so recipes can sit next to the data, but a name found in
    # both places must not silently shadow the shipped one
    if len(found) > 1 and found[0].resolve() != found[1].resolve():
        raise ValueError(
            f"experiment name {str(name_or_path)!r} exists both at {found[0]} and "
            f"{found[1]}. Pass the one you mean as an explicit path."
        )
    if found:
        return found[0]

    # cwd and repository coincide in a clone: list each root once
    searched = "\n  ".join(dict.fromkeys(str(r) for r in search_roots))
    # the recipes are package data: if they are missing the install itself is broken, which
    # is a different problem from a misspelt name
    if not SHIPPED_EXPERIMENTS.is_dir():
        raise FileNotFoundError(
            f"no experiment config named {name_or_path!r}, and the recipes that ship with "
            f"the package are not where they should be ({SHIPPED_EXPERIMENTS}). Reinstall "
            "it, pass an explicit path to a .yaml file, or build the configuration in "
            "Python -- Config() is the published model.\n"
            f"Searched:\n  {searched}"
        )
    known = sorted(
        p.relative_to(SHIPPED_EXPERIMENTS).with_suffix("").as_posix()
        for p in SHIPPED_EXPERIMENTS.rglob("*")
        if p.suffix in (".yaml", ".yml")
    )
    wanted_lower = str(name_or_path).lower()
    close = [k for k in known if wanted_lower in k.lower()]
    if not close:
        close = difflib.get_close_matches(
            str(name_or_path), [Path(k).name for k in known], n=8, cutoff=0.5
        )
    hint = (
        ("\nDid you mean:\n  " + "\n  ".join(close[:8]))
        if close
        else (
            f"\n{len(known)} recipes ship with the package ({', '.join(known)}); a bare name "
            "must match one of them, or a file under configs/experiments/ in the working "
            "directory."
        )
    )
    raise FileNotFoundError(
        f"no experiment config named {name_or_path!r}. Searched:\n  {searched}{hint}"
    )


def with_overrides(cfg: Config, overrides: dict[str, Any] | None) -> Config:
    """`cfg` with overrides applied and validated, for a configuration that already exists
    (one recovered from a checkpoint) where load_config does not apply."""
    return _apply_overrides(cfg, overrides).validate() if overrides else cfg


def load_config(
    name_or_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    _seen: list[str] | None = None,
) -> Config:
    # No name loads BASE; a name loads its YAML on top of its "extends" parent, then the
    # caller's overrides. _seen tracks the chain to catch cycles; only the outermost call
    # validates.
    is_root = _seen is None

    if name_or_path is None:
        cfg = BASE
        name = "base"
    else:
        path = _resolve_experiment_path(name_or_path)
        seen = list(_seen or [])
        key = str(path.resolve())
        if key in seen:
            chain = " -> ".join(seen + [key])
            raise ValueError(f"circular 'extends' in experiment configs: {chain}")
        seen.append(key)

        payload = _read_yaml(path)
        parent = payload.pop("extends", None)
        name = payload.pop("name", path.stem)

        cfg = load_config(parent, _seen=seen) if parent else BASE
        cfg = _apply_overrides(cfg, payload)
        cfg = dc.replace(cfg, name=name)

    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    return cfg.validate() if is_root else cfg


def config_to_dict(cfg: Any) -> Any:
    if dc.is_dataclass(cfg):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in dc.fields(cfg)}
    if isinstance(cfg, (tuple, list)):
        return [config_to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {str(k): config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg


# YAML or JSON by file extension.
def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config_to_dict(cfg)

    if path.suffix in (".yaml", ".yml"):
        import yaml

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
