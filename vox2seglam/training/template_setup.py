"""Builds the landmark template a training run needs when template_dir lacks it."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data.datasets import read_split
from ..data.landmarks import as_integer_labels, extract_landmarks
from .topology import build_topology
from ..utils.io import load_nifti, to_zyx

__all__ = ["ensure_template", "training_landmark_coords", "place_template"]

MEAN_FILE = "mean_landmark_coords_voxel_zyx.npy"


def training_landmark_coords(cfg: Config) -> tuple[np.ndarray, list[str]]:
    """(n, K, 3) ZYX voxel coordinates of the training subjects' landmarks."""
    ids = read_split(cfg.paths.resolve_split("train"))
    lm_dir = Path(cfg.paths.resolve_landmark_dir())
    landmark_ids = cfg.task.resolved_landmark_ids()
    X = []
    for sid in ids:
        arr, _ = load_nifti(lm_dir / f"{sid}{cfg.paths.landmark_suffix}")
        X.append(extract_landmarks(as_integer_labels(np.rint(to_zyx(arr))), landmark_ids))
    return np.stack(X).astype(np.float64), ids


def _record(tdir: Path, name: str, detail: dict) -> None:
    path = tdir / "auto_template.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    log[name] = {**detail, "written": datetime.now(UTC).isoformat(timespec="seconds")}
    path.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")


def fit_identity(cfg: Config) -> dict:
    """What a template is fitted on: the training subjects (count, split file, sha1 of the
    ordered id list) and the landmark files."""
    split = cfg.paths.resolve_split("train")
    ids = read_split(split)
    return {
        "fit_subjects": len(ids),
        "fit_split": Path(split).name,
        "fit_ids_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest(),
        "landmark_dir": Path(cfg.paths.resolve_landmark_dir()).name,
        "landmark_suffix": cfg.paths.landmark_suffix,
    }


def _check_existing(tdir: Path, files: list[str], identity: dict, lam) -> None:
    """A reused template file must come from the same training subjects and landmarks."""
    log_path = tdir / "auto_template.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {}
    for name in files:
        rec = log.get(name)
        if rec is None:
            raise RuntimeError(
                f"{tdir / name} has no record in {log_path.name}: its provenance "
                "cannot be checked. Delete it to have it rebuilt from the training split."
            )
        diff = {k: (rec.get(k), v) for k, v in identity.items() if rec.get(k) != v}
        if diff:
            raise RuntimeError(
                f"{tdir / name} was fitted on other data than this run "
                f"(recorded vs current: {diff}). Delete it, or set paths.template_dir."
            )
        if (
            name.startswith("edge_index_")
            and lam.graph_lambda != "auto"
            and float(rec.get("lambda", float("nan"))) != float(lam.graph_lambda)
        ):
            raise RuntimeError(
                f"{tdir / name}: built with lambda {rec.get('lambda')}, "
                f"model.lam.graph_lambda is {lam.graph_lambda}"
            )


def _usable(tdir: Path, cfg: Config) -> bool:
    """Whether tdir already holds every file this config needs, fitted on this run's subjects."""
    lam = cfg.model.lam
    kind, edge_file = lam.edge_source()
    wanted = (
        [MEAN_FILE]
        + ([lam.template_std_filename] if lam.perturb_template else [])
        + ([edge_file] if kind == "template" else [])
    )
    if not all((tdir / f).is_file() for f in wanted):
        return False
    try:
        _check_existing(tdir, wanted, fit_identity(cfg), lam)
    except RuntimeError:
        return False
    return True


def place_template(cfg: Config, quiet: bool = False) -> Config:
    """Decides where this run's template lives, and returns the config that says so.

    Left to the default, the template is looked for among the ones shipped with the package
    (named after the training split and the landmark files) and used if it was fitted on
    exactly this run's subjects. Otherwise it is fitted into <exp_dir>/template: the package
    directory is never written to, and a split that happens to be called train_ids.txt cannot
    be mistaken for the published one.
    """
    if cfg.paths.template_dir:
        return cfg
    shipped = Path(cfg.paths.resolve_template_dir())
    if _usable(shipped, cfg):
        return cfg
    local = (Path(cfg.paths.exp_dir) / "template").resolve()
    if shipped.is_dir() and not quiet:
        print(
            f"template: {shipped.name} ships with the package but was fitted on other "
            f"subjects than this run's training split; fitting one in {local}",
            flush=True,
        )
    return replace(cfg, paths=replace(cfg.paths, template_dir=str(local)))


def ensure_template(cfg: Config, quiet: bool = False) -> dict:
    """Write the template files this config needs and template_dir lacks, after checking the
    provenance of those it reuses; returns what was written."""
    lam = cfg.model.lam
    tdir = Path(cfg.paths.resolve_template_dir())
    kind, edge_file = lam.edge_source()
    wanted = {"mean": MEAN_FILE}
    if lam.perturb_template:
        wanted["std"] = lam.template_std_filename
    if kind == "template":
        wanted["edges"] = edge_file
    need = {k: f for k, f in wanted.items() if not (tdir / f).is_file()}
    identity = fit_identity(cfg)
    _check_existing(tdir, [f for k, f in wanted.items() if k not in need], identity, lam)
    if not need:
        return {}

    X, ids = training_landmark_coords(cfg)
    common = {**identity, "frame": "voxel index, ZYX"}
    # everything is computed before anything is written: a fit that fails part-way (too few
    # subjects for the graph, say) must not leave a half-built template behind for the next
    # run to trip over
    res = (
        build_topology(X, select=lam.graph_topology, lam=lam.graph_lambda)
        if "edges" in need
        else None
    )
    tdir.mkdir(parents=True, exist_ok=True)
    if "mean" in need:
        np.save(tdir / MEAN_FILE, X.mean(0).astype(np.float32))
        _record(tdir, MEAN_FILE, common)
    if "std" in need:
        np.save(tdir / need["std"], X.std(0, ddof=0).astype(np.float32))
        _record(tdir, need["std"], {**common, "std_ddof": 0})
    if "edges" in need:
        torch.save(torch.from_numpy(res["edge_index"]), tdir / need["edges"])
        _record(
            tdir, need["edges"], {**common, "graph_lambda": lam.graph_lambda, **res["provenance"]}
        )
    if not quiet:
        msg = ", ".join(need.values())
        extra = ""
        if "edges" in need:
            p = res["provenance"]
            extra = f" ({p['n_edges']} edges, lambda {p['lambda']:.4g} by {p['lambda_selection']['criterion']})"
        print(
            f"template: built {msg} in {tdir} from {len(ids)} training subjects{extra}", flush=True
        )
    return need
