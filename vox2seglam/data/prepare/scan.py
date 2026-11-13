"""Prepares one scan for the model: registration to the reference grid, skull stripping, N4."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import SimpleITK as sitk

from ... import assets
from .n4 import n4
from .register import PreparedScan, to_reference_grid
from .synthstrip import stripper_identity, synthstrip

__all__ = [
    "prepare_scan",
    "prepare_if_needed",
    "prepare_all",
    "preparation_status",
    "pending",
    "stale_reason",
    "UNREADABLE",
]

# what a scan that cannot be read or prepared raises (SimpleITK errors are RuntimeErrors)
UNREADABLE = (ValueError, RuntimeError, OSError)

ALREADY_STRIPPED = "already-stripped"


def _stat(path: Path) -> dict:
    info = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _provenance(path, reference, cfg, script, already_stripped: bool) -> dict:
    pre = cfg.preprocess
    return {
        "source": _stat(path),
        "reference": _stat(reference),
        "stripper": ALREADY_STRIPPED if already_stripped else stripper_identity(cfg, script),
        "settings": {
            "border_mm": pre.synthstrip_border_mm,
            "n4_shrink": pre.n4_shrink_factor,
            "n4_iterations": list(pre.n4_iterations),
        },
    }


def stale_reason(
    scan: PreparedScan, path, reference, cfg, script=None, *, already_stripped: bool = False
) -> str | None:
    """Why this preparation cannot be reused, or None if it can."""
    if not scan.complete:
        return "an earlier preparation did not finish"
    try:
        recorded = json.loads(scan.record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "its record is unreadable"
    current = _provenance(path, reference, cfg, script, already_stripped)
    if recorded.get("source") != current["source"]:
        return "the input file has changed"
    if recorded.get("reference") != current["reference"]:
        return "it was registered onto a different reference"
    if recorded.get("stripper") != current["stripper"]:
        return f"it was skull-stripped another way ({recorded.get('stripper')})"
    if recorded.get("settings") != current["settings"]:
        return "the preprocessing settings have changed"
    return None


def prepare_scan(
    path: Path,
    reference: Path,
    out_dir: Path,
    cfg,
    script=None,
    *,
    already_stripped: bool = False,
    keep_intermediates: bool = False,
    stage=None,
) -> PreparedScan:
    """Register, skull-strip and bias-correct one scan, unconditionally.

    stage(name): called before each step begins. Preparation takes the best part of a minute
    per scan and the steps are silent, so a caller that has a terminal says what is running
    rather than leaving it blank.
    keep_intermediates: keep the registered and the stripped volume. They are inputs to the
    next step and nothing reads them afterwards, so by default only what inference and the
    mapping back need is left behind (one volume instead of three, ~31 MB instead of ~109).
    """
    announce = stage or (lambda _name: None)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = PreparedScan.under(out_dir, path)
    out.record.unlink(missing_ok=True)  # the record is what says the rest is whole
    announce("registering onto the reference grid")
    aligned, transform = to_reference_grid(path, reference)
    sitk.WriteImage(aligned, str(out.aligned))
    sitk.WriteTransform(transform, str(out.transform))
    if already_stripped:
        # the caller stripped it themselves: carry the registered volume forward unchanged
        sitk.WriteImage(aligned, str(out.brain))
    else:
        announce("skull-stripping (SynthStrip)")
        synthstrip(out.aligned, out.brain, cfg, script)
        # an all-zero result is a failed strip; N4 and the z-score would accept it silently
        # (statistics, not GetArrayViewFromImage: a view of an inline-read image points at
        # freed memory, which segfaults on a full-size volume)
        stripped = sitk.StatisticsImageFilter()
        stripped.Execute(sitk.ReadImage(str(out.brain)))
        if stripped.GetMaximum() <= 0:
            raise SystemExit(
                f"{Path(path).name}: skull stripping left nothing. The volume is probably not "
                "registered onto the reference grid as expected -- check --reference."
            )
    announce("correcting the bias field (N4)")
    n4(out.brain, out.model_input, cfg)
    if not keep_intermediates:
        out.aligned.unlink(missing_ok=True)
        out.brain.unlink(missing_ok=True)

    record = _provenance(path, reference, cfg, script, already_stripped)
    record["written"] = datetime.now(UTC).isoformat(timespec="seconds")
    partial = out.record.with_suffix(".json.part")
    partial.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, out.record)  # atomic, and last
    return out


def preparation_status(
    path: Path,
    reference: Path,
    out_dir: Path,
    cfg,
    script=None,
    *,
    already_stripped: bool = False,
    force: bool = False,
) -> str | None:
    """None if this scan's preparation can be reused, else why one has to be made."""
    scan = PreparedScan.under(out_dir, path)
    if force:
        return "forced"
    if not Path(reference).is_file() or (
        not scan.model_input.exists() and not scan.record.exists()
    ):
        return "first time"
    return stale_reason(scan, path, reference, cfg, script, already_stripped=already_stripped)


def pending(
    inputs,
    reference: Path,
    out_dir: Path,
    cfg,
    script=None,
    *,
    already_stripped: bool = False,
    force: bool = False,
) -> list[Path]:
    """The inputs that still need preparing - which decides whether the skull stripper's
    weights are needed at all, since nobody should be offered a download for nothing."""
    return [
        p
        for p in inputs
        if preparation_status(
            p, reference, out_dir, cfg, script, already_stripped=already_stripped, force=force
        )
        is not None
    ]


def prepare_if_needed(
    path: Path,
    reference: Path,
    out_dir: Path,
    cfg,
    script=None,
    *,
    already_stripped: bool = False,
    force: bool = False,
    keep_intermediates: bool = False,
    stage=None,
) -> tuple[PreparedScan, str]:
    """The scan's preparation, made only if there is none that can be reused.

    Returns (scan, status): "reused", or why it was (re)made - "first time", "forced",
    or the reason the existing one could not be kept.
    """
    why = preparation_status(
        path, reference, out_dir, cfg, script, already_stripped=already_stripped, force=force
    )
    if why is None:
        return PreparedScan.under(out_dir, path), "reused"
    return prepare_scan(
        path,
        reference,
        out_dir,
        cfg,
        script,
        already_stripped=already_stripped,
        keep_intermediates=keep_intermediates,
        stage=stage,
    ), why


def prepare_all(
    paths,
    reference: Path,
    out_dir: Path,
    cfg,
    script=None,
    *,
    already_stripped: bool = False,
    force: bool = False,
    show=print,
    keep_intermediates: bool = False,
    stage=None,
) -> tuple[list[PreparedScan], dict[Path, str]]:
    """Prepares every scan; one that cannot be read or prepared (a truncated file, say) is
    reported and skipped so that it does not stop the rest of a cohort.

    Returns (the prepared scans, {path: reason} of those skipped). A missing asset is a setup
    problem, not a bad scan, and still raises.
    """
    scans, failed = [], {}
    for path in paths:
        try:
            scan, status = prepare_if_needed(
                path,
                reference,
                out_dir,
                cfg,
                script,
                already_stripped=already_stripped,
                force=force,
                keep_intermediates=keep_intermediates,
                stage=(lambda name, p=path: stage(p, name)) if stage else None,
            )
        except assets.AssetUnavailable:
            raise
        except UNREADABLE as err:
            # the first line names what failed, the last says why: SynthStrip reports "exit 1"
            # first and the reason -- an empty brain mask, a missing file -- several lines down,
            # and keeping only the first line threw away the part the user can act on
            said = str(err).strip()
            lines = [ln.strip() for ln in said.splitlines() if ln.strip()]
            failed[path] = (
                (lines[0] if len(lines) < 2 else f"{lines[0]} -- {lines[-1]}")
                if lines
                else repr(err)
            )
            show(f"  {path.name}: skipped - {failed[path]}")
            continue
        # a first preparation has no interesting reason; one that is redone always has one,
        # and that reason is the whole point of printing it
        if status == "reused":
            done = "already prepared"
        elif status == "first time":
            done = "prepared"
        else:
            done = f"prepared again ({status})"
        show(f"  {path.name}: {done}")
        scans.append(scan)
    return scans, failed
