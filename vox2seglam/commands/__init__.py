"""The commands, and the one entry point that dispatches to them.

`vox2seglam <command> [options]`. Each command keeps its own parser, so
`vox2seglam predict --help` prints exactly what that command accepts; this module only
chooses which one runs, and imports nothing heavy until it has.
"""

import importlib
import sys

from .. import __version__

__all__ = ["COMMANDS", "main"]

#: command name -> (module, what it does). The order is the order a user meets them.
COMMANDS: dict[str, tuple[str, str]] = {
    "predict": ("predict", "a scan of your own -> landmarks and segmentation, in its space"),
    "predict-prepared": ("predict_prepared", "the same, on volumes already on the model's grid"),
    "prepare": ("prepare", "register, skull-strip and N4 a whole cohort ahead of time"),
    "evaluate": ("evaluate", "score predictions against annotations of your own"),
    "fetch-assets": ("fetch_assets", "download the published files on purpose"),
    "export-annotations": ("export_annotations", "landmark label volumes -> one coordinate table"),
    "prepare-dataset": (
        "prepare_dataset",
        "native scans and annotations -> a training dataset on the grid",
    ),
    "train": ("train", "train one recipe into one run directory"),
    "score-split": ("score_split", "metrics of a checkpoint on one split"),
    "fit-postprocess": ("fit_postprocess", "fit the protocol post-processing of a checkpoint"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "usage: vox2seglam <command> [options]",
        "",
        "Subcortical segmentation and anatomical landmarks on a T1-weighted brain MRI.",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {what}" for name, (_, what) in COMMANDS.items()]
    lines += ["", "`vox2seglam <command> --help` describes one of them in full."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0

    name = argv[0]
    if name not in COMMANDS:
        close = [c for c in COMMANDS if c.startswith(name[:3])]
        raise SystemExit(
            f"vox2seglam: no command {name!r}"
            + (f". Did you mean {' or '.join(close)}?" if close else "")
            + f"\n\n{usage()}"
        )

    module = importlib.import_module(f".{COMMANDS[name][0]}", __name__)
    # each command built its parser expecting to be the program, and its --help and its
    # error messages name it; tell it what it is now called
    sys.argv = [f"vox2seglam {name}", *argv[1:]]
    return module.main()
