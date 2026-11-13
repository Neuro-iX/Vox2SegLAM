"""Argument groups and the start-up banner shared by the entry points."""

import sys

from . import __version__

__all__ = [
    "TASKS",
    "add_asset_flags",
    "add_model_flags",
    "add_precision_flag",
    "resolve_postprocess",
    "banner",
]

#: "Vox2SegLAM" as block letters, 74 columns wide, assembled once at import
_WORDMARK = (
    " __     __              ____   ____                 _         _     __  __",
    " \\ \\   / /  ___  __  __|___ \\ / ___|   ___    __ _ | |       / \\   |  \\/  |",
    "  \\ \\ / /  / _ \\ \\ \\/ /  __) |\\___ \\  / _ \\  / _` || |      / _ \\  | |\\/| |",
    "   \\ V /  | (_) | >  <  / __/  ___) ||  __/ | (_| || |___  / ___ \\ | |  | |",
    "    \\_/    \\___/ /_/\\_\\|_____||____/  \\___|  \\__, ||_____|/_/   \\_\\|_|  |_|",
    "                                             |___/",
)
_WIDTH = 79
_RULE = "#" * _WIDTH
_TAGLINE = "Subcortical segmentation and anatomical landmarks on a T1-weighted MRI"
_HOME = "github.com/Neuro-iX/Vox2SegLAM"


def banner(stream=None, force: bool = False) -> None:
    """Print the start-up banner, on a terminal only.

    A piped or redirected run gets nothing: the block letters are for a person watching a
    command start, and they have no business in a log file or in a test's captured output.
    `force` prints it anyway.
    """
    stream = stream or sys.stdout
    if not (force or stream.isatty()):
        return
    left = f"Version {__version__}"
    # both info lines are laid out inside the same width as the rule, the right-hand column
    # flush with its last "#", so nothing drifts if the version string grows
    gap = _WIDTH - 1 - len(left) - len(_HOME)
    lines = [
        _RULE,
        *_WORDMARK,
        "",
        _TAGLINE.center(_WIDTH - 1).rstrip(),
        f" {left}{' ' * gap}{_HOME}",
        _RULE,
        "",
    ]
    print("\n".join(lines), flush=True, file=stream)


# the flags that answer the download prompt in advance, for batch jobs and CI
def add_asset_flags(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--download",
        dest="download",
        action="store_true",
        default=None,
        help="download missing files without prompting",
    )
    group.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="never download; exit with instructions if a file is missing",
    )


# The subject lists of the published runs are published with the data, not with the code.
# A run whose config leaves paths.splits_dir empty fetches them, and asks before doing so:
# better here, before the dataset is read and the model built, than halfway into the startup.
def ensure_splits(cfg, download: bool | None = None) -> None:
    from . import assets

    if not cfg.paths.splits_dir:
        assets.resolve("splits", download=download)


# Where the fitted plane shifts of a checkpoint live. fit_postprocess.py writes
# `postprocess.json` beside the weights; the published record, having no directories, names the
# same file after the model it belongs to. Both spellings are tried before giving up, and the
# failure names what was looked for instead of raising from inside json.load.
def resolve_postprocess(checkpoint, given: str, fetched=None):
    from pathlib import Path

    if given:
        path = Path(given)
        if not path.is_file():
            raise SystemExit(f"--postprocess {path}: no such file")
        return path
    if fetched is not None:
        return fetched
    checkpoint = Path(checkpoint)
    tried = [
        checkpoint.with_name(f"{checkpoint.stem}-postprocess.json"),
        checkpoint.parent / "postprocess.json",
    ]
    for candidate in tried:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "--postprocess needs the fitted plane shifts of this checkpoint, and neither\n"
        + "\n".join(f"  {p}" for p in tried)
        + "\nexists. Give the file with --postprocess FILE, or fit one with "
        "`vox2seglam fit-postprocess`."
    )


# which published model to use when no --checkpoint is given
#: {task: number of structures its model segments}. The flag names the task rather than one of
#: its two outputs: "39 structures" said nothing about the 32 landmarks that come with them.
TASKS = {"subcortical": 16, "full-brain": 39}


def add_model_flags(parser) -> None:
    parser.add_argument(
        "--task",
        choices=tuple(TASKS),
        default="subcortical",
        metavar="{subcortical,full-brain}",
        help="which published model to use (default: %(default)s). "
        "subcortical: 16 structures and 16 landmarks, with the protocol "
        "post-processing. full-brain: 39 structures and 32 landmarks.",
    )


# Half precision is reached on its own when a GPU runs out of memory, so asking for it is a
# reproduction detail rather than a thing to offer: predict.py does not, predict_prepared.py
# does, because that is where a published measurement is repeated.
def add_precision_flag(parser) -> None:
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="numeric precision of the network on a GPU (default: float32, "
        "which reproduces the published numbers). Half precision needs "
        "about 30 %% less memory and moves landmarks by under 0.06 mm; it "
        "is also used automatically if the GPU runs out of memory.",
    )
