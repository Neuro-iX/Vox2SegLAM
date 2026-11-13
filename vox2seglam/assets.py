"""Locates the files a run needs that are too large for the repository, and fetches them."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

__all__ = [
    "RELEASE_URL",
    "ASSETS",
    "MODELS",
    "model_asset_names",
    "assets_dir",
    "asset_path",
    "release_url",
    "published_url",
    "resolve",
    "AssetUnavailable",
    "describe",
]

# Where the published files live. A record on an archive such as Zenodo has no directories, so
# each file is published under the flat name in `Asset.remote`. "{name}" in the URL is replaced
# by that name, which is what Zenodo's download links need; without it the name is appended.
# Empty until the record exists, then
#   https://zenodo.org/records/<id>/files/{name}?download=1
# $VOX2SEGLAM_RELEASE_URL overrides it (a mirror, a file:// copy).
RELEASE_URL = "https://zenodo.org/records/24205673/files/{name}?download=1"

#: Manifest inside the archive: {"version": str, "files": {path: {bytes, md5}}}
MANIFEST_NAME = "assets.json"

#: Directory under weights/ of each published model, by number of structures segmented
#: (16 structures go with 16 landmarks, 39 with 32).
MODELS: dict[int, str] = {16: "structures16_landmarks16", 39: "structures39_landmarks32"}
DEFAULT_MODEL = MODELS[16]


@dataclass(frozen=True)
class Asset:
    """One file. An empty `url` means the published archive; otherwise the file is fetched
    from that address and checked against `sha256`."""

    name: str
    relative_path: str  # where it is kept on this machine, under the assets directory
    what: str
    remote: str = ""  # its flat name in the published record; the basename by default
    url: str = ""
    sha256: str = ""
    size: int = 0
    members: tuple[str, ...] = ()  # files it holds, when published as a zip of a directory
    instead: str = ""  # how to supply this one by hand, printed when it is missing

    def path(self, root: Path | None = None) -> Path:
        return (root or assets_dir()) / self.relative_path

    def published_as(self) -> str:
        return self.remote or Path(self.relative_path).name

    def present(self, root: Path | None = None) -> bool:
        here = self.path(root)
        return all((here / m).is_file() for m in self.members) if self.members else here.is_file()


# Keyed by the name used in CLI flags and error messages.
ASSETS: dict[str, Asset] = {
    "checkpoint": Asset(
        "checkpoint",
        f"weights/{DEFAULT_MODEL}/best.pt",
        "the trained weights",
        remote=f"vox2seglam-{DEFAULT_MODEL}.pt",
        instead="pass --checkpoint",
    ),
    "config": Asset(
        "config",
        f"weights/{DEFAULT_MODEL}/config.yaml",
        "the configuration the weights were trained with",
        remote=f"vox2seglam-{DEFAULT_MODEL}.yaml",
        instead="pass --config",
    ),
    "postprocess": Asset(
        "postprocess",
        f"weights/{DEFAULT_MODEL}/postprocess.json",
        "the fitted protocol post-processing",
        remote=f"vox2seglam-{DEFAULT_MODEL}-postprocess.json",
    ),
    "reference": Asset(
        "reference",
        "reference/reference_template_mean.nii.gz",
        "the reference template volume, which every input is registered onto",
        instead="pass --reference",
    ),
    "splits": Asset(
        "splits",
        "splits",
        "the subject lists the published runs were trained on",
        remote="vox2seglam-splits.zip",
        size=5_914,
        instead="set paths.splits_dir to a directory of your own id lists",
        members=(
            "train_ids.txt",
            "val_ids.txt",
            "test_ids.txt",
            "hoa_cc_train_ids.txt",
            "hoa_cc_val_ids.txt",
            "hoa_cc_test_ids.txt",
            "afids_nnlm_train_ids.txt",
            "afids_nnlm_val_ids.txt",
            "afids_nnlm_test_ids.txt",
            # the data-efficiency subsets: no shipped recipe uses them, but splits.csv
            # names them and the campaign they belong to is part of the published record
            "afids_nnlm_train_n11_ids.txt",
            "afids_nnlm_train_n22_ids.txt",
            "afids_nnlm_train_n44_ids.txt",
            "afids_nnlm_train_n66_ids.txt",
            "README.md",
        ),
    ),
    # SynthStrip (Hoopes et al., NeuroImage 2022), the stripper the checkpoints were trained on
    "synthstrip": Asset(
        "synthstrip",
        "synthstrip/synthstrip.1.pt",
        "SynthStrip's weights, for skull stripping",
        url="https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
        sha256="37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33",
        size=30_851_709,
    ),
}


for _directory in MODELS.values():
    if _directory != DEFAULT_MODEL:
        ASSETS[f"checkpoint:{_directory}"] = Asset(
            f"checkpoint:{_directory}",
            f"weights/{_directory}/best.pt",
            "the trained weights",
            remote=f"vox2seglam-{_directory}.pt",
        )
        ASSETS[f"config:{_directory}"] = Asset(
            f"config:{_directory}",
            f"weights/{_directory}/config.yaml",
            "the configuration the weights were trained with",
            remote=f"vox2seglam-{_directory}.yaml",
        )


def model_asset_names(structures: int = 16, *, postprocess: bool = False) -> list[str]:
    """Asset names of the published model with this many structures. The protocol
    post-processing exists for the 16-structure model only."""
    if structures not in MODELS:
        raise ValueError(
            f"no published model segments {structures} structures; available: "
            f"{', '.join(str(n) for n in MODELS)}"
        )
    directory = MODELS[structures]
    if directory == DEFAULT_MODEL:
        return ["checkpoint", "config"] + (["postprocess"] if postprocess else [])
    if postprocess:
        raise ValueError(
            "--postprocess applies to the subcortical model only; omit it or use --task subcortical"
        )
    return [f"checkpoint:{directory}", f"config:{directory}"]


class AssetUnavailable(RuntimeError):
    """An asset is not on disk and could not (or may not) be fetched."""


def published_url(name_in_record: str) -> str:
    """Where one published file is fetched from: "{name}" in the base URL is replaced by the
    file's flat name (Zenodo's download links carry a query string after it), else appended."""
    base = release_url()
    return (
        base.replace("{name}", name_in_record) if "{name}" in base else f"{base}/{name_in_record}"
    )


def release_url() -> str:
    """Where the model's own assets are published: $VOX2SEGLAM_RELEASE_URL, else RELEASE_URL."""
    return (os.environ.get("VOX2SEGLAM_RELEASE_URL") or RELEASE_URL).rstrip("/")


def assets_dir() -> Path:
    """$VOX2SEGLAM_ASSETS (a shared copy, say on a cluster), else the XDG cache directory."""
    env = os.environ.get("VOX2SEGLAM_ASSETS")
    if env:
        return Path(env).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(cache).expanduser() / "vox2seglam"


def asset_path(name: str) -> Path:
    """Where an asset belongs, present or not."""
    if name not in ASSETS:
        raise KeyError(f"unknown asset {name!r}; known: {sorted(ASSETS)}")
    return ASSETS[name].path()


def _source(asset: Asset) -> str:
    """Where this asset would be fetched from."""
    return asset.url or release_url()


def describe(names=None) -> str:
    """Table of the assets and whether each is present (`--show-assets`)."""
    rows = [(n, ASSETS[n]) for n in (names or ASSETS)]
    width = max(len(n) for n, _ in rows)
    lines = [
        f"assets directory:  {assets_dir()}",
        f"published archive: {release_url() or '(not published yet)'}",
        "",
    ]
    for name, asset in rows:
        state = "present" if asset.present() else "missing"
        lines.append(f"  {name:<{width}}  {state:<8} {asset.path()}")
        if asset.url:
            lines.append(f"  {'':<{width}}  {'':<8} from {asset.url}")
    record = assets_dir() / MANIFEST_NAME
    if record.is_file():
        version = json.loads(record.read_text(encoding="utf-8")).get("version")
        if version:
            lines += ["", f"fetched from archive version {version} (see {record})"]
    return "\n".join(lines)


def _instructions(missing: list[str]) -> str:
    """What to do about assets that are not on disk."""
    lines = []
    for name in missing:
        asset = ASSETS[name]
        lines.append(f"  {asset.path()}")
        if asset.url:
            lines.append(f"      {asset.what}: download {asset.url}")
    if any(not ASSETS[n].url for n in missing):
        archive = release_url()
        lines.insert(
            0,
            f"Download them from {archive} and put them here:"
            if archive
            else "There is no published archive yet. Put your own copies here:",
        )
    else:
        lines.insert(0, "Put them here:")
    lines += [
        "",
        "Either way:",
        f"  - set VOX2SEGLAM_ASSETS to a directory that already holds them "
        f"(currently {assets_dir()})",
    ]
    lines += [
        f"  - {how}"
        for how in dict.fromkeys(ASSETS[n].instead for n in missing if ASSETS[n].instead)
    ]
    return "\n".join(lines)


def _human(size: int) -> str:
    """A size somebody reads before agreeing to download it."""
    return f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{max(size, 0) / 1e3:.0f} kB"


def _unpack(url: str, asset: Asset, root: Path, expected: str | None) -> None:
    """Fetch an archive and lay the files it holds out under the asset's directory. Only the
    members named in ASSETS are read, so the archive cannot write anywhere else."""
    here = asset.path(root)
    here.mkdir(parents=True, exist_ok=True)
    archive = here.with_name(here.name + ".zip.part")
    _download(url, archive, expected, "md5")
    try:
        with zipfile.ZipFile(archive) as zf:
            held = set(zf.namelist())
            absent = [m for m in asset.members if m not in held]
            if absent:
                raise AssetUnavailable(
                    f"{asset.published_as()} does not hold {', '.join(absent)}; it may be a "
                    f"version that does not carry them"
                )
            for member in asset.members:
                (here / member).write_bytes(zf.read(member))
    except zipfile.BadZipFile as err:
        raise AssetUnavailable(f"{asset.published_as()} is not a readable archive ({err})") from err
    finally:
        archive.unlink(missing_ok=True)


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _consent(missing: list[str], download: bool | None) -> bool:
    """Whether the network may be used; None asks a terminal and refuses without one."""
    if download is not None:
        return download
    if not sys.stdin.isatty():
        # nobody to answer in a batch job or CI: a prompt would hang or raise EOFError
        return False
    print("\nThis run needs files that are not on this machine yet:", flush=True)
    for name in missing:
        asset = ASSETS[name]
        size = f"{_human(asset.size)}, " if asset.size else ""
        host = urlparse(_source(asset)).netloc or "the published archive"
        print(f"  - {name:<11} {asset.what} ({size}from {host})", flush=True)
    print(f"They are stored under {assets_dir()}.", flush=True)
    try:
        answer = input("Download them now? [y/N] ").strip().lower()
    except EOFError:
        # isatty() said there was a terminal and there was not. Windows counts NUL among
        # its character devices, so a job whose stdin is /dev/null looks interactive to it;
        # the EOF is what settles the question. Refusing here turns a traceback into the
        # same instructions a Linux batch job already got.
        print("nobody to answer: not downloading.", flush=True)
        return False
    return answer in ("y", "yes")


def _download(url: str, destination: Path, expected: str | None, algorithm: str) -> None:
    """Fetch and verify one file; it reaches its final path only once complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    temporary = Path(temporary)
    os.close(handle)
    try:
        try:
            with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
        except (urllib.error.URLError, OSError) as err:
            raise AssetUnavailable(f"cannot download {url} ({err})") from err
        if expected:
            got = _hash(temporary, algorithm)
            if got != expected:
                raise AssetUnavailable(
                    f"{destination.name} does not match the manifest ({algorithm} {got}, "
                    f"expected {expected}); the download was corrupted or the file changed"
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest() -> dict:
    url = published_url(MANIFEST_NAME)
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as err:
        raise AssetUnavailable(f"cannot read {MANIFEST_NAME} from {release_url()} ({err})") from err


def _fetch(missing: list[str], root: Path) -> None:
    """Download the missing files and record where each came from."""
    recorded: dict[str, dict] = {}
    manifest: dict | None = None
    for name in missing:
        asset = ASSETS[name]
        if asset.url:
            print(f"  fetching {asset.published_as()} ({_human(asset.size)})", flush=True)
            _download(asset.url, asset.path(root), asset.sha256, "sha256")
            recorded[asset.relative_path] = {
                "bytes": asset.size,
                "sha256": asset.sha256,
                "source": asset.url,
            }
            continue
        manifest = manifest or _read_manifest()
        published = asset.published_as()
        entry = manifest.get("files", {}).get(published)
        if entry is None:
            raise AssetUnavailable(
                f"the archive's {MANIFEST_NAME} does not list {published}; it may "
                f"be a version that does not carry this asset"
            )
        print(f"  fetching {published} ({_human(entry.get('bytes', 0))})", flush=True)
        if asset.members:
            _unpack(published_url(published), asset, root, entry.get("md5"))
        else:
            _download(published_url(published), asset.path(root), entry.get("md5"), "md5")
        recorded[asset.relative_path] = {**entry, "source": release_url()}

    record = root / MANIFEST_NAME
    previous = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
    out = {
        **{k: v for k, v in previous.items() if k not in ("files", "fetched")},
        "files": {**previous.get("files", {}), **recorded},
        "fetched": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if manifest is not None:  # only the archive has a version
        out["source"], out["version"] = release_url(), manifest.get("version")
    temporary = record.with_suffix(".json.part")
    temporary.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, record)


def resolve(*names: str, download: bool | None = None) -> dict[str, Path]:
    """Local paths of the named assets, fetching missing ones if allowed.

    download=True fetches without asking, False never does, None asks a terminal. All or
    nothing: if any missing asset has no source, nothing is offered. Raises AssetUnavailable,
    with instructions, when an asset is missing and may not or could not be fetched.
    """
    root = assets_dir()
    wanted = {name: ASSETS[name].path(root) for name in names}
    missing = [name for name in names if not ASSETS[name].present(root)]
    if not missing:
        return wanted

    if all(_source(ASSETS[n]) for n in missing) and _consent(missing, download):
        try:
            _fetch(missing, root)
        except AssetUnavailable as err:
            raise AssetUnavailable(f"{err}\n{_instructions(missing)}") from err
        missing = [n for n in missing if not ASSETS[n].present(root)]
    if missing:
        raise AssetUnavailable(f"missing asset(s): {', '.join(missing)}.\n{_instructions(missing)}")
    return wanted
