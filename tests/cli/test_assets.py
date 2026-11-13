"""Finding and fetching the published assets."""

import hashlib
import json
import shutil
from dataclasses import replace

import pytest

from vox2seglam import assets


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv("VOX2SEGLAM_ASSETS", str(root))
    return root


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A published archive on disk: the assets, flat as a record holds them, and the manifest."""
    root = tmp_path / "archive"
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, asset in assets.ASSETS.items():
        payload = f"contents of {name}".encode() * 64
        (root / asset.published_as()).write_bytes(payload)
        files[asset.published_as()] = {
            "bytes": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
        }
    (root / assets.MANIFEST_NAME).write_text(
        json.dumps({"version": "2.0.0", "files": files}), encoding="utf-8"
    )
    monkeypatch.setattr(assets, "RELEASE_URL", root.as_uri())
    return root


def test_the_assets_directory_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VOX2SEGLAM_ASSETS", str(tmp_path / "shared"))
    assert assets.assets_dir() == tmp_path / "shared"

    monkeypatch.delenv("VOX2SEGLAM_ASSETS")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert assets.assets_dir() == tmp_path / "xdg" / "vox2seglam"

    monkeypatch.delenv("XDG_CACHE_HOME")
    assert assets.assets_dir().parts[-2:] == (".cache", "vox2seglam")


def test_an_asset_already_present_is_returned_without_asking(cache, monkeypatch):
    path = assets.ASSETS["reference"].path(cache)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"already here")
    monkeypatch.setattr(assets, "_consent", lambda *a: pytest.fail("should not ask"))
    assert assets.resolve("reference") == {"reference": path}


def test_without_a_published_archive_the_message_explains_what_to_do(cache, monkeypatch):
    monkeypatch.setattr(assets, "RELEASE_URL", "")
    with pytest.raises(assets.AssetUnavailable) as err:
        assets.resolve("checkpoint", "reference")
    message = str(err.value)
    assert "no published archive yet" in message
    assert "VOX2SEGLAM_ASSETS" in message
    assert str(assets.ASSETS["checkpoint"].path(cache)) in message  # the exact paths


def test_a_non_interactive_run_is_refused_and_never_prompted(cache, archive, monkeypatch):
    """With nobody to answer (batch job, container, CI) a prompt hangs or raises EOFError:
    refuse with instructions instead."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted without a tty"))
    with pytest.raises(assets.AssetUnavailable, match="missing asset"):
        assets.resolve("reference")


def test_a_terminal_that_turns_out_not_to_be_one_is_refused(cache, archive, monkeypatch):
    """isatty() can say yes and be wrong. Windows counts NUL among its character devices,
    so a job started with stdin=DEVNULL looks interactive to it and the prompt raises
    EOFError. The instructions must come out, not a traceback."""

    def eof(*_):
        raise EOFError("EOF when reading a line")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(assets.AssetUnavailable, match="missing asset"):
        assets.resolve("reference")
    assert not assets.ASSETS["reference"].path(cache).exists()


def test_a_terminal_is_asked_and_a_refusal_is_honoured(cache, archive, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    with pytest.raises(assets.AssetUnavailable):
        assets.resolve("reference")
    assert not assets.ASSETS["reference"].path(cache).exists()

    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert assets.resolve("reference")["reference"].is_file()


@pytest.mark.parametrize("answer", ["", "n", "no", "nope", "maybe"])
def test_only_an_explicit_yes_downloads(cache, archive, monkeypatch, answer):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: answer)
    with pytest.raises(assets.AssetUnavailable):
        assets.resolve("reference")


def test_the_flags_answer_in_advance(cache, archive, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("asked despite a flag"))
    with pytest.raises(assets.AssetUnavailable):
        assets.resolve("reference", download=False)
    assert assets.resolve("reference", download=True)["reference"].is_file()


def test_fetching_records_what_it_fetched(cache, archive):
    assets.resolve("checkpoint", "reference", download=True)
    record = json.loads((cache / assets.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert record["version"] == "2.0.0"
    assert record["source"] == archive.as_uri()
    assert set(record["files"]) == {
        assets.ASSETS["checkpoint"].relative_path,
        assets.ASSETS["reference"].relative_path,
    }  # the local layout
    assert "fetched" in record
    # a later fetch adds to the record rather than replacing it
    assets.resolve("config", download=True)
    grown = json.loads((cache / assets.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert len(grown["files"]) == 3


def test_a_corrupted_download_is_refused_and_leaves_nothing_behind(cache, archive):
    """A half-written file must not be accepted by later runs, like a truncated .gz."""
    manifest = json.loads((archive / assets.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["files"][assets.ASSETS["reference"].published_as()]["md5"] = "0" * 32
    (archive / assets.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(assets.AssetUnavailable, match="does not match the manifest"):
        assets.resolve("reference", download=True)
    target = assets.ASSETS["reference"].path(cache)
    assert not target.exists()
    assert not list(target.parent.glob("*.part"))  # no leftover temporary


def test_an_asset_absent_from_the_archive_says_so(cache, archive):
    manifest = json.loads((archive / assets.MANIFEST_NAME).read_text(encoding="utf-8"))
    del manifest["files"][assets.ASSETS["postprocess"].published_as()]
    (archive / assets.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(assets.AssetUnavailable, match="does not list"):
        assets.resolve("postprocess", download=True)


def test_an_unreachable_archive_falls_back_to_instructions(cache, monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "RELEASE_URL", (tmp_path / "nowhere").as_uri())
    with pytest.raises(assets.AssetUnavailable) as err:
        assets.resolve("reference", download=True)
    assert "cannot read" in str(err.value) and "VOX2SEGLAM_ASSETS" in str(err.value)


def test_describe_reports_presence_and_provenance(cache, archive):
    assert "missing" in assets.describe()
    assets.resolve("reference", download=True)
    text = assets.describe()
    assert "present" in text and "archive version 2.0.0" in text


def test_an_unknown_asset_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="checkpoint"):
        assets.asset_path("weigths")


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """The skull stripper's weights, served from a file:// 'upstream' with a pinned checksum."""
    payload = b"synthstrip weights" * 128
    source = tmp_path / "upstream" / "synthstrip.1.pt"
    source.parent.mkdir()
    source.write_bytes(payload)
    pinned = replace(
        assets.ASSETS["synthstrip"],
        url=source.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    monkeypatch.setitem(assets.ASSETS, "synthstrip", pinned)
    return source


def test_the_stripper_weights_are_pinned_to_their_authors(cache):
    """The address and checksum are fixed: the file is not ours to change."""
    asset = assets.ASSETS["synthstrip"]
    assert asset.url.startswith("https://surfer.nmr.mgh.harvard.edu/")
    assert len(asset.sha256) == 64 and asset.size > 30_000_000


def test_an_asset_with_its_own_source_needs_no_published_archive(cache, upstream, monkeypatch):
    monkeypatch.setattr(assets, "RELEASE_URL", "")
    path = assets.resolve("synthstrip", download=True)["synthstrip"]
    assert path.read_bytes() == upstream.read_bytes()


def test_its_checksum_is_verified_and_a_bad_file_leaves_nothing(cache, upstream):
    upstream.write_bytes(b"tampered")
    with pytest.raises(assets.AssetUnavailable, match="does not match"):
        assets.resolve("synthstrip", download=True)
    target = assets.ASSETS["synthstrip"].path(cache)
    assert not target.exists() and not list(target.parent.glob("*.part"))


def test_the_record_keeps_where_each_file_came_from(cache, archive, upstream):
    assets.resolve("reference", "synthstrip", download=True)
    files = json.loads((cache / assets.MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
    assert files["synthstrip/synthstrip.1.pt"]["source"] == upstream.as_uri()
    assert files["synthstrip/synthstrip.1.pt"]["sha256"]
    assert files["reference/reference_template_mean.nii.gz"]["source"] == archive.as_uri()


def test_nothing_is_offered_if_part_of_the_request_cannot_be_fetched(cache, upstream, monkeypatch):
    """Without a published archive the checkpoint cannot be had, so the stripper's weights alone
    are not offered either."""
    monkeypatch.setattr(assets, "RELEASE_URL", "")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("offered a doomed download"))
    with pytest.raises(assets.AssetUnavailable) as err:
        assets.resolve("checkpoint", "synthstrip")
    assert not assets.ASSETS["synthstrip"].path(cache).exists()
    assert "no published archive yet" in str(err.value)


def test_the_instructions_name_the_upstream_address_for_manual_download(cache, monkeypatch):
    monkeypatch.setattr(assets, "RELEASE_URL", "")
    with pytest.raises(assets.AssetUnavailable) as err:
        assets.resolve("synthstrip", download=False)
    assert assets.ASSETS["synthstrip"].url in str(err.value)


def test_the_prompt_says_what_is_fetched_how_big_and_from_where(
    cache, upstream, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assets.resolve("synthstrip")
    shown = capsys.readouterr().out
    assert "synthstrip" in shown and "skull stripping" in shown
    assert "Download them now?" not in shown  # that is the input() prompt, not stdout
    assert str(cache) in shown  # where it will be stored


def test_the_archive_location_can_be_overridden_for_a_mirror(archive, monkeypatch, tmp_path):
    mirror = tmp_path / "mirror"
    shutil.copytree(archive, mirror)
    monkeypatch.setattr(assets, "RELEASE_URL", "")  # nothing published...
    monkeypatch.setenv("VOX2SEGLAM_RELEASE_URL", mirror.as_uri())  # ...but a mirror is given
    assert assets.release_url() == mirror.as_uri()
    assert assets.resolve("reference", download=True)["reference"].is_file()


def test_fetch_assets_fills_the_cache_for_machines_without_a_network(
    tmp_path, archive, upstream, monkeypatch
):
    """Cluster compute nodes have no network: the files are fetched once, elsewhere, into a
    shared directory."""
    import subprocess

    from conftest import command, clean_env

    env = clean_env(
        tmp_path,
        VOX2SEGLAM_ASSETS=str(tmp_path / "shared"),
        VOX2SEGLAM_RELEASE_URL=archive.as_uri(),
    )
    done = subprocess.run(
        [*command("fetch_assets"), "reference", "--download"],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "shared" / "reference" / "reference_template_mean.nii.gz").is_file()
    assert "present" in done.stdout

    refused = subprocess.run(
        [*command("fetch_assets"), "weigths"], capture_output=True, text=True, env=env
    )
    assert refused.returncode != 0 and "unknown asset" in refused.stderr


def test_published_names_are_flat_and_unique():
    """A record on an archive such as Zenodo has no directories, so no published name may
    contain a separator, and two assets may not claim the same one."""
    names = [a.published_as() for a in assets.ASSETS.values()]
    assert not [n for n in names if "/" in n or "\\" in n]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(
    "base, expected",
    [
        ("https://example.org/files", "https://example.org/files/a.pt"),
        (
            "https://zenodo.org/records/1/files/{name}?download=1",
            "https://zenodo.org/records/1/files/a.pt?download=1",
        ),
    ],
)
def test_the_download_url_takes_a_name_placeholder(base, expected, monkeypatch):
    monkeypatch.setattr(assets, "RELEASE_URL", base)
    assert assets.published_url("a.pt") == expected
