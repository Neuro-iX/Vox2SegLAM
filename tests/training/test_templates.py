"""Automatic templates: built on the training split, refused when fitted on other subjects."""

import nibabel as nib
import numpy as np
import pytest

from vox2seglam.config import load_config
from vox2seglam.training.template_setup import ensure_template

N_LANDMARKS = 4


def make_dataset(root, n_subjects=12, seed=0):
    rng = np.random.default_rng(seed)
    (root / "Landmarks").mkdir()
    (root / "splits").mkdir()
    base = np.array([[10, 10, 10], [10, 20, 10], [20, 10, 20], [20, 20, 20]])
    ids = [f"s{i:02d}" for i in range(n_subjects)]
    for sid in ids:
        vol = np.zeros((32, 32, 32), np.uint8)
        for k, (z, y, x) in enumerate(base + rng.integers(-2, 3, base.shape), 1):
            vol[x, y, z] = k  # NIfTI is stored XYZ
        nib.save(nib.Nifti1Image(vol, np.eye(4)), root / "Landmarks" / f"{sid}_lm.nii.gz")
    (root / "splits" / "train_ids.txt").write_text("\n".join(ids), encoding="utf-8")
    return ids


def config(root, split="train_ids.txt"):
    return load_config(
        "baseline",
        overrides={
            "paths": {
                "data_root": str(root),
                "landmark_dir": "Landmarks",
                "landmark_suffix": "_lm.nii.gz",
                "splits_dir": str(root / "splits"),
                "split_train": split,
                "template_dir": str(root / "template"),
            },
            "task": {"n_landmarks": N_LANDMARKS},
        },
    )


def test_built_once_then_reused(tmp_path):
    make_dataset(tmp_path)
    written = ensure_template(config(tmp_path), quiet=True)
    assert set(written) == {"mean", "std", "edges"}
    assert np.load(tmp_path / "template" / "mean_landmark_coords_voxel_zyx.npy").shape == (4, 3)
    assert ensure_template(config(tmp_path), quiet=True) == {}


def test_refused_for_other_subjects(tmp_path):
    ids = make_dataset(tmp_path)
    ensure_template(config(tmp_path), quiet=True)
    (tmp_path / "splits" / "other_ids.txt").write_text("\n".join(ids[:-1]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fitted on other data"):
        ensure_template(config(tmp_path, "other_ids.txt"), quiet=True)


def test_a_skull_strip_cache_records_which_stripper_made_it(tmp_path):
    """The routes agree to within a voxel at the mask border, but a cache should still say
    which one made it."""
    import json

    from vox2seglam.config import load_config
    from vox2seglam.data.prepare.manifest import write_manifest

    write_manifest("skull_strip", tmp_path, ["sub-01"], load_config("baseline"))
    record = json.loads((tmp_path / "prepare.json").read_text(encoding="utf-8"))
    assert record["stripper"].startswith("builtin:nipreps-synthstrip==")
    assert record["parameters"] == {"border_mm": 1}  # unchanged: old caches still match


def test_a_published_recipe_takes_its_split_from_the_archive(tmp_path):
    """The subject lists are published with the data, not with the code. A recipe naming
    train_ids.txt means the published list, fetched once into the assets directory; it is not
    a file anyone happens to have in the working directory.

    Whether the shipped template is then reused cannot be checked here: that comparison is a
    hash of the published identifiers, which this repository does not carry.
    """
    from vox2seglam import assets

    cfg = load_config("hoa16f_med_jnt_s1")
    assert cfg.paths.splits_dir == ""
    with pytest.raises(assets.AssetUnavailable, match="splits"):
        cfg.paths.resolve_split("train")  # nothing in the cache, nothing published


def test_the_published_lists_are_read_from_the_archive_once_fetched(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile

    from vox2seglam import assets
    from vox2seglam.data.datasets import read_split

    record = tmp_path / "record"
    record.mkdir()
    held = assets.ASSETS["splits"]
    with zipfile.ZipFile(record / held.published_as(), "w") as zf:
        for member in held.members:
            zf.writestr(member, "sub-01\nsub-02\n")
    payload = (record / held.published_as()).read_bytes()
    (record / assets.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": "test",
                "files": {
                    held.published_as(): {
                        "bytes": len(payload),
                        "md5": hashlib.md5(payload).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(assets, "RELEASE_URL", record.as_uri())

    cfg = load_config("hoa16f_med_jnt_s1")
    assert assets.resolve("splits", download=True)
    for split in ("train", "val", "test"):
        path = cfg.paths.resolve_split(split)
        assert path.startswith(str(assets.assets_dir()))
        assert read_split(path) == ["sub-01", "sub-02"]


def test_a_split_named_like_the_published_one_gets_its_own_template(tmp_path):
    """train_ids.txt with _landmark_16.nii.gz is exactly how the published HOA template is
    named; a user's own data must not collide with it, nor write into the package."""
    from pathlib import Path

    from vox2seglam.training.template_setup import place_template

    ids = make_dataset(tmp_path)
    cfg = load_config(
        "baseline",
        overrides={
            "paths": {
                "data_root": str(tmp_path),
                "landmark_dir": "Landmarks",
                "landmark_suffix": "_landmark_16.nii.gz",
                "splits_dir": str(tmp_path / "splits"),
                "split_train": "train_ids.txt",
                "exp_dir": str(tmp_path / "run"),
            },
            "task": {"n_landmarks": N_LANDMARKS},
        },
    )
    assert Path(cfg.paths.resolve_template_dir()).name == "train_landmark_16"  # the collision
    placed = place_template(cfg, quiet=True)
    assert placed.paths.template_dir == str((tmp_path / "run" / "template").resolve())
    assert len(ids) == 12


def test_a_failed_fit_leaves_nothing_behind(tmp_path):
    from vox2seglam.training.topology import TooFewSubjects

    make_dataset(tmp_path, n_subjects=5)
    with pytest.raises(TooFewSubjects, match="graph_lambda"):
        ensure_template(config(tmp_path), quiet=True)
    assert not (tmp_path / "template").exists()
