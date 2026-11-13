"""Resuming a run from last.pt reproduces the uninterrupted one, bit for bit."""

import numpy as np
import pytest
import torch

from vox2seglam.config import load_config

GRID = (32, 48, 32)
N_LANDMARKS = 16
TRAIN = [f"sub-{i:02d}" for i in range(4)]
VAL = [f"sub-{i:02d}" for i in range(4, 6)]


def landmark_positions() -> np.ndarray:
    """16 distinct voxels, well inside the frame so augmentation can keep them there."""
    rng = np.random.default_rng(0)
    low, high = np.array([8, 12, 8]), np.array(GRID) - np.array([9, 13, 9])
    seen, out = set(), []
    while len(out) < N_LANDMARKS:
        p = tuple(int(v) for v in rng.integers(low, high))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return np.array(out)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A tiny cohort on disk: raw T1w, a 17-class segmentation and a landmark volume."""
    import nibabel as nib

    root = tmp_path_factory.mktemp("cohort")
    affine = np.diag([-0.7, 0.7, 0.7, 1.0])  # LAS, as the recipes expect
    base = landmark_positions()
    for d in ("MRI_Volumes", "Segmentation_Volumes_16", "Landmarks_Volumes_16"):
        (root / d).mkdir()

    def save(array_zyx, path, dtype):
        nib.save(
            nib.Nifti1Image(
                np.ascontiguousarray(array_zyx.transpose(2, 1, 0)).astype(dtype), affine
            ),
            str(path),
        )

    for n, sid in enumerate(TRAIN + VAL):
        rng = np.random.default_rng(100 + n)
        image = rng.random(GRID, dtype=np.float32)
        image[6:-6, 8:-8, 6:-6] += 1.0  # a bright block to normalise over
        save(image, root / "MRI_Volumes" / f"{sid}_T1w.nii.gz", np.float32)

        seg = np.zeros(GRID, np.int16)
        for label in range(1, 17):  # a slab per class, so none is empty
            seg[label + 6, 10:-10, 6:-6] = label
        save(seg, root / "Segmentation_Volumes_16" / f"{sid}.nii.gz", np.int16)

        lm = np.zeros(GRID, np.int16)
        jitter = rng.integers(-2, 3, base.shape)  # per-subject, so std is not zero
        for k, (z, y, x) in enumerate(base + jitter, start=1):
            lm[z, y, x] = k
        save(lm, root / "Landmarks_Volumes_16" / f"{sid}_landmark_16.nii.gz", np.int16)

    (root / "train_ids.txt").write_text("\n".join(TRAIN) + "\n", encoding="utf-8")
    (root / "val_ids.txt").write_text("\n".join(VAL) + "\n", encoding="utf-8")
    (root / "test_ids.txt").write_text("\n".join(VAL) + "\n", encoding="utf-8")
    return root


def config(dataset, exp_dir, epochs, workers=0):
    return load_config(
        "hoa16f_small_stg_s0",
        overrides={
            "task": {"patch_size": list(GRID)},
            "paths": {
                "data_root": str(dataset),
                "splits_dir": str(dataset),
                "template_dir": str(dataset / "template"),
                "exp_dir": str(exp_dir),
            },
            # no SynthStrip, no N4: the fixture writes what a prepared cache would contain
            "preprocess": {"skull_stripped": False, "bias_field_corrected": False},
            # all pairs: avoids fitting a graph topology (a cross-validated lasso) on 4 subjects
            "model": {"lam": {"graph_topology": "fully_connected"}},
            # every stage trains from the first epoch, so both runs exercise the same parameters
            "staging": {"enabled": False},
            "train": {"epochs": epochs, "eval_every": 1},
            "data": {"num_workers": workers, "eval_num_workers": 0},
        },
    )


def weights(trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


# workers=0 keeps the augmentation RNG in this process; workers=2 moves it into subprocess
# workers re-seeded by _worker_init_fn, which is the case checkpoint.py's comment is about
@pytest.mark.parametrize(
    "workers, method",
    [(0, None), (2, None), (2, "spawn")],
    ids=["0 workers", "2 workers", "2 workers, spawn (Windows)"],
)
def test_resuming_reproduces_an_uninterrupted_run(dataset, tmp_path, workers, method, monkeypatch):
    if method:
        monkeypatch.setattr("vox2seglam.data.loaders.worker_start_method", lambda: method)
    from vox2seglam.training import Trainer

    straight = Trainer(config(dataset, tmp_path / "straight", epochs=2, workers=workers))
    straight.fit()
    expected = weights(straight)

    first = Trainer(config(dataset, tmp_path / "resumed", epochs=1, workers=workers))
    first.fit()

    resumed = Trainer(config(dataset, tmp_path / "resumed", epochs=2, workers=workers))
    resumed.load_checkpoint(tmp_path / "resumed" / "last.pt")
    resumed.fit()
    got = weights(resumed)

    assert set(got) == set(expected)
    differing = [k for k in expected if not torch.equal(expected[k], got[k])]
    assert not differing, (
        f"{len(differing)} of {len(expected)} tensors differ after a resume with "
        f"num_workers={workers}, e.g. {differing[:3]} -- the resume is not bit-exact"
    )


def test_two_identical_runs_give_identical_weights(dataset, tmp_path):
    """The control: without this, the test above could pass on a model that never moves."""
    from vox2seglam.training import Trainer

    a = Trainer(config(dataset, tmp_path / "a", epochs=1))
    a.fit()
    before = weights(a)

    b = Trainer(config(dataset, tmp_path / "b", epochs=1))
    start = weights(b)
    b.fit()
    after = weights(b)

    assert all(torch.equal(before[k], after[k]) for k in before)  # reproducible
    assert any(not torch.equal(start[k], after[k]) for k in start)  # and it did train


def test_bfloat16_on_a_gpu_without_it_says_so(monkeypatch):
    from vox2seglam.training.trainer import _bf16_warning

    cfg = load_config("baseline")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: False)
    assert "train.amp_dtype=float16" in _bf16_warning(cfg, torch.device("cuda"))
    assert _bf16_warning(cfg, torch.device("cpu")) is None
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
    assert _bf16_warning(cfg, torch.device("cuda")) is None
