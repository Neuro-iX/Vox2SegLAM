"""A path that does not exist is a message, not a traceback.

Mistyping --input or --checkpoint is the most ordinary mistake there is, and these commands
used to answer it with a stack trace from inside pathlib. predict_prepared.py also used to
offer a 36 MB download before it ever looked at the path it had been given.
"""

import subprocess

import pytest

from conftest import command, ROOT, clean_env


def run(script, *args, home):
    return subprocess.run(
        [*command(script), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=clean_env(home),
    )


@pytest.mark.parametrize(
    "script, flag",
    [
        ("predict.py", "--input"),
        ("prepare.py", "--input"),
        ("predict_prepared.py", "--input"),
        ("prepare_dataset.py", "--scans"),
        ("evaluate.py", "--predictions"),
    ],
)
def test_an_input_that_is_not_there_is_named(script, flag, tmp_path):
    extra = {
        "prepare.py": ["--out-dir", tmp_path / "out"],
        "prepare_dataset.py": ["--landmarks", tmp_path / "l.csv", "--output", tmp_path / "out"],
    }.get(script, [])
    done = run(script, flag, tmp_path / "nowhere", *extra, home=tmp_path)
    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    assert "nowhere" in done.stdout + done.stderr


@pytest.mark.parametrize(
    "script", ["predict.py", "predict_prepared.py", "score_split.py", "fit_postprocess.py"]
)
def test_a_checkpoint_that_is_not_there_is_named(script, tmp_path):
    extra = ["--input", tmp_path] if script.startswith("predict") else []
    (tmp_path / "scan.nii.gz").write_bytes(b"")  # the path exists; the weights do not
    done = run(script, "--checkpoint", tmp_path / "none.pt", *extra, home=tmp_path)
    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    assert "none.pt" in done.stdout + done.stderr
