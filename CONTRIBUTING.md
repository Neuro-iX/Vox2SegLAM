# Contributing

## Reporting a problem

Open an issue with the version, the exact command, and the message in full:

```bash
python -c "import vox2seglam; print(vox2seglam.__version__)"
```

For something that looks wrong rather than broken, attach `<stem>_landmarks.mrk.json`. Its
`vox2seglam` key records which checkpoint ran, in what precision, on which reference grid, and
what the post-processing did — usually enough to tell a configuration problem from a defect.

Two things are always worth reporting. A segmentation that still leaves voxels outside the
admissible zones after `--postprocess`: the command re-checks the rules on the file it wrote,
so a non-empty `violating_voxels_after` means the enforcement failed, not that the model was
unlucky. And a prediction whose landmarks carry the wrong anatomical name, which would mean
the vocabulary and the model have come apart.

## Running the tests

262 tests, three minutes, no dataset: the volumes, the landmarks and the published
archive are all built synthetically.

```bash
pip install -e ".[test]"
pytest                   # everything
pytest tests/protocol    # one group
ruff check . && ruff format --check .
```

The five directories are not a mirror of `vox2seglam/`. They are named after the thing that
must not break, and each pulls in whatever modules that takes:

| | tests | what a failure there means |
|---|---|---|
| `cli/` | 101 | a command a user types stopped doing what it says |
| `model/` | 64 | the architecture, a loss, or a shipped recipe changed shape |
| `io/` | 45 | a coordinate, a geometry, or a file the release carries went wrong |
| `protocol/` | 36 | a rule, a plane, or the enforcement no longer holds |
| `training/` | 16 | a run stopped being reproducible, or a template lost its provenance |

So `postprocess/` has no directory of its own: enforcing the protocol is checked in
`protocol/`, the flag that triggers it in `cli/`, and the shape of what it writes in `io/`.
One module changed, three groups to run.

CI runs the suite on Linux, macOS and Windows, on Python 3.11 and 3.13, and `ruff`
once. Formatting is `ruff format`, with the line length pinned in `pyproject.toml`. It
also installs the package non-editably and imports it from an empty directory, because an
editable install hides a missing data file: the protocol, its vocabularies and the landmark
templates have to be packaged, not merely present in the clone.

## Offering a change

Keep it to one subject. A change in behaviour needs a test that fails without it, and the
test's name should say what breaks rather than what it calls. `pytest` and `ruff check .` must
pass.

Parameters live in `vox2seglam/config/`, not in the code: a new setting is a field there, with
its default, and `Config.validate` rejects the combinations the architecture cannot honour.

The published checkpoints belong to a paper. Corrections, portability and clarity are welcome;
a change to the method itself would make them no longer reproduce what was measured, so raise
it as an issue first.
