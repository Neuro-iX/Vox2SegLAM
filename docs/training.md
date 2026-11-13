# Training

Running the shipped recipes, and training on a cohort of your own.

## Running experiments

An experiment is a YAML file overriding only what it changes. No code is edited to run a
new configuration.

```bash
vox2seglam train --config hoa16f_med_jnt_s1           # a shipped recipe, by name
vox2seglam train --config hoa16f_med_jnt_s1 --set train.epochs=350 repro.seed=1
vox2seglam train --config hoa16f_med_jnt_s1 --resume
```

`--set` takes dotted keys, each validated against the schema so that a typo fails immediately
rather than silently doing nothing; `--resume` continues from `<exp_dir>/last.pt`.

**Reproducibility.** Runs are bit-exact by default (`repro.deterministic: true`): two runs of
the same recipe on the same hardware and software give identical weights, at about 1.6× the
time per epoch. Three limits are not visible from a config file. `data.num_workers` changes
the augmentation stream, so reproduce a run from its own `config.yaml`, which records the
count, and not from the recipe alone. The per-epoch seed is `repro.seed + epoch`, so the two
seeds of an experiment draw the same shuffling one epoch apart and their spread is
initialisation variance only. And the random-convolution contrast draws from the global torch
RNG rather than MONAI's, reproducible only because PyTorch seeds each worker.

## Training on your own cohort

**From native scans and annotations**, one step brings everything onto the reference grid and
writes a training configuration:

```bash
vox2seglam prepare-dataset --scans scans/ --segmentations labels/ \
    --landmarks landmarks.csv --output cohort/ --split 0.7 0.15 0.15
vox2seglam train --config cohort/cohort.yaml
vox2seglam score-split --checkpoint experiments/cohort/best.pt
vox2seglam predict --input new_scans/ --checkpoint experiments/cohort/best.pt \
    --reference <the reference you prepared with>
```

What you supply: the T1w scans; optionally a native label map each, 0 for background, left
out to train a landmark-only model; and the landmarks as a coordinate table with
`participant_id, landmark_id, x_mm, y_mm, z_mm` in each scan's own millimetres — what
`vox2seglam predict` writes and what any annotation program exports.

Each scan is registered rigidly to the reference template and resampled onto it, the
segmentation following by nearest neighbour so that no label is invented and your numbering
is kept, each landmark going to the nearest voxel. A landmark that falls off the grid, is
missing or lands on another's voxel stops the run with the subject named. About 50 seconds
per scan, and a second run skips what is finished.

**From volumes already on a grid**, copy `configs/examples/own_data.yaml` instead, set the
paths, `task.n_landmarks` and `task.n_classes`, and train with it. Each subject then needs a
T1w volume, a label volume, a landmark volume in which the voxel of landmark *k* holds the
value *k*, and the three split files. The landmark template is fitted on your training split
at the start of the run and kept with the run, never written into the package.

## Shipped recipes

**`baseline.yaml`** is the reference every other recipe extends: the code defaults, plus the
three `detach_*` flags set to `true`, which give the staged curriculum in which each stage
learns from its own loss. The code default is the end-to-end arm, so stating them here keeps
the recipes below meaning what they meant when they were run.

| Recipe | Task | Notes |
|---|---|---|
| `hoa16f_med_jnt_s1` | HOA, 16 landmarks + 17 classes | the **16-structure checkpoint on Zenodo**; end to end. The checkpoint records the ablation name the run was launched under, `abl_fully_joint_s1`; the two configurations are the same |
| `hoa40f_med_stg_s0` | HOA, 16 landmarks + 40 classes | the `hoa_cc` split (73 / 9 / 6); its U-Net is what the AFIDs model freezes |
| `afids_v2sl_med_stg_n88_s0` | AFIDs, 32 landmarks, no segmentation head | the **32-landmark checkpoint**; needs `hoa40f_med_stg_s0` first |
| `hoa16f_small_stg_s0` | HOA, as the baseline | 2.72 M parameters, not published: what the tests build, and the cheapest way to try the pipeline |

The campaign behind the paper was 126 runs over three sizes, two seeds, five AFIDs
training-set sizes and a set of ablations; those recipes are not shipped, describing
measurements rather than models anybody can download. Every recipe sets its own
`paths.exp_dir`, so two runs never share a directory. A `configs/experiments/` in the
working directory is searched first, so a recipe of your own needs no path:

```yaml
name: my_run
extends: baseline
paths:
  exp_dir: ./experiments/my_run
train:
  epochs: 350
model:
  backbone:
    encoder_channels: [32, 64, 128, 256, 512]
```

## What a run writes, and how to read it

The fully resolved configuration is written to `<exp_dir>/config.yaml` before the first epoch,
so a result is never separated from the settings that produced it; `last.pt`, `best.pt` and
the TensorBoard logs sit beside it. The curves are grouped by what they diagnose: `landmark/`
holds `mre_train` against `mre_val`, whose gap is the overfitting signal; `loss/` the
objective and its terms; `seg/` Dice overall and per class; `refine/` the fraction of
landmarks whose patch no longer contains the target, which nothing else would report; and
`optim/` the learning rates, branch weights, AMP scale and gradient norms, because a loss
curve alone cannot tell slow learning from vanished gradients.

```bash
tensorboard --logdir ./experiments/hoa16f_med_jnt_s1/tb
vox2seglam score-split --checkpoint experiments/hoa16f_med_jnt_s1/best.pt
```

`vox2seglam score-split` evaluates a checkpoint on its own test split and writes `test_metrics.json`
beside it, with the same `--passes` default as `vox2seglam predict-prepared`, so the metrics
describe what the command outputs.

---

## Data

The dataset is not redistributable. `paths.data_root` defaults to `data` at the repository
root: put the dataset there, link it, or pass `--set paths.data_root=/path/to/dataset`. Under
it, the shipped recipes expect raw T1w volumes in `MRI_Volumes/`, class maps in
`Segmentation_Volumes_16/` and landmark volumes in `Landmarks_Volumes_16/`, every directory
name and suffix being a configuration field. Only the raw volumes have to be supplied: skull
stripping and N4 are produced once by the pipeline and cached beside the source, each cache
carrying a `prepare.json` that records the tool, its version and its parameters.

Volumes must already be on the reference grid — **LAS, 0.7 mm isotropic, 240 × 304 × 240
(Z,Y,X)** — declared by `task.patch_size` and `task.voxel_spacing_mm`. Every metric in
millimetres is a voxel count times that spacing, read from the run's own configuration, so a
checkpoint trained on another grid is measured on its own.

`paths.splits_dir` is empty by default, meaning the subject lists the published runs were
trained on: nine files fetched once from the archive, published with the data rather than with
the code because they are identifiers of the source cohorts. Point it at a directory of your
own and nothing is fetched.

---

<sub>[← back to the README](../README.md)</sub>
