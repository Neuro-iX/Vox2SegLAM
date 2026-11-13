# The repository

Where every piece of the pipeline lives, and why it lives there.

## Principles

**One config, no hidden state.** Every parameter lives in `vox2seglam/config/`, including the
paths, so a run is described by the config it snapshots. Nothing a run computes comes from the
environment: it says where downloaded files live (`$VOX2SEGLAM_ASSETS`, `$XDG_CACHE_HOME`), where
they are fetched from (`$VOX2SEGLAM_RELEASE_URL`), and which skull stripper to run
(`$VOX2SEGLAM_SYNTHSTRIP`, for a site with its own); `$PYTORCH_CUDA_ALLOC_CONF` and
`$CUBLAS_WORKSPACE_CONFIG` are set only if you have not set them yourself. The CLI flags that are
not config overrides only choose where and how a script runs (`--device`, `--passes`). Changing
dataset, split, architecture, loss, optimiser or augmentation is a config edit.

**One responsibility per module.** The directory structure *is* the pipeline:

```
vox2seglam/
├── commands/               one module per command, and the entry point that dispatches
│                           to them: `vox2seglam <command>` is `commands/<command>.py`,
│                           with the hyphen written as an underscore
├── configs/
│   ├── experiments/          the recipes that produced the published models
│   └── examples/             own_data.yaml, the starting point for a cohort of your own
├── config/                 every hyperparameter, one module per package below
│   ├── task.py               the problem: classes, landmarks, grid, voxel spacing
│   ├── paths.py              where a run reads from and writes to, splits included
│   ├── data.py               → data/: loading, preprocessing, augmentation
│   ├── model.py              → models/
│   ├── loss.py               → losses/
│   ├── training.py           → training/
│   ├── root.py               Config, its cross-section checks, and the defaults
│   └── loading.py            load_config / save_config, and --set KEY=VALUE parsing
├── data/
│   ├── prepare/
│   │   ├── register.py         rigid registration onto the reference grid; landmarks and
│   │   │                       labels carried through it, in both directions
│   │   ├── synthstrip.py       skull stripping: built-in, a FreeSurfer install, or a container
│   │   ├── n4.py               N4 bias-field correction
│   │   ├── scan.py             prepares one scan and keeps a record of what made it
│   │   └── manifest.py         the prepare.json written beside each cache
│   ├── augmentation/
│   │   ├── builder.py          the MONAI chain: affine, contrast, intensity, resolution, noise
│   │   ├── contrast.py         random-convolution contrast
│   │   └── augmenter.py        moves the landmarks with the affine
│   ├── landmarks.py          landmark ground truth, from a label volume or a coordinate table
│   ├── sample.py             one subject → tensors, deterministically
│   ├── normalize.py          z-score, whole-volume or brain-masked
│   ├── datasets.py           subject reading, and the split files
│   └── loaders.py            batching, worker seeding, shuffling
├── models/
│   ├── backbone/
│   │   ├── unet.py             the residual U-Net and its feature taps
│   │   └── blocks.py           residual blocks, up- and downsampling
│   ├── landmark/
│   │   ├── graph.py            the two-step graph network over the landmark template
│   │   ├── graph_blocks.py     message passing and the latent state
│   │   ├── refiner.py          the local patch head: heatmap and offset
│   │   ├── film.py             FiLM conditioning on landmark identity and graph state
│   │   └── template.py         the mean coordinates and the edges, as model buffers
│   ├── sampling.py           trilinear feature sampling and patch extraction
│   └── vox2seglam.py         the whole network: one forward pass, two outputs
├── losses/
│   ├── dice_ce_seg.py        segmentation: Dice and cross-entropy
│   ├── wing.py               Wing loss on the coordinates
│   ├── wing_graph.py         the same, per graph step
│   ├── cls_reg_refiner.py    the refiner: heatmap classification and offset regression
│   ├── deep_supervision.py   the auxiliary decoder heads
│   ├── adaptive_weighting.py learned, uncertainty-based term weights
│   └── total.py              assembles the terms into the training objective
├── protocol/
│   ├── hoa_protocol.pql      the planes and the per-region rules
│   ├── pql_parser.py         .pql → planes and rules
│   ├── geometry.py           planes from landmarks and rule depths: one implementation
│   ├── constants.py          the vocabularies, and the label tables a prediction writes
│   └── data/                 landmarks.yaml, regions.yaml, regions40.yaml, and the
│                             reference landmark coordinates
├── training/
│   ├── trainer.py            the epoch loop
│   ├── stages.py             the curriculum: which branch trains when
│   ├── checkpoint.py         what a checkpoint holds, how the best epoch is picked
│   ├── template_setup.py     the landmark template, fitted on the training split
│   ├── topology.py           the landmark graph: graphical lasso, cross-validated, Kruskal
│   ├── init_coords.py        where the landmarks start, and how training perturbs it
│   ├── optim.py              optimisers, schedulers, the clipped gradient step
│   ├── reporting.py          TensorBoard scalars and the console line
│   └── seeding.py            seeding and determinism
├── postprocess/
│   ├── plane_shift.py        plane shift + protocol enforcement
│   └── fitting.py            the shift ranges, fitted on validation
├── evaluate/
│   ├── batch.py              torch, per batch, during training
│   ├── metrics.py            numpy, per finished subject, in millimetres
│   ├── validation.py         the pass that drives the metrics during training
│   └── statistics.py         paired comparison (Wilcoxon signed-rank)
├── inference/
│   ├── predictor.py          a checkpoint applied to a volume on the reference grid
│   ├── scan.py               the same on a scan of any grid, mapped back to its own space
│   ├── slicer.py             the landmarks as a 3D Slicer markups file
│   └── constants.py          the defaults a command line needs before torch is imported
├── utils/
│   ├── device.py             device selection and deterministic inference
│   ├── geometry.py           coordinates, patch grids, LPS/RAS
│   ├── io.py                 NIfTI and JSON
│   └── logging.py
├── assets.py                 where the published files belong, and the one place that uses
│                             the network
├── cli.py                    the flags and the banner the entry points share
└── templates/                the landmark templates fitted on the published splits
    ├── train_landmark_16/              HOA, 80 subjects, 16 landmarks (the default)
    ├── hoa_cc_train_landmark_16/       HOA 40-label split, 73 subjects
    └── afids_nnlm_train_afids_32/      AFIDs, 88 subjects, 32 landmarks

tests/                      pytest, synthetic data only, no dataset needed
├── cli/                    what a user runs: the commands, the Predictor, the downloads
├── protocol/               the rules, the planes, the enforcement, the metrics
├── training/               resuming bit for bit, determinism, the templates
├── model/                  the architecture, the losses, the recipes
└── io/                     coordinates, geometry, the dataset layout, the release metadata
```

A package under `vox2seglam/` is described by the module of the same name under `config/`.
Every command is listed below, and `vox2seglam <command> --help` describes each in full.

## The commands

One entry point, ten subcommands. `vox2seglam` on its own lists them; each explains itself
with `--help`, and each is also a module, so `python -m vox2seglam.commands.predict` runs the
same code from a clone.

| To | Run |
|---|---|
| predict on a scan of your own | `vox2seglam predict --input scan.nii.gz` |
| predict on volumes already on the model's grid | `vox2seglam predict-prepared --input volume.nii.gz` |
| prepare a cohort ahead of time, then predict later | `vox2seglam prepare`, then `vox2seglam predict --prepared-dir` |
| turn landmark label volumes into a coordinate table | `vox2seglam export-annotations` |
| build a training dataset from native scans and annotations | `vox2seglam prepare-dataset` |
| train | `vox2seglam train --config <recipe or file>` |
| measure a checkpoint on its test split | `vox2seglam score-split --checkpoint <run>/best.pt` |
| fit the protocol post-processing | `vox2seglam fit-postprocess --checkpoint <run>/best.pt` |
| fill a cache for machines without a network | `vox2seglam fetch-assets` |
| score predictions against your own reference | `vox2seglam evaluate` |

Hard Dice (`evaluate/batch.py`) is computed on the argmax label map, per class;
a class absent from both prediction and ground truth gives `nan` and is skipped by the
averages rather than counted as 0 or 1.

`UNet` returns features and class logits from one decoder pass: here they come from the
same computation, and producing them together is what keeps a single forward cheap enough
to train. The segmentation output stays optional - `model.seg_head.enabled=false` builds
none at all. It is deeply supervised: auxiliary 1×1×1 heads on the two decoder stages
before the final head (D7 at 1/4 and D8 at 1/2 resolution, `model.seg_head.deep_supervision_levels`),
their logits upsampled to full resolution and compared with the full-resolution ground
truth as in Dou et al. (MedIA, 2017); Dice + cross-entropy weights 1/7, 2/7 and 4/7
(D7, D8, final), halving per level.

`models/` splits along the same line as the configuration: `model.backbone` and
`model.seg_head` configure `backbone/`, `model.lam` configures `landmark/`.
`models/sampling.py` sits above both because both sample backbone features at
continuous coordinates - the graph at its current estimates, the refiner over a
patch around them.

---

<sub>[← back to the README](../README.md)</sub>
