# Changelog

## 1.0.0, 2026-11-13

First public release: joint subcortical segmentation and anatomical landmark detection, the
anatomical protocol and its post-processing, the two published models, and the HOA and AFIDs
experiments.

- `vox2seglam predict --input scan.nii.gz` takes a raw T1w scan to landmarks and a
  segmentation in the scan's own space. The scan is registered onto the reference grid,
  skull-stripped and bias-corrected as needed; that preparation is kept with a record of its
  inputs and reused on later runs.
- Two published models, chosen with `--task {subcortical,full-brain}`: 16 structures with 16
  landmarks (default) and 39 structures with 32 landmarks.
- Skull stripping needs neither FreeSurfer nor a container: the default route is
  `nipreps-synthstrip` with SynthStrip's own weights. `vox2seglam prepare --freesurfer-home`
  uses an installation instead.
- `--postprocess` enforces the anatomical protocol on the scan's own voxels, from the
  network's probabilities rather than from carried labels, and re-checks the rules on the
  file it writes before reporting them.
- `vox2seglam prepare-dataset` turns native scans, segmentations and a landmark coordinate
  table into a training dataset on the reference grid, with a ready configuration, optional
  splits and a per-subject registration check.
- Weights and the reference grid are published separately and fetched on demand, only with
  consent, verified against a checksum. `vox2seglam fetch-assets` fills the cache from a
  machine that has a network, for compute nodes that do not.
- The same scan predicted twice gives the same files, byte for byte: the reductions in the
  graph steps have no fixed order on a GPU, so a prediction asks torch for deterministic
  algorithms.
- 262 tests that need no dataset, on Linux, macOS and Windows and on Python 3.11 and 3.13.

[docs/reproducing.md](docs/reproducing.md) records how the two published checkpoints were
built, and where this code differs from the one that produced the paper's runs.
