# Vox2SegLAM 1.0.0: pre-trained models, reference grid and subject lists

Vox2SegLAM predicts subcortical structures and anatomical landmarks from a T1-weighted brain
MRI, and can enforce the anatomical protocol that puts a structure boundary on a landmark
plane. This record carries the trained models and everything the model needs to run. The code
is at <https://github.com/Neuro-iX/Vox2SegLAM> and fetches these files from here on demand.

The landmark annotations and the tables that describe the experiments are deposited separately,
at [doi:10.5281/zenodo.24206546](https://doi.org/10.5281/zenodo.24206546): a trained model is not a
dataset, and the two are cited for different reasons.

## Using them

Nothing has to be downloaded by hand. The first command that needs a file names it, asks once,
fetches it, checks it against the checksum in `assets.json` and keeps it under
`~/.cache/vox2seglam`:

```bash
git clone https://github.com/Neuro-iX/Vox2SegLAM.git && cd Vox2SegLAM && pip install -e .
vox2seglam predict --input my_scan.nii.gz --output-dir my_output/
```

Where the machine doing the work has no network, fetch them elsewhere into a shared directory
and point at it:

```bash
VOX2SEGLAM_ASSETS=/shared/vox2seglam vox2seglam fetch-assets
```

## Files

### Trained models

| File | Structures | Landmarks |
|---|---|---|
| `vox2seglam-structures16_landmarks16.pt` + `.yaml` | 16 | 16 |
| `vox2seglam-structures16_landmarks16-postprocess.json` | the fitted protocol post-processing of that model | |
| `vox2seglam-structures39_landmarks32.pt` + `.yaml` | 39 | 32 |

Each checkpoint holds what inference reads — the weights, the epoch and the name of the recipe
— and not the optimiser state. The `.yaml` beside it is the configuration the weights were
trained with, paths of the training machine reset, so that the architecture and the
preprocessing always match the weights. The 32 landmarks of the second model follow the
[AFIDs protocol](https://doi.org/10.1038/s41597-023-02330-9) and keep its numbering.

### Reference grid

`reference_template_mean.nii.gz`, the grid every scan is registered onto before the model
sees it: 240 × 304 × 240 voxels, 0.7 mm isotropic, LAS. It is the mean of the 80 training
subjects after registration.

### Subject lists

`vox2seglam-splits.zip`, the fourteen id lists the published runs were trained on, one subject
per line, with a README explaining how the split families relate. They are the experimental
protocol: without them a run cannot be reproduced. `splits.csv`, in the annotations record,
carries the same assignment as one table.

`assets.json` carries the size and md5 of every file the code fetches; `CHECKSUMS.txt` the
sha256 of everything here.

## What is not here

The MRI volumes. The HOA subjects are from the
[Human Connectome Project](https://www.humanconnectome.org/) under its data use terms, the
out-of-domain scans on OpenNeuro under each record's own licence. SynthStrip's weights are not
here either — the code downloads them from their authors, with a pinned checksum.

## Licence

**CC BY 4.0.** The software is MIT, in the repository. The models were trained on HCP data: if
you redistribute them or anything derived from them, check that the Human Connectome Project's
terms allow it.

## Citing

Rekik, A. *Vox2SegLAM: joint subcortical segmentation and anatomical landmark detection on
brain MRI*, version 1.0.0, 2026. doi:10.5281/zenodo.24205672

If you use the skull stripper, cite SynthStrip as well: A. Hoopes, J. S. Mora, A. V. Dalca,
B. Fischl, M. Hoffmann, *SynthStrip: skull-stripping for any brain image*, NeuroImage 260
(2022) 119474.
