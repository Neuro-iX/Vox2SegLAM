# The two deposits

The files too large for the repository, or that belong to the study rather than to the code,
are published on Zenodo in **two** records. A trained model is not a dataset: the weights carry
the HCP's terms and are what the software runs on, the annotations are what a reader needs to
redo the experiments, and the two are cited for different reasons.

| | record | what it holds | size |
|---|---|---|---|
| **Models** | [doi:10.5281/zenodo.24205672](https://doi.org/10.5281/zenodo.24205672) | the two trained models, their configurations, the fitted post-processing, the reference grid, the subject lists | 96 MB |
| **Annotations** | [doi:10.5281/zenodo.24206546](https://doi.org/10.5281/zenodo.24206546) | the landmark annotations and the tables that describe the experiments | 0.2 MB |

Both are **CC BY 4.0**; the software is MIT. Nothing has to be downloaded by hand: the first
command that needs a published file names it, asks once, fetches it from the models record,
checks it against the checksum in `assets.json` and keeps it under `~/.cache/vox2seglam`. The
annotations record is never downloaded by the code.

```bash
git clone https://github.com/Neuro-iX/Vox2SegLAM.git && cd Vox2SegLAM && pip install -e .
vox2seglam predict --input my_scan.nii.gz --output-dir my_output/

# where the machine doing the work has no network
VOX2SEGLAM_ASSETS=/shared/vox2seglam vox2seglam fetch-assets
```

[models.md](models.md) and [annotations.md](annotations.md) are the descriptions the two records
carry, each listing its files one by one. [metadata-models.json](metadata-models.json) and
[metadata-annotations.json](metadata-annotations.json) are the deposition metadata behind them.

## What is in neither

The MRI volumes. The HOA subjects are from the
[Human Connectome Project](https://www.humanconnectome.org/) under its data use terms, the
out-of-domain scans on OpenNeuro under each record's own licence; every table names its
subjects, so the images can be fetched from the source and matched by `participant_id`.
SynthStrip's weights are not deposited either — the code downloads them from their authors,
with a pinned checksum.

## Citing

Rekik, A. *Vox2SegLAM: joint subcortical segmentation and anatomical landmark detection on
brain MRI*, version 1.0.0, 2026. doi:10.5281/zenodo.24205672

If you use the skull stripper, cite SynthStrip as well: A. Hoopes, J. S. Mora, A. V. Dalca,
B. Fischl, M. Hoffmann, *SynthStrip: skull-stripping for any brain image*, NeuroImage 260
(2022) 119474.
