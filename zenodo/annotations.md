# Vox2SegLAM 1.0.0: landmark annotations, split assignment and label tables

The anatomical landmark annotations behind Vox2SegLAM, and the tables that say which subject
belongs to which split and what each class integer means. This is what a reader needs to redo
the experiments; the code never downloads from this record.

Vox2SegLAM predicts subcortical structures and anatomical landmarks from a T1-weighted brain
MRI. The code is at <https://github.com/Neuro-iX/Vox2SegLAM>; the trained models, the reference
grid and the subject lists are deposited at
[doi:10.5281/zenodo.24205672](https://doi.org/10.5281/zenodo.24205672).

## Files

| File | Subjects | Rows |
|---|---|---|
| `landmarks_hoa_100subjects.csv` | 100 HCP subjects, on the reference grid | 1600 |
| `landmarks_outofdomain_25subjects.csv` | 25 subjects of five public cohorts, in each scan's own space | 400 |
| `landmarks16_definition.csv` | the 16 landmarks: id, name, short name, side | 16 |
| `outofdomain_subjects.csv` | the 25 out-of-domain subjects: cohort, age, sex, scanner, voxel size, integrity | 25 |
| `splits.csv` | which subject belongs to which split file of the campaign | 463 |
| `vox2seglam-structures16_dseg.tsv` | what each integer of the 16-structure class map means | 17 |
| `vox2seglam-structures39_dseg.tsv` | the same for the 39-structure model, with the FreeSurfer label of each | 40 |

Both landmark tables carry `participant_id, landmark_id, x_mm, y_mm, z_mm` and
`coordinate_system` (LPS), which is what the code reads and what any annotation program can
export. The out-of-domain subjects come from five OpenNeuro cohorts (ds000115, ds003499,
ds003568, ds003763, ds006067), three scanner vendors, voxel sizes from 0.49 to 1.0 mm and ages
from 8 to 63; none was seen in training.

The two `_dseg.tsv` files are label lookup tables in the form BIDS defines for a discrete
segmentation: `index` and `name`, plus the short form and, for the 39-structure set, the
FreeSurfer label. The two sets are numbered differently, so a class map must be read with its
own table. `CHECKSUMS.txt` carries the sha256 of everything here.

## What is not here

The MRI volumes. The HOA subjects are from the
[Human Connectome Project](https://www.humanconnectome.org/) under its data use terms, the
out-of-domain scans on OpenNeuro under each record's own licence; every table names its
subjects, so the images can be fetched from the source and matched by `participant_id`.

## Licence

**CC BY 4.0.** These are subject identifiers and landmark coordinates, not imaging data and not
clinical variables.

## Citing

Rekik, A. *Vox2SegLAM 1.0.0: landmark annotations, split assignment and label tables*, 2026.
doi:10.5281/zenodo.24206546

The software and the trained models are cited as: Rekik, A. *Vox2SegLAM: joint subcortical
segmentation and anatomical landmark detection on brain MRI*, version 1.0.0, 2026.
doi:10.5281/zenodo.24205672
