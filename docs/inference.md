# Inference

What the prediction commands write, and which model they use.

## On a scan of your own

```bash
vox2seglam predict --input my_scan.nii.gz --output-dir my_output/ --postprocess
```

A raw T1w volume goes in, at any orientation and resolution. The model works on one fixed
grid — LAS, 0.7 mm, 240 × 304 × 240, skull-stripped, bias-corrected — so the scan is registered
onto it, stripped, corrected, predicted there, and every output mapped back. `--postprocess`
enforces the anatomical protocol on the segmentation and re-verifies it on the file it writes,
reporting one line per scan; anything other than `verified` names the structures still in
breach and is a defect worth reporting, the figures behind it being in the markups file under
`vox2seglam.postprocess`. It exists for the subcortical model only, and `--input` also takes a
directory.

A run over three scans leaves:

```
my_output/
├── 136833_T1w_landmarks.mrk.json      23 kB   3D Slicer markups, one list per scan
├── 136833_T1w_seg.nii.gz              49 kB   class map on the scan's own voxel grid
├── 144832_T1w_landmarks.mrk.json
├── 144832_T1w_seg.nii.gz
├── 792564_T1w_landmarks.mrk.json
├── 792564_T1w_seg.nii.gz
├── landmarks.csv                       5 kB   every subject's landmarks, in millimetres
├── structures.tsv                     589 B   what each integer of the class map means
└── prepared/                                 the registered, stripped, bias-corrected scans
```

Two files per scan, two tables for the run, and the prepared volumes beside them -- by far the
largest thing there, kept so that a second run skips the preparation. The markups file opens in
3D Slicer by drag and drop, labelled with the anatomy, and validates against Slicer's own
schema; beside the points, a `vox2seglam` key records each landmark in four coordinate systems,
the checkpoint, and what the post-processing did. The class map is a plain NIfTI, carrying the
scan's origin and orientation so that it overlays the image it came from.

The two tables are what a script reads. `landmarks.csv` is the form `vox2seglam prepare-dataset` turns
into a training set and `vox2seglam evaluate --landmarks` scores against:

```
participant_id,landmark_id,landmark_name,coordinate_system,x_mm,y_mm,z_mm,voxel_x,voxel_y,voxel_z
136833_T1w,1,Left Putamen Anterior Start,LPS,18.3699,-18.1729,-3.6042,144.8141,202.9613,87.7083
```

`structures.tsv` is a label lookup table in the form BIDS defines for a discrete segmentation:

```
index	name	abbreviation
1	Lateral Ventricle Left	LV_L
```

Predicting the same scan twice produces byte-identical files.

Preparation costs about 50 seconds and 32 MB a scan, so it is kept in `<output-dir>/prepared`
and reused by later runs writing there; it is redone, with the reason printed, when the scan,
the reference, the stripper or the settings have changed. A run with a *different*
`--output-dir` therefore prepares again — point `--prepared-dir` at the first one to share the
work, or prepare a whole cohort once:

```bash
vox2seglam prepare --input scans/ --out-dir prepared/
vox2seglam predict --prepared-dir prepared/ --output-dir first/ --postprocess
vox2seglam predict --prepared-dir prepared/ --output-dir second/ --task full-brain
```

## Which model

| command | structures | landmarks | test error |
|---|---|---|---|
| `--task subcortical` *(default)* | 16 | 16 | 1.103 mm, Dice 0.914 (HOA test, 10 subjects) |
| `--task full-brain` | 39 | 32 | 1.450 mm landmarks (AFIDs test, 22 subjects) |

Each is a 36 MB download. Both label sets are named in [structures.md](structures.md), and how
the two checkpoints were built is in [the changelog](../CHANGELOG.md).

## What it fetches

Three files, on first use, into `$VOX2SEGLAM_ASSETS` or `~/.cache/vox2seglam/`: the weights
with their configuration, the reference template, and SynthStrip's weights, 90 MB in all
(`--postprocess` fetches a fourth, the fitted post-processing, under 3 kB). The
command names each and asks once; a run with no terminal is never prompted and fails with the
paths it was looking for, and `--download` / `--no-download` answer in advance. Skull stripping
is SynthStrip, because that is what the checkpoints were trained on; the default route needs
nothing installed and `--already-stripped` skips it; `vox2seglam prepare --freesurfer-home`
uses a FreeSurfer installation instead. **On a cluster, fetch them where there is a network:**

```bash
VOX2SEGLAM_ASSETS=/shared/vox2seglam vox2seglam fetch-assets
vox2seglam fetch-assets --show-assets          # what is where, fetching nothing
```

## On volumes already on the reference grid

`vox2seglam predict-prepared` skips the registration, which is how the published measurements are
reproduced. Its inputs must already be on the reference grid, skull-stripped, bias-corrected
and in the training orientation; `--show-contract` prints what a checkpoint expects.

```bash
vox2seglam predict-prepared --checkpoint experiments/hoa16f_med_jnt_s1/best.pt \
    --input subject_T1w.nii.gz --output-dir predictions/ --save-segmentation
```

The same from Python, through `Predictor.from_checkpoint(...).predict_file(...)`, which reads
the configuration beside the checkpoint so that architecture and normalisation match the
weights.

<sub>[&larr; back to the README](../README.md)</sub>
