# The anatomical protocol

The rules that put a structure boundary on a landmark plane, and the post-processing that enforces them.

`vox2seglam/protocol/hoa_protocol.pql` describes the HOA protocol in a small language: planes
are defined from landmarks (`sagittal_plane(LM10)`, `axial_plane(LM13, LM5)`) and each region
gets side rules relative to them (`left_to`, `anterior_to`, `superior_to`, combined with `and`
and `or`), carrying the label a failing voxel falls back to:

```
PUT_L = (posterior_to(putamen_start_L) and
         strictly_anterior_to(put_nacc_contact_ant_L, NACC_L)) or (...)
```

It is a language rather than a YAML tree so that an anatomist can read and challenge it, and
its vocabulary is checked at parse time, so a typo is an error. `protocol/geometry.py` is the
single implementation of the planes and the rule test, shared by the post-processing, the
metrics and the ground-truth check: a plane built from one landmark is normal to the scanner
axis of its type, from two it is the plane of that type containing the line joining them.

## Post-processing

The planes are built from the predicted landmarks; one after the other, each may slide along
its normal within the interquartile range of this model's own plane errors on the validation
subjects, stopping where enforcing the protocol destroys the least network confidence. The
coronal planes that end a structure or a contact zone are instead put on the section of their
landmark, as in the reference. Every voxel then takes the most probable class the protocol
allows. These are the defaults of `vox2seglam fit-postprocess`:

```bash
vox2seglam fit-postprocess --checkpoint experiments/hoa16f_med_jnt_s1/best.pt
vox2seglam predict --checkpoint experiments/hoa16f_med_jnt_s1/best.pt \
    --input subject_T1w.nii.gz --postprocess
```

The ranges are fitted on the validation split and the test split is never read; the values
and their provenance go to `postprocess.json` beside the checkpoint. `--percentiles`,
`--section-planes` and `--tolerance` change the recipe, the last giving voxels to the planes
the training ground truth itself crosses, and 0 — the exact protocol — by default.

## Enforcing it on a prediction

`vox2seglam predict --postprocess` enforces these rules on a predicted segmentation, and the class map
it writes is strictly compliant **on the scan's own grid**, not only on the reference grid.

That map is *built* there rather than carried onto it: the network's float32 probabilities are
resampled trilinearly onto the scan's voxels and the enforcement runs on them, so no label is
moved by a nearest neighbour and each voxel takes the most probable class its rules admit. The
plane shifts still come from the fit on the reference grid, whose ranges are reference voxels;
refitting them here would no longer be the published post-processing. Enforcement is strict —
every plane exact, no tolerance, the fitted tolerance existing only so that the rules do not
correct the HOA 2.0 annotations where those cross their own planes. The rules are then
re-verified on that same grid; what the command prints is in [inference.md](inference.md).

<sub>[← back to the README](../README.md)</sub>
