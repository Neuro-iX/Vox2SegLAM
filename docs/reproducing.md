# Reproducing the published runs

How the two published checkpoints were built, and where this code differs from the one that
produced the numbers in the paper. Read this before comparing your own run against ours.

## The two checkpoints

**16 structures, 16 landmarks** is one run, `hoa16f_med_jnt_s1`. Its seed was chosen on the
test split: s1 reached 1.103 mm against 1.163 mm for s0 while the two were tied on
validation. The checkpoint records the ablation name the run was launched under,
`abl_fully_joint_s1`; the two configurations are the same.

**39 structures, 32 landmarks** was trained on a frozen U-Net taken from a 39-structure model
and kept no segmentation layers of its own. The published file carries those layers back,
grafted after checking that the 102 U-Net tensors are identical: the segmentation is then
that 39-structure model's and the landmarks are the 32-landmark model's, differing by about
1e-4 mm from the two runs taken separately. The AFIDs seeds were tied on test (1.4499 mm
each) and seed 0 was kept because its segmentation scored higher.

Both are single runs, so their numbers differ slightly from the seed-averaged ones reported
in the paper.

## Where this code differs from the campaign's

A pre-publication audit found four defects in the code that produced the runs. Three of them
change numbers, so a run of the campaign cannot be reproduced bit for bit with this release.

- **Last gradient-accumulation window** (`training/trainer.py`): with `train.grad_accum: 2`
  and an odd number of batches per epoch, the final batch's gradient was computed and then
  dropped. This affects `hoa_cc_train` (73 subjects) and the AFIDs `n11` subset (11 subjects,
  about 9 % of every epoch), which among the shipped recipes is `hoa40f_med_stg_s0`. Splits
  of 80, 88, 22, 44 and 66 subjects are unaffected.
- **Surface distances for a structure missing from one of the two maps**
  (`evaluate/metrics.py`): `distance_transform_edt` returned the distance to the edge of the
  crop, about 15 mm, which entered ASSD, HD, HD95 and their means. Such a structure now
  reports `nan`, which the mean skips; Dice, precision and recall are unchanged.
- **Section planes of the post-processing** (`postprocess/plane_shift.py`) are placed before
  the sliding planes are searched; their searched position used to serve as the reference for
  every plane later in file order. This affects the post-processed columns only.
- **`_fixed_term_names`** (`training/trainer.py`) grouped the loss terms wrongly when the
  U-Net is shared without a segmentation head. No published number changes: the 30 recipes
  concerned (`final/afids_v2l/`) set `loss.adaptive_weighting: false`, so the value was never
  used.

## Two ablations are not shipped

`no_segmentation` trained under a loss weighting the baseline does not share, and
`no_refiner` hit a curriculum gate that never opened without a refiner, so it trained one
graph step instead of two. Disregard any number reported from either.

<sub>[← back to the README](../README.md)</sub>
