# Structures and landmarks

What each published model predicts, and the label every class carries in the NIfTI it
writes. A run writes the same table as `structures.tsv` beside its output.

## 16 structures, 16 landmarks — the default model

The structures the protocol governs, and the landmarks on which it places their
boundaries. `--postprocess` exists for this model only. Label 0 is background.

| # | structure | # | landmark |
|---|---|---|---|
| 1 | Lateral Ventricle Left (`LV_L`) | 1 | Left Putamen Anterior Start |
| 2 | Lateral Ventricle Right (`LV_R`) | 2 | Right Putamen Anterior Start |
| 3 | Third Ventricle (`V3`) | 3 | Left NAcc-Putamen Anterior Contact |
| 4 | Nucleus Accumbens Left (`NACC_L`) | 4 | Right NAcc-Putamen Anterior Contact |
| 5 | Nucleus Accumbens Right (`NACC_R`) | 5 | Left NAcc-Putamen Posterior Contact |
| 6 | Putamen Left (`PUT_L`) | 6 | Right NAcc-Putamen Posterior Contact |
| 7 | Putamen Right (`PUT_R`) | 7 | Left NAcc Posterior Limit |
| 8 | Brainstem (`BRAINSTEM`) | 8 | Right NAcc Posterior Limit |
| 9 | Thalamus Left (`TH_L`) | 9 | Third Ventricle Anterior Start |
| 10 | Thalamus Right (`TH_R`) | 10 | Anterior Commissure |
| 11 | Inferior Horn of the Lateral Ventricle Left (`IH_L`) | 11 | Left Mammillary Body |
| 12 | Inferior Horn of the Lateral Ventricle Right (`IH_R`) | 12 | Right Mammillary Body |
| 13 | Anterior Ventral Diencephalon Left (`VDC_A_L`) | 13 | Left Continuity of Atrium |
| 14 | Anterior Ventral Diencephalon Right (`VDC_A_R`) | 14 | Right Continuity of Atrium |
| 15 | Posterior Ventral Diencephalon Left (`VDC_P_L`) | 15 | Posterior Commissure |
| 16 | Posterior Ventral Diencephalon Right (`VDC_P_R`) | 16 | Prepontine Fissure |

The markups file carries the landmark names, so a list opened in 3D Slicer shows the
anatomy rather than a number. Both vocabularies live in
[`vox2seglam/protocol/data/`](../vox2seglam/protocol/data) and are what the protocol's
rules are written against.

## 39 structures, 32 landmarks — `--task full-brain`

The full Harvard-Oxford Atlas 2.0 label set, with the 32 landmarks of the
[AFIDs protocol](https://doi.org/10.1038/s41597-023-02330-9), which keep its numbering.
No protocol post-processing. These labels are numbered differently from the 16 above —
label 1 is the left cerebral white matter here and the left lateral ventricle there — so
a class map must be read with its own table. `FS` is the label the same structure carries
in FreeSurfer, absent for the two ventral diencephalon halves, a split the protocol
introduces.

| # | structure | FS | # | structure | FS |
|---|---|---|---|---|---|
| 1 | Left Cerebral White Matter (`CWM_L`) | 2 | 21 | Right Globus Pallidus (`GP_R`) | 52 |
| 2 | Right Cerebral White Matter (`CWM_R`) | 41 | 22 | Brainstem (`BRAINSTEM`) | 16 |
| 3 | Left Cerebral Cortex (`CGM_L`) | 3 | 23 | Left Thalamus (`TH_L`) | 10 |
| 4 | Right Cerebral Cortex (`CGM_R`) | 42 | 24 | Right Thalamus (`TH_R`) | 49 |
| 5 | Left Cerebellum White Matter (`CBWM_L`) | 7 | 25 | Left Inferior Horn (`IH_L`) | 5 |
| 6 | Right Cerebellum White Matter (`CBWM_R`) | 46 | 26 | Right Inferior Horn (`IH_R`) | 44 |
| 7 | Left Cerebellum Cortex (`CBGM_L`) | 8 | 27 | Left Hippocampal Formation (`HF_L`) | 17 |
| 8 | Right Cerebellum Cortex (`CBGM_R`) | 47 | 28 | Right Hippocampal Formation (`HF_R`) | 53 |
| 9 | Left Lateral Ventricle (`LV_L`) | 4 | 29 | Left Amygdala (`AMY_L`) | 18 |
| 10 | Right Lateral Ventricle (`LV_R`) | 43 | 30 | Right Amygdala (`AMY_R`) | 54 |
| 11 | Cerebrospinal Fluid (`CSF`) | 24 | 31 | Left Ventral Diencephalon Anterior (`VDC_A_L`) | 28 |
| 12 | Third Ventricle (`V3`) | 14 | 32 | Right Ventral Diencephalon Anterior (`VDC_A_R`) | 60 |
| 13 | Fourth Ventricle (`V4`) | 15 | 33 | Left Ventral Diencephalon Posterior (`VDC_P_L`) | -- |
| 14 | Left Nucleus Accumbens (`NACC_L`) | 26 | 34 | Right Ventral Diencephalon Posterior (`VDC_P_R`) | -- |
| 15 | Right Nucleus Accumbens (`NACC_R`) | 58 | 35 | Corpus Callosum Posterior (`CC_POST`) | 251 |
| 16 | Left Caudate (`CAU_L`) | 11 | 36 | Corpus Callosum Mid Posterior (`CC_MPOST`) | 252 |
| 17 | Right Caudate (`CAU_R`) | 50 | 37 | Corpus Callosum Central (`CC_CENT`) | 253 |
| 18 | Left Putamen (`PUT_L`) | 12 | 38 | Corpus Callosum Mid Anterior (`CC_MANT`) | 254 |
| 19 | Right Putamen (`PUT_R`) | 51 | 39 | Corpus Callosum Anterior (`CC_ANT`) | 255 |
| 20 | Left Globus Pallidus (`GP_L`) | 13 | |  |  |

---

<sub>[&larr; back to the README](../README.md)</sub>
