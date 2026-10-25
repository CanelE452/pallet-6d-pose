# V249 SOURCE oracle RGB-reference diagnostic

Teacher-selected true-object bank is an oracle diagnostic, not deployable input. No optimizer/encoder/head training; reference-bank statistics fitted on SOURCE FIT-D. No threshold/best-view selection or original-query correction.

|View|D / E|AUC E=-targetcos|AUC E=-margin|D target>other|
|---|---:|---:|---:|---:|
|full_CLS|340 / 57|0.8305985552115583|0.8454592363261094|0.8823529411764706|
|fixed_centre_Q|340 / 57|0.9065531475748194|0.8950980392156863|0.788235294117647|
|centre_CLS|340 / 57|0.8133642930856553|0.9837461300309598|0.9529411764705882|

All3 views, every predeclared q-r stratum, per-object groups, unsupported/empty strata and full distributions are in [DESCRIPTIVE_TABLES.json](DESCRIPTIVE_TABLES.json). No view is selected.

All source roles/labels/q/L and frozen features unchanged; no q/r correction or owner filter.
Associated-r D reference is an equally weighted q crop feature, not a target-only clean visible cutout.
Teacher object ID chooses an oracle bank, not a deployable input or reference retriever.
SOURCE DEV is exposed/reused; no native physical gold, causal mechanism or independent generalization certification.
AUC direction fixed to E=-nominalcos and E=-margin. Wrong cosine is reported as a distribution only.
No optimizer, encoder/head forward, RGB/weights/current/held input, PnP or label/role modification.
