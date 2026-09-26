# Experiment roles

| Artifact family | Role |
| --- | --- |
| R0 | MAIN_BASELINE |
| R1_NAIVE/P43/P44 | NOT_COMPARABLE |
| Replay teacher last300 (9/38) | MAIN_CORRECTED_PSEUDO |
| pose_only RAW_LR4/RAW_LR5 | MAIN_RAW_PSEUDO |
| pose_only REF_LR4/REF_LR5 | MAIN_CORRECTED_STUDENT |
| pose_repeat RAW/REF ORDER43/44 | MAIN_ORDER_SENSITIVITY |
| pose_only/repeat SYN | SOURCE_UPDATE_CONTROL |
| Clean19 OLD_STUDENT/S0/S1/S2 | EXTENSION_OCCLUSION |
| H_MANUAL hard8 | EXTENSION_MANUAL_HARD |
| HMAN_SPECIFIC_GEO_LINEAR | EXTENSION_SELECTOR |
| ALL300 | DIAGNOSTIC_ONLY |
| FINAL_V2 visible66 | MAIN_Q1_FIXED_ID |
| HELDOUT128 | MAIN_REUSED_DEV |
| source256 | DIAGNOSTIC_ONLY |
| old reserves | PROVENANCE_AUDIT_AFTER_FREEZE |

## Core mapping

R0 / RAW_LR5 / REF_LR5 main. LR4 and ORDER43/44 complete sensitivity, SYN source-only controls. Same frozen Replay9/38 for Q1 and pseudo generation. Clean19 teacher never substituted. Newfits=0. Existing12 student fits reused.
