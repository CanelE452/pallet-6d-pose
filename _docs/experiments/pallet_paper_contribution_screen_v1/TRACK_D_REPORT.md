# Track D — D_COMPLEMENTARITY_NOT_PREDICTABLE

Teacher: frozen A local line seed1; corrected DHT seed1 is a descriptive control,
not the main teacher. Teacher train55980; out-of-training heldout1985 frames,
split by scenario into1187 train/417 calibration/381 test. Zero stored image-hash
and scenario overlap.15476 eligible edges from1978 usable frames.

Three hidden64/dropout0.1 MLPs each completed1500 updates. All fifteen calibration
seed/threshold combinations failed the coverage/harm safety combination. For
tau0.25, coverage was17.56%/12.24%/14.15%, mean gain0.373/0.382/0.377px, but harm
35.03%/38.44%/36.09% exceeded25%. These are calibration diagnostics, not selected
test performance. Each seed therefore froze abstention: test selected coverage0,
gain/harm undefined. No test threshold was tried to rescue the result.

On1884 overlapping control edges, local teacher mean normal gain0.12255px versus
corrected DHT -2.39420px. These are teacher-specific normal directions; not an
identical scalar error target or a student benefit claim.

Student fits0. D2 vs D1 NOT_RUN. Evidence level MECHANISM_ONLY. The float32 JSON
serialization correction occurred before any trust/student update; no tuning.
