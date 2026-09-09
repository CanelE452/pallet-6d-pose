# Final dimension-conditioning probe report

Status: **STOP — no deployable parity head.** This is a real-DEV diagnostic,
is not eligible for a paper final table, and did not tune on real labels.

[CONTRACT]

checkpoint: `challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt`

sha: `1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c`

commit: the checkpoint contains no commit. Current repository HEAD is
`f8472ac08cdbf748754dacbca4b54fb4c576d666`; reflog reconstruction suggests
`152012b6757d792f8e32179815523d14f10ca85c` at training time, but that is not a
checkpoint-bound or clean-source claim.

training runs: 24 tiny probes = 4 learned arms × 2 architectures × 3 seeds.
B4/B5 reused B3 without retraining. Full YOLO runs: 0.

YOLO parameter updates: 0. Python 3.10.20, torch 2.1.1+cu118, Ultralytics
8.4.60, NVIDIA RTX 3080.

[OBJECT TYPES]

plastic: `plastic_standard_110x130x11`; canonical `(x,y,z) =
(1.10, 0.11, 1.30) m`; cuboid diameter `1.7064876208 m`; legacy W/D/H tuple
`(1.10, 1.30, 0.11) m`; frozen 180-degree yaw evaluation symmetry.

wood: `wood_small_80x59x14`; canonical `(x,y,z) = (0.80, 0.14, 0.59) m`;
cuboid diameter `1.0038426171 m`; legacy W/D/H tuple `(0.80, 0.59, 0.14) m`.
Wood symmetry is unreviewed and its intrinsics are sensor-profile-scaled, so
wood pose evaluation is blocked.

The authoritative registry remained unchanged at SHA-256
`0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627`.
Derived diameter, legacy-order, and evidence fields are recorded in the probe
snapshot rather than mutating that authority.

[PHASE A ORACLE]

Population: frozen Plastic DEV140. A1 is explicitly `ORACLE`,
`GT_USED_FOR_PARITY`, `PAPER_ELIGIBLE = FALSE`.

| metric | A0 current | A1 oracle parity | A3 GT-v2 keypoint replay |
| --- | ---: | ---: | ---: |
| selector accuracy | 59.29% | 100.00% | 100.00% |
| PnP solve rate | 138/140 | 138/140 | 140/140 |
| pose valid | 138/140 | 138/140 | 140/140 |
| Restricted ADD-S AUC | 0.264757 | 0.314657 | 0.999307 |
| symmetry-aware R median | 8.067 deg | 3.671 deg | 0.000 deg |
| t median | 0.1326 m | 0.1055 m | ~0 m |
| symmetry-aware yaw median | 7.898 deg | 3.008 deg | ~0 deg |

Session-cluster 95% bootstrap intervals (1,000 replicates) are in
`phase_a_oracle/ORACLE_BOOTSTRAP.json`. They are wide because there are only
seven session clusters.

A3 is a circular GT-v2 annotation/PnP consistency replay, not an independent
annotation/intrinsics noise-floor estimate. The separate GT-keypoint static
mapping check passed 140/140. Predicted-keypoint A2 agreed with GT parity on
118/138 solved frames; A2 is a GT-pose-error oracle, so those disagreements do
not imply a static mapping defect.

The labelled A4 plastic-image/wood-dimension control drove AUC to 0 and median
translation error to 0.8903 m, confirming that the metric is dimension
sensitive. The reverse wood pose direction was not run.

[ORACLE GATE]

G1: PASS — A1 parity is 100%; A0 fails the frozen selector gate (overall
59.29%, NIGHT 46.43%, minimum session 33.33%).

G2: PASS — 4/5 conditions: AUC +0.0499; rotation −54.50%; translation
−20.45%; yaw −61.91%; pose-valid delta 0 pp.

G3: PASS under the implemented primary-population `ALL` interpretation. The
user protocol fixed the threshold but did not explicitly say whether G3 also
ranges over every subgroup; G1, by contrast, names its subgroup checks. Under
a strict every-subgroup interpretation G3 would FAIL: NIGHT median translation
worsened from 0.5716 m to 0.8110 m (+41.88%), while NIGHT AUC remained 0.088.
In that reading, Phases B/C are exploratory extra evidence rather than a
gate-authorized continuation. Either interpretation leads to the same final
no-deploy/no-full-training decision.

G4: PASS — 38/57 selector-error frames (66.67%) met the frozen recovery rule.

verdict: `ORACLE_PARITY_HEADROOM_PRESENT`.

[SYNTH DATA]

unique dimensions: 40,000 exact fixed-axis xyz triplets; 39,933 unique after
millimetre rounding.

aspect ratios: 40,000 exact x/z ratios; 38,070 unique at six decimals.

parity balance: SHORT 20,625/40,000 (51.56%), LONG 19,375/40,000 (48.44%).
Every split is within 40–60%.

asset-disjoint: the tiny-head split is source-asset-ID disjoint — TRAIN 20,281,
DEV 9,624, TEST 10,095 — but only provisionally. Two source meshes are absent
on this machine, so topology separation is not certified. More importantly,
the frozen YOLO was already trained on the original G38 membership spanning all
four assets. All 10,095 probe TEST frames were exposed upstream: 9,591 were in
the original G38 train split and 504 in its val split. Synthetic TEST is held
out from the tiny probe head only, not from the feature extractor, and is not
an end-to-end held-out claim.

new render required: **yes before any plastic+wood/multishape claim.** Wood
`z=0.59 m` is below the observed synthetic minimum `z=0.82777 m`; no current
frame is within 40% on all wood axes. `DIM_PROBE_SYNTH` must be materialized and
frozen first. No invented render membership was created. The present
plastic-only STOP conclusion does not require that render.

Plastic-neighborhood caveat: source TRAIN has only 36 frames within 20% on all
three axes of the real plastic dimensions and none within 10%; the nearest
plastic-like frame is in probe TEST. The result therefore rejects this current
G38 split/probe as deployable, not dimension conditioning in every possible
data design.

[PARITY PROBE]

Selected architecture/seed was locked using synthetic DEV only. Values below
are all-frame accuracy; missed real detections count as failures. Plastic ALL
is the frozen COMMON_DEV_PLASTIC_POS128 model-comparison population. Wood is an
out-of-range parity diagnostic only.

| arm | SYNTH TEST* | PLASTIC ALL (128) | NIGHT (28) | WOOD (45) |
| --- | ---: | ---: | ---: | ---: |
| B0 dims-only | 49.97% | 55.47% | 28.57% | 62.22% |
| B1 kp-only | 80.94% | 53.91% | 92.86% | 84.44% |
| B2 image+kp | 81.13% | 51.56% | 82.14% | 91.11% |
| B3 image+kp+dims | 85.47% | 60.16% | 39.29% | 88.89% |
| B4 shuffled dims | 82.83% | 53.91% | 39.29% | 66.67% |

`*` Synthetic TEST is probe-head-heldout only, with the upstream-exposure and
topology caveats above.

Phase-C gate on COMMON128: P1 FAIL, P2 FAIL, P3 FAIL (worst session 16.67%),
P4 PASS (+11/128 = +8.59 pp versus B2), P5 PASS (+8/128 = +6.25 pp versus B4),
P6 FAIL, P7 PASS, P8 PASS, P9 not reached.

[POSE RECOVERY]

Population: COMMON_DEV_PLASTIC_POS128, with A0/A1 re-summarized from the frozen
Phase-A per-frame artifact on the exact same 128 IDs.

| arm | Restricted ADD-S AUC | pose valid | R median | t median | yaw median |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 current | 0.272363 | 126/128 | 9.046 deg | 0.1289 m | 8.409 deg |
| B1 kp-only | 0.187605 | 126/128 | 11.739 deg | 0.1398 m | 11.490 deg |
| B2 image+kp | 0.190410 | 126/128 | 19.649 deg | 0.1403 m | 19.313 deg |
| B3 image+kp+dims | 0.273602 | 126/128 | 7.323 deg | 0.1178 m | 6.950 deg |
| A1 oracle | 0.320719 | 126/128 | 4.079 deg | 0.1024 m | 3.055 deg |

oracle recovery ratio: `(0.273602 - 0.272363) / (0.320719 - 0.272363)
= 0.02561`, or **2.56%**, versus the required 70%.

[COUNTERFACTUAL]

correct dims: B3 77/128 = 60.16%.

shuffled dims: B4 69/128 = 53.91%; exact overall delta +8/128 = +6.25 pp.
The derangement is bound to the registered COMMON128+Wood45 union; all 12
FT_EVAL_LEAK-excluded DEV140 rows remain untouched. It changed the actual
triplet for 31/128 COMMON128 rows. On those rows B3 was 70.97% versus B4 45.16%
(+25.81 pp); the 97 unchanged-triplet rows had identical predictions and
56.70% accuracy.

wrong type: B5 61/128 = 47.66%. This labelled control changes only the parity
head input; PnP retains the true registry dimensions, so no wrong-scale pose
claim is made.

[LATENCY]

YOLO: `NOT_RUN_NO_GATE_WINNER`

parity head: `NOT_RUN_NO_GATE_WINNER`

PnP: `NOT_RUN_NO_GATE_WINNER`

end-to-end: `NOT_RUN_NO_GATE_WINNER`

relative increase: `NOT_APPLICABLE`

P1–P8 did not pass, so P9 was not authorized. No deployment wrapper or
integration artifacts were created; this is required by the Phase-D gate.

[CAUSE]

`PARITY_INFORMATION_NOT_RECOVERED_FROM_CURRENT_FEATURES`

Q1: the current selector is a meaningful overall pose bottleneck: the oracle
gate passed and A1 added 0.0499 AUC. It is not a complete explanation of NIGHT
pose behavior, where AUC did not improve and translation worsened.

Q2: dimensions alone are insufficient (B0 is chance-like on synthetic TEST and
55.47% on real). Image+keypoints+dimensions has measurable incremental signal,
but none of B1/B2/B3 is deployable.

Q3: no. B3 recovered only 2.56% of oracle AUC headroom.

Q4: the oracle gap plus the B3–B4 dependence leaves a rationale to investigate
a representation that exposes spatial parity information, but it is not
evidence to adopt keypoint-head dimension conditioning now. Current frozen
features fail the real gate.

[NEXT]

Do not deploy B1/B2/B3 and do not add a wrapper. If multishape evidence is
needed, first create the specified 200-image render smoke test and then the
8,000-image `DIM_PROBE_SYNTH`, with independent dimension/topology/texture/
background/viewpoint sampling and a genuinely unseen TEST. Separately approve
and audit wood symmetry/intrinsics before wood pose metrics.

Only after that, and under separate approval, design (do not silently
implement) either a zero-initialized FiLM experiment or a spatial parity
encoder. Full YOLO fine-tuning remains forbidden until separately approved.

[VERIFY]

real used for tuning = 0

real used for final evaluation only = DEV diagnostic only; paper-final eligible
= false

checkpoint unchanged = yes; before/after SHA identical

YOLO predictions unchanged = yes; candidate count/confidence/box/keypoint max
absolute differences are exactly 0; mapped source-logit confidence difference
is `8.63e-8`

existing GT modified = 0; audited Plastic140 and Wood45 source snapshots match

existing geometry registry modified = 0

new full training = 0

YOLO parameter updates = 0

forbidden threshold metric computed/reported = 0
