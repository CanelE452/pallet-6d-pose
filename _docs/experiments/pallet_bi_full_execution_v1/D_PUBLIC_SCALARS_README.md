# D diagnostic and S2 calibration saved aggregates

This document projects already saved, sealed aggregate metadata. It adds no fit, reference decode, pose metric, solve, observation forward, threshold search, or causal test. Its observation counts are not API or neural forward counts. It documents completed limited D diagnostics, S2 calibration, and the now-completed S2 full banks, own nine H1 fits, all238 real predictions per arm, and sealed scalar evaluation/independent analyses. It also records the already sealed F configuration choice. It does not certify F own fits/predictions or whole-campaign completion; consult those separate final stage and prediction seals.

## D: four-cell diagnostic and public scalar tables

D uses the limited two-WD-family observation solver, not the original full D1/120-success H1 system. The representative observation seed is 20261018. The canonical graph is the effective post-dedup COMMON factor graph, including fixed typed partials. Its logical factor slots and native target association are held fixed; donor source provenance is recorded honestly. This is algorithmic q/covariance sensitivity conditional on that graph. It does not identify a pure physical observation intervention, certify physical owner identity, or establish independent sources. Transported Q1S1 reproduces the actual BI parsed packet only on the 103 exactly shared-source rows. The remaining 135 source-changing rows stay in the full denominator 238 and in explicit strata.

The original variant, `D_SIGMA_GAMMA_CONFOUNDED`, transported covariance donor reliability Gamma as well as Sigma. The census found 19 changed factors in 18 frames. The repaired variant, `D_FIXED_COMMON_GAMMA`, fixes Gamma to COMMON while transporting q and/or Sigma. Both variants and their original output history are retained. The correction used 72 family dispatches, 64 new optimizer calls, and 916 exact numerical-packet reuses. Those units differ; 916 reuses are not new optimizer calls.

The total after-minus-before change is decomposed into (a) change within the BEFORE-selected WD family and (b) automatic family-readout contribution. Both fixed families are reported. This decomposition is algorithmic, not physical ownership attribution. Strict-source-compatible 103, source-changing 135, full 238, the four original reporting populations, and recording groups remain distinguishable. D saved summaries did not retain fallback/accepted/rejected counts; those are unavailable rather than zero. Continuous summaries use available values, preserve tails and failures, and never zero-impute absent poses. PRIMARY192 equals TEST49 plus REMAINDER143; CAL46 is separate, and ALL238 is supplementary.

All six saved public scalar CSV tables are linked below. Their complete hashes and row counts are authenticated by the private D export seal whose hash is listed below; the public METRICS_SCHEMA.json and SCALAR_CHECKS.json provide the public schema and checks. No R/t poses, raw references, RGB, or observation banks are included.

| Table | Rows | Content |
|---|---:|---|
| [CONTRAST_COUNTS.csv](tables/D_PUBLIC_SCALARS_V3/CONTRAST_COUNTS.csv) | 92 | Paired recoveries/damages, family switches, total/within-family/readout joint-success changes. |
| [DECOMPOSITION_ROWS.csv](tables/D_PUBLIC_SCALARS_V3/DECOMPOSITION_ROWS.csv) | 5712 | Per-ID saved scalar decomposition, including strict/source-changing identity and recording group. |
| [DECOMPOSITION_SUMMARY.csv](tables/D_PUBLIC_SCALARS_V3/DECOMPOSITION_SUMMARY.csv) | 828 | Mean, sample SD/variance, median, P90, P95, maximum for decomposition components. |
| [FIXED_FAMILY_METRICS.csv](tables/D_PUBLIC_SCALARS_V3/FIXED_FAMILY_METRICS.csv) | 368 | Both fixed-family scalar comparisons and denominators. |
| [GAMMA_CORRECTION.csv](tables/D_PUBLIC_SCALARS_V3/GAMMA_CORRECTION.csv) | 12 | Paired saved scalar effects of the reliability confound repair. |
| [POPULATION_METRICS.csv](tables/D_PUBLIC_SCALARS_V3/POPULATION_METRICS.csv) | 138 | Four-cell joint success, continuous distributions, and translation/rotation tails. |

## D: other stored observation seeds

All rows below summarize the existing saved COMMON/BI observations over the same 238 real IDs. No new observation forward or pose solve was run to make these summaries. Only seed 20261018 has the registered four-cell pose diagnostic; these three metadata rows are not three independent full-pipeline repeats. The strict counts express packet/source compatibility, not certified physical correctness.

| Saved observation seed | Frames | q changed | Effective covariance changed | Source/factor divergence | Strict compatible |
|---|---:|---:|---:|---:|---:|
| 20261018 | 238 | 135 | 135 | 135 | 103 |
| 20261019 | 238 | 125 | 125 | 125 | 113 |
| 20261020 | 238 | 111 | 111 | 111 | 127 |

## S2: frozen calibration recipe and population

S2 fixes the observed coordinates and dimensions to COMMON seed 20261018. CAL is development calibration data, not independent validation or an unseen real test set. It contains 46 frames in four recording groups, 241 effective full 2D factors (B=222, I=19), and 26 fixed typed partial-factor groups. The partial groups and their revoked full archives are excluded from the total-error floor fit and remain unchanged. Recording-group frame counts are REC_012=5, REC_021=18, REC_027=3, REC_042=20. These are saved factor/frame counts, not inference-call counts.

The protocol was written before current CAL reference access and fit, after the campaign plan freeze. The registered lexical whitelist returned only 46 CAL numeric objects. The access receipt honestly records that the containing file bytes, which also contain PRIMARY, were read/scanned; PRIMARY numeric objects were neither decoded nor supplied to calibration. The target is the current canonical derived cuboid C2 reference, not a physical material-line/owner oracle. Its provenance differs from historical minimal calibration references. No PRIMARY covariance fit, bias correction, or shrinkage selection was performed.

The zero-centered 80% two-dimensional ellipse target is eᵀ Sigma⁻¹ e <= -2 log(0.2) = 3.2188758248682006. Errors are not recentered; fitted bias parameters=0. Weights are equal recording group, then frame, then effective full point. Conditional input covariance must be strictly SPD. A nonnegative isotropic floor is added; invalid input is rejected without eigenvalue clipping or fitted rotation. The smallest floor f>=1 px² satisfying the weighted 80th-percentile condition is found with 60 bisection steps, upper bound 1e8 px². An empty role falls back to the pooled floor; zero training points raises instead of fabricating a zero floor.

Let rho = clip(nonself_scale_px / scale_CAL_px, 0.5, 2), with full-fit scale_CAL=668.9397863142003 px. C0 retains the current conditional/effective covariance, including pre-existing exact-observation covariance overrides. CP/CB instead use the frozen new-distribution conditional covariance plus the new floor, replacing old fitted covariance once. Both policies retain the fixed typed partial covariance. All compared full-factor reliability weights are one after the separately recorded C0 unit-weight input correction; CP/CB packets were bitwise unchanged by that correction, and this floor fit is Gamma-independent.

- CP: Sigma_CP = Sigma_conditional + f_pool rho² I₂.
- CB: Sigma_CB = Sigma_conditional + [n_role/(n_role+20) f_role_raw + 20/(n_role+20) f_pool] rho² I₂.

The pseudo-count 20 and scale clip were fixed before current-reference access. They were not selected by PRIMARY or by these leave-group summaries. Total-error calibration does not certify physical source correctness or a probability of correct ownership.

| Full-CAL parameter | Saved value |
|---|---:|
| Pooled floor, px² | 115.879824742 |
| B raw / shrunk floor, px² | 115.879824742 / 115.879824742 |
| I raw / shrunk floor, px² | 75.5579259534 / 96.2358227682 |
| B / I shrink weights | 0.917355371901 / 0.487179487179 |
| Effective fitted floor parameters, CP / CB | 1 / 2 |
| Raw fitted scalar parameters, CP / CB | 1 / 3 |
| Bias / learned orientation / prior parameters | 0 / 0 / 0 |

## S2: four leave-recording-group-out summaries

Each fold refits only the same declared scalar-floor recipe on the other CAL recording groups and evaluates the held-out group. Upstream observations/models remain frozen in all four folds. This is descriptive calibration validation within the development CAL set, not independent whole-pipeline validation. No group, large residual, or failed observation is removed. Values just above one reflect saved floating-point arithmetic, not clipping.

| Held-out group | Frames / full factors | Train full factors | Fitted pooled floor px² | C0 coverage / NLL | CP coverage / NLL | CB coverage / NLL |
|---|---:|---:|---:|---:|---:|---:|
| REC_012 | 5 / 27 | 214 | 216.845034231 | 0.506666666667 / 5.47028373742 | 1 / 6.14886449678 | 1 / 6.12451068235 |
| REC_021 | 18 / 90 | 151 | 216.845034231 | 0.449338624339 / 8.67921073207 | 1 / 7.26074908568 | 1 / 7.25345769193 |
| REC_027 | 3 / 6 | 235 | 41.6705887412 | 0.388888888889 / 10.5079974131 | 0.166666666667 / 9.19621393676 | 0.166666666667 / 9.23952129635 |
| REC_042 | 20 / 118 | 123 | 115.879824742 | 0.360714285714 / 543.589908739 | 0.785714285714 / 85.0210429966 | 0.785714285714 / 85.3264229575 |

| Scope | C0 coverage / NLL | CP coverage / NLL | CB coverage / NLL |
|---|---:|---:|---:|
| Full CAL fit (in sample) | 0.426402116402 / 142.061850155 | 0.80753968254 / 27.2533368366 | 0.805224867725 / 27.3710113089 |
| Combined held-out CAL folds | 0.426402116402 / 142.061850155 | 0.738095238095 / 26.906717629 | 0.738095238095 / 26.985978157 |

CP and CB have identical combined held-out 80% coverage (0.7380952380952381). CP has the lower combined held-out NLL (26.90671762895612 versus 26.985978157035465) with fewer effective parameters. These saved descriptive results do not establish a downstream pose/H1 improvement and do not justify a physical owner claim.

## S2: completed full-path outcomes and actual frozen F choice

The corrected matched C0/CP/CB full candidate banks cover real238 and synthetic1003 each, with six arm/domain feature seals. Each has its own three H1 fits and238 real predictions, sealed before new reference scoring. Canonical and independent evaluation agree (independent verification PASS; maximum saved-summary difference 8.753886504564434e-12). The post-bank, selection, and core-diagnostic outputs are now closed and hash-bound. This section copies their saved scalar summaries only.

All three arms use the new matched fullnormal backend, not an original120-success D1 parity adapter. Typednormal-only seed/guards exclude revoked archived2D and tangent residuals; original search caps and both families/visited endpoints are retained. The188-feature framework remains, but old B profile/edge energy and their gap channels are explicitly unavailable through validity masks for all matched arms; no trustworthy zero cost is fabricated. AB reduces to A, and G13 is an explicit G10 alias. Thus the whole gap to original H1 includes this backend/feature contract as well as observation distribution, bank and own-head changes. It is not a pure covariance effect versus original120, nor a physical owner causal claim.

Joint success is one returned pose with T<=5cm and R<=5degrees under unchanged C2, with full binary denominators. Seed order below is20261018/20261019/20261020; these are head initialization repeats on frozen observations, not independent real data.

| Population | C0 joint successes | CP joint successes | CB joint successes |
|---|---:|---:|---:|
| PRIMARY192 | 117/119/118 | 115/112/112 | 111/114/113 |
| TEST49 | 41/42/41 | 42/43/43 | 43/43/43 |
| REMAINDER143 | 76/77/77 | 73/69/69 | 68/71/70 |
| CAL46 | 34/35/36 | 34/35/35 | 35/35/35 |

CP has fewer PRIMARY joint successes than matched C0 in all three seeds; CB also has fewer than C0. TEST49 moves in the opposite direction from REMAINDER143: CP minus C0 is +1/+1/+2 on TEST and -3/-8/-8 on REMAINDER. CB versus CP is mixed across PRIMARY seeds (-4/+2/+1). These are net differences of saved marginal counts, not inferred per-ID recoveries/damages. TEST49 is nested within reused-development PRIMARY192, not an independent new test, and CAL46 was used for calibration.

For the preregistered representative seed20261018, continuous summaries use all192 available PRIMARY errors, sample variance/SD(ddof1), and preserve every tail; no pose is assigned zero error.

| Representative PRIMARY arm | T mean cm | T variance cm² | T SD cm | R mean degrees | R variance degrees² | R SD degrees | T>50 / T>100 / R>45 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 11.318281315 | 772.678656244 | 27.7970979824 | 10.6549537002 | 638.443420318 | 25.2674379453 | 8 / 5 / 20 |
| CP | 11.1427186468 | 756.127511043 | 27.4977728379 | 10.2747676815 | 614.219795902 | 24.7834581102 | 7 / 5 / 19 |
| CB | 12.7517604031 | 861.515373058 | 29.3515821219 | 11.2289806232 | 723.949467419 | 26.9063090635 | 10 / 6 / 20 |

Representative CP lowers T/R means and sample variances versus C0 while joint success falls117→115. CB has111 joint successes and larger T/R means/variances than CP. All36 arm/seed/population summaries have noPose0 and selected-output fallbacktrue0/unknown0. Those fallback fields do not certify all internal optimizer endpoints as accepted; selected-output acceptance/rejection/timeout fields are unknown, while the separate core archive preserves primitive statuses.

The improved CAL80% coverage/NLL above therefore does not imply improved PRIMARY joint success. The completed PRIMARY post-bank counts are C0 generated/unique/eligible good148/148/148, CP148/148/148 and CB150/150/150. Selection losses for the three seeds are C0 31/29/30, CP33/36/36 and CB39/36/37. This distinguishes candidate absence from eligible-but-unselected good poses; it does not identify the causal feature, physical correspondence, or identical sets of good candidate IDs. Local Jacobian rank remains local and is not proof of global pose uniqueness.

The actual frozen F choice is S2_CP, selected by the registered highest median own synthetic VAL joint success, then lower median own VAL loss, then measured production wall. The wall tiebreak was not used.

| S2 configuration | Own VAL joint success seeds18/19/20, each /235 | Median own VAL joint | Median own VAL loss |
|---|---:|---:|---:|
| S2_C0 | 47/47/45 | 47 | 0.10743919156 |
| S2_CP | 48/48/48 | 48 | 0.11785734727 |
| S2_CB | 46/48/48 | 48 | 0.11926387065 |

CP, CB, and S3_BI tie on median VAL joint success48. CP has the lower median VAL loss0.11785734727027568, versus CB0.11926387064951531 and S3_BI0.23336538374423982. C0 has a lower median loss but median joint success47, so it does not pass the first criterion. This records the controller’s existing choice; CP is not the PRIMARY-best S2 arm. PRIMARY was used only for the declared all-negative fallback bit(false), not for configuration ranking.

The S2 choice maps to no PF or material auxiliary action. F retains its own frozen COMMON/BI observation policies, own CAL fit and fullnormal core; neither S2_CP real performance nor its calibration model is transferred as F evidence. F’s own six heads and real prediction seals are separate requirements. Independent new real test count remains0.

Direct C0→CP and CP→CB paired recoveries/damages, continuous deltas, and recording-cluster intervals were not recalculated for this supplement. The currently closed S2 canonical PAIRED file compares each arm with original H1 only. Use the already scheduled [final all-population paired contrasts](tables/PUBLIC_CONTRASTS30/ALL_POPULATION_PAIRED_CONTRASTS.json) for the final direct comparisons; marginal net counts above must not substitute for those paired transitions. The [all40-arm metrics](tables/PUBLIC_METRICS40/ALL_POPULATION_METRICS.csv) supply the complete per-seed/four-population means, sample variances/SD and tails when their final export is sealed.

## Source receipts and input hashes

The following hashes identify the saved evidence used for this text. Private source files are not published by this document. The private binding receipt also verifies the complete six-CSV D artifact hash map and the S2 packet seal entries for SUMMARY, MODEL, PROTOCOL, and CAL access.

| Saved evidence | SHA-256 |
|---|---|
| `S2_CALIBRATION/PACKET_SEAL.json` | `4e7df3f2ab91a783343fb00a40cf9b31559f19f6d37be4745cc2cda06bc05c5c` |
| `S2_CALIBRATION/SUMMARY.json` | `328914695b125de5355c34c1942247b73e122375f18cd71353846373a1da7cae` |
| `S2_CALIBRATION/MODEL.json` | `01db47b51d6bb8cfda10572f78525825ef04cb5f576095a89315f71d1242965a` |
| `S2_CALIBRATION/PROTOCOL.json` | `54de960226603fde185d50ae57e93cfb103770eb9ae9ffa7bc9ce656c2957188` |
| `S2_CALIBRATION/CAL_REFERENCE_ACCESS.json` | `de1ca5a565d46ea0cd2873ae7fa7844bfed092228b40b183fd501664f1eb81ae` |
| `S2_UNIT_WEIGHT/CORRECTION_CENSUS.json` | `86273aa93004cd2da62e23b108a7c239564c613406ba45f142391c8555cd7514` |
| `D_PUBLIC_SCALARS_V3/EXPORT_SEAL.json` | `e1d60838683a76e5fac03252d3cfb478eaee629643eeef0b560e33b1c6b43663` |
| `D_PUBLIC_SCALARS_V3/SCOPE_AND_LIMITATIONS.json` | `7f655f256f54f7c3f514e314cb135a80986cb26df26b1b96da0d996540f17e2d` |
| `D_STRICT/SUMMARY.json` | `866986d2398901527d4ce6b5924be26686383ffedf990f42f64a16e5ae0e5ab1` |
| `D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json` | `cf97d8403422bf6ddc3f8d9323e68737d7046da707c7dae5cafca214c210ddb8` |
| `PUBLIC_RECOVERY_PREPARATION_V1/D_OBSERVATION_SEED_METADATA.json` | `bc10ed74d4c41667fb22a444a98c3413d3bea1af7938a8288bd21942d769b55e` |

This supplement and its input-binding receipt were produced by saved-metadata projection only. No public copy or publication action is certified here.

Additional closed S2/F-choice source hashes for the completed-outcome section:

| Saved evidence | SHA-256 |
|---|---|
| `F_SYSTEM_CHOICE.json` | `2bf98be8998aae12cc1e0622b9bafde4fa054b392177965f8ce26da97f286215` |
| `S2_BANK_ANALYSIS/ANALYSIS_SEAL.json` | `59a70f8928be622b202f1a20199fe2e95042f75e5b865db1a6476c255f68498c` |
| `S2_BANK_ANALYSIS/SUMMARY.json` | `e16ad55141d4c38bb2c1ecdd0a0ab0dac09eb1c4ea2d1b5916b6f48fe0575bb6` |
| `S2_CORE_DIAGNOSTICS/DIAGNOSTIC_SEAL.json` | `e4e92438b95b6a80455aab3b3fe96e3633cfff63eb851d2f7ff03b17407fd085` |
| `S2_CORE_DIAGNOSTICS/SUMMARY_BY_POPULATION_AND_STRATUM.json` | `a7819a95db11900129b935424236d73c097bdcf007bedb1afd8a778e68125f3f` |
| `S2_EVALUATION/EVALUATION_SEAL.json` | `bd96d2b16b21e6feaf189a5083c6daf122279b3487083cd0b750234763ff592b` |
| `S2_EVALUATION/PAIRED.json` | `3d9649af705a84d90a1e34768030f0e62a5cb62b5103f2eb6347d2128b9efabd` |
| `S2_EVALUATION/SUMMARY.json` | `5470ebe29517c88de7d7282dc86b46f34be501a1711f37230b4f7de88f885b39` |
| `S2_INDEPENDENT/INDEPENDENT_PAIRED.json` | `a2bbf6e0329cf46f7237d0587b0017b57ff7299f59a3621b8a86992a9b58f3b5` |
| `S2_INDEPENDENT/INDEPENDENT_SUMMARY.json` | `e4329b41fe4b0827f6f31ad6e84db978b76d9c516a691674da40a5bed10f02f7` |
| `S2_INDEPENDENT/INDEPENDENT_VERIFICATION.json` | `0e5e28a92f92ce0fd089c2f240f63a12100881a417cd3a42864f1e9711bda2cd` |
| `S2_SELECTION_DIAGNOSTIC/SELECTION_DIAGNOSTIC_SEAL.json` | `d240862b0c04b2cf44a0a3fb2316cc4e09df77c992b29a9e392f71758c79d910` |
| `S2_SELECTION_DIAGNOSTIC/SUMMARY.json` | `dca713eeab401cf038facf2ec0f15ebea23bf51b859cbf19e0ac448e96b3d51f` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/ALL_REAL_PREDICTIONS_SEAL.json` | `72c3dfd24f5f2d6a70296cfdafe4f7f72d0a5c1373fd54293ffb092f408fcdd5` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/C0/prediction/PREDICTION_SEAL.json` | `5ec77860dd120ff27a25def30863d26da3043dbfb69a09d7bdabfd372c452c33` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/C0/training/TRAINING_MANIFEST.json` | `971a66bda673b81e42225507652efd1cc0c10e4c1acfb4a3a52a077fdbe1c3ca` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/C0/training/TRAINING_PROTOCOL.json` | `d2d9d6f578e411510bfb3589683add2c3bfa5641950a4c6392b503b599f3fb5a` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/CB/prediction/PREDICTION_SEAL.json` | `fc22c12e02e650c6bb4586fa19b341e355981c17e64a73acf00aa32e58602f61` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/CB/training/TRAINING_MANIFEST.json` | `f2f921944248d77f1876176575b0e33d745cc23b67f2dc98ade0cce878684fe5` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/CB/training/TRAINING_PROTOCOL.json` | `d2d9d6f578e411510bfb3589683add2c3bfa5641950a4c6392b503b599f3fb5a` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/CP/prediction/PREDICTION_SEAL.json` | `8b5273dd1b6c7cf60f4381885215e978fe521ade1a2b36cd4fa830ccd3dec36b` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/CP/training/TRAINING_MANIFEST.json` | `055d6347a7670e48270c4e3fcbffbc4b168e8ea59800839ced21bf91a474fdbf` |
| `S2_UNIT_WEIGHT_FOUR_CPU_H1/CP/training/TRAINING_PROTOCOL.json` | `d2d9d6f578e411510bfb3589683add2c3bfa5641950a4c6392b503b599f3fb5a` |
