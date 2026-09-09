# Pre-registered spatial fusion follow-up

This follow-up was frozen on 2026-08-29 KST, before extracting a spatial
feature cache or training any model in this directory.  It does not rewrite
the completed Phase-C contract or its negative result.  The first draft was
audited and tightened before any Phase-E result existed; the durable freeze is
the SHA-256 pair recorded in `PROTOCOL_FREEZE.json`.

## Scope and claim boundary

The completed Phase C showed that late concatenation of one selected 64-D
classification cell, 26-D keypoint geometry and 4-D dimensions did not recover
real parity.  This follow-up tests whether retaining a small spatial
neighbourhood changes that result.

The primary diagnostic population remains `COMMON_DEV_PLASTIC_POS128`.  Wood
parity is reported only as out-of-range exploratory evidence; wood pose and a
multishape claim remain blocked.  Real GT is never used for training, epoch,
seed, architecture, threshold or fusion-method selection within Phase E.

This is not a prospectively blind real experiment.  The same COMMON128 labels
and G38 TEST labels/results were already opened in Phase C, and the spatial
follow-up was motivated by that failure.  Phase E only promises that it will
not re-access those labels until its own source-only method lock is durable.
Both populations are permanently labelled `ADAPTIVELY_REUSED_DEV_DIAGNOSTIC`
and cannot support model promotion.

The available G38 split is reused so the comparison can run now.  This has two
known limitations: the frozen YOLO saw every synthetic TEST image upstream,
and G38 has only 36 source-TRAIN samples within 20% of the plastic dimensions
on every axis and none within 10%.  Consequently this is a matched frozen-head
diagnostic, not an end-to-end held-out result.

This Phase-E run is irrevocably exploratory.  It cannot be retroactively
promoted by later creating a dataset or changing a Boolean gate.  A future
promotion attempt must use a new contract, retrain/reselect every method on a
new `DIM_PROBE_SYNTH`, and use a new untouched real holdout.  That synthetic
dataset is not present, and the current shell has no callable Isaac Sim or
Blender runtime.  Full-YOLO training, unfreezing, wrapper work and deployment
are zero-authorized in this contract, regardless of the Phase-E numbers.

## Frozen feature recipe

- Primary checkpoint and prediction contract are unchanged.
- Discover the active one2one pose head by capability, never by a hard-coded
  executable module path.
- Use the highest-confidence final detection, exactly as Phase A/C.
- At the final detection's exact source `(level,row,column)`, extract a
  row-major `7 x 7 x 64` local patch from the immediate producer of the active
  one2one classification readout.
- Zero-pad at feature-map boundaries and store a 49-token validity mask.
- The center token must equal the prior 64-D Phase-C feature before float16
  storage; confidence, box and all nine keypoints must equal the frozen cache.
- Store patches as float16 in a separate cache.  Do not alter the completed
  Phase-C cache.
- Normalize each of the 64 channels using valid patch tokens from synthetic
  TRAIN only.  KP and dimension normalization reuse the Phase-C TRAIN-only
  statistics.

The local patch is a spatial parity probe around the detection source cell. It
is not called keypoint-head conditioning and it does not update YOLO.

## Arms

All arms share the same token encoder, position/level embeddings and
`[128, 32]` SiLU classifier.  The classifier always receives pooled spatial
features, 26-D KP geometry and a 4-D late-input slot.

1. `S0_SPATIAL_NO_DIMS`: the late dimension slot is zero and no adapter sees
   dimensions.
2. `S1_SPATIAL_CONCAT`: correct dimensions enter only through the late 4-D
   slot.  This is the matched concat baseline.
3. `S2_SPATIAL_FILM`: S1 plus a zero-init residual channel-wise FiLM adapter
   over all 49 spatial tokens.
4. `S3_SPATIAL_CROSS_ATTENTION`: S1 plus one four-head, 32-D dimension-query
   cross-attention operation over the 49 tokens.  Its 32-to-128 residual
   projection is zero-init.

For a given seed, S1/S2/S3 common weights initialize identically and their
step-zero logits must match within `1e-7`.  FiLM and attention adapter parameter
counts must be within 5% of one another.  Dropout is zero.

## Training and source-only selection

- Seeds: 0, 1, 2.
- AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Balanced cross entropy and the same class-balanced sampler.
- 40 epochs, effective batch 512, identical per-epoch sampled indices across
  methods for a given seed.
- Per run, select the earliest epoch with maximum synthetic-DEV balanced
  accuracy.
- Score a method by the mean of its three selected-epoch synthetic-DEV
  balanced accuracies; a single lucky seed cannot select the mechanism.
- S1 concat is the default.  S2 or S3 becomes eligible only if its three-seed
  mean is at least 0.02 above S1.  If both are eligible, S3 is chosen only when
  its mean is at least 0.02 above S2; otherwise prefer S2.  Exact comparisons
  use unrounded floats.
- For every arm, choose its reported artifact as the median-DEV seed; ties go
  to the lower seed.  Synthetic TEST reports each arm's locked median artifact.
  E1-E9 and real use the locked primary median artifact and the locked S0
  median artifact.  No ensemble or best-seed substitution is allowed.  These
  artifacts are diagnostic, not deployment authorization.
- Write and hash `SOURCE_SELECTION_LOCK.json` before Phase E re-accesses
  synthetic TEST or real labels.  The lock explicitly records that both were
  historically opened in Phase C.  TEST and real results never change it.
- Decision threshold is fixed at 0.5.

## Controls and metrics

Report synthetic TEST accuracy, balanced accuracy, AUROC and confusion matrix.
On real DEV report COMMON128 overall/DAY/NIGHT/session plus the existing
DEV140 and wood diagnostic views.  Feed each selected parity output to the
same PnP diagnostic used by Phase C.

Synthetic TEST reports all four locked median artifacts and deterministic
seed-42 shuffled dimensions for all three conditioned arms; every shuffled
TEST row must receive a different dimension triplet *and* a different
normalized 4-D dimension feature (at least one float32 component must differ
exactly).  Real is re-opened once for only the locked primary median artifact,
the locked S0 median artifact, and the primary's shuffled/wrong-dimension
controls.  The losing conditioned methods are never evaluated on real in
Phase E.

Real shuffling keeps the registered COMMON128+Wood45 control, but because only
31/128 plastic rows change triplet, it is descriptive and not a diagnostic
gate.  Wrong-object dimensions are also descriptive and never enter PnP scale.

## Frozen diagnostic gates

These E-gates describe whether the source-locked method would have solved the
reused DEV diagnostic.  Passing them does not make a promotion claim.

- E1: Plastic COMMON128 at least 122 correct of 128.
- E2: Plastic NIGHT at least 26 correct of 28.
- E3: COMMON128 session minima, expressed as exact counts:
  `eval_outside >=9/10`, `eval_noapril >=11/12`, `eval_cad >=16/18`,
  `eval_pallet07 >=23/27`, `eval_pallet09 >=29/33`,
  `eval_night08 >=11/12`, and `eval_night09 >=14/16`.
- E4: at least 7 more correct COMMON128 frames than S0 (the integer form of
  an improvement of at least five percentage points).
- E5: on the current 10,095-row synthetic TEST, correct dimensions yield at
  least 505 more correct rows than the 100%-changed shuffled control.  This is
  a Phase-E diagnostic and does not replace historical Phase-C P5.
- E6: Restricted ADD-S AUC recovery
  `(AUC_candidate - 0.27236328125) / (0.32071875 - 0.27236328125) >= 0.70`,
  equivalently `AUC_candidate >= 0.306212109375` on COMMON128.
- E7: pose-valid count at least 126/128, the frozen A0 count.
- E8: candidate count/confidence/float32 box/float32 keypoint max absolute
  difference exactly zero, and the float32 patch center equals the prior 64-D
  feature exactly before float16 storage.  Float16 quantization is reported
  but is not used for this exactness assertion.
- E9: if and only if E1-E8 pass, compare S0 and the primary branch on the pinned
  RTX 3080 with batch one, 20 warmups and 200 timed iterations.  Both p50 and
  p95 parity-branch latency increases must be at most 5%.  Use CUDA events,
  synchronize before and after each timed iteration, keep already-cached
  normalized features on device, exclude YOLO/feature extraction and host
  transfer, and compute `(primary_ms - S0_ms) / S0_ms`.  This is diagnostic
  only and never triggers full training.

Adapter parity counts every parameter in FiLM's dimension MLP, and every
parameter in attention's token/query projections, MHA and residual projection.
The relative difference is `abs(F - X) / max(F, X)` and must not exceed 0.05.

The maximum possible verdict is
`EXPLORATORY_SPATIAL_FUSION_DIAGNOSTIC_COMPLETE`.  A future, separately
approved protocol needs a new 8,000-frame dataset (6,000/1,000/1,000; exact
50:50 parity in each split), asset and canonical-topology hash separation,
zero upstream-checkpoint exposure for TEST, 100% independently reconstructed
parity/metadata agreement, independent nuisance sampling, frozen dimension
strata, a 200-frame smoke audit, and a new untouched real holdout.  Merely
auditing that future data may not promote or reuse any Phase-E checkpoint.
