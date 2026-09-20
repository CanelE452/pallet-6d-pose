# Next bounded stage: learn whole-view utility on synthetic supervision only

Status: planned after completed `views` experiment; selector is NOT trained yet.

## Evidence motivating the intervention

On fixed ordinary-plastic194, the predeclared GT-free whole-view medoid recovered
23/281 matched baseline>20px corners to <=10px but damaged24/511 baseline<5px
corners to >10px. The1percent damage guard failed. Full960/1280 alone also damage
many good points and must not be silently adopted. A GT best-complete-view
diagnostic recovered72 with4 damaged; it is not a deployable result or a promise
that a source-trained selector will attain this performance.

A subsequent, explicitly GT-only diagnostic added a90degree **prediction
channel reassignment** to each view while keeping official scoring C2. It
recovered106/281 with4 damaged. For the user's original chair/cone example,
native consensus remains262.87px mean (R0264.90), but the diagnostic best
FLIP640+reindex prediction is8.97px. These are not deployed choices: the
diagnostic uses GT. This evidence requires the next selector to cover both
view selection and channel identity, not merely choose among native views.
Artifact: `views/IDENTITY_ORACLE_DIAGNOSTIC.json` in the result root.

The new goal is large-error recovery, not another favourable median. Keep all194
images, official C2, fixed baseline detection metadata, and canonical-identity
recovery/damage scoring. No evaluation GT may train/calibrate the selector.

## Planned source-only ranking test

1. Reuse the existing scenario-disjoint synthetic split from
   `_docs/experiments/pallet_posefix_utility_selector_v1/SPLIT.json` as an input
   population, not its trained selector or results. Deterministically choose up
   to512 train rows and128 validation rows with seed20260921. Keep clean/stress
   variants of every source image together. Verify hashes, scenario separation,
   and exclusion of all current real194 image hashes. Existing R0 validation and
   historical source-development exposure must be disclosed.
2. Use `pallet_posefix_replay_v1.source.SourceData` only as a read-only synthetic
   RGB/GT adapter. Prepared RGB already has reflect100 padding: remove the
   original border exactly once and subtract100 from prepared-space GT before
   calling the new view pipeline, which applies its own border. Check this
   coordinate contract before generating any supervision. Use the existing
   per-row approved symmetry from DIMENSION_SIDECAR; do not pretend all source
   objects are C2 or add C4 equivalence to ordinary plastic.
3. Generate the same six actual R0 candidates on clean RGB and one fixed stress
   RGB condition (photometric degradation/occlusion, coordinates unchanged).
   Candidate generation/cropping receives only RGB and predicted R0 boxes. No
   GT-centered crop or GT-derived coordinate perturbation. Synthetic GT is
   accessed only after candidates and input features exist. Add the explicit
   native/reindexed90 prediction hypotheses. Reassignment changes the output
   channels and is NOT an additional allowed scoring symmetry. If source
   identity-error coverage is insufficient, stop this particular fit and
   record it; do not assume perfect synthetic detections teach identity repair.
4. Learn per-candidate mean-error reduction versus R0, in percent of predicted
   box diagonal, from GT-free view identity, predicted confidences, cross-view
   disagreement, relative coordinates and local image evidence. Start with one
   fixed RandomForestRegressor configuration (300 trees, min_samples_leaf15,
   max_depth10, random_state20260921, n_jobs4); do not run a real-eval sweep.
   R0 utility is identically0 by definition. Output one complete actual view,
   never an average or an oracle per-corner splice.
5. Calibrate a conservative acceptance threshold only on synthetic validation
   using the fixed grid[0,.25,.5,1,2,4,8]percent-diagonal utility. Choose maximum
   large-corner recovery subject to <=1percent good-corner damage; ties prefer
   the higher threshold. If none passes, use exact R0 and report failure.
6. Freeze model/configuration/threshold and ALL real194 choices before scoring.
   Report every predeclared stage and the same recovery/damage counts. Do not
   reinterpret the source validation guard as a target-domain guarantee.
7. Only a useful GT-free correction teacher can justify a later pseudo-label
   regeneration/student-distillation experiment. Current original pseudo
   labels, final model and evaluation annotations remain untouched. This stage
   alone does not complete the self-training/large-error goal.

## Known limits / stopping interpretation

The previous local per-corner utility selector had weak clean-source and real
transfer; this is a different whole-view candidate task, not an assumption that
another selector must work. Synthetic source may not cover real axis confusion.
If source ranking transfers poorly, preserve that result and revisit the need
for verified real training support rather than using real evaluation labels as
training targets. No depth, CAD, extra physical-shape or X-crossing rejection is
introduced. No original model/annotation/source artifact is edited.
