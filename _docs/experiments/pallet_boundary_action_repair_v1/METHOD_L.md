# L: actual pose actions and the 2×2 comparison

This document describes the preregistered implementation. Actual execution and
accuracy belong to the training manifest and final results; this text does not
declare an unexecuted arm complete.

The two factors are synthetic input production (historical D0 versus the
reproduced automatic D1 procedure) and action/target packaging (historical H0
versus actual-action H1). Every checkpoint is evaluated again on the same common
real feature bytes. The historical 115/192 result remains an old-bank comparator;
it is not copied into the common-bank D0/H0 cell. New B-repair candidates are not
added to the L comparison.

H0 retains the original 31 signed-log scalars, 31 validity bits, eligible-candidate
mean/max, log candidate count and none bit: 188 inputs. Its target is uniform over
eligible jointly successful candidates, or none when there is no such candidate.
None returns the original N0 pose. The three historical MLP checkpoints are
reused without training. Their normalization and weights remain unchanged.

H0 also preserves the archived eligible-candidate order, with the virtual none
entry appended last. Its original `np.argmax` selects the first entry on an exact
score tie. H0 therefore follows the archived order on ties; none has no special
priority. Candidate ID strings are not model inputs, but reordering tied H0
entries can change the selected action. No H0 order-invariance claim is made.

H1 first constructs unique eligible actual poses, including available N0. Exact
numeric registration tolerances remain maximum rotation-entry difference 1e-7
and maximum translation-coordinate difference 1e-7 times object scale. No C2 or
90-degree equivalence is used for deduplication. Source aliases survive. An
available N0 absent from the archived candidates is added with explicitly invalid
features. An absent baseline feature is never replaced by an invented score.
The final input bit is explicitly **is_N0**, replacing H0's none bit; there is no
virtual none action. H0 and H1 therefore differ in action/target packaging as well
as loss. Both models use the same train-only scalar normalization for a given D.

Each action receives one binary label from its own unchanged pose: center error
≤5 cm and rotation error ≤5° under that synthetic row's original C1/C2 convention.
Multiple ones and all-zero frames remain. BCEWithLogits is averaged over valid
actions inside each frame, then over frames with at least one action. Zero-action
frames remain in the label ledger and evaluation denominator, but cannot
contribute a training loss. Missing/padded/ineligible actions do not enter pooling
or loss. H1 inference selects one maximum-score actual pose; exact ties prefer N0,
then deterministic numerical physical-pose order independent of candidate IDs.
No available action produces NO_POSE.
The broad `tie_rule` shorthand in the immutable `LEARNING_PLAN.json` describes
this H1 ordering. H0's legacy behavior is the separate rule above. This wording
clarification changes neither the frozen plan bytes nor either implementation.

The MLP is 188→64 ReLU→64 ReLU→1, with **16,321 parameters**. The role-specific
affine control sees the same information: the is_N0 bit routes between two
187→1 affine scorers, with **376 parameters**. Its context coefficients can differ
for N0 versus other actions. Common additive context still cancels among actions
of the same role. Constructed-weight tests demonstrate this distinction; this
does not claim that a learned model actually exploits context effectively.

The pair models are trained separately, restricting actual actions to N0/E0
aliases. Identical poses are still one action with both aliases. A pair run
requires an input seal certifying independently executed E0. It cannot be enabled
by copying N0 into an E0 field. The D1 producer's execution receipt, state and
observation journal establish the scope of this certification.
Pair training/inference pools only the retained pair actions, while reusing the
same D1 scalar normalization. This experiment still pays for producing the
common full bank; it does not measure a separately optimized two-expert producer
or establish its potential deployment cost savings.

Six new groups × three fixed seeds give at most 18 new training attempts. Seeds
are 20261018/19/20 and the representative is always 20261018. AdamW, lr 1e-3,
weight decay 1e-4, batch 32 frames, maximum 50 epochs and patience 5 are fixed.
Checkpoint selection uses strictly lower corresponding validation loss. CE and
BCE magnitudes are not compared as measures of pose quality. Every best/last
checkpoint, seed, learning curve and technical failure is preserved. The run
ledger reserves an attempt before optimizer creation and enforces the 18-run cap.

Every newly trained model's epoch records train/validation actual joint pose success, center/rotation
distributions, the fixed large-error cutoffs, source counts, missing-good-action
counts, and frame-balanced action accuracy/Brier score. H0 softmax scores against
actual binary output success are descriptive diagnostics, not its uniform CE
target or calibrated probabilities. The success objective does not distinguish a
6 cm failure from a 100 cm failure; final continuous-error and tail comparisons
remain necessary.
The historical D0/H0 training recorded CE curves but not per-epoch pose metrics;
only its archived best/final checkpoints are available. Those absent historical
metrics are reported unavailable. Reusing that model does not invent a complete
past pose learning curve or authorize retraining it outside this run budget.

The implementation takes the principle of evaluating the returned expert/action
from [Mozannar and Sontag, 2020](https://proceedings.mlr.press/v119/mozannar20b.html)
and the separation of softmax and one-vs-all score behavior from
[Verma et al., 2023](https://proceedings.mlr.press/v206/verma23a.html). The original
papers and their surrogate definitions were read. This finite-pose BCE experiment
does not reproduce their complete classifier/deferral surrogate, conformal
procedure, calibration or theoretical guarantees. The stable binary loss is
implemented with the installed PyTorch BCE-with-logits primitive; its reduction
is explicitly controlled to preserve frame balance.

The same `feature_profile` function summarizes D0, D1 and real candidate counts,
sources, validity/missingness and signed-log feature distributions. State or raw
covariance fields absent from an input are reported unavailable, not inferred or
filled with zero to claim procedural parity.
