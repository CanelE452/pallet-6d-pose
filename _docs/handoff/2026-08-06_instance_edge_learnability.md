# Handoff — Instance-Aware 12-Edge Learnability Screen

Next session starts here.  Everything below was measured in the 08-05/08-06
session; none of it needs re-deriving.  Phase A of the instruction should pass
almost immediately.

## Why this experiment

`O12` oracle places corners at 98.7% / 96.1% within 20px using **no corner
heatmap and no top-K**, while `O5` reaches 2.7% / 0.6%.  Five semantic classes
cannot identify a corner even with perfect ground truth, because `top_width`
covers two parallel edges and conditioning on three classes selects six edges.
The open question is whether the twelve-edge representation can be **learned**
from images.  O12 is an oracle; it proves capacity, not learnability.

## Confirmed facts

```
topology rule       corners whose 3D coords differ in exactly one axis
                    -> exactly 12 edges, 3 incident per corner,
                       class counts 2/2/2/2/4 (top_w, top_d, base_w, base_d, vertical)
source              challenge/scripts/annotate_pnp.py :: make_pallet_keypoints_3d
                    NEVER hand-write the edge list; generate and assert

O12 decoder         score_i(x,y) = -exp(mean dist to its 3 incident edges / tau)
tau                 5 belief pixels, fixed
grid                50 x 50 belief space, global argmax

oracle reference    O12  eval56 <=20px 98.7%  median 4.68px  PnP 56/56
                         wood   <=20px 96.1%  median 7.99px  PnP 45/45
                    O5   eval56 <=20px  2.7%  median 148.66px  PnP 0
                         wood   <=20px  0.6%  median 292.53px  PnP 0

A1 checkpoint       weights/paper_s2_pdg/A1/epoch_003.pth
                    run_state completed=True, epoch 3/3, steps 7329
A0 checkpoint       weights/paper_s2_stageB/net_epoch_0057.pth
                    sha256 c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896

feature taps        F100 = vgg[17] ReLU, 256 x 100 x 100   (runtime confirmed)
                    F50  = vgg[26] = net.vgg(x),  128 x 50 x 50
                    do not hardcode without re-asserting the shapes

PPD recipe source   scripts/stage0/paper_s2_ppd_long_run.py
                    run_state: weights/paper_s2_ppd_t2_screen/{L0,M0,M1}, epoch 20/20
                    prior timing ~64 s/epoch on 3,039 synthetic frames
                    -> 20 epochs ~21 min per arm; 7 arms ~2.5 h detached

canonical sets      eval56 56 frames, objects[0].split == "eval" on all 56
                      challenge/data/01_real/eval_canonical/_outside_eval_manual_gt        22
                      challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt   12
                      challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt     22
                    wood 45 frames, split is "<none>" -- a separate manual-GT set,
                      not covered by the split=="eval" rule
                    manifests: data/pallet/results/paper_s2_eval56/{eval56,wood}_manifest.json
```

## Rules that were violated before and must hold

- Never draw a development, diagnostic or probe subset from
  `data/_eval_sets/*combined`.  Print the frame paths and `objects[0].split`
  before using any new subset.
- A1's VGG and belief/affinity stages stay frozen; assert trainable A1
  parameters == 0 and that the A1 parameter delta is 0 after training.
- Checkpoint selection uses synthetic validation only.  eval56 and wood are
  opened once, after the checkpoint is fixed.
- O12 decoding must not receive a corner heatmap or top-K candidates.

## Suggested reduced first pass

If context or time is tight, run **L12-F50 seed 1 only** to answer Q1 and Q2.
Expand to seeds 2-3, L12-MS and L5-CTRL after there is a signal.  The full
instruction's 7 arms are ~2.5 h of detached training plus a large runner, 34
new tests and 14 reports.

## Prior audit trail

```
_docs/audits/eval56_summary/canonical_corner_audit/
    PCS_NET_ARCHITECTURE_DECISION.md          corner bottleneck, PCS-Net STOP
    capacity_audit/PCR_NET_CAPACITY_GATE.md   (QUALIFIED banner: near verdict overturned)
    spatial_relational_line_audit/
        FINAL_CAPACITY_GATE.md                near spatial, far linear
        FINAL_ARCHITECTURE_GATE.md            far nonlinear, O5 vs O12   <- most recent
_docs/audits/eval56_summary/pdg_unified_program/   Stage 1 (A1/A2) and its correction
_docs/history/2026-08-05.md, 2026-08-06.md
```

Current standing decision: **INSTANCE_AWARE_LINE_BRANCH_FIRST**, with spatial
HCRM secondary and `ENCODER_REPRESENTATION_CHANGE_REQUIRED` withdrawn.
