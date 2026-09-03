# Data split plan

Every overlap below was measured on **image SHA256 of the raw bytes**, recomputed
rather than read from a manifest. Two cross-checks passed: 1,000 declared adaptation
hashes matched recomputation with zero disagreements, and the 8,031 pool images
reproduced the `ADAPTATION_POOL_LOCK` invariant 2,227 + 5,804 exactly.

## The one fact that decides this plan

```text
real frames with camera_facing_pnp.axis_assignment_confirmed == True    0
```

Verified independently across all 2,004 pose-bearing real annotations:

```text
axis_assignment_confirmed = False          700   (explicit)
field absent (pre-v2 legacy)             1,304
                                         -----
confirmed True                               0
```

There is no axis-confirmed real pose ground truth anywhere in the repository. Not in
PAPER_EVAL, not in the legacy sets, not in the newest captures. Every population
question below inherits this.

## Real pose-GT inventory, deduplicated by image SHA256

2,004 pose-bearing JSONs collapse to 1,176 unique frames once label mirrors and
duplicated copies are merged.

```text
object                      pose_status                 ADAPT  FREE  PAPER_EVAL  total
────────────────────────────────────────────────────────────────────────────────────
plastic_110x130x11          LEGACY_v1_NO_POSE_STATUS      45    166        123    334
  (the paper object)        UNCONFIRMED_SIGNED_AXIS        0      1         71     72
plastic_SQ_110x110x15       LEGACY_v1_NO_POSE_STATUS     243      0          0    243
  (challenge only)          UNCONFIRMED_SIGNED_AXIS        0    402          0    402
wood_80x59x14               LEGACY_v1_NO_POSE_STATUS       0      0         44     44
                            UNCONFIRMED_SIGNED_AXIS        0      0         81     81
────────────────────────────────────────────────────────────────────────────────────
                                                         288    569        319  1,176
gt_source: manual 899 / apriltag 243 / pseudo 34
```

## Overlap with PAPER_EVAL, measured

```text
set                              unique   ∩ PAPER_EVAL
──────────────────────────────────────────────────────
PAPER_EVAL_319                      319            319
DEV_PLASTIC_POS140                  140            128    <- selector diagnostic population
DEV_WOOD_POS45                       45             45
COMMON_DEV_MULTISHAPE_POS           173            173
FINAL_WS_POSITIVE_ALL               229            146
INCOMING_DAY_20260830            29,028             68
INCOMING_NIGHT_20260830          13,583             78
LIVE_CAPTURE_GT                     402              0
live-capture raw                 15,145              0
capturenight05/06/07              1,624              0
```

Two consequences that are easy to miss:

```text
1  the raw bytes of PAPER_EVAL's 146 newer frames still sit in incoming/sessions/.
   Treating incoming as a candidate pool contaminates automatically.
   The incoming manifest's own duplicate_in_active_evaluation flag is false on all
   42,611 rows and is STALE — it was written when the active set was DEV 173.
   Do not trust that flag.

2  PAPER_EVAL is already spent on selector development as well. The selector
   diagnostic ran on DEV_POS140, and 128 of those 140 are inside PAPER_EVAL.
```

## Overlap with the adaptation pool, measured

```text
                              unique   ∩ ADAPT_1000   ∩ ADAPT_8031
──────────────────────────────────────────────────────────────────
PAPER_EVAL_319                   319              0              0   <- lock claim reproduced
INCOMING_DAY / NIGHT          42,611              0              0
LIVE_CAPTURE_GT                  402              0              0
all real pose-GT (1,176)       1,176              -            288
```

`ADAPTATION_POOL_LOCK.json`'s `adapt_sha_intersect_eval_sha: 0` reproduces as true.
But 288 pose-GT frames do sit inside the pool — `pallet11_gt` 243 and
`capturenight01-04` 45 — and cannot evaluate a self-training arm. The 243 are
excluded on separate grounds anyway: their AprilTag GT is known broken.

## A — SELECTOR_SOURCE

```text
STATUS   AVAILABLE     n = 60,000   synthetic only
```

```text
PROBE_METADATA_60K.jsonl                60,000 rows, images on disk
  images = datasets/g38_legacy_v1v2_p0_tex20k
  train 55,980 / val 4,020, sha1-based split, reproducible
  labels: front_face_type (short/long), front_face_class_short0_long1,
          perm_v4[8], fixed_renderer_dims, fixed_axis_reconstruction
  parity: train long 29,503 / short 26,477
  PROBE_METADATA_60K_AUDIT.json: 60,000/60,000 parity, stems unique

LEGACY_SPATIAL_METADATA.jsonl           20,000, same schema
challenge/data/02_synthetic/training/v3 10,000  location + quaternion
  addon_v1                               6,000  + camera world pose
  truncation_addon_v1                    6,000  + keypoints_3d_world (full 6D)
  addon_v1_train / addon_v1_val          6,000  ★ 12,000 dangling symlinks, unusable
  v1, v2                                     0  directories only, no JSON
```

Pixel overlap with real is zero by construction. Three caveats:

```text
dangling symlinks   addon_v1_train/val point at pre-2026-08-14 paths. Recoverable,
                    but unusable as they stand.
perm_v4             synthetic PnP is exactly 180 degrees off pose_transform, so any
                    pose-based threshold selection on synthetic is invalid without
                    the perm_v4 correction.
already measured    this selector has been trained on this source. Synthetic dev
                    balanced accuracy 0.904, AUROC 0.966; the dims-shuffle ablation
                    drops it to 0.745, confirming the signal comes from dimensions.
                    The same family scores 0.59-0.65 on real DEV_POS140.
```

That last line is the finding: **the bottleneck is sim-to-real transfer, not the
amount of source data.** Adding synthetic frames does not address it.

## B — SELECTOR_DEV

```text
STATUS   NOT AVAILABLE
```

Frames of the paper object with zero PAPER_EVAL overlap and zero pool overlap:

```text
session (legacy_unverified, role DEV_UNVERIFIED)     n   lighting
────────────────────────────────────────────────────────────────
night_eval_manual_gt                                43   night
outside_eval_manual_gt                              32   day
forklift_20260528_manual_gt                         25   day
capturepalletcad_manual_gt                          15   day
capture0403noapril_manual_gt                         6   day
────────────────────────────────────────────────────────────────
                                                   121   day 78 / night 43
```

Verified: `legacy_unverified` holds 409 pose frames, of which `pallet11_gt` 243 are
excluded (broken AprilTag GT) and `capturenight01-04` 45 are inside the adaptation
pool. 409 − 243 − 45 = 121.

Three independent disqualifications, any one of which is sufficient:

```text
1  no selector labels.   All 409 have axis_assignment_confirmed absent. The
                         selector's ground truth — which footprint axis faces the
                         camera — does not exist for these frames. A fresh human
                         pass would have to create it.

2  contract forbids it.  DATASET_CONTRACT.json defines DEV_UNVERIFIED as
                         "preserved legacy annotation, not paper eligible".

3  sample too small.     The registered gates are overall >= 0.95 and night >= 0.90.
                         At n = 121 the 95% interval around 0.95 is about +/- 3.9 pp;
                         at night n = 43 it is about +/- 6.5 pp. A night result of
                         0.90 could not be distinguished from 0.84 or 0.96.
                         "121 frames exist, therefore it is possible" would be false.
```

Two adjacent candidates also fail:

```text
eval_outside DEV_SUPPORT 12   members of DEV_POS140; already spent on the selector
live_capture_gt 402           object is the square 110x110x15, so the W/D parity
                              question does not even arise; and it is registered as
                              a training set in REAL_FT_V1_METHOD_LOCK
```

## C — POSE_FINAL

```text
STATUS   NOT AVAILABLE     inventory 0
```

```text
FINAL_POS / FINAL_NEG / FINAL_PLASTIC_POS / FINAL_WOOD_POS / FINAL_ALL_POS
    all items 0, all carry an unavailable_reason

physical FINAL workspace   229 frames, of which 146 are absorbed into PAPER_EVAL by
                           SHA; the remaining 83 are unannotated plastic_night_01
```

And the decisive one: with zero axis-confirmed frames, opening a final population
today would force the evaluator to receive the GT axis assignment again. That is
precisely the leak that made `5cm5deg` read 30.4 percent instead of 19.3 percent.

### Candidate pools reachable without new capture — all with zero annotations

```text
candidate                        n      ∩PE  ∩ADAPT  session-independent  object
──────────────────────────────────────────────────────────────────────────────────
incoming 20260830 remainder  42,465       0       0   NO                 plastic+wood
capturenight05/06/07 rest     1,581       0       0   YES                110x130x11
capture02/03/0403middle       3,470       0       0   YES                unverified
vdoframes                     2,362       0       0   YES                unverified
wood/_annotate_1837+1843      190        45       0   NO                 wood
```

The 42,465 incoming frames are SHA-clean but **not session-independent**, measured:
day frames are numbered 000000–029027 and night 029028–042610 — one continuous
recording — and `final/positive/sessions/*/session.json` states
`promoted_from_sessions: [real_unlabeled_{day,night}_20260830]`. PAPER_EVAL's 146
came from there. The workspace's own
`adaptation_evaluation_separation.gates` requires `capture_session_id overlap == 0`,
which this violates.

Budget note for any new annotation: the existing QA quarantined 21 of 161 legacy
frames (13.0 percent) for annotation defects. Assume roughly 1.15x the target count.

## Verdict

```text
NEW_CAPTURE_REQUIRED = YES
```

Session-independent remaining stock is nighttime plastic only (1,581 frames from
capturenight05/06/07). Daytime plastic and wood have zero session-independent
inventory.

Two qualifications, stated so the decision is not made on a half-truth:

```text
if separation is relaxed to SHA-only — the same standard PAPER_EVAL itself used —
then NEW_CAPTURE_REQUIRED becomes NO, but NEW_ANNOTATION_REQUIRED is still YES,
because the candidate pools hold zero pose GT.

and annotation alone does not open POSE_FINAL. Without an axis-confirmation
procedure, new annotations save as UNCONFIRMED_SIGNED_AXIS — which is exactly what
happened to the most recent 146 and 402 frames.
```

A reduced night-only variant (annotating capturenight05/06/07) is reachable today,
but the pose table would then be nighttime-only and the wood cross-shape column
could not be filled.
