# Honest Evaluation Protocol

Frozen definition of the honest pose/keypoint metric and the per-bin reporting format,
plus a function-level map onto the existing eval code. No new evaluator is needed — this
documents WHICH functions already compute the honest number and HOW to bin/report it.

---

## 1. Selected reproj vs. honest full-8 reproj (★ STAGE20 lesson)

Two "reprojection errors" exist and they are NOT interchangeable:

```
name             what it compares                          verdict
--------------------------------------------------------------------------------
selected reproj  PnP-input keypoints  ↔  the SAME points   RED HERRING. self-consistency
(pnp_reproj_          reprojected from the solved pose      of the fit; a wrong pose that
 _click)              (pred ↔ pred-via-PnP)                 fits its own inliers scores LOW.
honest full-8    solved-pose 8 corners (GT dims) ↔         TRUTH. pred ↔ GT. the actual
(pnp_honest8)         GT projected_cuboid (pred ↔ GT)       pose error a user would see.
```

- **Report honest8. Never headline selected reproj.** (STAGE20: selected reproj looks
  great while the pose is visibly wrong; honest8 is the metric that moved with real quality.)
- Also matches memory `evaluate-on-val-convention-bug`: naive same-index reproj gave 130px+
  for every model (convention mismatch) — order-free honest8 is the corrected metric.

Code (all in `data/pallet/eval_results/stage16_truncation_addon/capturecad_b2_eval/eval_capturecad_b2.py`,
reused by stage25 + paper_stage_a via importlib):
- `eval_frame()` L91:
  - `pnp_reproj_click = pose["reproj_error_px"]`  ← **selected** (do not report).
  - `pnp_honest8`: solve_pose 8 corners with GT dims → `hungarian(pa, gt8)` → mean
    ← **honest**, order-free vs GT projected_cuboid.
- `solve_pose` = `challenge/scripts/annotate_pnp.py::solve_pose` (order-free W/D swap,
  GT dims). ★ flat pallet → must use SOLVEPNP_ITERATIVE (EPnP diverges; checklist rule).

---

## 2. Order-free matching + honest8 cross-check

- Corner error uses **order-free Hungarian** assignment, not same-index:
  `eval_capturecad_b2.hungarian(pred, gt)` (scipy `linear_sum_assignment` on the pairwise
  distance matrix). Same-index comparison is FORBIDDEN (memory `evaluate-on-val-convention-bug`
  + checklist: same-idx under-scores every model to 130px+).
- honest8 ALSO runs through Hungarian on the solved-pose corners vs GT
  (`hungarian(pa, gt8)`), so the corner metric and the pose metric use the same order-free
  rule → cross-consistent.
- ⚠ Hungarian cross-check artifact at **extreme proximity**: when two corners project
  within a few px (flat/edge-on views), Hungarian can swap them and report a spuriously low
  error. Guard = report honest8 alongside per-corner (front/rear) medians; a swap shows up
  as front≈rear collapse. Cross-check honest8 against the raw same-order error on the
  subset of frames where GT corners are well-separated (frsep large) to detect swaps.

---

## 3. reflect-pad parity

- **Official = no-pad (aspect)**, because paper_base_v2 was trained no-pad (train/infer
  parity). `pad_frame(img, 0)` = no-op → 400/min aspect resize.
- pad100 (`pad_frame` reflect-pad, `belief_to_orig_pad` inverse map `*(W+2P)/W − P`) is a
  coverage/robustness variant ONLY (memory: dope-inference-needs-reflect-padding). Report
  it in a separate row, never as the headline.
- Every model in a comparison MUST use the same pad (STAGE25 uses pad100 for all; the
  official table uses A_nopad for all) — mixing pads across models is invalid.

---

## 4. Per-bin output spec (freeze)

Each metric block is reported over these bins so detection and localization stay separate
and failure modes are attributable:

```
bin axis         values                         source function
--------------------------------------------------------------------------------
truncation       V=8 (full-view) | V<8          summarize_breakdown() -> ["V8"],["Vlt8"]
                                                 (paper_stage_a_pad_ablation.py L60-61,
                                                  split on row["v_geom"])
face             front (idx0-3) | rear (idx4-7)  summarize() front_med / rear_med
                                                 (matched-GT-idx bucket, stage25 summarize)
camera angle     elev bins deg:                  elev_dist() + summarize per bin
                 -90~5 / 5~10 / 10~15 /          (elev via stage18 elev_from_pose /
                 15~25 / 25~90                    elev_from_world; edges = ELEV_BINS)
scale/near       near (large projected size) |   bucket on projected cuboid diagonal /
                 large vs far                     mask_bbox size (add: pc_diag bins)
```

Per bin report: `n, det%, front_med, rear_med, corner_med, worst2_med, pnp%, honest8_med,
good% (corner<10px), gross% (corner>20px)`. Functions: `stage25_paperbase_eval.summarize()`
(honest8_med, good%/gross%, det%, pnp%) and `paper_stage_a_pad_ablation.summarize_breakdown()`
(V8/Vlt8/elev_bins wrapper). Aggregators: `eval_capturecad_b2.agg()`.

### false-accept count (ADD to summarize output)
- Definition: a frame that is **detected + pnp_ok but honest8 > T** (default T=20px) — the
  model confidently returns a pose that is actually wrong. This is the metric that catches
  "looks detected, pose is garbage" and must be reported per bin.
- Not yet emitted by `summarize()`. Add: `false_accept = sum(1 for r in rows if r["n_det"]>=k
  and r["pnp_ok"] and r["pnp_honest8"] > T)` and `false_accept_pct` over detected frames.
  Store the raw count in the bin JSON so precision-at-detection is auditable.

---

## 5. Function map (quick reference)

```
concern                 module.function
--------------------------------------------------------------------------------
reflect-pad             eval_capturecad_b2.pad_frame / belief_to_orig_pad
order-free corner       eval_capturecad_b2.hungarian
pose solve (ITERATIVE)  challenge/scripts/annotate_pnp.solve_pose  (via APNP)
honest8 (pred↔GT)       eval_capturecad_b2.eval_frame -> pnp_honest8
selected (red herring)  eval_capturecad_b2.eval_frame -> pnp_reproj_click   (DO NOT headline)
per-frame agg           eval_capturecad_b2.agg
bin summarize           stage25_paperbase_eval.summarize (+ per_corner_dists)
V8/Vlt8/elev breakdown  paper_stage_a_pad_ablation.summarize_breakdown / elev_dist
elevation               stage18_elevation_threshold.elev_from_pose / elev_from_world
```

★ final-test set stays SEALED — this protocol is exercised on dev sets (filter-val N=123,
handannot17 N=17) only. Re-running eval is out of scope for this Ubuntu prep; this doc
fixes the definitions so the eventual run is honest and order-free.
