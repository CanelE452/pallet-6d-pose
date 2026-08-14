# Official Baseline Protocol — paper_base_v2 (no-pad/aspect)

Frozen, citation-ready definition of the paper's quantitative baseline and how it
relates to the pad-100 qualitative demo and the B2 private-CAD reference. Numbers are
**cited from prior measurements — no re-run** (paper_stage_a/eval.json, stage25_paperbase_eval).
Supersedes / consolidates `data/pallet/eval_results/paper_stage_a/PAPER_STAGE_B_official_baseline.md`.

- weights: `weights/paper_base_v2/final_net_epoch_0060.pth`
  (procedural 19,308 frames, palletobj-free, scratch 60ep, no-pad/aspect training,
   truncation_aug_prob=0.0 — verified in `weights/paper_base_v2/header.txt`).
- source: `data/pallet/eval_results/paper_stage_a/eval.json` (PAPER_STAGE_A) and
  `data/pallet/eval_results/stage25_paperbase_eval/` (STAGE25). No re-execution.

---

## 1. Protocol split (what is official vs. what is a demo/reference)

```
role                         config             use in paper
----------------------------------------------------------------------------------
OFFICIAL quantitative        paper_base_v2      main results table. train/infer parity
  baseline                   A_nopad (aspect)   (model trained no-pad -> eval no-pad).
QUALITATIVE / fallback       paper_base_v2      coverage demo + robustness ablation ROW
  coverage demo              C_pad100           only. reflect-pad100 for truncation/近접.
PRIVATE-CAD upper /          B2 (mask-aux,      reference ceiling ONLY. B2 saw v3/addon
  reference (NOT main)       v3+addon replay)   replay -> not a fair paper comparison.
```

- **Official = A_nopad.** Rationale: the model was trained no-pad, so no-pad eval is the
  parity-correct measurement, and it is the purest (highest good%, lowest gross%). Any
  paper improvement (self-train / filter / aug) is measured as recovery of the gap below.
- **pad100 is NOT the official number.** reflect-pad100 raises detection/coverage on
  truncated/near frames but lowers purity; it belongs in a robustness-ablation row and in
  qualitative figures, never in the headline metric. (memory: dope-inference-needs-reflect-padding
  — pad helps truncation, is neutral-to-harmful on fully-contained frames.)
- **B2 is off-limits as paper main.** B2 = stage11_16k mask-aux finetune that replayed
  v3 + addon_v1 (palletobj scans). It is an *internal* ceiling showing what private-CAD
  replay buys; citing it as a paper baseline would leak the challenge track.

---

## 2. Official numbers (A_nopad, cited from paper_stage_a/eval.json)

filter-val (N=123, **primary** representative set):
```
preprocess   det%  front   rear  corner  honest8  good%  gross%   PnP%
----------------------------------------------------------------------
A_nopad ★      68   16.4   34.7    27.5     31.7   28.7    48.1     71
  V=8          77   15.3   34.2    27.2     28.4   28.6    47.7     79
  V<8          12   54.5   64.6    60.0     70.2   31.2    62.5     18
C_pad100(demo) 79   16.2   33.5    22.6     25.1   21.8    51.6     79
```

handannot17 (N=17, high-elevation-biased, **qualitative** only):
```
preprocess   det%  front   rear  corner  honest8  good%  gross%   PnP%
----------------------------------------------------------------------
A_nopad        24    5.8    9.3     7.6     28.0   76.9     3.8     24
C_pad100(demo) 47    9.1   17.0    13.2     18.9   45.5    21.8     47
```

Consistent across every pad and set: **rear > front** median error (rear is the bottleneck).

---

## 3. paper_base_v2 vs. B2 — detection vs. localization decomposition

★ The gap to B2 is **localization, not detection.** Evidence: STAGE25 evaluates all
models under identical conditions (pad100, order-free Hungarian + solve_pose + honest
full-8 reproj, per-frame K). On filter-val (N=123, overall):

```
model          det%  front   rear  corner  worst2  honest8  good%  gross%
-------------------------------------------------------------------------
paper_base_v2    79   16.2   33.5    22.6    50.9     25.1   21.8    51.6
B2               75   12.1   16.8    13.5    26.1     15.6   29.1    29.1
Δ (paper−B2)     +4   +4.1  +16.7    +9.1   +24.8     +9.5   −7.3   +22.5
```

Reading:
- **Detection (coverage) is EQUAL+**: paper_base_v2 det 79% ≥ B2 75%. paper_base_v2 is
  NOT under-detecting — it finds the pallet at least as often as the private-CAD model.
- **Localization is where it loses**: corner 22.6 vs 13.5 (≈9px), rear 33.5 vs 16.8
  (≈17px), honest8 25.1 vs 15.6 (≈9.5px), gross% 51.6 vs 29.1. The deficit is
  concentrated in **rear corners at low camera angle (flat-view depth collapse)**, not in
  finding the object. (matches history 2026-07-05 verdict + memory
  corner01-premise-debunked-rear-is-bottleneck: REAR/low-elev flat-view is the real lever.)
- Interpretation: the paper_base_v2 → B2 gap is exactly "the cost of NOT replaying
  v3/addon private scans." Because that gap is pure localization on rear/low-angle, the
  paper's improvements (rear-aware synthetic data + mask-aux + geometric-filter self-train)
  target localization, and are measured as recovery of this rear/honest8 gap — **with
  detection held constant** so gains cannot be faked by lowering the detection threshold.

---

## 4. Reporting rules (freeze)

- Headline table: **A_nopad** only. Report det% AND (corner, rear, honest8, good%, gross%)
  so detection and localization are never conflated.
- Any pad100 number is labelled "reflect-pad100 (coverage demo / robustness ablation)".
- B2 appears only in a clearly-labelled "private-CAD reference (not paper main)" row.
- filter-val (N=123) = primary; handannot17 (N=17, high-elev) = qualitative, small-sample
  — never a headline. All sets are small (outside44 / night43 / manual36) — state N.
- ★ final-test set stays SEALED; none of the above touches it. This doc uses filter-val +
  handannot17 (dev sets) only.
