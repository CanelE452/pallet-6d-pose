# Table 3 — Pseudo-label selection ablation

The table keeps two questions apart: does the selection rule improve the
labels, and does the student improve. Only the first is a property of the
filter; the second is what the paper is about.

```text
Selection rule                           Keypoint[px]  gross20     Det   AUROC   FPR95
──────────────────────────────────────────────────────────────────────────────────────
Naive self-training                             7.120    0.180   0.981  0.9913  0.0558
Confidence                                      7.037    0.194   0.987  0.9923  0.0469
+ reprojection consistency                      7.044    0.194   0.987  0.9920  0.0487
+ keypoint-removal consistency                  6.999    0.194   0.987  0.9911  0.0502
+ horizontal-flip consistency (full)            7.210    0.197   0.984  0.9953  0.0283
```

## Do the selection signals separate good labels from bad?

Post-hoc diagnostic, measured against evaluation GT. Development evidence.

```text
signal                    frame-level AUC   corner-level AUC
────────────────────────────────────────────────────────────
box_conf                           0.7245—
kp_conf                 —             0.6023
r_flip                  —             0.6567
r_remove                —             0.6317
s_flip                             0.7345—
s_remove                           0.7256—
s_reproj                           0.7413—
valid_corners                      0.5197—
combined                           0.8116             0.7259
```

The signals are informative but weak, and they are far from a clean
separation. Two facts constrain how much can be claimed:

- the per-keypoint confidence floor removes **0 corners** — every
  supervised corner already clears it, so confidence gating is inert at
  the corner level;
- the combined frame-level discriminator reaches an AUC below 0.82 on a
  population where roughly half of the frames contain a gross error.

Better separation of the labels did not become a better student. That is
the finding this table exists to support.
