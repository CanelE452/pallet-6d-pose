# Table 3. Symmetry supervision ablation

## Panel A — rectangular C2, reused DEV319

| Arm | Matched median (px) | Matched P90 (px) | Full PCK10 (%) | E_sym |
|---|---|---|---|---|
| N2_DIM_ONLY | 5.7777 | 42.4595 | 68.587 | 0.04842188 |
| N3_DIM_SYM | 5.7782 | 42.1338 | 68.587 | 0.04841909 |

## Panel B — square C4, separate reused DEV155

| Arm | Matched median (px) | Matched P90 (px) | Full PCK10 (%) | E_sym |
|---|---|---|---|---|
| S0_FIXED | 1.6536 | 5.1696 | 97.141 | 0.00316741 |
| S1_SYM | 1.5319 | 4.1916 | 98.921 | 0.00261229 |

Symmetry supervision ablation; arithmetic means of three per-seed statistics. Panel A: rectangular C2 reused DEV319, N2 fixed-index vs N3 symmetry-aware supervision. Panel B: separate real-supervised secondary square C4 reused DEV155, S0 fixed-index vs S1 symmetry-aware supervision; all 155 frames matched, 1,236 valid corners. Dimensions are constant within the square cohort, so this panel does not test dimension utility. C2/C4 are explicit object/task contracts, not automatically inferred from dimensions. Panels are not pooled; metric definitions follow Table 1.
