# Final experiment plan

Supersedes the question structure in `_docs/paper/EXPERIMENTS.md` for paper-facing
interpretation. That document is preserved as the historical design record.

## Research questions

```text
Q1  How strong is synthetic-only supervision for real-world pallet detection
    and keypoint estimation?

Q2  Does self-training with unlabeled target RGB improve real-domain detection
    coverage and confidence ranking?

Q3  Does the same adaptation also improve fine keypoint localisation?

Q4  Do confidence, projective consistency, and equivariant consistency identify
    better pseudo-labels, and does improved pseudo-label selection translate into
    a better student?

Q5  What limits the benefit of single-frame self-training when localisation does
    not improve?
```

Every main experiment answers exactly one of these. A result that answers none is
cut or moved to the Appendix.

## Main tables — four, no more

### Table 1 — Main comparison (Q1, Q2, Q3)

```text
row                         what it is
──────────────────────────────────────────────────────────────────────────
DOPE same-data control      reference architecture on the same data (Tier C)
Synthetic-only R0           the baseline every arm is measured against
Synthetic-replay control    additional optimisation, no real adaptation
Naive self-training         all teacher predictions above the floor
Confidence self-training    confidence pre-filter only
Full consistency filter     confidence + projective + equivariant
```

Real-FT is **not a row.** It is a reference upper bound trained with real
supervision and belongs in a footnote or a separate reference block, never in the
same column block as the unlabeled-adaptation arms.

```text
columns
──────────────────────────────────────────────────────────────────────────
Detection            coverage on the evaluation population
Detection AP         exact metric name copied from the evaluator
Ranking AUROC        positives against the real negative set
FPR95                at 95 percent true-positive rate
Keypoint error       original-image pixel error, paired population
```

Column rules: the metric name in the header must be the evaluator's own name.
`AP50`, `AP50-95`, and ranking AP are different quantities and are never
interchanged. The keypoint column states its unit and that the comparison is paired.

### Table 2 — Daytime and nighttime adaptation (Q2, Q3)

```text
rows     R0 / Naive / Confidence / Full filter
columns  Day detection   Night detection   Day localisation   Night localisation
```

This table carries the paper's headline message on its own: nighttime detection
improves substantially, and localisation does not improve in either condition.

### Table 3 — Pseudo-label selection ablation (Q4)

```text
rows     Confidence
         + projective (reprojection consistency)
         + single-keypoint-removal reprojection consistency
         + horizontal-flip keypoint consistency (full)

columns  Retention              fraction of teacher predictions kept
         Pseudo-label quality   measured against evaluation GT, diagnostic
         Student localisation   downstream, paired
```

The table exists to keep two questions visibly apart:

```text
does the filter improve label purity?      may be yes
does the student improve?                  measured separately
```

Reporting only the first would misrepresent the result.

### Table 4 — Robustness (Q3, Q5)

```text
condition axes    material (plastic / wood)
                  lighting (daytime / nighttime)
                  occlusion (clean / occluded)
                  truncation
                  distance (near / mid / far)
```

Keep the main-text version narrow. Additional condition columns move to the
Appendix rather than widening the printed table.

## Appendix — diagnostic interventions (Q5)

One compressed table covering the development follow-ups.

```text
intervention                  proxy or mechanism result     student localisation
────────────────────────────────────────────────────────────────────────────────
per-keypoint masking          regression reduced            R0 not beaten
true-ignore at the loss       keypoint confidence recovered R0 not beaten
geometry repair               few repairable keypoints      not trained
reliability weighting         label quality improved        no localisation gain
multi-view teacher consensus  tail worsened                 not trained
```

Numbers are filled from `generated/TABLE_FINAL_DIAGNOSTIC.md`, which reads them from
the original result artifacts.

The purpose of this table is **not** "we tried five things and failed." It is
"candidate failure mechanisms were isolated one at a time and ruled out." Every row
is labelled development evidence.

## Evidence-tier rule for every table

```text
Tier A   contract frozen before the result was seen   -> main tables
Tier B   designed after seeing PAPER_EVAL             -> Appendix / Discussion only
Tier C   reference or upper bound                     -> labelled as reference
```

A Tier B result is never described as an independent confirmation or as a held-out
improvement.

## Prohibited after the fact

```text
replacing the Proposed arm with a better-scoring arm discovered later
promoting a development variant into the main comparison
re-tuning any threshold once an evaluation result has been seen
```
