# Final paper outline

Target venue: IEEE Sensors Journal.
This outline supersedes the section plan in `_docs/paper/EXPERIMENTS.md` for all
paper-facing interpretation. That file is preserved as a historical design document.

## Structure

```text
1  Introduction
2  Related Work
3  Methodology
   3.1  Problem formulation
   3.2  Synthetic source supervision
   3.3  Target-domain self-training
   3.4  Confidence and consistency signals
   3.5  Exposure-matched adaptation protocol
4  Experimental Setup
   4.1  Dataset and splits
   4.2  Evaluation protocol
   4.3  Baselines
   4.4  Metrics
5  Results
   5.1  Main comparison                          -> Table 1
   5.2  Daytime and nighttime adaptation         -> Table 2
   5.3  Pseudo-label selection ablation          -> Table 3
   5.4  Robustness                               -> Table 4
6  Analysis and Discussion
   6.1  The detection-localisation trade-off
   6.2  Why improved pseudo-label purity does not transfer
   6.3  Semantic-axis ambiguity near square projections
   6.4  Teacher ceiling
7  Limitations
8  Conclusion
```

## What goes where

```text
section  content                                     evidence tier
──────────────────────────────────────────────────────────────────────────
3        frozen V1 study contract only               A
5.1-5.4  V1 frozen arms + reference baselines        A (+ C for references)
6        V2-V5, FAST teacher, separability, axis     B  (labelled diagnostic)
7        every limitation in LIMITATIONS section     —
Appendix diagnostic intervention table, extra
         robustness columns, protocol details        B
```

**Method (Section 3) describes the frozen protocol, not the experiment history.**
V2 through V5 and the teacher probes are development follow-ups; they appear in
Section 6 and the Appendix, never as successive versions of the proposed method.

## Ordering rule for Section 5

The negative localisation result appears in **5.3**, before the ablation and before
the diagnostics. It is not deferred to the end and it is not placed after the
explanations that soften it.

## Reverse check

Every paragraph must trace:

```text
result -> research question (Q1-Q5) -> contribution (C1-C3) -> top-level claim
```

If a paragraph cannot complete that chain it is deleted or moved to the Appendix.
"We ran it, so it goes in" is not a reason to include a result.

## Cross-references

```text
research questions        FINAL_EXPERIMENT_PLAN.md
claims and prohibitions   PAPER_CLAIM_LOCK.md
evidence classification   EVIDENCE_LEDGER.md
numbers and their source  generated/RESULT_SOURCE_MAP.json
terminology               PAPER_CLAIM_LOCK.md, terminology section
```
