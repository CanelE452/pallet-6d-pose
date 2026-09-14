# Population and provenance limits

The source training stream is the exact historical cached synthetic P stream. `POPULATION_AND_METRIC_LOCK.json` records actual partition counts. D uses55,915 matched usable training rows and the original per-seed96,000 exposure order. No real features/labels or negative image normalization enter the optimizer. Synthetic selection and heldout are reused historical partitions, not independent of every R0/probe decision.

DEV319 contains311 matched frames and2,756 supervised matched keypoints; the full GT point denominator is stored in the population lock. The8 unmatched frames remain in all-GT PCK. Seed repetitions are not new frames.13 observed sessions support conditional session-level uncertainty, not a population-generalization guarantee.

2D and pose artifacts use different frame-ID spellings. `analysis_panel.py` makes a bijection using actual canonical image paths before coverage/paired computations. An initial new adapter's exclusion list compared the two spellings directly; it was corrected before final reporting. Numerical2D primary and stored canonical pose values were unaffected. Historical artifacts were not changed.

All raw provenance includes sources and hashes; large cached features are mmap read-only, bounded by existing completion/shard manifests and recorded sizes/mtime. Core weights, evaluator inputs, source/code/selection files are byte-bound. The record does not claim a fresh full-byte verification of every terabyte-scale feature array.

No independent new capture membership was supplied. Existing FINAL manifests explicitly mark membership unavailable. History discovery covers recorded prior uses and must be supplemented by human capture-lineage review before any confirmation claim.
