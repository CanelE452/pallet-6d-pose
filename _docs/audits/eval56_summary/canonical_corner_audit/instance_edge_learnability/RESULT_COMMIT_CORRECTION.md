# Result metadata: what was corrected and what was not

## Not an error

The `tests` field of `COMPLETE` was reported as "0/0".  It was not wrong.  `0` is
the pytest **exit status**, and the same field already carried the counts in its
captured tails:

```
new_tests_returncode  0      new_tests_tail  ... "38 passed, 3 warnings in 4.85s"
full_returncode       0      full_tail       ... "491 passed, 10 warnings in 50.75s"
```

What was wrong was the one-line summary `[test] new=0 full=0`, which reads as a
count.  The stored metadata was accurate throughout.

## Re-verified

```
python -m pytest -q challenge/tests/test_instance_edge_learnability.py    38 passed
python -m pytest -q challenge/tests                                      491 passed
```

`COMPLETE.tests` now also carries parsed `passed / failed / skipped` and the
exact commands, so the field cannot be misread again.  No result number was
changed.

## Metric semantics, stated explicitly

```
R4    a frame with at least 4 line-generated corners within 20px of GT
PnP   the solver returned a finite pose from the generated correspondences
```

`PnP` is **not** a claim of pose accuracy.  EPnP returns a pose from any four or
more correspondences whether or not they are correct, so `R4 = 0.0%` together
with a non-zero finite-PnP count is not a contradiction.  Always with the
denominator:

```
eval56 finite PnP   20/56      wood finite PnP   32/45      (L12-MS best seed)
R4                   0/56                         0/45
```

## Verdict labels

```
automatic    DIRECT_12EDGE_HEAD_NOT_LEARNABLE
primary      DIRECT_12EDGE_FIELD_LOCALIZATION_FAIL
secondary    SYNTHETIC_TO_REAL_TRANSFER_COLLAPSE
```

The automatic label is the Phase G rule applied verbatim and is kept for
traceability, but it understates the result and is not used alone.  Channel
identity is learned (12/12 active, minimum channel recall 0.997, Hungarian
alignment identical to fixed), the fixed incidence topology demonstrably works
(+38pp over shuffled incidence), occlusion is not the cause (1.7pp between
all-visible and any-occluded corners over 47k samples), and multi-scale is not
the cause (1.2pp on untouched, and F50 is better on canonical).  What fails is
precise line localization, and that failure is amplified on real data from 44%
to 2.5%.
