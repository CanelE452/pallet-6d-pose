# eval44-clean and wood -- not spent

Phase M runs only after a candidate clears the D13 gate.  None did, so neither
holdout was evaluated.

Membership was still computed and frozen so a later run cannot redefine it:

```
E44  eval56 minus the 12 frames shared with N87        44 frames
W45  wood                                              45 frames
D13 inter E44 = 0      C13 inter E44 = 0
D13 inter W45 = 0      C13 inter W45 = 0
```

Both holdouts are disjoint from the development sets, so when a candidate does
appear they will be clean one-shot evaluations.
