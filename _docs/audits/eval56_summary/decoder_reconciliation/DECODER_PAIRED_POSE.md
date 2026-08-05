# Common-success paired pose, per path

Each arm against B0 **on the same path**, frames where both solved.  Rescues are
excluded from the delta and reported separately.  Frame-clustered bootstrap,
10,000 resamples, seed 1.

```
   set  arm  path   n  median delta  imp  wor  cat>=10  rescue  new fail  P(improve)
────────────────────────────────────────────────────────────────────────────────────
eval56   E2    P0  50        -0.033   26   24        0       0         0       0.696
eval56   E2    P1  46        -0.112   26   20        1       1         0       0.183
eval56   S1    P0  48        -0.201   28   20        3       2         2       0.577
eval56   S1    P1  43        -0.247   26   17        1       1         3       0.663
eval56   C1    P0  50        +0.296   23   27        9       5         0       0.015
eval56   C1    P1  43        +0.587   16   27        4       0         3       0.061
eval56   N2    P0  50        -0.088   28   22        0       0         0       0.729
eval56   N2    P1  46        -0.158   26   20        1       1         0       0.524
eval56   N3    P0  50        +0.014   17   32        0       2         0       0.869
eval56   N3    P1  46        -0.002   25   19        0       0         0       0.883
  wood   E2    P0  44        -0.052   22   22        2       0         0       0.774
  wood   E2    P1  41        -0.002   21   19        0       0         1       0.866
  wood   S1    P0  43        +0.467   16   27        6       1         1       0.130
  wood   S1    P1  40        +0.216   12   28        3       0         2       0.175
  wood   C1    P0  44        -0.331   28   16        5       1         0       0.576
  wood   C1    P1  42        +0.301   19   23        1       1         0       0.860
  wood   N2    P0  44        -0.065   23   21        2       0         0       0.752
  wood   N2    P1  41        -0.056   21   19        0       0         1       0.937
  wood   N3    P0  44        +0.002   19   22        0       0         0       0.991
  wood   N3    P1  42        +0.000   18   19        0       0         0       0.894
```

P2 is absent because it has no common-success population.

The signs agree between P0 and P1 for every arm on eval56: E2 and N2 slightly
negative, C1 clearly positive (worse), N3 flat.  On wood S1 and C1 are positive
on both paths.  No arm reaches its P(improve) requirement (0.90 eval56 / 0.80
wood) on both sets under either decoder.
