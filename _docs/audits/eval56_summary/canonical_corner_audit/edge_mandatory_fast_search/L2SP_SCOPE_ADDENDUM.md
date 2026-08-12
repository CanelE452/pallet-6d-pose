# What the L2-SP result is allowed to be called

`f3af68f` stands.  `REGULARIZED_LATE_A1_OVERCONSTRAINED`.

## Established

```
pretrained anchoring strongly reduced drift
    weight  ||W - W0|| / ||W0||  ended at 6.66%
    feature F50 relative drift flattened at 0.92 from 5,000 steps onward,
            while the cosine kept falling from 0.8054 to 0.6168
specialization was substantially reduced
    D2/D0 1.1108 / 1.1545 against F1's 1.4255 / 1.2994
    73.96% and 48.39% of the distance to 1 removed
F1's held-out accuracy was not preserved under this fixed coefficient
    angle median +26.35%, offset median +12.60%
the accuracy/generalization tradeoff moved
```

## Not established

```
lambda is generally too large
a smaller lambda would solve the problem
weight drift causes specialization
the L2-SP family is invalid
F1's specialization is caused by appearance
```

The coefficient was fixed once, before the run, by a gradient balance at a
train-only state, and never swept.  One point on one axis says nothing about the
shape of that axis, and trying a weaker lambda *because this result was seen*
would be selecting on the dev set.

```
DO_NOT_OPEN_LAMBDA_SWEEP = True
```

## The correlation that keeps appearing, and what it is not

```
                   angle median   D2/D0 angle
F1 late unfreeze     2.070244       1.4255
L1 low-rank          2.450502       1.2336
S1 F1 + L2-SP        2.615707       1.1108
R1 role depth        2.909729       1.0461
F2 adapter           3.084991       1.0445
F0 frozen            3.735687       1.0691
```

Accuracy and specialization have moved together across six arms.  That is six
observations from six differently-built runs, not a law and not a mechanism.
Promoting it to "specialization is the price of accuracy" would be reading a
trend line as a constraint.

## The next question

```
DOES_TRAIN_ONLY_APPEARANCE_CONSISTENCY
REDUCE_SPECIALIZATION_WITHOUT_SUPPRESSING_USEFUL_FEATURE_ADAPTATION ?
```

Every constraint tried so far has limited *where* adaptation happens, *how many*
parameters it may use, or *how far* the weights may move.  The next screen limits
none of those: F1's late features adapt freely, and the requirement is instead
that two photometric views of the same geometry produce the same structural-line
distribution.

If that helps, the most it will support is that the consistency factor improved
held-out geometry.  It will not show that appearance caused the specialization.
