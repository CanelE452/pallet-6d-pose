# What the low-rank result is allowed to be called

`f321345` stands unchanged.

```
LATE_A1_LOW_RANK_INSUFFICIENT
LOW_RANK_A1_SPECIALIZES = True
```

## Established

```
rank-8 internal adaptation changes the late A1 features strongly
    F50 relative drift 3.9068, cosine against frozen 0.3724
parameter count 113,664 = 2.2665% of the F1 full unfreeze
the delta magnitude was NOT small
    final ||dW|| / ||W0||   [19] 1.2539  [21] 1.1628  [23] 1.3521  [25] 1.1916
    all four above 1
offset reduction 43.50% -- clears the 40% threshold
angle  reduction 34.40% -- misses it
D0 angle cleared 40% at 43.15% while D2 angle did not, at 34.40%
```

## The phrasing that must not be used

```
"low rank was too restrictive"
"low rank kept the backbone close to pretrained"
"full unfreeze is required"
"internal adaptation necessarily specializes"
```

The first two are contradicted by the measurement in this very run: every one of
the four effective delta kernels ended up larger than the frozen kernel it
corrects, and F50 moved almost four times its own norm.  Nothing about that arm
kept the backbone near its pretrained state.

```
LOW_RANK = LOW-DIMENSIONAL STRUCTURAL CONSTRAINT,
NOT SMALL-MAGNITUDE CONSTRAINT
```

The third is unsupported -- one rank, one formulation, one learning rate, and
F1's signal is unchanged either way.  The fourth generalises two arms into a
necessity; what was observed is that both arms adapting inside the late
convolutions specialized and both adapting outside them did not.

## The next question

```
DOES_EXPLICIT_PRETRAINED_WEIGHT_ANCHORING
PRESERVE_F1_ACCURACY_WHILE_REDUCING_SPECIALIZATION ?
```

Because magnitude was never constrained in any arm so far, the untested factor
is a penalty on it.  Not a new component, not a different location, not a
different capacity: F1 exactly, plus an L2-SP term pulling the late weights
toward their pretrained values rather than toward zero.  One factor.

Whatever comes out, "specialization is caused by weight drift" will not follow.
The most an L2-SP arm can support is that explicit anchoring changes the
accuracy/generalization tradeoff.
