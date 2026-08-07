# R1C: belief-space edge fusion has capacity; Round-1 lost it in credit assignment

Post-validation diagnostic. Round-1 stands unchanged at
`NO_EDGE_ARCHITECTURE_PASS`; nothing in its files was edited.

## Why Round-1's ORACLE row could not answer this

The Round-1 ORACLE fed ground-truth edges at inference into an EGCR that had been
**trained** against a noisy PEQ. It measured that pipeline's response to clean
input, not the fusion's intrinsic capacity. Here EGCR is trained on
teacher-forced ground-truth edges from the start, on the same search2k, same
frozen A1, same 166 steps.

## Result, validation512 (frozen manifest b236bd3d)

```
        near<=20   all<=20   ID1+2     R4    far>50
C0        0.3784    0.4065   0.5615    254    0.1614
F1        0.8794    0.9131   0.9326    512    0.0081
F2        0.8774    0.9092   0.9268    508    0.0112
```

F1 gains +50.1pp on near<=20, +37.1pp on ID1+2, R4 from 254 to **512 of 512**,
and cuts far>50px from 16.1% to 0.8%. Every gate clears by an order of magnitude.

```
FUSION_CAPACITY_VALID
JOINT_TRAINING_CREDIT_ASSIGNMENT_FAIL
```

Belief-space edge fusion is not the bottleneck and the belief-residual line does
not close. What failed in Round-1 was joint training: with PEQ producing 16-degree
orientations and 6.6-cell offsets, EGCR learned to ignore the proposal channel,
and no later injection of clean edges could undo that. Adding HCRM on top (F2)
costs a little rather than helping, so the near-corner residual has nothing to add
once edges are correct.

## The stronger finding: skip belief space entirely

Ground-truth edges through CIGM straight into the verified `annotate_pnp.solve_pose`:

```
solved              508 / 512
reprojection median   0.033 px
reprojection p90      0.062 px
```

Sub-tenth-pixel pose from twelve edge roles and a fixed incidence, with no belief
map and no decoder in the path. The belief detour costs accuracy rather than
adding any.

## Not deployable

Both F1/F2 and the PnP row consume ground-truth edges at inference. They are
capacity oracles. They say what is reachable if edge geometry is predicted well;
they say nothing about predicting it, which is exactly where Round-1 failed at
16 degrees and 0.9% edge-only accuracy.

## Direction this fixes

The next architecture is edge-line query -> CIGM -> direct PnP, and the whole
problem becomes edge localisation. That is a cleaner target than the previous
three attempts: the dense field, HCRM and PEQ all tried to move a belief map, and
this shows the belief map was never the place to intervene.
