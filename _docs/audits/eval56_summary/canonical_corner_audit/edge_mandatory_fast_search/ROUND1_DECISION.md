# Round-1: NO_EDGE_ARCHITECTURE_PASS

validation512 opened once, under the protocol committed at c49973c before any
forward. Nothing was changed after seeing a number.

## Result

```
                near<=20   all<=20   ID1+2     R4    far>50   edge-only<=20
C0_A1            0.3784    0.4065   0.5615    254   0.1614          --
E1 NORMAL        0.3774    0.4067   0.5518    253   0.1562      0.0093
E1 ZERO          0.3784    0.4065   0.5615    254   0.1614      0.0093
E1 SHUFFLE       0.3750    0.4055   0.5508    253   0.1553      0.0054
E1 ORACLE        0.3813    0.4092   0.5576    254   0.1526      1.0000
E2 NORMAL        0.3774    0.4065   0.5518    253   0.1582      0.0093
E2 ZERO          0.3789    0.4067   0.5664    254   0.1616      0.0093
E2 ORACLE        0.3818    0.4092   0.5625    254   0.1538      1.0000

geometry (both arms)   orientation median 16.0 deg   offset median 6.6 cell
                       half-length relative error 0.60
                       query activity 12/12          CIGM finite 100%
```

## Gates

```
        G1  G2  G3  G4  G5  G6  G7  G8  EDGE_USE
E1      ok  --  --  ok  --  --  ok  --  ok
E2      ok  --  --  ok  --  --  ok  --  ok
```

G2 orientation 16.0 against 15, G3 offset 6.6 against 5, G5 edge-only 0.9%
against 20%, G6 no gain, G8 shuffle costs 0.4pp against 10.

## What failed, and what did not

The queries are alive and the geometry module is sound. All twelve roles stay
active, CIGM returns a finite corner on every frame, and feeding ground-truth
edges through the same CIGM and the same trained EGCR puts 100% of edge-only
corners within 20px. The pipeline downstream of PEQ works.

PEQ itself does not localise. Orientation is off by 16 degrees at the median and
the perpendicular offset is 6.6 cells -- more than a corner's Gaussian is wide --
so only 0.9% of edge-generated corners land within 20px. That is
`EDGE_QUERY_LOCALIZATION_FAIL`.

Role semantics are not used either. Shuffling all twelve roles through a
derangement costs 0.4 points of edge-only accuracy, where the gate asks for 10.
A head that had learned which role is which would not be indifferent to having
them permuted.

The most informative row is ORACLE. With perfect edge geometry the final belief
moves by +0.3pp on near <= 20px and **-0.4pp on ID1+2**, and R4 stays at 254.
So even a perfect edge branch does not convert into corner or pose gain through
this fusion: `EDGE_FUSION_NOT_CONVERTED`. Fixing PEQ localisation would not have
rescued Round-1.

`EDGE_USE` passes only in the degenerate sense that ZERO differs from NORMAL by
essentially nothing in both directions -- the edge contribution is present but
too small to matter, which is the same story from the other side.

## Decision

```
NO_EDGE_ARCHITECTURE_PASS
```

Round-2 and the full 20k run are not executed. Per protocol, Round-1 failure is
failure; any new configuration becomes a separate R1C experiment and cannot enter
Round-1 selection.

## Standing

```
untouched   0 opened
eval56      0 opened
wood45      0 opened
final-test  UNOPENED
```

## Read alongside

This is the third architecture in the same family to fail on the same axis. The
dense twelve-edge field failed because the correct line left the top-5 on real
data; Spatial HCRM failed because a residual that moved 72 of 80 corners changed
no metric; PEQ fails because even ground-truth edge geometry does not convert
through fusion. The common finding is not that edges carry no information -- the
O12 oracle at 98.7% and this ORACLE row at 100% edge-only both say they do -- but
that a corner-belief head fed by A1 does not turn that information into pose.
