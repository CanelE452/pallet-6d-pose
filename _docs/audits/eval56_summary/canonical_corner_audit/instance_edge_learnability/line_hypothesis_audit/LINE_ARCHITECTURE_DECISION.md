# Decision: MIXED_LINE_FAILURE

```
{
 "decision": "MIXED_LINE_FAILURE",
 "primary_arm": "L12-MS",
 "verdicts": {
  "HYPOTHESIS_PRESENT_SYNTHETIC": false,
  "HYPOTHESIS_PRESENT_CANONICAL": false,
  "SELECTION_FAILURE": false,
  "ENDPOINT_SEGMENT_FAILURE": false,
  "HYPOTHESIS_TRANSFER_COLLAPSE": true,
  "DENSE_LOCALIZATION_ABSENT": true
 },
 "gates": {
  "J1": {
   "edge_top5_strict_line": false,
   "edge_top5_strict_segment": false,
   "corner_triplet_top5": false,
   "oracle_corner_le20": true
  },
  "J2": {
   "edge_top5_strict_line": false,
   "edge_top5_strict_segment": false,
   "corner_triplet_top5": false,
   "oracle_corner_le20": false
  },
  "J3": {
   "oracle_over_s0>=20pp": true,
   "s1_over_s0>=10pp": true,
   "precondition": false
  },
  "J4": {
   "top5_strict_line>=60%": true,
   "top5_strict_segment<35%": false,
   "l1_share>=40%": "False"
  },
  "J5": {
   "synthetic_strict>=70%": false,
   "canonical_strict<30%": true,
   "synthetic_oracle>=50%": true,
   "canonical_oracle<15%": true
  },
  "J6": {
   "top5_strict_line<60%": false,
   "triplet_top5<40%": true,
   "oracle_le20<40%": false
  }
 },
 "architecture": {
  "twelve_edge_representation": "VALID",
  "dense_predictor": "STOP",
  "parametric_extractor": "STOP",
  "CIGM": "STOP",
  "line_only_branch": "STOP",
  "fusion": "STOP",
  "spatial_hcrm": "NEXT"
 }
}
```

## What the label does and does not say

```
Q1  correct line present but unselected     PARTLY, synthetic only
Q2  correct line absent from the map        YES, dominant on canonical
Q3  line right, segment extent wrong        NO -- rejected
Q4  synthetic-to-real transfer collapse     YES
```

### The selection component is real but cannot carry the branch

```
                       S0 top-1   S1 topology   oracle top-5
synthetic untouched      15.5%       39.5%          97.1%
```

S1 gains 24 points over S0 with no ground truth, and the oracle shows the
geometry is essentially exact once the right three lines are in hand.  A better
selector is therefore worth something.  It is not worth the branch: triplet
availability is 19.6% against a 60% gate, so for four corners in five there is
nothing correct to select.

### Endpoints were not the problem

```
top-5 strict infinite line   61.3%      L1_INFINITE_LINE_ONLY   627 / 53,223  = 1.2%
top-5 strict segment         60.1%
```

Within 1.3 points of each other.  When the infinite line is right the extent is
right too, so an endpoint or parametric-segment representation would not have
recovered this.  H3 and Case 2 are rejected on the data rather than deferred.

### The canonical oracle numbers are not usable

Oracle-selected corners exist only where all three incident edges carry a strict
top-5 line: 0.45% of eval56 corners and 0.46% of wood, which is 6 and 5 corners
across three seeds.  The 41.7% and 0.0% that follow are reported for
completeness and carry no weight.  The canonical finding rests on availability --
13.9% and 15.1% of edges, 0.45% of corner triplets -- which is measured over
1,998 and 1,080 edges.
