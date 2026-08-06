# Synthetic to canonical transfer

```
{
 "untouched": {
  "arm": "L12-MS",
  "seed": 1,
  "set": "untouched",
  "edges": 17741,
  "corners": 11832,
  "edge_top1_strict_line": 0.3282791274449016,
  "edge_top5_strict_line": 0.615636097176033,
  "edge_top5_loose_line": 0.7500140916521053,
  "edge_top1_strict_segment": 0.3269826954512147,
  "edge_top5_strict_segment": 0.6058846739191703,
  "edge_top5_loose_segment": 0.740037201961558,
  "corner_triplet_top1": 0.03972278566599054,
  "corner_triplet_top5": 0.1910074374577417,
  "corner_triplet_loose_top5": 0.40297498309668695,
  "oracle_n": 2260,
  "oracle_le20": 0.9699115044247788,
  "oracle_median": 4.626409936324759,
  "s0_n": 11832,
  "s0_le20": 0.15965179175118324,
  "s0_median": 142.206087545512,
  "s1_n": 11832,
  "s1_le20": 0.40001690331304934,
  "s1_median": 134.19224612082223
 },
 "canonical": {
  "eval56": {
   "arm": "L12-MS",
   "seed": 1,
   "set": "eval56",
   "edges": 666,
   "corners": 448,
   "edge_top1_strict_line": 0.07957957957957958,
   "edge_top5_strict_line": 0.15015015015015015,
   "edge_top5_loose_line": 0.34984984984984985,
   "edge_top1_strict_segment": 0.06606606606606606,
   "edge_top5_strict_segment": 0.08858858858858859,
   "edge_top5_loose_segment": 0.2552552552552553,
   "corner_triplet_top1": 0.0,
   "corner_triplet_top5": 0.008928571428571428,
   "corner_triplet_loose_top5": 0.044642857142857144,
   "oracle_n": 4,
   "oracle_le20": 0.25,
   "oracle_median": 32.99735877562581,
   "s0_n": 448,
   "s0_le20": 0.015625,
   "s0_median": 200.82511219704793,
   "s1_n": 448,
   "s1_le20": 0.017857142857142856,
   "s1_median": 228.61727419627638
  },
  "wood": {
   "arm": "L12-MS",
   "seed": 1,
   "set": "wood",
   "edges": 535,
   "corners": 360,
   "edge_top1_strict_line": 0.08411214953271028,
   "edge_top5_strict_line": 0.16261682242990655,
   "edge_top5_loose_line": 0.27102803738317754,
   "edge_top1_strict_segment": 0.07850467289719626,
   "edge_top5_strict_segment": 0.13644859813084112,
   "edge_top5_loose_segment": 0.22429906542056074,
   "corner_triplet_top1": 0.002777777777777778,
   "corner_triplet_top5": 0.005555555555555556,
   "corner_triplet_loose_top5": 0.03611111111111111,
   "oracle_n": 2,
   "oracle_le20": 0.0,
   "oracle_median": 37.601234705089304,
   "s0_n": 360,
   "s0_le20": 0.0,
   "s0_median": 345.7939676585749,
   "s1_n": 360,
   "s1_le20": 0.002777777777777778,
   "s1_median": 350.7198089047138
  }
 },
 "canonical_mean": {
  "edge_top5_strict_line": 0.15638348629002835,
  "edge_top5_strict_segment": 0.11251859335971486,
  "corner_triplet_top5": 0.007242063492063492,
  "oracle_le20": 0.125,
  "s0_le20": 0.0078125,
  "s1_le20": 0.010317460317460317
 },
 "J5": {
  "synthetic_strict>=70%": false,
  "canonical_strict<30%": true,
  "synthetic_oracle>=50%": true,
  "canonical_oracle<15%": true
 }
}
```
