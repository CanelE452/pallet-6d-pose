# Synthetic learnability

```
arm         s              eval    <=20px    median       R4      PnP       F1   act
------------------------------------------------------------------------------------
L5-CTRL     1               val    0.1833     164.2        0  0.09282    0.369     5
L5-CTRL     1      val_shuffled    0.1833     164.2        0  0.09282    0.369     5
L5-CTRL     1       val_aligned    0.1833     164.2        0  0.09282    0.369     5
L5-CTRL     1         untouched    0.1831     162.8        0  0.08012   0.3525     5
L12-F50     1               val    0.4144     131.3    0.556   0.4632   0.5317    12
L12-F50     1      val_shuffled    0.1036     160.2  0.001914   0.9541   0.5317    12
L12-F50     1       val_aligned    0.4144     131.3    0.556   0.4632   0.5317    12
L12-F50     1         untouched    0.4183     131.8   0.5592   0.4332   0.5377    12
L12-F50     2               val    0.4105     131.8   0.5388   0.4373   0.4833    12
L12-F50     2      val_shuffled   0.09797     159.6  0.004785   0.9005   0.4833    12
L12-F50     2       val_aligned    0.4105     131.8   0.5388   0.4373   0.4833    12
L12-F50     2         untouched    0.4077     132.1   0.5509   0.4319   0.4878    12
L12-F50     3               val    0.4212     132.2   0.5703   0.4153   0.5379    12
L12-F50     3      val_shuffled    0.1081     161.7  0.002871    0.911   0.5379    12
L12-F50     3       val_aligned    0.4212     132.2   0.5703   0.4153   0.5379    12
L12-F50     3         untouched    0.4253     128.1   0.5909   0.3937   0.5415    12
L12-MS      1               val    0.4411     114.6   0.6574   0.3005   0.5437    12
L12-MS      1      val_shuffled   0.05921     160.9        0   0.9378   0.5437    12
L12-MS      1       val_aligned    0.4411     114.6   0.6574   0.3005   0.5437    12
L12-MS      1         untouched    0.4423     113.8   0.6629    0.312   0.5481    12
L12-MS      2               val     0.444     114.2    0.644   0.3627    0.526    12
L12-MS      2      val_shuffled   0.07225       159  0.0009569   0.9627    0.526    12
L12-MS      2       val_aligned     0.444     114.2    0.644   0.3627    0.526    12
L12-MS      2         untouched    0.4367     121.1   0.6305   0.3629   0.5282    12
L12-MS      3               val    0.4396     120.7   0.6354   0.4048   0.5475    12
L12-MS      3      val_shuffled   0.08852       163        0   0.9158   0.5475    12
L12-MS      3       val_aligned    0.4396     120.1   0.6344   0.3914   0.5474    12
L12-MS      3         untouched    0.4366     122.3   0.6109   0.4175   0.5506    12
```

```
{
 "L5-CTRL": {
  "gate": {
   "checks": {
    "1_val_le20>=0.70": false,
    "2_untouched_le20>=0.60": false,
    "3_r4": false,
    "4_pnp": false,
    "5_fixed_beats_shuffled>=30pp": false,
    "6_fixed_vs_aligned<=15pp": true,
    "7_active_channels==12": false,
    "8_min_channel_recall>=0.30": true,
    "9_seed_range<=15pp": true
   },
   "passed": false
  },
  "taxonomy": [
   "CHANNEL_COLLAPSE",
   "DIRECT_HEAD_FAILURE"
  ],
  "seed_range_le20": 0.0
 },
 "L12-F50": {
  "gate": {
   "checks": {
    "1_val_le20>=0.70": false,
    "2_untouched_le20>=0.60": false,
    "3_r4": false,
    "4_pnp": false,
    "5_fixed_beats_shuffled>=30pp": true,
    "6_fixed_vs_aligned<=15pp": true,
    "7_active_channels==12": true,
    "8_min_channel_recall>=0.30": true,
    "9_seed_range<=15pp": true
   },
   "passed": false
  },
  "taxonomy": [
   "LOCALIZATION_FAILURE"
  ],
  "seed_range_le20": 0.01064593301435407
 },
 "L12-MS": {
  "gate": {
   "checks": {
    "1_val_le20>=0.70": false,
    "2_untouched_le20>=0.60": false,
    "3_r4": false,
    "4_pnp": false,
    "5_fixed_beats_shuffled>=30pp": true,
    "6_fixed_vs_aligned<=15pp": true,
    "7_active_channels==12": true,
    "8_min_channel_recall>=0.30": true,
    "9_seed_range<=15pp": true
   },
   "passed": false
  },
  "taxonomy": [
   "LOCALIZATION_FAILURE"
  ],
  "seed_range_le20": 0.004425837320574166
 }
}
```
