# Canonical transfer

Line-only corner generation: no corner heatmap and no top-K enter the decoder.

```
arm         s                eval    <=20px    median       R4    PnP    reproj
-------------------------------------------------------------------------------
L5-CTRL     1              eval56   0.02679     248.3        0      4     235.1
L5-CTRL     1     eval56_shuffled   0.02679     248.3        0      4     235.1
L5-CTRL     1      eval56_aligned   0.02679     248.3        0      4     235.1
L5-CTRL     1                wood  0.002778       389        0     21     317.4
L5-CTRL     1       wood_shuffled  0.002778       389        0     21     317.4
L5-CTRL     1        wood_aligned  0.002778       389        0     21     317.4
L12-F50     1              eval56   0.04464     182.7        0     20     202.8
L12-F50     1     eval56_shuffled    0.0558     225.7        0     40     199.3
L12-F50     1      eval56_aligned   0.04464     182.7        0     20     202.8
L12-F50     1                wood         0       346        0     20       301
L12-F50     1       wood_shuffled   0.02222     374.5        0     41     297.7
L12-F50     1        wood_aligned         0       346        0     20       301
L12-F50     2              eval56   0.02009       219        0     22     226.8
L12-F50     2     eval56_shuffled   0.09152       192        0     43     190.3
L12-F50     2      eval56_aligned   0.02009       219        0     22     226.8
L12-F50     2                wood         0     369.1        0     19     295.6
L12-F50     2       wood_shuffled   0.01111     369.3        0     41       314
L12-F50     2        wood_aligned         0     369.1        0     19     295.6
L12-F50     3              eval56   0.04911     209.4        0     12     162.4
L12-F50     3     eval56_shuffled   0.07812     222.1        0     40     196.8
L12-F50     3      eval56_aligned   0.04911     209.4        0     12     162.4
L12-F50     3                wood  0.005556     368.1        0     12     283.2
L12-F50     3       wood_shuffled   0.01944     372.5        0     39     324.4
L12-F50     3        wood_aligned  0.005556     368.1        0     12     283.2
L12-MS      1              eval56   0.01116     212.9        0     19     239.6
L12-MS      1     eval56_shuffled   0.04688     221.6        0     41     184.8
L12-MS      1      eval56_aligned   0.01116     212.9        0     19     239.6
L12-MS      1                wood         0     361.5        0     24     316.5
L12-MS      1       wood_shuffled  0.008333     406.9        0     44     357.5
L12-MS      1        wood_aligned         0     361.5        0     24     316.5
L12-MS      2              eval56   0.02455     233.5        0     20     243.3
L12-MS      2     eval56_shuffled   0.04241     228.3        0     52     179.4
L12-MS      2      eval56_aligned   0.02455     233.5        0     20     243.3
L12-MS      2                wood         0       381        0     32     303.6
L12-MS      2       wood_shuffled   0.01667     393.7        0     43       331
L12-MS      2        wood_aligned         0       381        0     32     303.6
L12-MS      3              eval56   0.02009     220.8        0     13     245.5
L12-MS      3     eval56_shuffled   0.07143     237.1  0.01786     39     256.4
L12-MS      3      eval56_aligned   0.01562     227.1        0     11     250.4
L12-MS      3                wood         0     367.7        0     23     292.6
L12-MS      3       wood_shuffled  0.005556     409.1        0     41     348.5
L12-MS      3        wood_aligned         0       368        0     23     287.4
```

```
{
 "oracle_o12": {
  "eval56": {
   "set": "eval56",
   "mode": "O12",
   "n_frames": 56,
   "le20": 0.9866071428571429,
   "le50": 0.9955357142857143,
   "gt100": 0.0,
   "median": 4.683842999069036,
   "pnp": 56,
   "reference_le20": 0.987,
   "reference_pnp": 56,
   "le20_delta": -0.00039285714285708373,
   "parity": true
  },
  "wood": {
   "set": "wood",
   "mode": "O12",
   "n_frames": 45,
   "le20": 0.9611111111111111,
   "le50": 0.9722222222222222,
   "gt100": 0.013888888888888888,
   "median": 7.992492955535445,
   "pnp": 45,
   "reference_le20": 0.961,
   "reference_pnp": 45,
   "le20_delta": 0.00011111111111117289,
   "parity": true
  }
 },
 "oracle_o5": {
  "eval56": {
   "set": "eval56",
   "mode": "O5",
   "n_frames": 56,
   "le20": 0.026785714285714284,
   "le50": 0.08482142857142858,
   "gt100": 0.6941964285714286,
   "median": 148.6557611762965,
   "pnp": 0,
   "reference_le20": 0.027,
   "reference_pnp": 0,
   "le20_delta": -0.00021428571428571547,
   "parity": true
  },
  "wood": {
   "set": "wood",
   "mode": "O5",
   "n_frames": 45,
   "le20": 0.005555555555555556,
   "le50": 0.013888888888888888,
   "gt100": 0.9583333333333334,
   "median": 292.5339347551767,
   "pnp": 0,
   "reference_le20": 0.006,
   "reference_pnp": 0,
   "le20_delta": -0.00044444444444444436,
   "parity": true
  }
 },
 "local_corner_reference": {
  "available": true,
  "source": "data/pallet/results/paper_s2_eval56/decoder_reconciliation/compatibility_calibration/pdg_unified_program/stage1_failure_audit/canonical_reeval.csv",
  "A0|eval56": {
   "n": 56,
   "pnp": 50,
   "r4": 0.8392857142857143,
   "corner_median_px": 8.613709148486386,
   "reproj_median_px": 11.557804521813573
  },
  "A0|wood": {
   "n": 45,
   "pnp": 44,
   "r4": 0.9777777777777777,
   "corner_median_px": 9.327208269502345,
   "reproj_median_px": 9.28390307559009
  },
  "A1|eval56": {
   "n": 56,
   "pnp": 52,
   "r4": 0.8928571428571429,
   "corner_median_px": 8.073741352426572,
   "reproj_median_px": 10.16083835698247
  },
  "A1|wood": {
   "n": 45,
   "pnp": 44,
   "r4": 0.9333333333333333,
   "corner_median_px": 9.453489173505664,
   "reproj_median_px": 8.956907488450696
  },
  "A2|eval56": {
   "n": 112,
   "pnp": 102,
   "r4": 0.8125,
   "corner_median_px": 9.348314547956312,
   "reproj_median_px": 16.269942240420345
  },
  "A2|wood": {
   "n": 90,
   "pnp": 88,
   "r4": 0.9666666666666667,
   "corner_median_px": 9.548024974288946,
   "reproj_median_px": 8.776553788810869
  }
 },
 "gates": {
  "L5-CTRL": {
   "checks": {
    "1_eval56_le20>=50%_oracle": false,
    "2_wood_le20>=50%_oracle": false,
    "3_eval56_pnp>=50%_oracle": false,
    "4_wood_pnp>=50%_oracle": false,
    "5_fixed_beats_shuffled>=20pp": false,
    "6_active_channels==12": false,
    "7_fixed_vs_aligned<=20pp": true,
    "8_no_collapse": true,
    "9_occluded>=40%_of_visible": false,
    "10_two_of_three_seeds": false
   },
   "passed": false,
   "fraction": {
    "eval56": {
     "le20_fraction": 0.027149321266968323,
     "pnp_fraction": 0.07142857142857142,
     "r4_fraction": 0.0
    },
    "wood": {
     "le20_fraction": 0.002890173410404624,
     "pnp_fraction": 0.4666666666666667,
     "r4_fraction": 0.0
    }
   },
   "seeds_meeting_direction": 0
  },
  "L12-F50": {
   "checks": {
    "1_eval56_le20>=50%_oracle": false,
    "2_wood_le20>=50%_oracle": false,
    "3_eval56_pnp>=50%_oracle": false,
    "4_wood_pnp>=50%_oracle": false,
    "5_fixed_beats_shuffled>=20pp": false,
    "6_active_channels==12": true,
    "7_fixed_vs_aligned<=20pp": true,
    "8_no_collapse": true,
    "9_occluded>=40%_of_visible": true,
    "10_two_of_three_seeds": false
   },
   "passed": false,
   "fraction": {
    "eval56": {
     "le20_fraction": 0.049773755656108594,
     "pnp_fraction": 0.21428571428571427,
     "r4_fraction": 0.0
    },
    "wood": {
     "le20_fraction": 0.005780346820809248,
     "pnp_fraction": 0.26666666666666666,
     "r4_fraction": 0.0
    }
   },
   "seeds_meeting_direction": 0
  },
  "L12-MS": {
   "checks": {
    "1_eval56_le20>=50%_oracle": false,
    "2_wood_le20>=50%_oracle": false,
    "3_eval56_pnp>=50%_oracle": false,
    "4_wood_pnp>=50%_oracle": true,
    "5_fixed_beats_shuffled>=20pp": false,
    "6_active_channels==12": true,
    "7_fixed_vs_aligned<=20pp": true,
    "8_no_collapse": true,
    "9_occluded>=40%_of_visible": true,
    "10_two_of_three_seeds": false
   },
   "passed": false,
   "fraction": {
    "eval56": {
     "le20_fraction": 0.024886877828054297,
     "pnp_fraction": 0.35714285714285715,
     "r4_fraction": 0.0
    },
    "wood": {
     "le20_fraction": 0.0,
     "pnp_fraction": 0.7111111111111111,
     "r4_fraction": 0.0
    }
   },
   "seeds_meeting_direction": 0
  }
 }
}
```
