# Fixed, ground-truth-free selection

S0 takes each channel's top-1. S1 searches top-5 cubed per corner and picks lexicographically by intersection residual, then condition number, then candidate score sum. No weighted objective, no tuned coefficient.

```
    arm  seed       set  s0_le20  s0_median  s1_le20  s1_median  oracle_le20
L12-F50     1    eval56 0.040359 193.142079 0.049327 213.842471     0.000000
L12-F50     1 untouched 0.144016 145.635700 0.377197 143.770095     0.977638
L12-F50     1       val 0.145813 146.553920 0.362679 146.279281     0.967905
L12-F50     1      wood 0.002778 330.494551 0.000000 357.486458     0.000000
L12-F50     2    eval56 0.015625 204.354569 0.017857 243.102100     0.000000
L12-F50     2 untouched 0.124662 144.159104 0.317022 151.243367     0.976712
L12-F50     2       val 0.123804 144.042076 0.316746 152.930209     0.989331
L12-F50     2      wood 0.005556 358.459479 0.002778 377.671692     0.000000
L12-F50     3    eval56 0.029148 222.772971 0.026906 239.342071     0.000000
L12-F50     3 untouched 0.169907 144.066292 0.373457 145.760676     0.977978
L12-F50     3       val 0.164833 142.797852 0.364593 149.206206     0.970464
L12-F50     3      wood 0.002778 347.587575 0.008333 375.928811     0.000000
 L12-MS     1    eval56 0.015625 200.825112 0.017857 228.617274     0.250000
 L12-MS     1 untouched 0.159652 142.206088 0.400017 134.192246     0.969912
 L12-MS     1       val 0.157057 144.245350 0.392464 143.776073     0.967949
 L12-MS     1      wood 0.000000 345.793968 0.002778 350.719809     0.000000
 L12-MS     2    eval56 0.013393 221.118423 0.013393 262.345742     1.000000
 L12-MS     2 untouched 0.151961 143.216552 0.389959 141.346824     0.968973
 L12-MS     2       val 0.152392 142.853420 0.389474 140.200014     0.968542
 L12-MS     2      wood 0.008333 321.886563 0.002778 378.585002     0.000000
 L12-MS     3    eval56 0.020089 207.806684 0.013393 231.392815     0.000000
 L12-MS     3 untouched 0.152637 141.992525 0.393932 146.014135     0.973310
 L12-MS     3       val 0.142823 141.906059 0.397249 140.238818     0.979191
 L12-MS     3      wood 0.008333 349.392054 0.002778 352.451961     0.000000
```

## finite PnP (not pose accuracy)

```
    arm  seed       set   mode  complete_frames  frames  finite_pnp  reproj_median
L12-F50     1       val oracle                0    1045           0            NaN
L12-F50     1       val     s0             1045    1045         514     142.828322
L12-F50     1       val     s1             1045    1045         814     132.184747
L12-F50     1 untouched oracle                3    1479           3       2.816618
L12-F50     1 untouched     s0             1479    1479         750     146.709090
L12-F50     1 untouched     s1             1479    1479        1113     131.764855
L12-F50     1    eval56 oracle                0      56           0            NaN
L12-F50     1    eval56     s0               55      56          22     279.065308
L12-F50     1    eval56     s1               55      56          47     216.033844
L12-F50     1      wood oracle                0      45           0            NaN
L12-F50     1      wood     s0               45      45          24     329.434412
L12-F50     1      wood     s1               45      45          38     314.057715
L12-F50     2       val oracle                0    1045           0            NaN
L12-F50     2       val     s0             1045    1045         527     141.294163
L12-F50     2       val     s1             1045    1045         814     137.798511
L12-F50     2 untouched oracle                2    1479           2       2.505598
L12-F50     2 untouched     s0             1479    1479         768     142.683262
L12-F50     2 untouched     s1             1479    1479        1123     136.243251
L12-F50     2    eval56 oracle                0      56           0            NaN
L12-F50     2    eval56     s0               56      56          28     262.398768
L12-F50     2    eval56     s1               56      56          56     233.565623
L12-F50     2      wood oracle                0      45           0            NaN
L12-F50     2      wood     s0               45      45          22     403.571441
L12-F50     2      wood     s1               45      45          42     324.152272
L12-F50     3       val oracle                5    1045           5       2.521568
L12-F50     3       val     s0             1045    1045         509     138.974681
L12-F50     3       val     s1             1045    1045         813     135.963106
L12-F50     3 untouched oracle                1    1479           1       4.100130
L12-F50     3 untouched     s0             1478    1479         698     139.711292
L12-F50     3 untouched     s1             1478    1479        1160     132.652857
L12-F50     3    eval56 oracle                0      56           0            NaN
L12-F50     3    eval56     s0               55      56          29     277.121175
L12-F50     3    eval56     s1               55      56          52     209.803455
L12-F50     3      wood oracle                0      45           0            NaN
L12-F50     3      wood     s0               45      45          27     368.878816
L12-F50     3      wood     s1               45      45          40     324.831450
 L12-MS     1       val oracle                2    1045           2       4.573082
 L12-MS     1       val     s0             1045    1045         373     143.996531
 L12-MS     1       val     s1             1045    1045         780     135.794047
 L12-MS     1 untouched oracle                9    1479           9       3.965598
 L12-MS     1 untouched     s0             1479    1479         604     136.793893
 L12-MS     1 untouched     s1             1479    1479        1066     132.962247
 L12-MS     1    eval56 oracle                0      56           0            NaN
 L12-MS     1    eval56     s0               56      56          32     279.174407
 L12-MS     1    eval56     s1               56      56          55     195.363586
 L12-MS     1      wood oracle                0      45           0            NaN
 L12-MS     1      wood     s0               45      45          14     333.555630
 L12-MS     1      wood     s1               45      45          43     306.051797
 L12-MS     2       val oracle                1    1045           1       2.592815
 L12-MS     2       val     s0             1045    1045         454     142.137495
 L12-MS     2       val     s1             1045    1045         784     136.533200
 L12-MS     2 untouched oracle                6    1479           6       2.662673
 L12-MS     2 untouched     s0             1479    1479         676     139.549496
 L12-MS     2 untouched     s1             1479    1479        1099     131.819977
 L12-MS     2    eval56 oracle                0      56           0            NaN
 L12-MS     2    eval56     s0               56      56          33     258.676958
 L12-MS     2    eval56     s1               56      56          54     229.161302
 L12-MS     2      wood oracle                0      45           0            NaN
 L12-MS     2      wood     s0               45      45          30     328.648247
 L12-MS     2      wood     s1               45      45          42     325.218920
 L12-MS     3       val oracle                3    1045           3       1.972153
 L12-MS     3       val     s0             1045    1045         474     140.702120
 L12-MS     3       val     s1             1045    1045         765     132.145429
 L12-MS     3 untouched oracle                5    1479           5       2.538696
 L12-MS     3 untouched     s0             1479    1479         693     141.520917
 L12-MS     3 untouched     s1             1479    1479        1097     134.767067
 L12-MS     3    eval56 oracle                0      56           0            NaN
 L12-MS     3    eval56     s0               56      56          29     209.019415
 L12-MS     3    eval56     s1               56      56          51     214.116323
 L12-MS     3      wood oracle                0      45           0            NaN
 L12-MS     3      wood     s0               45      45          25     386.130844
 L12-MS     3      wood     s1               45      45          41     350.790271
```
