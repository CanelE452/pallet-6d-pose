# 정본 119장 — Ours vs 각 baseline PAIRED (ADD m / yaw deg)

```
dom            vs               met   N   other    Ours   better        p
------------------------------------------------------------------
outside_ft_p07 Synthetic only   ADD  14   0.292   0.290   7/14     0.4631
outside_ft_p07 Synthetic only   yaw  14   5.258   3.990   8/14     0.5416
outside_ft_p07 Naive ST         ADD  16   0.290   0.290  10/16     0.4037
outside_ft_p07 Naive ST         yaw  16   6.237   4.397  10/16     0.4037
outside_ft_p07 Reproj+flip ST   ADD  15   0.334   0.317   7/15     0.9780
outside_ft_p07 Reproj+flip ST   yaw  15   7.935   4.399   9/15     0.1688
--------------------------------------------------------
outside_ft_p09 Synthetic only   ADD  20   0.941   0.412  13/20     0.0400
outside_ft_p09 Synthetic only   yaw  20  16.110  14.057  11/20     0.3300
outside_ft_p09 Naive ST         ADD  27   0.447   0.350  18/27     0.0187
outside_ft_p09 Naive ST         yaw  27  11.096  13.862  12/27     0.8223
outside_ft_p09 Reproj+flip ST   ADD  30   0.517   0.342  21/30     0.0017
outside_ft_p09 Reproj+flip ST   yaw  30  12.748  13.844  18/30     0.1460
--------------------------------------------------------
outside_fv     Synthetic only   ADD  16   0.126   0.158   6/16     0.7057
outside_fv     Synthetic only   yaw  16   4.411   5.293   4/16     0.0654
outside_fv     Naive ST         ADD  19   0.241   0.170  11/19     0.2101
outside_fv     Naive ST         yaw  19   6.151   3.727   6/19     0.2253
outside_fv     Reproj+flip ST   ADD  19   0.545   0.170  14/19     0.0046
outside_fv     Reproj+flip ST   yaw  19   6.274   3.727   7/19     0.2753
--------------------------------------------------------
night_ft       Synthetic only   ADD  23   0.281   0.272  14/23     0.1045
night_ft       Synthetic only   yaw  23   5.758   2.507  13/23     0.3146
night_ft       Naive ST         ADD  24   0.376   0.277  18/24     0.0022
night_ft       Naive ST         yaw  24   3.798   2.552  13/24     0.1208
night_ft       Reproj+flip ST   ADD  26   0.439   0.297  19/26     0.0525
night_ft       Reproj+flip ST   yaw  26   2.410   2.757  12/26     0.7835
--------------------------------------------------------
night_ft_n08   Synthetic only   ADD  12   0.160   0.208   6/12     0.7910
night_ft_n08   Synthetic only   yaw  12   3.304   2.239   8/12     0.1099
night_ft_n08   Naive ST         ADD  12   0.336   0.208   9/12     0.0161
night_ft_n08   Naive ST         yaw  12   1.838   2.239   5/12     0.7910
night_ft_n08   Reproj+flip ST   ADD  12   0.204   0.208   8/12     0.2661
night_ft_n08   Reproj+flip ST   yaw  12   2.167   2.239   6/12     0.5186
--------------------------------------------------------
night_ft_n09   Synthetic only   ADD  11   0.446   0.333   8/11     0.0420
night_ft_n09   Synthetic only   yaw  11   8.126   5.084   5/11     0.8984
night_ft_n09   Naive ST         ADD  12   0.470   0.322   9/12     0.0640
night_ft_n09   Naive ST         yaw  12  16.802   8.728   8/12     0.0923
night_ft_n09   Reproj+flip ST   ADD  14   0.469   0.350  11/14     0.1531
night_ft_n09   Reproj+flip ST   yaw  14   3.744   8.841   6/14     0.5016
--------------------------------------------------------
noapril        Synthetic only   ADD  11   0.056   0.053   7/11     0.2061
noapril        Synthetic only   yaw  11   0.886   0.735   8/11     0.0244
noapril        Naive ST         ADD  11   0.061   0.053  11/11     0.0010
noapril        Naive ST         yaw  11   0.946   0.735  10/11     0.0068
noapril        Reproj+flip ST   ADD  11   0.083   0.053   9/11     0.0049
noapril        Reproj+flip ST   yaw  11   1.188   0.735  10/11     0.0029
--------------------------------------------------------
```

cad 제외(검출 붕괴 N<=2). noapril 은 포화라 차이 거의 없음 예상.
소표본 — p 값은 예비적.
