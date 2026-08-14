# PalletGraph-6D — Oracle Line Utility Screen

- checkpoint `net_epoch_0057.pth` SHA `c0055fe7c4210f63…` (불변)
- git HEAD `0956b0a36638cf00ab0d6a1e88ecc399ebe658bd`
- baseline gate: GT-2D 87/87, predicted 70/87, yaw 6.025°, reproj 23.162px
- final-test open count **0**
- close-range 정의(지표 확인 전 고정): `bbox_area_ratio_top_25pct`


> **P2/P3 는 oracle 이다.**  GT pose 로 그린 line 이므로 inference 결과가 아니다.  P2(amodal)는 자기가림 edge 를 포함할 수 있어 'visible line' 이 아니다.


## 1. Arm 정의

```
arm  points   solver      line                         
───────────────────────────────────────────────────────
P0   ep57 D0  OpenCV PnP  -                            
P1   ep57 D0  DGP         lambda_line=0                
P2   ep57 D0  DGP         oracle AMODAL                
P3   ep57 D0  DGP         oracle ASSOCIATED            
P4   ep57 D0  DGP         generic edge (class-agnostic)
```

## 2. DGP point-only parity (P0 -> P1)

```
check                         value  
─────────────────────────────────────
pose_success_delta_frames     0      
yaw_median_delta_deg          0.7090 
reproj_median_delta_px        -0.9384
n_common_success              70     
nan_inf_count                 0      
F5_safe_pose_success_drop_pp  0.0000 
passed                        False  
```

[확인] parity FAIL — solver 교체 자체가 결과를 바꾸는 정도를 먼저 분리했다.


## 3. Arm x subset 절대 지표

```
arm      subset              n   pose_success_rate  yaw_mod180_median  rotation_sym_median  translation_median_m  corner_sym_median_m  reproj_fixed_gt_median_px  fallback_rate
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
P0       ALL                 87  0.8046             6.025              6.555                0.4194                0.4516               23.162                     0.0000       
P0       truncated           17  0.3529             6.574              7.792                1.220                 1.220                55.135                     0.0000       
P0       non_truncated       70  0.9143             6.025              6.555                0.4017                0.4265               20.853                     0.0000       
P0       close_range         22  0.5909             7.671              9.091                0.2475                0.2611               51.971                     0.0000       
P0       F1_NO_RESPONSE      24  0.2917             7.697              10.089               0.1878                0.2611               57.608                     0.0000       
P0       F2_CONFIDENT_WRONG  35  1.000              9.945              9.938                0.6232                0.8365               40.148                     0.0000       
P0       F5_safe             23  1.000              2.070              3.213                0.3832                0.3959               9.980                      0.0000       
P0       outside             44  0.8182             6.655              6.798                0.5029                0.5848               30.661                     0.0000       
P0       night               43  0.7907             4.779              5.435                0.4005                0.4099               21.384                     0.0000       
P1       ALL                 87  0.8046             6.734              7.147                0.2983                0.3326               22.223                     0.1954       
P1       truncated           17  0.3529             6.616              7.838                1.246                 1.255                54.997                     0.6471       
P1       non_truncated       70  0.9143             6.734              7.147                0.2685                0.2907               20.101                     0.0857       
P1       close_range         22  0.5909             7.744              10.491               0.4122                0.4122               51.842                     0.4091       
P1       F1_NO_RESPONSE      24  0.2917             9.101              10.491               0.2614                0.2762               57.608                     0.7083       
P1       F2_CONFIDENT_WRONG  35  1.000              9.156              10.165               0.6224                0.8295               40.723                     0.0000       
P1       F5_safe             23  1.000              2.509              3.207                0.1683                0.1725               9.431                      0.0000       
P1       outside             44  0.8182             6.449              6.681                0.3206                0.3757               23.213                     0.1818       
P1       night               43  0.7907             7.872              8.215                0.2701                0.2974               21.137                     0.2093       
P2       ALL                 87  0.8046             7.320              7.620                0.3802                0.4108               19.669                     0.1954       
P2       truncated           17  0.3529             6.171              7.417                1.237                 1.246                53.744                     0.6471       
P2       non_truncated       70  0.9143             7.320              7.620                0.3035                0.3806               18.641                     0.0857       
P2       close_range         22  0.5909             7.652              9.874                0.3913                0.3924               46.807                     0.4091       
P2       F1_NO_RESPONSE      24  0.2917             9.471              11.020               0.1868                0.2609               57.117                     0.7083       
P2       F2_CONFIDENT_WRONG  35  1.000              8.928              9.619                0.6499                0.8156               40.533                     0.0000       
P2       F5_safe             23  1.000              2.316              2.800                0.2127                0.2146               9.916                      0.0000       
P2       outside             44  0.8182             6.501              6.843                0.4430                0.4600               21.035                     0.1818       
P2       night               43  0.7907             7.927              8.286                0.3035                0.3537               19.464                     0.2093       
P3       ALL                 87  0.8046             7.437              7.689                0.3309                0.3781               20.970                     0.1954       
P3       truncated           17  0.3529             6.187              7.696                1.238                 1.247                53.619                     0.6471       
P3       non_truncated       70  0.9143             7.437              7.689                0.2997                0.3445               19.386                     0.0857       
P3       close_range         22  0.5909             7.679              10.194               0.3204                0.3206               44.911                     0.4091       
P3       F1_NO_RESPONSE      24  0.2917             9.229              10.484               0.1903                0.2610               57.546                     0.7083       
P3       F2_CONFIDENT_WRONG  35  1.000              9.473              10.436               0.6526                0.8149               40.677                     0.0000       
P3       F5_safe             23  1.000              1.862              2.766                0.1877                0.1897               8.801                      0.0000       
P3       outside             44  0.8182             6.997              7.176                0.3808                0.4045               23.855                     0.1818       
P3       night               43  0.7907             7.678              8.032                0.3035                0.3287               19.913                     0.2093       
P4       ALL                 87  0.8046             7.257              8.139                0.3531                0.3902               21.918                     0.1954       
P4       truncated           17  0.3529             6.005              7.009                1.237                 1.237                54.209                     0.6471       
P4       non_truncated       70  0.9143             7.683              8.139                0.3099                0.3582               20.016                     0.0857       
P4       close_range         22  0.5909             7.016              8.990                0.3705                0.3708               51.351                     0.4091       
P4       F1_NO_RESPONSE      24  0.2917             8.256              9.403                0.2106                0.2579               57.575                     0.7083       
P4       F2_CONFIDENT_WRONG  35  1.000              10.125             10.685               0.6510                0.8209               40.920                     0.0000       
P4       F5_safe             23  1.000              2.304              2.941                0.1933                0.2006               10.301                     0.0000       
P4       outside             44  0.8182             6.865              7.282                0.4204                0.4455               21.728                     0.1818       
P4       night               43  0.7907             8.008              8.437                0.3139                0.3577               21.918                     0.2093       
P2_f025  ALL                 87  0.8046             7.129              7.508                0.3323                0.4009               22.072                     0.1954       
P2_f025  truncated           17  0.3529             6.241              7.504                1.237                 1.245                53.713                     0.6471       
P2_f025  non_truncated       70  0.9143             7.129              7.508                0.3116                0.3637               19.485                     0.0857       
P2_f025  close_range         22  0.5909             7.636              10.044               0.3334                0.3338               48.529                     0.4091       
P2_f025  F1_NO_RESPONSE      24  0.2917             9.200              10.179               0.2286                0.2611               57.360                     0.7083       
P2_f025  F2_CONFIDENT_WRONG  35  1.000              9.221              10.136               0.6437                0.8185               40.810                     0.0000       
P2_f025  F5_safe             23  1.000              2.598              3.003                0.1726                0.1749               9.878                      0.0000       
P2_f025  outside             44  0.8182             6.475              6.954                0.4266                0.4471               23.227                     0.1818       
P2_f025  night               43  0.7907             7.695              7.977                0.3116                0.3311               20.179                     0.2093       
P2_f100  ALL                 87  0.8046             6.542              6.940                0.3658                0.3986               21.580                     0.1954       
P2_f100  truncated           17  0.3529             6.097              7.326                1.226                 1.233                53.509                     0.6471       
P2_f100  non_truncated       70  0.9143             6.542              6.940                0.3262                0.3824               20.145                     0.0857       
P2_f100  close_range         22  0.5909             7.668              9.704                0.3747                0.3748               46.438                     0.4091       
P2_f100  F1_NO_RESPONSE      24  0.2917             9.050              11.464               0.1858                0.2601               56.794                     0.7083       
P2_f100  F2_CONFIDENT_WRONG  35  1.000              8.920              9.608                0.6745                0.8085               39.998                     0.0000       
P2_f100  F5_safe             23  1.000              1.932              2.267                0.2753                0.2774               10.363                     0.0000       
P2_f100  outside             44  0.8182             6.468              6.945                0.4481                0.4797               21.220                     0.1818       
P2_f100  night               43  0.7907             6.542              6.940                0.3382                0.3628               21.715                     0.2093       
P3_f025  ALL                 87  0.8046             7.213              7.680                0.3376                0.3835               21.512                     0.1954       
P3_f025  truncated           17  0.3529             6.302              7.647                1.240                 1.249                53.827                     0.6471       
P3_f025  non_truncated       70  0.9143             7.213              7.680                0.2957                0.3507               19.925                     0.0857       
P3_f025  close_range         22  0.5909             7.712              10.554               0.3594                0.3599               51.060                     0.4091       
P3_f025  F1_NO_RESPONSE      24  0.2917             9.050              10.584               0.1869                0.2609               57.515                     0.7083       
P3_f025  F2_CONFIDENT_WRONG  35  1.000              9.256              10.303               0.6452                0.8195               40.944                     0.0000       
P3_f025  F5_safe             23  1.000              2.186              2.814                0.1940                0.1959               9.853                      0.0000       
P3_f025  outside             44  0.8182             6.533              7.119                0.3911                0.4319               22.059                     0.1818       
P3_f025  night               43  0.7907             7.827              8.088                0.3118                0.3507               21.063                     0.2093       
P3_f100  ALL                 87  0.8046             6.895              7.322                0.3425                0.3779               22.075                     0.1954       
P3_f100  truncated           17  0.3529             6.092              7.575                1.240                 1.249                53.669                     0.6471       
P3_f100  non_truncated       70  0.9143             6.895              7.322                0.3384                0.3527               20.203                     0.0857       
P3_f100  close_range         22  0.5909             7.593              10.148               0.3283                0.3283               43.553                     0.4091       
P3_f100  F1_NO_RESPONSE      24  0.2917             9.316              10.188               0.1900                0.2615               57.776                     0.7083       
P3_f100  F2_CONFIDENT_WRONG  35  1.000              8.801              8.860                0.6725                0.8164               40.178                     0.0000       
P3_f100  F5_safe             23  1.000              1.851              2.578                0.2127                0.2143               10.213                     0.0000       
P3_f100  outside             44  0.8182             6.558              7.207                0.4041                0.4287               22.337                     0.1818       
```

## 4.P2 Oracle gate — **FAIL**

```
subset              n   n_common_success  yaw_reduction_pct  corner_reduction_pct  pose_success_gain_pp  aggregate_pass  paired_yaw_improved_fraction  paired_corner_improved_fraction  paired_corroborated  subset_pass
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
truncated           17  6                 6.722              0.6947                0.0000                False           0.5000                        0.5000                           False                False      
F1_NO_RESPONSE      24  7                 -4.059             5.545                 0.0000                False           0.4286                        0.7143                           True                 False      
close_range         22  13                1.179              4.806                 0.0000                False           0.3077                        0.6154                           False                False      
F2_CONFIDENT_WRONG  35  35                2.492              1.672                 0.0000                False           0.4571                        0.4286                           False                False      
ALL                 87  70                -8.700             -23.509               0.0000                False           0.4857                        0.4857                           False                False      
```

[주의] `*_reduction_pct` 는 subset **집계 median** 비교이고, `paired_*_improved_fraction` 은 **같은 frame** 이 개선된 비율이다.  집계 median 은 frame 순위가 바뀌기만 해도 움직이므로 paired 가 진실이다.


guards:
```
guard                         value 
────────────────────────────────────
F5_safe_pose_success_drop_pp  0.0000
clean_yaw_worsening_pct       8.700 
nan_inf_count                 0.0000
fallback_increase_pp          0.0000
```

## 4.P3 Oracle gate — **FAIL**

```
subset              n   n_common_success  yaw_reduction_pct  corner_reduction_pct  pose_success_gain_pp  aggregate_pass  paired_yaw_improved_fraction  paired_corner_improved_fraction  paired_corroborated  subset_pass
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
truncated           17  6                 6.488              0.6181                0.0000                False           0.6667                        0.5000                           True                 False      
F1_NO_RESPONSE      24  7                 -1.406             5.497                 0.0000                False           0.4286                        0.5714                           False                False      
close_range         22  13                0.8362             22.227                0.0000                True            0.5385                        0.6154                           False                False      
F2_CONFIDENT_WRONG  35  35                -3.464             1.762                 0.0000                False           0.5429                        0.4571                           False                False      
ALL                 87  70                -10.438            -13.700               0.0000                False           0.5571                        0.4571                           False                False      
```

[주의] `*_reduction_pct` 는 subset **집계 median** 비교이고, `paired_*_improved_fraction` 은 **같은 frame** 이 개선된 비율이다.  집계 median 은 frame 순위가 바뀌기만 해도 움직이므로 paired 가 진실이다.


guards:
```
guard                         value 
────────────────────────────────────
F5_safe_pose_success_drop_pp  0.0000
clean_yaw_worsening_pct       10.438
nan_inf_count                 0.0000
fallback_increase_pp          0.0000
```

## 4.P4 Oracle gate — **FAIL**

```
subset              n   n_common_success  yaw_reduction_pct  corner_reduction_pct  pose_success_gain_pp  aggregate_pass  paired_yaw_improved_fraction  paired_corner_improved_fraction  paired_corroborated  subset_pass
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
truncated           17  6                 9.238              1.362                 0.0000                False           0.6667                        0.5000                           True                 False      
F1_NO_RESPONSE      24  7                 9.293              6.648                 0.0000                False           0.5714                        0.7143                           True                 False      
close_range         22  13                9.394              10.048                0.0000                True            0.5385                        0.5385                           False                False      
F2_CONFIDENT_WRONG  35  35                -10.576            1.030                 0.0000                False           0.4286                        0.4286                           False                False      
ALL                 87  70                -7.758             -17.339               0.0000                False           0.4571                        0.4857                           False                False      
```

[주의] `*_reduction_pct` 는 subset **집계 median** 비교이고, `paired_*_improved_fraction` 은 **같은 frame** 이 개선된 비율이다.  집계 median 은 frame 순위가 바뀌기만 해도 움직이므로 paired 가 진실이다.


guards:
```
guard                         value 
────────────────────────────────────
F5_safe_pose_success_drop_pp  0.0000
clean_yaw_worsening_pct       14.092
nan_inf_count                 0.0000
fallback_increase_pp          0.0000
```

## 5. Paired delta + session-cluster bootstrap CI

```
comparison  subset              metric          reference_median  candidate_median  percent_delta  paired_mean_delta  ci_low   ci_high  n_common_success
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
P0->P1      ALL                 yaw_mod180_deg  6.025             6.734             11.767         0.5166             0.1091   0.9636   70              
P0->P1      ALL                 corner_sym_m    0.4516            0.3326            -26.361        -0.0435            -0.0792  0.0151   70              
P0->P1      ALL                 pose_success    0.8046            0.8046            -              0.0000             -        -        87              
P0->P1      truncated           yaw_mod180_deg  6.574             6.616             0.6426         -2.240             -5.341   -0.1664  6               
P0->P1      truncated           corner_sym_m    1.220             1.255             2.792          0.1009             0.0197   0.2142   6               
P0->P1      truncated           pose_success    0.3529            0.3529            -              0.0000             -        -        17              
P0->P1      close_range         yaw_mod180_deg  7.671             7.744             0.9465         -1.568             -3.217   0.4046   13              
P0->P1      close_range         corner_sym_m    0.2611            0.4122            57.884         0.0148             -0.0115  0.0401   13              
P0->P1      close_range         pose_success    0.5909            0.5909            -              0.0000             -        -        22              
P0->P1      F1_NO_RESPONSE      yaw_mod180_deg  7.697             9.101             18.240         -0.3962            -2.144   2.553    7               
P0->P1      F1_NO_RESPONSE      corner_sym_m    0.2611            0.2762            5.786          0.0548             0.0045   0.1337   7               
P0->P1      F1_NO_RESPONSE      pose_success    0.2917            0.2917            -              0.0000             -        -        24              
P0->P1      F2_CONFIDENT_WRONG  yaw_mod180_deg  9.945             9.156             -7.933         0.9253             0.5816   1.291    35              
P0->P1      F2_CONFIDENT_WRONG  corner_sym_m    0.8365            0.8295            -0.8361        -0.0043            -0.0364  0.0446   35              
P0->P1      F2_CONFIDENT_WRONG  pose_success    1.000             1.000             -              0.0000             -        -        35              
P1->P2      ALL                 yaw_mod180_deg  6.734             7.320             8.700          -0.2635            -0.7637  0.1105   70              
P1->P2      ALL                 corner_sym_m    0.3326            0.4108            23.509         0.0089             -0.0131  0.0240   70              
P1->P2      ALL                 pose_success    0.8046            0.8046            -              0.0000             -        -        87              
P1->P2      truncated           yaw_mod180_deg  6.616             6.171             -6.722         0.4321             -0.3468  1.693    6               
P1->P2      truncated           corner_sym_m    1.255             1.246             -0.6947        -0.0046            -0.0212  0.0073   6               
P1->P2      truncated           pose_success    0.3529            0.3529            -              0.0000             -        -        17              
P1->P2      close_range         yaw_mod180_deg  7.744             7.652             -1.179         0.1833             -0.6384  0.5772   13              
P1->P2      close_range         corner_sym_m    0.4122            0.3924            -4.806         -0.0022            -0.0041  0.0021   13              
P1->P2      close_range         pose_success    0.5909            0.5909            -              0.0000             -        -        22              
P1->P2      F1_NO_RESPONSE      yaw_mod180_deg  9.101             9.471             4.059          0.1634             0.0324   0.3079   7               
P1->P2      F1_NO_RESPONSE      corner_sym_m    0.2762            0.2609            -5.545         -0.0212            -0.0689  0.0005   7               
P1->P2      F1_NO_RESPONSE      pose_success    0.2917            0.2917            -              0.0000             -        -        24              
P1->P2      F2_CONFIDENT_WRONG  yaw_mod180_deg  9.156             8.928             -2.492         -0.3892            -0.9830  0.4218   35              
P1->P2      F2_CONFIDENT_WRONG  corner_sym_m    0.8295            0.8156            -1.672         0.0012             -0.0046  0.0095   35              
P1->P2      F2_CONFIDENT_WRONG  pose_success    1.000             1.000             -              0.0000             -        -        35              
P1->P3      ALL                 yaw_mod180_deg  6.734             7.437             10.438         -0.2054            -0.5663  0.1710   70              
P1->P3      ALL                 corner_sym_m    0.3326            0.3781            13.700         0.0037             -0.0098  0.0134   70              
P1->P3      ALL                 pose_success    0.8046            0.8046            -              0.0000             -        -        87              
P1->P3      truncated           yaw_mod180_deg  6.616             6.187             -6.488         0.0060             -0.3987  0.5605   6               
P1->P3      truncated           corner_sym_m    1.255             1.247             -0.6181        -0.0202            -0.0592  0.0015   6               
P1->P3      truncated           pose_success    0.3529            0.3529            -              0.0000             -        -        17              
P1->P3      close_range         yaw_mod180_deg  7.744             7.679             -0.8362        -0.0796            -0.7352  0.1068   13              
P1->P3      close_range         corner_sym_m    0.4122            0.3206            -22.227        -0.0089            -0.0227  0.0108   13              
P1->P3      close_range         pose_success    0.5909            0.5909            -              0.0000             -        -        22              
P1->P3      F1_NO_RESPONSE      yaw_mod180_deg  9.101             9.229             1.406          0.2507             -0.0001  0.6521   7               
P1->P3      F1_NO_RESPONSE      corner_sym_m    0.2762            0.2610            -5.497         -0.0123            -0.0469  0.0134   7               
P1->P3      F1_NO_RESPONSE      pose_success    0.2917            0.2917            -              0.0000             -        -        24              
P1->P3      F2_CONFIDENT_WRONG  yaw_mod180_deg  9.156             9.473             3.464          -0.1563            -0.5261  0.5329   35              
P1->P3      F2_CONFIDENT_WRONG  corner_sym_m    0.8295            0.8149            -1.762         -0.0008            -0.0145  0.0091   35              
P1->P3      F2_CONFIDENT_WRONG  pose_success    1.000             1.000             -              0.0000             -        -        35              
P1->P4      ALL                 yaw_mod180_deg  6.734             7.257             7.758          -0.0522            -0.3828  0.2970   70              
P1->P4      ALL                 corner_sym_m    0.3326            0.3902            17.339         0.0083             -0.0107  0.0206   70              
P1->P4      ALL                 pose_success    0.8046            0.8046            -              0.0000             -        -        87              
P1->P4      truncated           yaw_mod180_deg  6.616             6.005             -9.238         -0.2106            -0.6633  0.3839   6               
P1->P4      truncated           corner_sym_m    1.255             1.237             -1.362         -0.0339            -0.1022  0.0149   6               
P1->P4      truncated           pose_success    0.3529            0.3529            -              0.0000             -        -        17              
P1->P4      close_range         yaw_mod180_deg  7.744             7.016             -9.394         -0.3213            -1.044   -0.1418  13              
P1->P4      close_range         corner_sym_m    0.4122            0.3708            -10.048        0.0009             -0.0054  0.0086   13              
P1->P4      close_range         pose_success    0.5909            0.5909            -              0.0000             -        -        22              
P1->P4      F1_NO_RESPONSE      yaw_mod180_deg  9.101             8.256             -9.293         -0.0783            -0.2223  0.2538   7               
P1->P4      F1_NO_RESPONSE      corner_sym_m    0.2762            0.2579            -6.648         -0.0065            -0.0244  0.0009   7               
P1->P4      F1_NO_RESPONSE      pose_success    0.2917            0.2917            -              0.0000             -        -        24              
P1->P4      F2_CONFIDENT_WRONG  yaw_mod180_deg  9.156             10.125            10.576         -0.0202            -0.3058  0.5812   35              
P1->P4      F2_CONFIDENT_WRONG  corner_sym_m    0.8295            0.8209            -1.030         -0.0012            -0.0117  0.0076   35              
P1->P4      F2_CONFIDENT_WRONG  pose_success    1.000             1.000             -              0.0000             -        -        35              
P3->P4      ALL                 yaw_mod180_deg  7.437             7.257             -2.426         0.1531             -0.0503  0.3838   70              
P3->P4      ALL                 corner_sym_m    0.3781            0.3902            3.201          0.0045             -0.0046  0.0122   70              
P3->P4      ALL                 pose_success    0.8046            0.8046            -              0.0000             -        -        87              
P3->P4      truncated           yaw_mod180_deg  6.187             6.005             -2.941         -0.2166            -0.7925  0.2563   6               
P3->P4      truncated           corner_sym_m    1.247             1.237             -0.7490        -0.0137            -0.0697  0.0328   6               
P3->P4      truncated           pose_success    0.3529            0.3529            -              0.0000             -        -        17              
P3->P4      close_range         yaw_mod180_deg  7.679             7.016             -8.630         -0.2417            -0.4063  -0.1685  13              
P3->P4      close_range         corner_sym_m    0.3206            0.3708            15.661         0.0098             -0.0062  0.0192   13              
P3->P4      close_range         pose_success    0.5909            0.5909            -              0.0000             -        -        22              
P3->P4      F1_NO_RESPONSE      yaw_mod180_deg  9.229             8.256             -10.550        -0.3290            -0.3983  -0.1943  7               
P3->P4      F1_NO_RESPONSE      corner_sym_m    0.2610            0.2579            -1.219         0.0058             -0.0125  0.0225   7               
P3->P4      F1_NO_RESPONSE      pose_success    0.2917            0.2917            -              0.0000             -        -        24              
P3->P4      F2_CONFIDENT_WRONG  yaw_mod180_deg  9.473             10.125            6.874          0.1361             0.0258   0.2536   35              
P3->P4      F2_CONFIDENT_WRONG  corner_sym_m    0.8149            0.8209            0.7455         -0.0004            -0.0041  0.0036   35              
P3->P4      F2_CONFIDENT_WRONG  pose_success    1.000             1.000             -              0.0000             -        -        35              
```

## 6. 그림

- `oracle_line_pose_recovery.png`
- `close_range_pose_recovery.png`
- `yaw_mod180_distribution.png`
- `generic_vs_semantic_lines.png`
- `oracle_line_examples.png`

## 7. 한계

- [확인] P2/P3 는 GT pose 로 만든 oracle 이므로 달성 가능한 **상한**이다.
- [확인] strict N87 소표본이고 subset(F1 24 / F2 35 / truncated 17)은 더 작다.
- [확인] yaw median 은 common-success frame 위에서만 비교했다.
- [확인] Canny 는 sweep 으로 고르지 않고 중간 설정 하나를 고정했다 (사용: (100, 200)).

