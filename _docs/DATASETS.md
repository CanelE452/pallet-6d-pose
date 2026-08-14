# 데이터셋 목록

이미지는 `.gitignore` 로 저장소에 올리지 않는다. 이 표는 실제 디스크를
실측한 것이고, 폴더마다 `DATASET.md` 에 더 자세한 카드가 있다.

총 96개 데이터셋 · 이미지 659,127장 · 222.9G

재생성: `python scripts/data_prep/validate/gen_dataset_cards.py`

## challenge/data/01_real

24개 · 이미지 904장 · 388.4M

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
capturepallet07_augmented                         275      275   112.2M
pallet11_gt                                       250      243   105.0M
capturepalletcad_manual_gt                         44       44    21.2M
capturepallet09_manual_gt                          36       36    15.9M
capturepallet07_manual_gt                          27       27    11.6M
capturepallet02_pseudo_gt                          26       24    11.5M
_outside_eval_manual_gt                            25       54    11.3M
capturenight09_manual_gt                           25       25    14.7M
wood_pallet_20260618_183705_manual_gt              25       25     3.7M
capturepallet08_manual_gt                          20       20     9.2M
wood_pallet_20260618_184309_manual_gt              20       20     5.8M
capture0403noapril_manual_gt                       18       18     4.9M
capturenight08_manual_gt                           17       17    10.2M
capturenight07_manual_gt                           16       16     9.6M
capturenight06_manual_gt                           15       15     8.6M
capturenight05_manual_gt                           12       12     7.3M
capturepallet09_pseudo_gt                          11       11     5.3M
_night_eval_manual_gt                              10       43     5.9M
capturepallet03_manual_gt                           8        8     3.5M
capturepallet04_manual_gt                           6        6     2.6M
capturenight04_manual_gt                            5        5     2.5M
capturepallet02_manual_gt                           5        5     2.2M
capturepallet05_manual_gt                           5        5     2.2M
capturepallet08_pseudo_gt                           3        3     1.4M
```

## challenge/data/02_synthetic

5개 · 이미지 105,992장 · 36.1G

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
v3                                             30,000   20,002    10.0G
v1                                             19,993    9,997     7.8G
v2                                             19,990    9,994     7.8G
truncation_addon_v1                            18,009    6,001     5.2G
addon_v1                                       18,000   12,002     5.4G
```

## challenge/data/03_derived

13개 · 이미지 109,377장 · 52.0G

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
truncation_crops_dope                          46,149   46,119    15.6G
yolo_pose                                      28,968        0    12.6G
yolo_pose_padded                               28,968        0    21.1G
yolo_pose_cropaug_v2_padded                     1,797        0     1.1G
yolo_pose_cropaug_padded                        1,049        0   704.8M
truncation_crops_palletobj                        754      748   209.3M
truncation_crops                                  491      485   179.9M
truncation_crops_synth                            400      394   105.8M
_train_pallet07_aug                               275      276   112.3M
yolo_pose_manual                                  219        0   110.3M
yolo_pose_manual_padded                           219        0   216.4M
_train_manual_pseudo                               63       64    27.9M
_train_capturepallet07                             25       26    10.6M
```

## challenge/data/04_results

10개 · 이미지 173장 · 53.8M

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
cropaug_truncation_eval                            76        1    20.5M
challenge_ft_forklift_eval                         32        1     6.0M
ab_crop_eval                                       15        0     6.4M
forklift_cropaug_v2_frames                         12        0     5.1M
_verify_truncation_aug                              8        0     2.3M
ab_crop_eval_cropaug                                8        0     3.8M
ab_crop_eval_cropaug_v2                             8        0     4.0M
forklift_cropaug_infer_frames                       6        0     2.5M
forklift_cropaug_v2_NEW_infer_frames                6        0     2.5M
_sanity                                             2        0   779.7K
```

## data/pallet/training_data

22개 · 이미지 377,767장 · 113.3G

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
paper_release                                 120,000   40,000    28.2G
v2_prod40k_clean_merged                       120,000   40,000    28.2G
achieve                                        81,255   81,233    37.0G
paper_4pallet_mask_v1                          30,000   10,000     9.2G
mixed_v8_train                                  9,000    9,000     3.6G
v4_split_base                                   4,000    4,000     1.8G
aug_trunc_v2                                    2,979    2,971     1.1G
aug_squash_v2                                   2,220    2,212   916.5M
mixed_v8_train_2k                               2,000    2,000   811.6M
paper_4pallet_mask_v1_2k                        2,000    2,000   878.5M
val                                             1,500    1,500   491.2M
aug_scale_v2                                    1,133    1,125   459.2M
paper_s2_fullpool_r1                              513      513   246.3M
paper_s2_pl_outside                               381      381   171.9M
paper_s2_pl_reproj_flip                           232      232   110.5M
paper_s2_fullpool_r2                              192      192    91.9M
paper_s2_plrf_outside                             191      191    86.9M
paper_s2_pl_night                                 107      107    63.5M
paper_s2_plrf_night                                39       39    23.1M
paper_s2_full7_pl19_r1                             19       21     9.0M
paper_s2_pl_noapril                                 4        4     1.1M
paper_s2_plrf_noapril                               2        2   573.7K
```

## data/pallet/real_unlabeled_ralph

1개 · 이미지 5,808장 · 2.7G

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
real_unlabeled_ralph                            5,808        0     2.7G
```

## data/pallet/real_unlabeled_ralph1500

1개 · 이미지 1,500장 · 713.2M

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
real_unlabeled_ralph1500                        1,500        0   713.2M
```

## data/pallet/real_unlabeled_ralph_cad

1개 · 이미지 500장 · 230.9M

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
real_unlabeled_ralph_cad                          500        0   230.9M
```

## data/pallet/real_unlabeled_ralph_night

1개 · 이미지 500장 · 291.9M

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
real_unlabeled_ralph_night                        500        0   291.9M
```

## data/pallet/real_unlabeled_ralph_noapril

1개 · 이미지 170장 · 46.9M

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
real_unlabeled_ralph_noapril                      170        0    46.9M
```

## data/pallet/real_unlabeled_ralph_outside

1개 · 이미지 500장 · 214.8M

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
real_unlabeled_ralph_outside                      500        0   214.8M
```

## data/pallet/raw_data

16개 · 이미지 55,936장 · 17.0G

```
dataset                                        images     json     size
────────────────────────────────────────────────────────────────────────
outside                                        20,733       32     5.1G
night                                          18,268        0     5.6G
_outside_all                                    2,976        0     1.3G
vdoframes                                       2,362        0     2.3G
real_pool_all                                   2,324        0   271.1M
real_data                                       1,924        0   147.3M
capture0403middle                               1,887    1,449   262.5M
_night_all                                      1,624        0   949.1M
capture02                                       1,452        0   360.7M
wood                                            1,165        0   395.1M
capture03                                         571        0   142.3M
pallet_extract_conf60                             425        0   145.0M
capture0403noapril                                188        0    51.7M
_procedural_textures                               21        0    10.8M
internet_pallet_data                               10        0     1.6M
models_usd                                          6        0    47.6M
```
