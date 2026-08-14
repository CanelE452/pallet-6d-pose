# challenge/data 트리 스냅샷 — 재편 전 (Phase 0)

2026-08-14 기록. `data/` 는 `.gitignore` 라 git 이 보호하지 않는다.
폴더 재편(01_real/02_synthetic/03_derived/04_results) 이전 상태이며,
이동이 잘못됐을 때 되돌릴 기준이다.

## 최상위 항목 (이동 대상 단위)
```
name                                           size    files     json
───────────────────────────────────────────────────────────────────
ab_crop_eval                                   6.5M       17        0
ab_crop_eval_cropaug                           3.8M       10        0
ab_crop_eval_cropaug_v2                        4.1M       10        0
capture0403noapril_manual_gt                   5.0M       36       18
capturenight01_manual_gt                       4.0K        0        0
capturenight03_manual_gt                       4.0K        0        0
capturenight04_manual_gt                       2.5M       10        5
capturenight05_manual_gt                       7.4M       24       12
capturenight06_manual_gt                       8.7M       30       15
capturenight07_manual_gt                       9.7M       32       16
capturenight08_manual_gt                        11M       34       17
capturenight09_manual_gt                        15M       50       25
capturepallet01_manual_gt                      4.0K        0        0
capturepallet01_pseudo_gt                      8.0K        0        0
capturepallet02_manual_gt                      2.3M       10        5
capturepallet02_pseudo_gt                       12M       50       24
capturepallet03_manual_gt                      3.6M       16        8
capturepallet03_pseudo_gt                      8.0K        0        0
capturepallet04_manual_gt                      2.7M       12        6
capturepallet04_pseudo_gt                      8.0K        0        0
capturepallet05_manual_gt                      2.3M       10        5
capturepallet05_pseudo_gt                      8.0K        0        0
capturepallet06_pseudo_gt                      8.0K        0        0
capturepallet07_augmented                      114M      550      275
capturepallet07_manual_gt                       12M       79       27
capturepallet08_manual_gt                      9.4M       40       20
capturepallet08_pseudo_gt                      1.4M        6        3
capturepallet09_manual_gt                       17M       73       36
capturepallet09_pseudo_gt                      5.4M       22       11
capturepallet10_pseudo_gt                      8.0K        0        0
capturepalletcad_manual_gt                      22M       88       44
challenge_ft_forklift_eval                     6.1M       33        1
cropaug_truncation_eval                         21M       77        1
_eval_real_gt_merged                           1.8M        3      222
forklift_20260528_manual_gt                    4.0K        0        0
forklift_cropaug_infer_frames                  2.6M        6        0
forklift_cropaug_v2_frames                     5.2M       12        0
forklift_cropaug_v2_NEW_infer_frames           2.6M        6        0
_live_captures                                 4.0K        0        0
_night_eval_manual_gt                          216K       43       43
_outside_eval_manual_gt                        320K       54       54
pallet11_gt                                    107M      493      243
_sanity                                        792K        3        0
_train_capturepallet07                          11M       51       26
training                                        37G   177392    63996
_train_manual_pseudo                            29M      127       64
_train_pallet07_aug                            114M      551      276
truncation_crops                               183M      976      485
truncation_crops_dope                           16G    92268    46119
truncation_crops_palletobj                     213M     1502      748
truncation_crops_synth                         108M      794      394
_verify_truncation_aug                         2.4M        8        0
wood_pallet_20260618_183705_manual_gt          204K       25       25
wood_pallet_20260618_184309_manual_gt          164K       20       20
yolo_pose                                       13G    57938        0
yolo_pose_cropaug_padded                       712M     2100        0
yolo_pose_cropaug_v2_padded                    1.1G     3596        0
yolo_pose_manual                               112M      440        0
yolo_pose_manual_padded                        218M      440        0
yolo_pose_padded                                22G    57938        0
───────────────────────────────────────────────────────────────────
forklift_cropaug_infer.mp4                      19M   (file)
forklift_cropaug_v2_infer.mp4                   20M   (file)
forklift_cropaug_v2_NEW_infer.mp4               19M   (file)
forklift_ft_s2_infer.mp4                        15M   (file)
forklift_ft_s2_PAD160_infer.mp4                 21M   (file)
forklift_ft_s2_PAD_infer.mp4                    20M   (file)
forklift_otftrunc_infer.mp4                     15M   (file)
forklift_otftrunc_PAD160_infer.mp4              21M   (file)
forklift_otftrunc_PAD_infer.mp4                 21M   (file)
holdout_stems.txt                              4.0K   (file)
yolo_pose_padded_convert.log                   4.0K   (file)
```

## 코드 참조 집계 (재편 시 깨질 수 있는 지점)
```
challenge/data 를 참조하는 파일   92개
참조 라인 총수                    272
그중 테스트                       test_eval_set_canonical.py
                                  test_decoder_reconciliation.py
                                  test_threshold_audit.py

참조 상위 (이것만 풀면 214/272 = 79%)
    129 challenge/data/training
     23 challenge/data/capturepalletcad_manual_gt
     23 challenge/data/capturepallet07_manual_gt
     21 challenge/data/capture0403noapril_manual_gt
     18 challenge/data/_outside_eval_manual_gt
     16 challenge/data/capturepallet03_manual_gt
     12 challenge/data/yolo_pose
     11 challenge/data/truncation_crops_dope
     10 challenge/data/_night_eval_manual_gt
     10 challenge/data/capturepallet09_manual_gt
     10 challenge/data/capturenight0
      9 challenge/data/capturepallet0
```

## 되돌리는 법

Phase 1 은 `mv` + 원위치 `ln -s` 다. 잘못되면 링크를 지우고 원래 이름으로 되돌린다.

```bash
cd challenge/data
find . -maxdepth 1 -type l -delete          # 심볼릭 링크만 제거
mv 01_real/eval_canonical/* .               # 각 구획에서 원위치로
mv 01_real/manual_gt/* . ; mv 01_real/pseudo_gt/* . ; ...
rmdir -p 01_real/* 02_synthetic 03_derived 04_results 2>/dev/null
pytest challenge/tests/test_eval_set_canonical.py   # 정본 56장 복구 확인
```

위 표의 files/json 개수가 복구 후에도 일치해야 한다.
