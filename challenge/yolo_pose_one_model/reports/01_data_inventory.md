# 01 — Data inventory

생성: `python challenge/yolo_pose_one_model/scripts/discover_and_audit.py`  
registry: `challenge/yolo_pose_one_model/manifests/all_samples.csv` (60612 rows)

모든 경로는 repo-relative. G 는 상수, T 는 `challenge/data_paths.py` 의 `synth.v1/v2`,
R 은 `01_real/manual_gt/*` + `01_real/eval_canonical/*` 글롭으로 찾았다.

## 도메인별 개수
```
domain            samples   image 없음
generic_synth       40000          0
target_synth        19991         23
real                  621         62
```

## 해상도 (헤더 판독, 도메인당 표본 400)
```
generic_synth: 640x480, 960x540, 720x480, 560x560
target_synth: 640x480
real: 640x480, 1280x720
```

## annotation_quality
```
generic_synth: {'exact_synthetic': 40000}
target_synth: {'exact_synthetic': 19985, 'negative': 6}
real: {'manual_direct': 346, 'manual_inferred': 32, 'legacy_mixed': 243}
```

## real 세션 상세
```
session_id                                  n  img  kp9               dims(w,h,d)  split
_night_eval_manual_gt                      43   10   36          (1.3, 0.11, 1.1)  {'-': 33, 'train': 10}
_outside_eval_manual_gt                    54   25   52          (1.3, 0.11, 1.1)  {'-': 29, 'eval': 22, 'train': 3}
capture0403noapril_manual_gt               18   18   18          (1.3, 0.11, 1.1)  {'-': 6, 'eval': 12}
capturenight04_manual_gt                    5    5    5          (1.1, 0.11, 1.3)  {'-': 5}
capturenight05_manual_gt                   12   12   11          (1.3, 0.11, 1.1)  {'-': 12}
capturenight06_manual_gt                   15   15   13          (1.1, 0.11, 1.3)  {'-': 15}
capturenight07_manual_gt                   16   16   12          (1.3, 0.11, 1.1)  {'-': 16}
capturenight08_manual_gt                   12   12   12          (1.3, 0.11, 1.1)  {'eval': 12}
capturenight09_manual_gt                   16   16   16          (1.3, 0.11, 1.1)  {'eval': 16}
capturepallet02_manual_gt                   5    5    5          (1.3, 0.11, 1.1)  {'-': 5}
capturepallet03_manual_gt                   8    8    8          (1.3, 0.11, 1.1)  {'-': 8}
capturepallet04_manual_gt                   6    6    4          (1.3, 0.11, 1.1)  {'-': 6}
capturepallet05_manual_gt                   5    5    5          (1.3, 0.11, 1.1)  {'-': 5}
capturepallet07_manual_gt                  27   27   26          (1.1, 0.11, 1.3)  {'eval': 27}
capturepallet08_manual_gt                  18   18   17          (1.1, 0.11, 1.3)  {'-': 18}
capturepallet09_manual_gt                  33   33   32          (1.1, 0.11, 1.3)  {'eval': 33}
capturepalletcad_manual_gt                 40   40   29          (1.1, 0.11, 1.3)  {'eval': 18, 'train': 12, '-': 10}
pallet11_gt                               243  243    0          (1.1, 0.15, 1.1)  {'-': 243}
wood_pallet_20260618_183705_manual_gt      25   25   25         (0.8, 0.14, 0.59)  {'-': 25}
wood_pallet_20260618_184309_manual_gt      20   20   20         (0.8, 0.14, 0.59)  {'-': 20}
```

과제 팔레트 판별 기준: GT height == 0.11 m (0.15 = pallet11 정사각, 0.14 = wood 소형은 다른 물체).
