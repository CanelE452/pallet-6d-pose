# challenge/data 지도

2026-08-14 재편. 분류 기준은 **어떻게 생긴 데이터인가** 다 — 찍은 것(01) /
렌더한 것(02) / 그 둘을 가공한 학습 입력(03) / 모델이 뱉은 것(04).

경로를 코드에 직접 쓰지 말고 `challenge/data_paths.py` 를 쓴다.

```python
from challenge.data_paths import EVAL_CANONICAL, get
get("eval_cad")          # challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt
```
```bash
python challenge/data_paths.py --get eval_cad [--abs]
python challenge/data_paths.py --list
python challenge/data_paths.py --check      # 전 경로 존재 검사
```

## 옛 경로는 더 이상 없다

2026-08-14 재편 때 호환용 symlink 71개를 잠시 뒀다가, 참조를 모두 전환한 뒤
제거했다. `challenge/data/capturepallet07_manual_gt` 같은 옛 경로는 이제 존재하지
않으므로 아래 구조의 실제 경로를 쓴다.

## 01_real — 실제로 찍은 것

전부 `png + json` 쌍이다. 즉 촬영본과 어노테이션이 한 폴더에 같이 있다.
**라벨 없는 순수 촬영본은 여기 없고** `data/pallet/real_unlabeled_ralph*` 에 있다.

```
eval_canonical/    ★정본 평가셋 56장. 다른 셋으로 대체 금지
  _outside_eval_manual_gt          22
  capture0403noapril_manual_gt     12
  capturepalletcad_manual_gt       22
    → 규칙 objects[0].split == "eval" / 상세 _docs/EVAL_SET_CANONICAL.md
    → 강제 challenge/tests/test_eval_set_canonical.py

manual_gt/         손 어노테이션 GT
  capturepallet01~09_manual_gt · capturenight01~09_manual_gt
  _night_eval_manual_gt · pallet11_gt (243장)
  wood_pallet_20260618_{183705,184309}_manual_gt
  forklift_20260528_manual_gt
    → 01/03(night) · 01(pallet) · forklift_20260528 은 현재 비어 있다(보존)

pseudo_gt/         이미지는 촬영본, 라벨만 모델 산출
  capturepallet01~10_pseudo_gt

augmented/         촬영본 증강
  capturepallet07_augmented        275장

_live_captures/    비어 있음(보존)
```

## 02_synthetic — 렌더한 것 (37G)

photogrammetry 로 스캔한 **본인 파렛트**(`data/pallet/scan_cleanup/pallet_full.obj`)
기반. **challenge 전용이며 논문 트랙에서 쓰지 않는다.**

```
training/
  v1   9,997 프레임   part_000~003, 접두어 train_palletobj_v1
  v2   9,994
  v3  20,002          mask 는 JSON mask_rle 사용, batch_XXX/mask/*.png 금지
  addon_v1 · addon_v1_train · addon_v1_val · truncation_addon_v1
```

## 03_derived — 위 둘을 가공한 학습 입력 (53G)

```
truncation_crops · _dope(16G) · _palletobj · _synth
yolo_pose(13G) · _padded(22G) · _manual · _manual_padded · _cropaug_padded · _cropaug_v2_padded
_train_manual_pseudo · _train_pallet07_aug · _train_capturepallet07   (각 _manifest.json + train/val)
_eval_real_gt_merged      219장 병합본
holdout_stems.txt
```

## 04_results — 모델 산출물 (데이터 아님)

```
ab_crop_eval · _cropaug · _cropaug_v2
cropaug_truncation_eval · challenge_ft_forklift_eval
forklift_cropaug_infer_frames · forklift_cropaug_v2_frames · forklift_cropaug_v2_NEW_infer_frames
_sanity · _verify_truncation_aug
forklift_*.mp4  9개 (171M)
yolo_pose_padded_convert.log
```

## 되돌리기

재편 이전 트리는 `_docs/audits/2026-08-14_challenge_data_tree_snapshot.md` 에
항목별 파일수와 함께 기록돼 있다. `data/` 는 `.gitignore` 라 git 복구가 없으므로
그 표가 유일한 기준이다.
