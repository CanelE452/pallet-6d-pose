# 01. YOLO26 pose 모델 — truncation 강건 (crop-aug)

## 문제
forklift가 팔레트에 접근하면 팔레트가 화면 가장자리에 **잘려(truncation)** 보임.
기존 모델은 잘린 팔레트 검출/PnP가 약함.

## 핵심 인사이트
```
1. YOLO pose 포맷: keypoint visibility 0(화면밖)/2(보임). v=0 은 loss에서 제외
   → 비패딩(nopad) 학습은 잘린 코너를 "안 가르침".
2. padding(100px reflect): 가장자리 밖 100px 이내 코너가 padded canvas 안으로
   들어와 v=2 학습타깃 → 잘린 기하까지 학습. 그래서 truncation에 강함.
3. crop 증강: full 팔레트 frame을 잘라 truncation 샘플을 합성 증식.
   ★ 실제 카메라는 좌우 패닝 → 측면(L/R) 잘림 위주. top 잘림은 비현실적이고
     degenerate(얇은 띠) 주범 → 제외.
```

## 데이터 구성 (최종 모델 cropaug_v2)
```
train 1755 frame
  ─ 실사진(real) 613 ──────────────────────────────────────
     원본 manual GT (capture*)        177
     forklift 수동 annotation          32
     real crop (capture*+forklift 측면잘림) 404
  ─ 합성(synthetic) 1142 ─────────────────────────────────
     mixed_v8 crop                    394
     palletobj crop (진짜 스캔 팔레트) 748
val 42 = holdout (crop 미포함, 비교 일관성)
※ holdout 42 frame 파생 real crop 81개는 leakage 방지로 제외
```
- crop 방향 가중치: 측면(L/R) ~75%, 하단 ~20%, 상단 ~3% (`gen_truncation_crops.py CUT_WEIGHTS`)
- degenerate 필터: in-image kp ≥5 AND visible bbox 면적 ≥10% AND min(변)≥50px

## 모델 인벤토리
```
모델                              방식          학습데이터              경로
────────────────────────────────────────────────────────────────────────────────────
yolo26n_pose_v1 (base)            합성 pretrain palletobj+mixed_v8(27.5k) challenge/weights/yolo26n_pose_v1/
yolo26n_pose_v1_ft_pad_ho         padding       real manual 177          runs/pose/challenge/weights/...
yolo26n_pose_v1_ft_nopad_ho       비패딩        real manual 177          runs/pose/challenge/weights/...
yolo26n_pose_v1_ft_cropaug        pad+crop      1007(real+mixed_v8 crop) runs/pose/challenge/weights/...
yolo26n_pose_v1_ft_cropaug_v2 ★   pad+crop      1755(+palletobj crop)    runs/pose/challenge/weights/...
yolo26n_pose_v1_ft_pad_ep20       padding       real manual 177 (20ep)   runs/pose/challenge/weights/...
```
★ = 배포 기본 모델 (`pallet_jetson_deploy/models/pallet_pose_cropaug_v2.pt`)

## 학습 설정 (cropaug_v2)
```
base:       yolo26n_pose_v1/weights/best.pt
optimizer:  SGD lr0=1e-4 lrf=0.01
epochs:     120 (best epoch 91), patience 40, batch 16, imgsz 640
aug:        mosaic=1.0 close_mosaic=10 scale=0.5 translate=0.1 fliplr=0.5 hsv erasing=0.4
pad:        100px reflect (convert_to_yolo_pose --pad 100)
env:        pallet-yolo26 (ultralytics 8.4.60), GPU RTX 3080
스크립트:    challenge/yolo_pose/scripts/ft_cropaug_v2.sh
데이터셋:    challenge/data/03_derived/yolo_pose_cropaug_v2_padded/
```

## 평가 — truncation 강건성 (holdout 42, crop level 0/1/2, 모두 pad 추론)
```
lvl  지표            pad_ho(crop X)  cropaug v1   cropaug v2(palletobj+)
──────────────────────────────────────────────────────────────────────
 0   det% / PnP%     92.9 / 83.3     100 / 90.5   100 / 92.9
 0   reproj_in med   8.9             8.9          8.6
 0   5cm5° / ADD     23.8% / 103     16.7% / 130  11.9% / 151
 1   det%/reproj_all 95.2 / 15.5     100 / 13.4   100 / 14.7
 2   det% / PnP%     85.7 / 76.2     100 / 83.3   100 / 83.3   ← 심한 truncation
 2   reproj_in med   9.7             8.2          8.5
```

## 결론
```
✅ truncation 검출·PnP 강건성: crop-aug ≫ crop 없음 (L2 det 85.7→100%, PnP 76→83%)
✅ keypoint 품질(reproj median): crop-aug 동등/우수
⚠️ clean(L0) pose 정밀도(5cm5°/ADD): synthetic 비중↑로 소폭 후퇴 (n=42 노이즈 큼)
   → 원인은 synthetic 비중. palletobj(진짜 팔레트) 추가도 clean 회복은 못함.
실배포: truncation만 보면 v1/v2 동급. 거리 정밀도는 03(depth fusion)이 해결.
```

세부: `_docs/history/2026-06-02.md`, 메모리 `truncation-side-cut-bias`, `cropaug-synthetic-ratio-tradeoff`.
