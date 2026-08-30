# PURPOSE — DOPE on the frozen final dataset

[소비처] 논문 백본 비교 표. "왜 YOLO26 인가" 에 답하는 유일한 통제 비교 —
        같은 학습 데이터·같은 조건에서 DOPE 와 YOLO26n 을 나란히 놓는다.

[문장] "최종 학습셋(g38_legacy_v1v2_p0_tex20k, 55,980장)으로 DOPE 를 학습하면,
       같은 데이터로 학습한 YOLO26n(CLEANSTART_60EP_SEED42)과 비교 가능한
       백본 대조가 성립한다."

## 왜 이 run 이 필요한가

현재 DOPE 체크포인트는 전부 `mixed_v8_train` 등 **다른 데이터**로 학습됐다.
그대로 표에 넣으면 백본 차이와 데이터 차이가 섞여 "왜 YOLO26 인가" 에 답할 수 없다.
데이터를 최종셋으로 동결했으므로, DOPE 를 같은 데이터로 다시 학습해야 통제가 성립한다.

## 통제 조건

```
학습 데이터   g38_legacy_v1v2_p0_tex20k 와 동일 프레임 집합 (train 55,980 / val 4,020)
              G38 38,002 + P0 8,989 + TEX 8,989
real 감독     없음 (합성만)
fine-tune     없음 (scratch/imagenet init 에서 60ep)
비교 상대     YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42 (같은 데이터, 60ep, seed 42)
```

## 무엇을 바꾸지 않는가

- 새 loss 0 · 새 architecture 0 · 새 keypoint convention 0
- sigma 는 기존 계약 유지 (sigma<1 은 gradient vanishing — `_docs/method/step1_synthetic_data.md` 3.6)
- keypoint convention = camera-facing 0123 (projected_cuboid 8 + centroid = 9)

## 착수 시점에 아는 한계

- **평가셋이 아직 완성되지 않았다.** 이 run 은 학습까지만 하고, pose 지표는 내지 않는다.
  완료 판정은 체크포인트 존재로만 한다.
- DOPE 와 YOLO26n 은 출력 표현이 다르다(belief map vs 직접 회귀).
  "같은 데이터" 는 통제되지만 "같은 학습 예산" 은 아니다 — epoch 수만 맞췄다.
- seed 1개. seed 산포는 재지 않는다.
