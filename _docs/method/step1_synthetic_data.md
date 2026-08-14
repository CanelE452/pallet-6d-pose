# Step 1 — 합성 데이터 + DOPE 학습 (paper_base)

> camera-facing 0123. 폐기 v8 버전은 `archive/step1_synthetic_data.md`.

## 목표

인터넷 무료 3D 팔레트 모델 기반 합성 데이터로 DOPE 를 학습하되,
**비율 강건성**(처음 본 파렛트 대응)과 **truncation 강건성**(잘린 이미지)을
확보한 논문용 base 모델 `paper_base` 를 만든다.

## 학습 데이터 (경로 확정)

```
합성 base    data/pallet/training_data/mixed_v8_train      9,000장  camera-facing v4
truncation   challenge/data/03_derived/truncation_crops_dope/pretrain 8,831장  crop+padding
squash       [미생성] 비율 강건 증강
제외         challenge/data/02_synthetic/training/v1·v2 (내 파렛트)
```

## 1) 비율 강건성 — squash 증강 [TODO]

처음 본 파렛트는 aspect ratio 가 제각각인데 우리는 특정 비율 합성만 학습 →
일반화 약함. 해결: 학습 이미지를 여러 비율로 **squash(찌부)/stretch** 증강.

- ⚠️ 이미지 변형 시 **JSON 꼭짓점(projected_cuboid)도 동일 변형 동기** 필수.
- 좌표 변환이라 `3d-expert` 위임으로 증강 스크립트 작성 + 검증.
- 변형 범위/분포(어느 비율까지), 원본:증강 비중은 실험으로 결정.

## 2) truncation 강건성 — crop + padding

9 keypoint 다 보이는 이미지를 crop 해 일부 코너가 화면 밖으로 나간 상황 합성 →
DOPE 로더가 padding 영역 확보 후 **화면 밖 코너의 belief map 을 padding 영역에
그려 supervise** (8/8 supervised 검증). 잘려도 9점 회귀 → PnP 6점 안정.

- 기존 자산 `truncation_crops_dope/` 재활용 (dope_cropaug 방식).
- 효과(과제 트랙 검증): real truncation PnP 23→99%, det 13→94%.
- 측면(L/R) 잘림 위주 (top 잘림은 비현실적·degenerate) — memory `truncation-side-cut-bias`.

## 3) 학습 설정

- DOPE VGG-19, 9 belief + 16 affinity, sigma=4.0 (sigma<1 gradient vanishing).
- finetune 은 누적 epoch (memory `dope-finetune-cumulative-epoch`).
- 중간 산출물 `dope_cropaug_pretrain`(squash 없음) → squash 추가 후 재학습 = paper_base.

## 체크리스트

- [ ] squash 증강 데이터 생성 (JSON 동기) — 3d-expert
- [ ] camera-facing v4 변환 정합성 최종 검증 — 3d-expert
- [ ] paper_base 학습 (합성 + squash + truncation)
