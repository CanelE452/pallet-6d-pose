# 정정된 real supervision 이 처음 본 촬영의 위치추정을 움직이는가

계열 문서.  실험 폴더 `challenge/yolo_pose_one_model/next_accuracy_v2/`.
사전등록 `data/pallet/results/next_accuracy_v2/METHOD_LOCK.json` (sha256 95fe48bef176b804).

## 1. 제안

**가설.** 학습 라벨을 정본 필드(`keypoint_annotations`)로 바꾸면, 그 자체로
처음 본 촬영(held-out 세션)의 2D 코너 위치추정이 개선된다.
근거: `projected_cuboid` 는 `live_capture_gt` 851장에서 camera-facing 0123 규약을
198장(23.3%) 어기고, 기존 real FT 데이터셋(`live_gt_v4`/`v5`)의 base 라벨은
696장 중 321장(46.1%)이 그 필드에서 나왔다.

**방법.** R0(합성만)에서 출발해 같은 536 프레임을 두 라벨 계약으로 각각 FT.
촬영단위(폴더) split, held-out 297장(9 폴더). seed 3.
바뀌는 축은 학습 라벨 필드 하나뿐 — 평가 GT·recipe·augmentation·스텝은 고정.

**판정 지표.** held-out 적격 keypoint(`source == manual_click`, 1,251점)의
index-wise 2D 오차 중앙값, 앙각 `<8` / `8-15` 층별.
짝지은 차이의 **세션 클러스터 95% CI** 로 판정한다.

**예상 실패 모드.**
- 라벨 계약을 고쳐도 held-out 이 안 움직인다 → 이전 real-FT 이득이 같은 세션
  분포(interleave) 때문이었을 가능성.
- FT 가 R0 보다 나빠진다 → 536장 소량 FT 의 과적합. augmentation 계약 확인.
- 두 arm 이 구분 안 된다 → 라벨 필드는 index 일관성만 바꾸고 위치는 안 바꾼다.

**중단 기준.** `FT_CONTRACT − R0` 의 `<8` 층 CI 가 0 을 포함하면
`REAL_SUPERVISION_LEVER_NOT_REPRODUCED` 로 기록하고 stage_2(앙각 구성 ablation)를
실행하지 않는다.  arm 을 추가하지 않고 D(matched capacity)로 이동한다.

**이 모집단에서 금지되는 보고.** axis / yaw / full 6D — 851장 전부
`n_pose_candidates = 2` 라 적격 프레임이 0장이다.  가림 층화도 불가
(`occlusion_level` 이 전부 `unknown`).  근거 `_docs/audits/next_accuracy_v2/GT_PARTITION.md`.

## 2. 결과

### 2.1 §11 corrected real-FT — REPRODUCED (2026-09-06, 20.6분)

held-out 297장(9 폴더), 적격 keypoint 1,251점, index-wise 2D 중앙값:

```text
arm             <8       8-15     검출
R0            4.37 px   6.83 px   295/297
FT_LEGACY     2.74 px   3.33 px   294/297
FT_CONTRACT   1.87 px   2.69 px   297/297

세션클러스터 95% CI (양수 = 앞쪽 우세)
contract - R0      <8    +2.70 [+1.73, +3.04]  0 배제  -> gate 통과
contract - legacy  <8    +0.71 [+0.31, +1.46]  0 배제
contract - legacy  8-15  +0.63 [-0.37, +1.32]  0 포함 -> INCONCLUSIVE
```

라벨 계약 수정 자체의 효과는 **저앙각에서만 확정**된다.

### 2.2 §13 앙각 구성 ablation — NOT_SUPPORTED (10.1분)

```text
2x2 (중앙값 px)      eval <8   eval 8-15
train L (<8) 137장     2.18       2.74
train M (8-15) 137장   2.37       3.43
```

대각 우세가 없다 — 8-15 로 학습해도 8-15 평가에서 낫지 않다.
"저앙각을 겨냥하면 저앙각이 특별히 좋아진다" 는 기각.
관측된 것은 "저앙각 프레임이 두 층 모두에서 더 나은 학습 데이터" 라는 약한 형태다
(세션별 부호 15/16 이 L 우세, CI 배제는 8-15 층 하나).

### 2.3 예상 실패 모드 대조

- "라벨 계약을 고쳐도 held-out 이 안 움직인다" -> 틀렸다.  움직였다.
- "FT 가 R0 보다 나빠진다" -> 안 났다.  검출도 297/297 로 올랐다.
- "두 arm 이 구분 안 된다" -> 부분적으로 맞다.  `<8` 은 INCONCLUSIVE 다.

### 2.4 인용 시 필수

1. **seed 3개는 독립 복제가 아니다** — 모델 텐서가 비트 동일(파라미터 753개,
   최대 절대차 0.000e+00).  arm 당 유효 run 은 1.  CI 는 학습 난수가 아니라
   held-out 모집단 불확실성이다.  ([[ultralytics-seed-does-not-reach-dataloader]])
2. **held-out 이 강한 일반화 시험이 아니다** — 촬영그룹 4개가 train·held-out
   양쪽에 다 있다.  1.87 px 는 "처음 본 촬영" 값이지 "처음 본 현장" 값이 아니다.
3. **axis / yaw / 6D 보고 금지** — 851장 전부 `n_pose_candidates = 2`.
4. secondary "gross(>25px) 프레임 비율" 은 프레임 지표를 중앙값으로 잡아
   robust 해진 탓에 0% 로 나왔다 — **미측정**으로 둔다.

### 2.5 다음

`_docs/audits/next_accuracy_v2/FINAL_DECISION.md` 의 결정표.
다음 실험 후보는 D-cheap(안 쓰인 저앙각 oblique pool 5,000장을 source 에 섞기,
렌더 0), 실패 시 E(matched capacity).
