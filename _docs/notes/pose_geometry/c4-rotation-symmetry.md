# 정사각 팔레트 회전 대칭 (C4) 계열

> 🟡 진행 중 · 갱신 2026-09-06 · 소비처: 과제 트랙 배포 모델 교체 결정
> (HuggingFace `CanelE452/pallet-pose-yolo26n-c4`)
> **한 줄**: 정본 라벨 재학습은 사전등록 gate PASS 지만, 실체는 index 일관성 회복이고
> C4 loss 자체의 위치정확도 이득은 아직 NOT_ESTABLISHED.

과제용 팔레트(`plastic_standard_110x110x15`)는 정사각 footprint 에 4면 포크 진입이라
yaw 90도 회전이 등가다. 그 등가를 학습 손실과 평가 지표에서 어떻게 다룰지의 계열.

계약: `challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json`
(`equivalent_yaw_degrees = [0, 90, 180, 270]`, 2026-09-06 FROZEN)

---

## 1. 제안 — 정본 라벨 + flip 제외 재학습 (2026-09-06 착수)

### 가설

앞선 C4 학습(F0/F1/F2)은 라벨 소스가 `projected_cuboid` 였고, 그 필드는
camera-facing 0123 규약을 198/851(23.3%)에서 어긴다. 정본 필드
`keypoint_annotations`(위반 0/847)로 데이터셋을 다시 만들면 index 일관성이 올라가
코너 위치추정이 개선된다.

동시에 좌우 flip 증강을 뺀다. 반사는 C4 회전군의 원소가 아니라서 phase 를 반전시키고
(실측: 원본 phase 270 44.4% → flip 본 phase 90 39.2%), 배포 장면에 거울상 팔레트는
없다. 대신 확대축소·색감·밝기를 on-the-fly 로 강화한다.

### 방법

```
데이터  live_gt_v6_clean   train 원본만 / val 은 F2 와 같은 프레임(interleave 6)
        라벨은 keypoint_annotations (새 load_kps 기본)
        파일 증강 없음 (flip·noise·crop 전부 제외)

증강    scale 0.25 -> 0.40,  hsv_v 0.35 -> 0.50,  hsv_s 0.50 -> 0.60
        hsv_h 0.015, mosaic 0.3, close_mosaic 10  (base 계약 유지)
        fliplr 0.0, flipud 0.0

학습    1단계  yolo26n-ft -> [indexed FT 120ep] -> stage1_indexed
        2단계  stage1     -> [C4 FT 120ep]      -> stage2_c4
        120 epoch 은 F2(2,088 x 40ep = 83,520 샘플)와 총 업데이트를 맞춘 값

loss    ChallengeC4PoseLoss — 4개 회전 순열에 대해서만 min, centroid(8) 불변,
        object 단위 branch 선택, Hungarian 미사용
```

2단계로 나눈 근거: C4 는 loss 가 아니라 init 이 face phase 를 정한다
(memory `c4-init-decides-face-phase-not-loss` — 합성 base 에서 바로 C4 를 걸면 270도로
40% 돌아가고, indexed 를 거치면 7%). 라벨 규약이 바뀌었으므로 그 규약을 indexed 로
먼저 익힌 base 가 필요하다.

### 판정 지표 (사전등록 — 결과를 보고 고치지 않는다)

주 지표는 **C4-equivalent median** (위치 정확도, phase 무관).
비교는 F2·v4 를 **같은 새 val 로 재채점**해서 한다 (base 가 다르므로 1:1 이 아니라
같은 잣대 비교).

```
PASS      c4_median(stage2) <= c4_median(F2) - 0.15px
     AND  true_collapse(stage2) <= true_collapse(F2)
     AND  detected(stage2) >= detected(F2)
NEUTRAL   |차이| < 0.15px 이고 collapse·검출이 나빠지지 않음
FAIL      그 외
```

0.15px 근거: 같은 트랙에서 F0(2.00) 와 v4(1.90) 의 차이가 0.10px 였고 그것을
유의미하다고 보지 않았다. 그보다 한 단계 위를 문턱으로 둔다.

부 지표 (판정에 쓰지 않고 기록만): fixed-index median — 배포는 index 일관성이
필요하므로 C4 가 좋아지고 fixed 가 나빠지면 배포 승격을 보류한다.

### 예상 실패 모드

- 라벨 위반 23%가 병목이 아니었다 → 개선 없음. 그러면 원인감사
  (`accuracy-root-cause-v1`)의 "병목은 저앙각 코너 위치추정" 과 정합하고,
  loss·라벨 축은 닫는다.
- train 696 이 너무 적어 120 epoch 에서 과적합 → val 곡선으로 확인.
- 새 라벨 규약이 배포 규약과 어긋남 → 착수 전 k-회전 거리 검사로 차단.

### 중단 기준

C4 계열에서 3회 연속 개선 실패면 loss 축을 접고 데이터 축(저앙각 flat-view 수집)으로
복귀한다. F1/F2 는 DEPLOYABLE_WITH_CAVEAT 미승격이라 미달성 1회로 셌었다.

2026-09-06 정본 라벨 재학습(§2)은 사전등록 gate 기준 **PASS** 다 — 실패 카운트를
0으로 리셋한다. 다만 이 PASS 의 실체는 index 일관성 회복이지 C4 loss 고유 효과가
아니다(§2의 3번, NOT_ESTABLISHED). "C4 loss 가 위치 정확도를 올리는가"라는 원래
가설 자체는 여전히 미해결이므로, 다음에 이 질문을 다시 테스트한다면 그 결과가
진짜 카운트 대상이다.

## 2. 결과 — 정본 라벨 + flip 제외 재학습 (2026-09-06, PASS)

두 단계 모델을 이전 C4 트랙의 두 체크포인트와 **같은 새 val 155장**으로 재채점했다.

```
stage1_indexed   1단계 결과물 — 정본 keypoint_annotations 라벨로 index 만 고정 FT
                 (C4 loss 미적용). checkpoint: clean_label/stage1_indexed/weights/best.pt
stage2_c4        2단계 결과물 — stage1_indexed 에서 이어 C4 loss 로 FT
                 (이번 착수가 궁극적으로 배포 후보로 노린 모델)
F2               비교 대상, 현재 HuggingFace 배포본의 학습 원본
                 (challenge_c4_track/F2/weights/best.pt, projected_cuboid 라벨
                 + flip 증강 + C4 loss — §1 이 문제 삼은 옛 라벨 계열)
v4               비교 대상, 그 이전 세대 base
                 (runs_live_gt/ft_live_gt_v4/weights/best.pt, indexed FT, C4 loss 미적용)
```

출처: `challenge/yolo_pose_one_model/challenge_c4_track/clean_label/RESULT.json`

| arm | 검출 | fixed-index median (px) | fixed p90 (px) | C4-equivalent median (px) | C4-equiv p90 (px) | true_collapse | symmetry_rescued | phase0 비율 | pose mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1_indexed | 155/155 | 1.80 | 3.78 | 1.79 | 3.55 | 1 | 2 | 98.7% (153/155) | 0.978 |
| stage2_c4 | 155/155 | 1.90 | 3.82 | 1.88 | 3.78 | 1 | 2 | 98.7% (153/155) | 0.960 |
| F2 (현 배포본) | 154/155 | 4.59 | 180.55 | 2.67 | 5.31 | 1 | 55 | 57.1% (88/154) | 0.323 |
| v4 (이전 세대 base) | 154/155 | 25.40 | 171.48 | 1.89 | 3.51 | 4 | 71 | 50.6% (78/154) | 0.295 |

지표 정의(`evaluate_c4_arms.py::summarize`):
- **fixed-index** = 배포가 실제로 읽는 값 — index 를 라벨 그대로(phase 0) 고정하고
  잰 오차. phase 가 틀리면 이 값이 그대로 커진다.
- **C4-equivalent** = 4개 회전 permutation(0/90/180/270도) 중 최소 오차 — phase 를
  안다고 가정한 상한. "위치 정확도" 를 보려면 이 지표를 봐야 한다.
- **symmetry_rescued** = fixed 오차 >20px 인데 C4-equivalent 오차 <5px — 점 자체는
  맞았는데 라벨/예측의 phase 가 90·180·270도로 돌아간 경우.
- **true_collapse** = C4-equivalent 로도 20px 를 넘는 경우 — phase 문제가 아니라
  실제 위치추정이 크게 틀린 경우.
- **phase0 비율** = 4개 permutation 중 phase 0 가 최소오차였던 프레임 비율. index
  일관성의 직접 측정치.

### 사전등록 gate 대조

```
PASS      c4_median(stage2) <= c4_median(F2) - 0.15px    1.88 <= 2.67-0.15=2.52   충족 (여유 0.64px)
     AND  true_collapse(stage2) <= true_collapse(F2)      1 <= 1                   충족
     AND  detected(stage2) >= detected(F2)                155 >= 154               충족
```

`RESULT.json.verdict = "PASS"`, detail: "C4 median 1.88 vs F2 2.67 (-0.79px) |
collapse 1 vs 1 (+0) | 검출 155 vs 154 (+1) | fixed median 1.90 vs 4.59".

### 판정을 어떻게 읽어야 하는가 (과대해석 방지)

1. **개선된 것은 index 일관성이다.** phase0 비율이 stage1/stage2 모두 98.7% 인
   반면 F2 57.1%, v4 50.6% — 절반 가까이가 반대 위상에 붙어 있었다. 그 결과
   fixed-index median 이 F2 4.59px → stage2 1.90px, p90 은 180.55px → 3.82px 로
   떨어졌다. 배포가 실제로 읽는 값이 이것이므로, 이번 개선의 실체는 여기다.

2. **위치 정확도 자체는 거의 그대로다.** phase 를 안다고 가정한 상한인
   C4-equivalent median 으로 보면 stage2 1.88px 대 v4 1.89px 로 사실상 동일하다.
   F2 대비 -0.79px 개선은 F2 가 55장에서 phase 가 반대로 붙은(symmetry_rescued 55)
   착시가 크다. "코너 위치추정 정확도가 좋아졌다" 로 읽으면 틀린다.

3. **C4 loss 자체의 이득은 확인되지 않았다 (NOT_ESTABLISHED).** stage1(C4 loss
   없이 정본 라벨로 index 만 고정 FT) 의 C4-equivalent median 1.79px 가 stage2
   (C4 loss 적용) 1.88px 보다 근소하게 낫다. 차이 0.09px 는 사전등록 gate 문턱
   0.15px 에 못 미쳐 구분되지 않는다 — C4 loss 가 해롭다는 뜻이 아니라, 이
   실험에서는 이득도 손해도 입증되지 않았다는 뜻이다. 라벨 위반율을
   23.3%→0.8% 로 낮추고 나니 C4 loss 가 흡수할 phase 모호성이 거의 남지 않은
   것으로 보인다 [추정].

4. **적용범위 — val 155 는 새 세션 일반화 근거가 아니다.** 이 val 은 F2 와 같은
   interleave 세션(같은 촬영 회차에서 train/val 을 섞어 나눈 분할)이라 과제
   트랙의 의도된 비교 계약이지만, 처음 보는 촬영 세션에 대한 일반화 근거로
   인용해서는 안 된다(관련 memory: `live-gt-ft-split-and-aug-decide-the-verdict`).
   일반화를 말하려면 SEALED holdout(105장) 이 필요하고, 그건 봉인 소진이라 사용자
   결정 사항이다.

### 배포 판단 (참고 — 최종 결정 아님)

네 arm 전체에서 `stage1_indexed`(C4 loss 미적용, index 고정 FT 만)가 fixed·
C4-equivalent 의 median·p90·pose mAP50-95(0.978) 전부에서 가장 좋다. 다만 HF
배포본 교체 여부는 사용자 결정 대기 — 위 3번(NOT_ESTABLISHED)과 4번(적용범위)을
감안해야 한다.

근거: `_docs/history/2026-09-06.md` "정본 라벨 + flip 제외 C4 재학습 착수" /
"결과 — PASS" 절. 착수 전 검증(라벨 차이 46.8%, LR 위반 0.8% vs 23.3%, k-회전
거리 검사)도 같은 절에 있다.
