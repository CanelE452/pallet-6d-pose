# METHOD_LOCK (초안) — Stage A

> 상태: **초안. 승인 전. GPU 학습 미착수.**
> 이 문서는 결과를 보기 전에 확정되어야 하고, 확정 후에는 게이트를 올리지 않는다.

## 0. 승인 없이는 실행하지 않는 것

새 synthetic 렌더 · exact-square C4 데이터 생성 · 10/25/50% C4 구성 실험 ·
60 epoch full training · self-training · DiffPnP 재실험 · PnP solver sweep ·
기존 R0/paper 결과 덮어쓰기.

## 1. 고정 사실 (Stage 0 확정분)

```
CURRENT_HEAD          903f1b3bb4e473908c7ed52335c834029e206afa
R0 checkpoint         .../YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt
  sha256              970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7
train dataset         challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k
  train/val           55,980 / 4,020
metadata manifest     .../spatial_concat_scratch/PROBE_METADATA_60K.jsonl
  sha256              20b863b75e763da04ca4ea7fcb270765ec262969725c696ea4ce3a0fae77a71e
평가 모집단           PAPER_EVAL 319 (role=DEV, held_out_final:false)
GT                    GEOMETRY_RESOLVED_POSE_GT.json  sha256 798959a6...
예측 캐시             predictions/R0.json             sha256 a05069d5...
pose object contract  sha256 a4c2918b4b0e9c97f94332d2e7e35132a8cbe0e738db25d92ea55e0d81210dbd
```

고정 상수:
```
pinv rcond            1e-6   (1e-3~1e-8 에서 결과 무감응 — 확인됨)
centroid index        8      translation-risk 에서 제외, keypoint loss 에서는 유지
exact-square tol      1e-6   저장 float 진단용, 물리적 C4 근거 아님
λ_geo                 gradient-norm 비 0.05 로 결정, screen 내내 고정  [추정][미검증]
```

## 2. arm (4 + 1 대조군)

```
A0  stock control     현행 YOLO26 pose loss, 고정 index, λ_geo = 0
A1  symmetry baseline per-sample C1/C2 (C4 는 데이터에 없다), translation-risk 없음
A2  proposed          A1 + L_TR (per-coordinate anisotropic)
A2iso ★등방 대조군    A1 + 같은 프레임 가중을 **스칼라 배율**로만 적용
A3  prior-art LC      A1 + Linear-Covariance 기준선
```

> ★ 2026-09-06 재배정: 초안이 `A2b` 라고 부른 등방 대조군은 **`A2iso`** 로 이름을
> 바꿨다. 사용자 지시의 `A2b` 는 다른 것(prior-art LC baseline)을 가리키며,
> 그 정본 정의는 **`METHOD_LOCK_A2B.md`** 다. 충돌 시 그 문서가 이긴다.
> 이 초안의 A0/A1/A2/A3/A2iso 는 전부 **실행 보류** — A2b 결과 이후에만 연다.

★ `A2iso` 를 두는 이유: Stage 0D 에서 `L_TR` 의 프레임 순위 결정력의 대부분이
Jacobian(시점 기하)에서 왔다(기하 인자 단독 rho 0.537 vs 잔차 인자 0.163).
그렇다면 이 항의 상당 부분은 **프레임 단위 hard-example 재가중**과 구별되지 않는다.
`A2 ~= A2iso` 로 나오면 "correspondence 방향을 가중한다" 는 주장은 근거를 잃고,
남는 것은 기하 유도 재가중이라는 훨씬 작은 기여다. 이 구분 없이 A2 의 이득을
방향 가중 덕이라고 부르지 않는다.

## 3. screen 예산

R0 checkpoint 에서 이어붙이는 **5 epoch continuation**. 모든 arm 이
동일한 init / train frames / steps / batch / optimizer / LR / augmentation /
device / update 수를 받는다. augmentation 은 Option B(geometry-changing 전부 OFF,
mosaic=0, color 만). 처음부터 60 epoch 를 돌리지 않는다.

## 4. primary metrics (PAPER_EVAL 319)

translation median/p90 · depth median/p90 · lateral median ·
rotation median/p90 · yaw median · 2D corner median/p90 · detection coverage.
ADD AUC 는 subgroup normalization 정의 문제가 해소되기 전까지 screen primary 로
쓰지 않는다. 반드시 ALL / plastic / wood / Low / Far / Occlusion / Clean 을 낸다
(subgroup 은 겹치므로 독립 causal factor 로 부르지 않는다).

## 5. 사전등록 게이트 — screen [추정][미검증]

A2 대 A1:
```
depth median              >= 5% 개선
AND total translation med  >= 5% 개선
AND rotation median 악화    <  5%
AND 2D corner median 악화   <  5%
```
전부 만족 -> full confirmatory 후보로 승격.
하나도 못 움직이면 `TRANSLATION_RISK_NO_SCREEN_SIGNAL` 로 종료.
결과를 보고 λ 를 무한 sweep 하지 않는다.

## 6. 사전등록 게이트 — full [추정][미검증]

A2 대 A1 에서 translation median >= 10% 개선 AND depth median >= 10% 개선,
방향이 대부분의 run/session 에서 동일, rotation/yaw/detection/corner 에 큰 손상 없음.
15~20% 이상이면 강한 신호지만 **사전 게이트를 결과 후에 올리지 않는다**.

replicate 는 `seed` 인자만 바꾸고 부르지 않는다 — ultralytics 의 seed 가
dataloader 에 도달하지 않은 이력이 있다(memory `ultralytics-seed-does-not-reach-dataloader`).
첫 100 batch 의 sample-id 해시가 실제로 다른지 로그로 증명한다.

## 7. 잡음 바닥 — 게이트 해석의 전제

memory `selftrain-pseudo-draw-noise-floor`: 대조군끼리도 rotation 6.7% ·
translation 14.3% 흔들린 이력이 있다. 그 실험과 조건이 다르지만
(여기는 고정 init·고정 데이터·동일 update 수), **10% 미만 차이를 단일 run 으로
주장하지 않는다** 는 규칙은 그대로 적용한다. full 게이트가 10% 인 이유다.

## 8. 실패 분기 (§45)

```
Stage 0 surrogate 상관 실패        -> loss 구현 중단           [해당 없음: PASS 0.828]
A2 < A1                           -> hypothesis reject, square 데이터 생성 금지
LC 는 좋고 A2 는 실패             -> 기존 LC 채택, novel 주장 금지
A2 synthetic 개선 real 319 악화   -> sim2real 문제, loss 항 추가 금지
depth 개선 rotation 악화          -> translation-only 가중의 trade-off, full covariance 검토
A2 ~= A2iso                       -> ★ 방향 가중 주장 철회, 기하 재가중으로만 기술
```

## 9. Stage B 전제

Stage A 가 실패하면 Stage B(정사각 합성 비율)는 **하지 않는다**.
그리고 Stage B 를 열더라도 `SYNTHETIC_DIMENSION_AUDIT.md` 의 사실 —
현재 학습셋의 exact-square 가 1 장이라는 것 — 때문에 C4 데이터는
**새로 생성해야만** 존재한다. 기존 near-square 4,219 장을 C4 로 승격하는 경로는
없다.
