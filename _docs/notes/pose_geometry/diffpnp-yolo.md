# 학습측 DiffPnP — 미분가능 PnP 를 YOLO pose loss 에 넣는다 (diffpnp_yolo_v1)

계열 코드: `diffpnp_yolo_v1`
착수: 2026-09-06

## 1. 제안

**가설.** 논문 평가는 예측 keypoint 에 `SQPnP + RefineLM` 을 걸어 pose 를 읽는다. 그 읽기
연산을 학습 loss 안으로 가져오면(미분가능 Gauss-Newton PnP), keypoint 가 pose 기준으로
정렬되어 실제 pose 정확도가 좋아지는가.

평가측 교체(`solver_swap_v1`)는 REJECT 로 끝났다. 남은 갈래가 이것이다.

**방법.** 예측 2D 로 GN PnP 를 풀고, 그 pose 가 만드는 카메라 좌표 코너를 참조 pose 의
코너와 Huber 로 비교한다. 참조 pose 는 **GT 2D 로 푼 PnP** 라서 규약 오프셋(perm_v4 180도,
memory `synthetic-pose-eval-needs-gt-axis-perm-v4`)이 예측·참조 양쪽에서 상쇄된다.

**판정 지표.** PAPER_EVAL 319 의 rotation median · translation median · IoU3D median ·
symmetry-aware ADD AUC. 게이트는 `DIFFPNP_SCREEN_METHOD_LOCK.json` 에 결과 전에 고정.

**예상 실패 모드.** 예측 2D 적합도를 낮추면 translation 은 당겨지고 rotation 이 깨진다
— 2026-08 predseed 스크린과 2026-09-06 `solver_swap_v1` 이 같은 방향을 보였다.

**중단 기준.** 스크린이 REJECT 면 60 epoch full 로 확대하지 않는다. `SOLVER_LOSS_TRACK`
은 이미 CLOSED 였고 이번은 사용자 지시로 다시 여는 것이다.

**누수 경고.** PAPER_EVAL 319 는 반복 사용된 development set. held-out 아님.

## 2. 배선에서 실제로 막힌 것 (전부 수치로 확인)

증강이 걸린 YOLO 학습에 PnP 항을 붙이는 건 좌표계 문제 세 겹이었다.

**(가) 증강된 이미지에는 원본 K 가 안 맞는다.** mosaic·scale·translate 가 영상면을
바꾼다. 영상면 affine `A` 는 투영에 그대로 곱해지므로 (`A K X / z`) **K' = A K** 가 정확한
보정이다. `A` 는 소스 `projected_cuboid` 와 배치 GT keypoint 대응에서 최소제곱으로 복원한다
(테스트로 정확성 확인).

**(나) 좌우 반전은 keypoint 인덱스를 바꾼다.** `fliplr=0.5` 이고 `flip_idx` 가
`[1,0,3,2,5,4,7,6,8]` 이다. 이걸 반영 안 하면 대응이 어긋나 affine 이 안 맞는다.
두 대응을 다 풀고 잔차가 낮은 쪽을 쓰도록 고쳤다. 감독률 **55% → 98.9%**.

**(다) mosaic 은 4장을 합치고 `im_file` 을 대표 1장만 남긴다.** 그래서 나머지 3장의
인스턴스는 사이드카와 어긋난다. affine 잔차 게이트(0.5px)가 이들을 걸러낸다 — 즉 마스킹된
감독이 된다. R0 레시피(mosaic 켬)에서 감독률 **19.4%**, mosaic 끄면 **98.9%**.

측정값 (스모크, 35 batch × 2 head):

```
설정                     감독 인스턴스   affine 거절   감독률
mosaic 켬 · flip 미처리        1,219       10,296      10.6%
mosaic 끔 · flip 미처리        3,378        2,751      55.1%
mosaic 끔 · flip 처리          6,061           66      98.9%
mosaic 켬 · flip 처리          2,232        9,262      19.4%
```

**(라) end2end 경로에서 항이 조용히 빠질 뻔했다.** YOLO26 은 `E2ELoss` 래퍼를 쓰고, 이
래퍼는 `__call__` 이 아니라 `loss()` 를 부른다. `__call__` 만 덮었더니 배치 파일 경로가
전달되지 않아 항이 **한 번도 계산되지 않았다**(첫 스모크에서 통계 0). `loss()` 도 덮어
one2many·one2one 두 벌 모두에 들어가는 것을 확인했다.

## 3. lambda 선택 (Q0)

loss **값** 비율로 lambda 를 잡지 않았다 (memory `aggregate-scale-statistic-does-not-
transfer-to-pose`). 두 항이 `pred_kpts` 에 만드는 **기울기 노름** 비율로 잡았다.

```
lambda=1 에서 기울기 비율 중앙값   0.0449   (목표 대역 5%)
5% 에 맞는 lambda 중앙값           1.13     IQR 0.079 ~ 3.78
배치당 감독 인스턴스 중앙값          15
```

배치별 산포가 48배로 크다 — 이 대역 선택은 정밀하지 않다. λ\* = 1.0 채택.

## 4. 결과

`FINAL = REJECT` · 승격 없음 · full 60epoch 확대 안 함 (중단 기준대로).
산출물 `data/pallet/results/diffpnp_yolo_v1/DIFFPNP_SCREEN_{RESULT.json,REPORT.md}`.

### 한 문장

합성 val 은 두 arm 이 소수점 넷째 자리까지 같은데, 실제 pose 는 네 지표가 전부 나빠졌다 —
pose 읽기 연산을 학습에 넣는 것이 실제 pose 를 **악화**시킨다.

### 수치 (PAPER_EVAL 319, 짝지은 같은 프레임)

| 지표 | 대조 λ=0 | 처치 λ=1 | 상대변화 |
|---|---:|---:|---:|
| rotation median (deg) | 2.6139 | 3.1928 | +22.15% |
| translation median (cm) | 6.9523 | 8.5838 | +23.47% |
| IoU3D median | 0.6018 | 0.5573 | −7.40% |
| symmetry-aware ADD AUC | 0.4230 | 0.3830 | −9.44% |
| axis accuracy | 0.7398 | 0.7147 | −3.39% |

사전등록 판정 = REJECT, 개선 0/4. 최대 상대변화 23%로 `NO_CHANGE` 판별 문턱(1e-6)과는
거리가 멀다 — 실체 있는 악화다.

### ★ 진단 — 합성에서는 아무 일도 일어나지 않았다

```
                        대조 λ=0    처치 λ=1
synthetic val pose mAP50-95   0.91969    0.91953     (-0.02%)
synthetic val box  mAP50      0.99299    0.99317
학습 시간                      4,589s     5,144s     (+12% = 항이 실제로 돌았다는 방증)
```

합성 val 은 **구분이 안 된다**. 그런데 실제 데이터에서는 7~23% 나빠진다. memory
`paper-s2-six-screens-all-reject-final-base-dope` 의 "공통 벽"(합성 목적함수는 내려가는데
real 전이 실패)과 정확히 같은 모양이고, `arch-baseline-synthetic-cannot-select-architecture`
("합성으로는 못 고른다")를 한 번 더 확증한다.

### 같은 방향의 네 번째 독립 재현

| 언제 | 무엇을 | 결과 |
|---|---|---|
| 2026-07 | DOPE StageA/B, DiffPnP3D λ0.005 학습 | PARTIAL (honest8 flat, 3-8° 회귀) |
| 2026-08 | predicted-seed DiffPnP, 평가 refine | REJECT (observed −47%, GT reproj +4.5%) |
| 2026-09-06 | 평가 solver 교체 (`solver_swap_v1`) | REJECT (rot +1.48%) |
| 2026-09-06 | **학습 loss 로 투입 (이 문서)** | **REJECT (네 지표 전부, 7~23%)** |

"예측 2D 에 pose 를 더 잘 맞출수록 실제 pose 가 나빠진다" 가 네 번째로 재현됐다.
학습측·평가측 양쪽이 닫혔으므로 이 계열은 종료한다.

### 적용범위 · 한계 (과장 금지)

- **seed 1개.** 두 arm 이 같은 seed·같은 데이터 순서를 썼으므로 짝 비교는 성립하지만,
  seed 산포는 재지 않았다. 효과 크기(7~23%)가 corner 계열 seed 산포(약 0.5%,
  memory `line-branch-seed-variance-exceeds-effect`)보다 훨씬 크다는 점만 근거다.
- **10 epoch.** R0 은 60 epoch 이다. 60 epoch 에서 뒤집힐 가능성은 배제하지 못한다.
  다만 중단 기준상 확대하지 않는다.
- **감독률.** 이 스크린은 `close_mosaic=10 >= epochs` 라 mosaic 없이 돌았고 감독률이
  약 99% 였다. 즉 DiffPnP 에 가장 유리한 조건에서도 졌다.
- PAPER_EVAL 319 는 development set — held-out 아님.

## 5. self-training 에 얹기 (사용자 요청, 2026-09-06)

앞의 두 실험은 (가) 평가 read-out solver 교체 (나) **합성** 학습에 DiffPnP 투입이었다.
아직 안 해본 칸이 하나 남아 있다 — SQPnP 로 학습한 base 를 **pseudo-label 로
self-training 할 때** DiffPnP 를 같이 켜는 것.

### 설계

```
base       R0 (FROZEN, SQPnP 기반 합성 학습)
데이터셋   paper_selftrain_v1/R5_PROPOSED — PL 1,440 노출 + 합성 replay 1,440 노출
레시피     run_paper_selftraining.py TRAIN_ARGS 그대로 · exposure lock 예산
           (10 epoch · 900 optimizer update · batch 32 · lr 0.002 · fliplr 0.0 · mosaic 0.15)
arm        ST_C_LAMBDA0 (λ=0) vs ST_T_DIFFPNP (λ=0.1432) — 유일 변수 λ
```

### 참조 pose 를 어디서 얻는가 (이 실험의 핵심 설계)

PL 에는 GT 가 없다. 참조는 **PL keypoint 에 SQPnP+RefineLM 을 건 결과**다 — 평가가 pose 를
읽는 연산과 같은 연산이므로 "SQPnP 로 학습한 것을 기준으로 삼는다"가 성립한다.

★ 정직한 한계: 이 참조는 **교사 자신의 좌표에서 읽은 pose** 다. 교사가 모르는 것을
알려주지 못한다. 이 항의 기전은 새 정보 주입이 아니라 **2D 오차를 pose 영향도로
재가중**하는 것이다. 실험은 그 재가중이 실제로 도움이 되는지를 본다.

사이드카: 1,683 / 1,699 프레임 (replay 1,440 + PL 243). PL 제외 사유는 가시 코너 부족 12,
참조 재투영 > 12px 4. PL 참조 재투영 median **2.00px** / p90 4.23px.
치수는 `metric_split_lock` §3.2 [LOCKED] 1.10×1.30×0.12 m, W/D 배정은 프레임마다 재투영이
낮은 쪽(평가의 selector 와 같은 방식).

### Q0 — 이 설정은 합성 때보다 훨씬 건강하다

```
                              합성 학습(§3)   self-training(여기)
lambda=1 기울기 비율               0.045           0.349
5% 대응 lambda 중앙값               1.13            0.143
       IQR                    0.079~3.78      0.112~0.200   (산포 48배 → 1.8배)
배치당 감독 인스턴스                  15             110.5     (7배)
corner 오차(대각선 정규화)           1.58            0.075     (Huber 이차 구간 안)
```

합성 스크린은 감독이 희박하고 λ 대역이 48배로 흩어져 있었다. 여기서는 감독이 7배 많고
대역이 좁다 — DiffPnP 에 훨씬 유리한 조건이다.

### 결과 — REJECT (세 draw 전부)

`FINAL = REJECT` · 첫 draw 의 개선은 **표집 잡음**이었다.

먼저 계획에 없던 검증이 하나 나왔다. **λ=0 대조군이 동결 정본 R5_PROPOSED 와 7개 지표
전부 오차 `0.000e+00` 으로 일치**한다. 내 DiffPnP 학습 하네스가 원본 self-training
파이프라인과 동일하다는 뜻이고, 따라서 처치군의 차이는 λ 만의 것이다.

#### 처치 효과 — 세 draw 에서 방향이 전부 뒤집힌다

| 지표 | D1 (기본) | D2 (P43) | D3 (P44) | 방향 |
|---|---:|---:|---:|---|
| rotation median° | +5.56% | −6.82% | −0.98% | 엇갈림 (2/3 개선) |
| translation median cm | −8.84% | +3.02% | −2.54% | 엇갈림 (2/3 개선) |
| IoU3D median | +1.48% | −1.08% | −1.25% | 엇갈림 (1/3 개선) |
| ADD AUC | +2.58% | −0.62% | −3.20% | 엇갈림 (1/3 개선) |
| 사전등록 판정 | REJECT (3/4) | REJECT (1/4) | REJECT (2/4) | |

D1 만 보면 "translation −8.8%, ADD AUC +2.6% 로 총합 pose 가 좋아졌다" 로 읽힌다.
D2·D3 에서 그 방향이 뒤집힌다. **단일 draw 로 판정했으면 오보였다.**

#### ★ 부산물 — pseudo-label 표집 draw 의 잡음 바닥이 크다

처치를 빼고 **대조군(λ=0)끼리만** 비교한 값이다. 같은 레시피·같은 base·같은 seed 인데
pseudo-label 표집만 다르다.

```
지표                     D1        D2        D3      폭
rotation median deg    2.5345    2.5999    2.7107   +6.7%
translation median cm  8.8265    7.6472    8.3098  +14.3%
IoU3D median           0.5868    0.5963    0.5913   +1.6%
ADD AUC                0.4001    0.4186    0.4161   +4.5%
```

DiffPnP 효과(±1~9%)가 **대조군 자체의 draw 산포보다 작다**. self-training arm 비교에서
이 정도 크기의 차이를 단일 draw 로 주장하면 안 된다는 뜻이다.
(적용범위: R5_PROPOSED, 3 draw, PAPER_EVAL 319 에서 측정. 다른 arm 은 미측정.)

#### seed replicate 는 여기서 무효다

`args.seed` 는 ultralytics dataloader 에 도달하지 않아, R0 고정 가중치에서 이어가는 이
설정은 seed 를 바꿔도 비트 동일한 모델을 낸다
([[ultralytics-seed-does-not-reach-dataloader]]). 유효한 replicate 는 **pseudo-label
표집 draw** 를 바꾸는 것이고, repo 의 P43/P44 데이터셋이 그 방식이다.

#### 다섯 번째 재현

self-training 칸은 감독 7배·λ 대역 1.8배로 **DiffPnP 에 가장 유리한 조건**이었는데도
개선이 재현되지 않았다. 평가측·합성학습측·self-training측이 모두 닫혔다.
