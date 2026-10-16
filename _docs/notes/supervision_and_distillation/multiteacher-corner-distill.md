# 여러 교사 + 국소 코너 증류 (multiteacher_corner_distill_v1)

계열 코드: `multiteacher_corner_distill_v1`
착수: 2026-09-05

## 1. 제안

**가설.** 합성 데이터만으로 학습한 단일 교사(이하 기준 모델 R0, YOLO26n-Pose)는 실제 영상 일부에서
keypoint 를 구조적으로 크게 틀린다. 이 오차는 기존 geometry filter / confidence / consistency 로
걸러지지 않고, 같은 교사의 hard pseudo-coordinate 를 다시 학생에게 먹이면 교사의 편향이 그대로 전달된다.
서로 다른 representation 을 가진 여러 교사와 실제 RGB 의 국소 코너 증거를 쓰면
그 편향을 넘는 pseudo supervision 을 만들 수 있는가?

**방법 (게이트 순서, 싼 것 먼저).**
- Gate A 여러 교사 상보성 감사 (학습 0) — oracle-best-teacher 상한이 최고 단일 교사보다 실제로 위인가.
- Gate B 국소 코너 증거 감사 (학습 0) — R0 예측 주변 실제 RGB 에 더 정확한 코너 후보가 존재하는가.
- Gate C 국소 코너 전문가 짧은 파일럿 학습 — B 가 통과할 때만.
- Gate D 융합 pseudo-target 품질 감사 → D2 단일 학생 증류 (900 optimizer update).
- Gate E 도메인 편향 진단 → AdaBN / residual adapter.

**판정 지표.** 2D: detection coverage, pooled keypoint median px, p90 px, gross20, gross40.
6D: PoseCov, rotation median, yaw median, translation cm, IoU3D, symmetry-aware ADD AUC.
불확실성: frame bootstrap + recording/session-cluster bootstrap.
게이트별 통과 임계는 `METHOD_LOCK.json` 에 결과를 보기 전에 고정한다.

**예상 실패 모드.**
- 여러 모델이 같은 방향으로 틀린다 (상보성 없음) → Gate A 에서 종료.
- 정답 후보는 있는데 GT 없이 고를 신호가 없다 (oracle only) → Gate B/C 로는 못 넘어간다.
- 국소 좌표는 좋아지는데 학생에게 전달되지 않는다.

**중단 기준.** Gate A 실패 → 여러 교사 융합 학습 금지. Gate B 실패 → 국소 전문가 학습 금지.
Gate C 실패 → adapter 로 구조하지 않는다. 융합 target 품질이 R0 보다 나쁘면 학생 학습 금지.

**누수 경고.** 평가에 쓰는 PAPER_EVAL 319 장은 이미 반복 사용된 development set 이다.
결과가 좋아져도 held-out / confirmed / final / SOTA 로 부르지 않는다. 상태는 DEVELOPMENT_METHOD_SIGNAL.

## 2. 결과

`FINAL_CASE = CASE_B ORACLE_COMPLEMENTARITY_ONLY` · 승격 없음 · DEVELOPMENT_METHOD_SIGNAL.
전체 산출물은 `data/pallet/results/multiteacher_corner_distill_v1/final/`.

### 한 문장

정답은 여러 교사 안에 실제로 있는데, GT 없이 그것을 고를 신호가 어디에도 없었다.

### 게이트별

```
게이트                              판정                                    근거
──────────────────────────────────────────────────────────────────────────────────────────
A 여러 교사 상보성                  STRONG                                  R0 gross 코너의 30.3% 를
                                                                            다른 교사가 10px 로 맞힘
A 파라미터 없는 융합                실패                                    좌표 median p90 72.7 ·
                                                                            medoid 69.9 (R0 43.9)
A 불일치의 오류 예측                AUC 0.79                                R0 자신의 confidence 0.658
B 국소 코너 증거                    STRONG (그러나 밀도의 산물)             균등난수 대비 lift 5px +0.015
B 고전 CV 선택기                    ORACLE_ONLY_HEADROOM                    gross 구제율 0.000
C 국소 전문가                       STOP                                    median -11.1% 이나 p90 -0.7%
D 융합 target 품질                  FAIL                                    usable 에서 R0 를 못 이김
D2 학생 증류                        NOT_RUN                                 게이트가 차단
E 도메인 편향                       DOMAIN_SEPARABLE_BUT_NOT_ERROR_LINKED   AUROC 1.0000 / 오차 AUC 0.482
E adapter                           NOT_RUN                                 두 번째 조건 불통과
```

### 기전 — 네 게이트가 같은 벽을 가리킨다

교사 불일치는 "어디가 틀렸는지" 를 잘 안다(AUC 0.79). 그런데 "무엇이 맞는지" 는 모른다.
그래서 불일치로 거른 usable 부분집합은 **R0 가 이미 맞히는 자리**가 되고
(R0 median 6.97 → 5.80, p90 60.31 → 20.14), 거기서 융합은 R0 를 이기지 못한다.
국소 RGB 도 대안을 못 준다 — 학습된 전문가조차 R0 의 gross 코너 498개 중 1개만 구제했다.
도메인 축은 오차와 무관하다. pseudo-label 은 좌표를 요구하는데 이 트랙이 만든 것은 신뢰도뿐이다.

### 그래도 남는 사실 두 개

- 국소 전문가는 중앙값을 실제로 개선한다 — 6.36 → 5.66 px (11.1%), harm 1.45%,
  ADDsym AUC 0.4285 → 0.4373. 꼬리를 못 건드려 게이트는 못 넘었다.
- 교사 불일치는 값싸고 강한 오류 탐지기다. 라벨 생성이 아니라 거부(rejection) 축에 쓸 수 있다.
- 미라벨 풀의 불일치 중앙값이 DEV_EVAL 의 7배다. self-training 병목이 방법이 아니라
  **adaptation pool 의 구성**일 수 있다. (둘 다 이 트랙의 잠긴 범위 밖이라 실행하지 않았다.)

### 다시 하지 말아야 할 것

좌표 median·medoid 융합 / 반경 12~64px 고전 CV 코너 선택기 / source 에서 보정한 합의 임계를
미라벨 풀에 적용 / Pose26 sigma 기반 교사 가중 / 합성-실제 도메인 정렬.

### 실행 이력

첫 Gate C 실행이 77분 만에 산출물 0 으로 정지했다. 원인 두 겹 — step 마다 PNG 를 콜드로
디코딩(3.1초/step)한 것과, torch 를 import 한 부모를 fork 한 프로세스 풀에서 OpenMP 런타임이
100% CPU 로 스핀한 것. crop 사전추출 + spawn + 스레드 1 로 고쳐 0.02초/step 이 됐고
양 arm 모두 5,000 update 를 완주했다. 계약(구조·예산·jitter·arm·임계)은 바꾸지 않았다.
