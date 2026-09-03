# 증거 등급 — MAIN / SUPPORTING / EXPLORATORY / EXCLUDE

> 이 문서는 `_docs/paper/final/EVIDENCE_LEDGER.md` 를 **대체하지 않는다**.
> 원장은 lock commit 과 result commit 의 시간 순서로 Tier A/B/C 를 이미 확정했다.
> 여기서는 그 등급을 **논문 배치**(main table / supporting / appendix / 제외)로
> 옮기고, 원장 이후에 생긴 트랙(pose closure · site · depth · temporal · V1B)을
> 같은 규칙으로 이어 붙인다.  충돌하면 원장이 이긴다.

```
MAIN         결과를 보기 전에 계약이 얼어 있었고(Tier A), 본문 표에 들어간다
SUPPORTING   기제 분석·부분모집단·진단.  본문을 뒷받침하되 확증으로 쓰지 않는다
EXPLORATORY  post-hoc·표본 부족·모집단 미확정.  부록 또는 각주
EXCLUDE      논문에 넣지 않는다
```

## MAIN

```
결과                                    근거                                          Tier
────────────────────────────────────────────────────────────────────────────────────────
R0 (synthetic-only)                     lock 15f0cb5 → result 89129c2, lead 4h11m       A
R0_CONT (source-only continuation)      lock c3a2581 → result 89129c2, lead 3h46m       A
R1 naive / R2 confidence /              세 lock 이 arms[] 에 이름을 미리 박아 두고       A
R3 reprojection / R4 removal /          init checkpoint 를 sha256 으로 고정,
R5 full consistency                     GT_USED_FOR_SELECTION=false
main 2D 표 (AP50-95 · AP50 ·            동일 arm JSON 에서 직접                          A
  pooled supervised keypoint median)
main 6D pose 표 (PoseCov · AxisAcc ·    GT 규칙(GT_AXIS_RESOLUTION_LOCK)이 결과를        A
  R · yaw · t · IoU3D · ADDsym AUC)     보기 전에 얼렸고 selector 는 손대지 않음.
                                        R0 replay 가 기존 캐시를 정확히 재현
session-cluster paired bootstrap        POSE_PAIRED_BOOTSTRAP.json, 10,000, seed        A
  (6D)                                  20260903.  24 개 metric block 전부 산출
```

★ **2026-09-04 확정.**  6D pose 표는 MAIN 이다.  `PAPER_CLAIM_LOCK.json` 이
`POSE_METRICS_STATUS = REPORTABLE` 로 amendment 됐고 historical first pass 는 보존됐다.
표를 싣되 개선은 주장하지 않는다 — 개선 방향으로 session-cluster 구간이 0 을 배제한
metric block 은 0/24 다.

## SUPPORTING

```
결과                                    쓰임                                          Tier
────────────────────────────────────────────────────────────────────────────────────────
pseudo-label 필터 품질(분리도·retention) 필터 신호가 무작위가 아님(claim F/G)             B
축 oracle 진단(ORACLE vs MAIN path)      selector 가 병목임을 보이는 상한선               B/C
site-matched 소규모 arm 평가            A8_DAY_ONLY 를 site 정합 88 프레임에서 평가.     B
  (SITE_ENVIRONMENT_AUDIT_V1,             **완료됨.** recording cluster 7 개 전부에서
   SITE_A_ARM_EVALUATION 88 frames)        구간이 0 을 포함 = 해소된 개선 없음
material / lighting 부분모집단           일반화 범위 서술.  선택 근거로 쓰지 않는다        B
ranking 결과(AUROC · FPR95)             claim C.  **이번에 구간을 채웠다** —              A 점추정
  + 이번에 계산한 부트스트랩 구간         G1 참조.  구간 자체는 사후 계산이므로 B          + B 구간
A8 day/night 부록                       lock 이 아니라 EXPERIMENTS.md 에 등록            A(단서)
DOPE M1 baseline row                    lock 15f0cb5 → result 8e6ccbc, lead 13h20m      A
```

## EXPLORATORY

```
결과                                    왜 EXPLORATORY 인가
────────────────────────────────────────────────────────────────────────────────────────
temporal 109 frames                     정식 모집단 계약 아래 적격 centre 가 **0 개**.
                                        FORMAL_TEMPORAL_PILOT = POPULATION_LIMITED.
                                        앞선 FAILED_TO_IMPROVE 는 lock 이 배제한
                                        모집단에서 계산된 값이라 정식 결과가 아니다
depth 진단(Gate 0 · 0B · 센서 검증)      FINAL = NOT_READY_FOR_GATE1.  4 조건 중 3 개는
                                        확실히 충족, 4 번째는 명세된 측정으로는 답할 수
                                        없다.  "depth 보정이 정확도를 올린다" 는 주장 없음
옛 DOPE multihead 진단                   합성 모집단 기반, 실제 전이 미확인
V2 · V3 · V4 · V5 선택 트랙              lock 이 스스로 DEV-INFORMED 라고 적었거나
                                        (V4) lock 과 result 가 같은 커밋이라 순서 불가
FAST / STRONG teacher probe             method lock 없음.  계약과 결과가 한 커밋
FAST_6D_SCREEN_V1  S1 · S3 · S4         POST-STOP 탐색.  전부 음성이고 S3 는 **정확히
FAST_6D_SCREEN_V1B C1 · L2 · L3 · L4    0 개** 선택을 바꿨다(=DOPE 병목이 공간 게이팅이
                                        아니라 peak 정밀도라는 음성 증거)
full-site 2,227 프레임 수량 확대 학습     NOT_RUN_AND_NOT_PLANNED.  기존 site-matched
                                        가설의 새 과학적 질문이 아니라 pseudo-label
                                        수량 확대에 가깝고, method search 종료 이후
                                        실행하지 않기로 결정했다
```

## EXCLUDE

```
결과                                    제외 사유
────────────────────────────────────────────────────────────────────────────────────────
AprilTag 기반 옛 GT(pallet11_gt 등)      gt_source 가 깨짐.  이미 폐기된 계보
temporal 의 정식 결과 표기               모집단이 0 이므로 표를 만들 수 없다.
                                        진단 서술로만 남긴다
V1 S2 의 full-cuboid bbox 해석           YOLO 가 학습한 semantics 가 아니다.
                                        V1B C1 이 이를 대체하고, C1 도 음성이다
tight-strip DOPE crop                    edge-on 팔레트를 wide strip 으로 만들어 붕괴
미검증 dimension 주장                     probe 가 keypoint 국소화가 아니라 classifier
                                        결과를 쟀다.  DIMENSION_HEADROOM = WEAK 까지만
v8 object-frame 계열 전체                 폐기된 좌표계로 학습된 데이터
REAL_FT_V1                              학습된 적 없음.  라벨 소스에 106/402 좌우 순서
                                        위반, 187/402 90도 순열.  Tier A 로 올릴 수 없다
REALFT_A / REALFT_B / REALFT_LV1V2       실제 감독으로 학습된 상한선이지 통제된 비교가
                                        아니다.  unlabeled adaptation arm 과 **같은 열
                                        블록에 절대 넣지 않는다**
```

### REALFT 를 한 번 더 짚어두는 이유

2D 표에서 유일하게 크게 움직인 것이 REALFT 다 —
`AP50-95 0.7688 → 0.8399`, `pooled keypoint median 6.616 → 5.990 px`.
자기-학습 계열 일곱 arm 중 어느 것도 이 근처에 오지 못했다.  이것은 논문의
서사를 **강화**한다(가짜 라벨의 신뢰도가 미세 국소화로 옮겨가지 않는다).
그러나 REALFT 는 같은 현장·같은 팔레트의 실제 라벨로 학습됐고 그 라벨 계보에는
문서화된 규약 문제가 있으므로, **상한선**으로만 인용하고 일반화 주장에 쓰지 않는다.
