==================================================
OVERNIGHT 6D DECISION
==================================================

START_HEAD
f0059ad9137fde795b624ad50965b7fca34d29c3
  fast6d: screen bbox and structural cues for full 6D pose

END_HEAD
f71a290  (V1B 결과 커밋)
  이 보고서 자체는 그 다음 CLOSURE 커밋에 들어간다 — 자기 해시를 담을 수 없어
  END_HEAD 는 측정이 끝난 시점의 head 로 적는다.  CLOSURE_COMMIT 은 터미널 요약 참조.


[V1B CONTRACT]

lock commit     6cb2b00   측정 전 단독 커밋, 원격 확인 완료
result commit   f71a290
LOCK_PRECEDES_RESULT = YES
status          POST_STOP_EXPLORATORY_CORRECTION
                EXPERIMENT_STOP_LOCK.json 수정 0
population      PAPER_EVAL positive 319 — 개발에 반복 사용된 셋
new training 0 · new checkpoint 0 · depth 0 · parameter sweep 0 · self-training 0


[C1 CORRECTED BBOX]

status   실행 완료, 예외 0, C1_UNRESOLVED_OBSERVABLE_BOX 0
N        319 (전 프레임에서 화면 안 코너 4 개 이상 확보)

```
arm                       R      Yaw    t cm     IoU3D    ADDsym   reproj
C0 frozen YOLO R0       2.262   1.231   7.897   0.6032   0.4285    3.55
C1 관측 semantics bbox   2.262   1.231   9.816   0.5372   0.3793    4.25
```

session CI   IoU3D  [-0.0903, -0.0233]  ★0 배제
             ADDsym [-0.0725, -0.0191]  ★0 배제

verdict  **STOP**
  ΔIoU3D -0.0660 · ΔADDsym -0.0492 — 둘 다 +0.020 게이트에 반대 방향
  translation 7.90 -> 9.82 cm (+24%) 로 5% 한도 초과

핵심   V1 이 **틀린** 상자(투영된 8 코너 전부)를 맞출 때 -0.0216 이었는데,
       YOLO 가 실제로 학습한 **맞는** 상자(화면 안 코너만)를 맞추니 -0.0660 이다.
       predicted reprojection 도 3.55 -> 4.25 px 로 같이 나빠졌으므로 목적함수
       과적합 함정이 아니라 bbox 와 keypoint 가 서로 다른 물체 위치를 말한다는 뜻.


[LINE FUSION]

checkpoint seed1   screen_A1_CORNER_LINE_FINAL40K_seed1/step_25000.pth
                   sha256 97e33a940ac41edd...
checkpoint seed2   screen_A1_CORNER_LINE_FINAL40K_seed2/step_25000.pth
                   sha256 0d138ed3b543abc0...
historical lambda  seed1 3.0 · seed2 1.0
                   theta_posealigned_d0.json 에서 읽음.  실행 시점에 lock 값과 대조
solver 상수        HUBER_PX 5.0 · MAX_NFEV 60 — canonical 구현 그대로

geometry parity    PASS, line_population = ALL
  P1 model point 생성기 두 개가 좌표까지 동일 (max diff 0.0)
  P2 GT pose 투영 vs 수동 어노 index-wise 중앙값
       plastic 0.94 px   (90도 재라벨 대조군 162.87 px)
       wood    0.45 px   (90도 재라벨 대조군 165.10 px)
  P3 0~3 이 near face 인 비율   plastic 1.000 · wood 1.000
  -> WOOD_LINE_STATUS = OK.  wood 를 억지로 섞은 것이 아니라 게이트를 통과했다

unit test status   LINE_FUSION_IMPLEMENTATION_GATE = PASS
  mh_fusion.run_tests 를 손대지 않고 OUT 만 우회해 실행.  T1~T7 이 과거 기록과
  **비트 동일**.  read-only artifact 무변경 확인

line cache         319/319 프레임, 예외 0, support 중앙값 12/12

```
seed1  lambda 3.0                R      Yaw    t cm     IoU3D    ADDsym   reproj
L0  frozen YOLO R0             2.262   1.231   7.897   0.6032   0.4285    3.55
L2  line rotation, t 고정       4.680   2.371   7.897   0.5434   0.3714    7.04
L3  line rotation + t refit    4.680   2.371   8.431   0.5132   0.3623    6.47
L4  yaw-only + t refit         3.048   2.035   7.891   0.6034   0.4194    4.51

seed2  lambda 1.0
L0  frozen YOLO R0             2.262   1.231   7.897   0.6032   0.4285    3.55
L2  line rotation, t 고정       2.718   1.358   7.897   0.5890   0.4234    3.96
L3  line rotation + t refit    2.718   1.358   8.106   0.5879   0.4234    3.91
L4  yaw-only + t refit         2.401   1.344   8.200   0.5965   0.4252    3.71
```

arm 이 실제로 움직였다 — 회전이 중앙값 3.084°(seed1) / 0.836°(seed2) 돌았다.
no-op 이 아니라 실재하는 음성 결과다.


[LINE PAIRED]   10,000 재표본 · seed 20260904 · 13 세션 클러스터

```
seed1                delta      frame CI95            session CI95
L2-L0  IoU3D       -0.0598   [-0.0745, -0.0243]   [-0.0922, -0.0119] ★
L2-L0  ADDsym      -0.0571   [-0.0708, -0.0445]   [-0.0980, -0.0366] ★
L3-L0  IoU3D       -0.0900   [-0.1176, -0.0515]   [-0.1346, -0.0460] ★
L3-L0  ADDsym      -0.0661   [-0.0838, -0.0494]   [-0.1112, -0.0388] ★
L3-L0  R deg       +2.418    (악화)
L3-L0  t cm        +0.534    (악화)
L4-L0  IoU3D       +0.0002   [-0.0098, +0.0252]   [-0.0098, +0.0253]
L4-L0  ADDsym      -0.0091   [-0.0173, -0.0015]   [-0.0212, -0.0004] ★

seed2
L2-L0  IoU3D       -0.0142   [-0.0218, +0.0040]   [-0.0179, +0.0072]
L2-L0  ADDsym      -0.0051   [-0.0070, -0.0033]   [-0.0114, -0.0028] ★
L3-L0  IoU3D       -0.0152   [-0.0253, +0.0085]   [-0.0258, +0.0097]
L3-L0  ADDsym      -0.0051   [-0.0093, -0.0011]   [-0.0139, -0.0001] ★
L3-L0  R deg       +0.456    (악화)
L3-L0  t cm        +0.209    (악화)
L4-L0  IoU3D       -0.0067   [-0.0149, +0.0112]   [-0.0116, +0.0132]
L4-L0  ADDsym      -0.0032   [-0.0062, -0.0004]   [-0.0062, -0.0000] ★
```

★ = session-cluster 구간이 0 을 배제.  양의 방향으로 배제된 것은 하나도 없다.


[MATERIAL]

```
                         n      R    t cm     IoU3D    ADDsym
plastic  C0            194   2.13   10.47    0.5857   0.3448
plastic  C1            194   2.13   13.34    0.5071   0.2837
plastic  L3 seed1      194   4.23   10.43    0.4953   0.2932
plastic  L3 seed2      194   2.52   10.58    0.5803   0.3430
wood     C0            125   2.98    4.20    0.6256   0.4230
wood     C1            125   2.98    5.21    0.6133   0.4016
wood     L3 seed1      125   5.63    5.15    0.5399   0.3223
wood     L3 seed2      125   3.28    4.01    0.5995   0.4094
```

wood 가 seed1 에서 가장 크게 무너진다.  line 모델의 학습 풀은 합성 BROAD 계열
(FINAL_SYNTH_TRAIN_V1, 40,000)이며 **wood 종횡비 노출량은 이번 작업에서 확인하지
않았다** — 그 미확인을 안고 읽어야 한다.


[LIGHTING]   manifest 의 기존 paper_domain 필드.  319 중 120 장만 라벨돼 있다

```
                         n      R    t cm     IoU3D    ADDsym
daytime    C0           70   2.48   11.03    0.5636   0.2903
daytime    C1           70   2.48   12.04    0.5031   0.2540
daytime    L3 seed1     70   3.85   11.05    0.4906   0.2667
daytime    L3 seed2     70   2.81   11.08    0.5455   0.2821
nighttime  C0           50   3.03   12.59    0.5324   0.2884
nighttime  C1           50   3.03   15.30    0.4306   0.2280
nighttime  L3 seed1     50   4.81   10.45    0.4591   0.2706
nighttime  L3 seed2     50   3.25   12.16    0.5208   0.2911
UNLABELLED             199   — 세션 이름으로 추측하지 않았다
```

어느 부분모집단에서도 양의 신호가 없다.  부분모집단은 선택에 쓰지 않았다.


[RUNTIME]

YOLO      미측정
line      미측정
hybrid    미측정

사전등록 trigger 미충족 — lock 은 "L3 가 게이트를 통과하거나 두 seed 모두 양의
방향" 일 때만 벤치마크를 돌린다.  두 seed 모두 음의 방향이므로 돌리지 않았다.
결과가 나쁜데 런타임을 재는 것은 선택 근거가 없는 계산이다.
RUNTIME_DONE = NO


[PROMOTION]

C1:  STOP
L3:  STOP        (두 seed 독립 평가, cherry-pick 없음)
L2:  STOP        (진단 대조)
L4:  STOP        (기제 진단 — 유일하게 중립, 아래 해석 참조)

CORRECTED_FAST6D = NO_PROMOTABLE_SIGNAL

PROMOTED_METHOD_CANDIDATE = 없음


[PAPER CONSEQUENCE]

negative 이므로 새 experiment search 를 종료하고 PAPER_FRAMING_CLOSURE_V1 을
자동 수행했다.  산출물은 `data/pallet/results/paper_framing_closure_v1/`.

```
PAPER_EVIDENCE_TIER.md        MAIN / SUPPORTING / EXPLORATORY / EXCLUDE 배정
PAPER_MAIN_CLAIMS.md          claim A~H 유지 + lock 이후 달라진 세 가지
PAPER_NO_CLAIMS.md            새로 금지되는 문장 여섯 개
PAPER_TABLE_PLAN.md           기존 생성 표의 배치와 필수 단서
PAPER_FIGURE_PLAN.md          기존 네 그림 유지 + 부록 그림 후보 하나
PAPER_REVIEWER_GAP_AUDIT.md   ★ 문서 간 불일치 하나 발견 — 아래
PAPER_FRAMING_DECISION.md     실험 종료 근거와 남은 사용자 결정 다섯 개
PAPER_STATIC_STAT_AUDIT.json  §29 정적 감사 G1~G5
```

★★ **가장 중요한 발견 — 사용자 결정이 필요하다**

```
_docs/paper/final/PAPER_CLAIM_LOCK.json   POSE_METRICS_STATUS = "BLOCKED"
_docs/paper/final/LIMITATIONS.md §3       "pose 열을 표에서 제거했다"
                                          "rotation, translation, yaw, ADD, ADD-S,
                                           3D IoU 에 대한 어떤 주장도 이 논문에
                                           나오지 않는다"
그런데
_docs/paper/final/generated/TABLE_FINAL_POSE.md   7 arm 의 6D 표가 이미 존재
POSE_CLOSURE_STATUS.json                          POSE_METRICS_STATUS = "REPORTABLE"
                                                  (second pass, 09-03)
```

현재 상태는 **두 번째 pass**가 맞다 — 차단 사유는 selector 가 아니라 GT 축의
부재였고, 결과를 보기 전에 얼린 규칙으로 geometry-resolved GT 를 만들면서
selector 를 전혀 건드리지 않고 6D 지표를 보고할 수 있게 됐다.

바뀌지 않는 것: `can_claim_6d_improvement = false`.
**24 개 session-cluster 구간이 전부 0 을 포함한다.**

claim lock 은 자율 작업이 편집하는 파일이 아니라 고치지 않았다.  선택은 사용자 몫:
  A  claim lock 과 LIMITATIONS §3 을 갱신하고 6D 표를 본문에 넣는다
  B  6D 표를 부록으로 내리고 §3 을 "차이가 갈리지 않는다" 로 다시 쓴다
어느 쪽이든 표가 생성돼 있는데 문서가 "제거했다" 고 말하는 지금 상태로 두지 말 것.

부수적으로 채운 공백: `LIMITATIONS.md` §8 이 "ranking 차이에 구간이 없다" 고
적은 것을 frozen per-frame 점수만으로 메웠다(새 추론 0).

```
paired R5 - R0   AUROC  +0.00318  [+0.00009, +0.00690]   0 배제
                 FPR95  -0.01339  [-0.02566, +0.00558]   0 포함
```
negative 행에 session_id 가 비어 있어 완전한 session-cluster 구간은 여전히
계산 불가(BLOCKED_MISSING_ARTIFACT) — lock 의 UNAVAILABLE 판단은 옳았다.
이 구간은 점추정을 본 뒤 계산됐으므로 Tier B 다.


[WHAT NOT TO DO NEXT]

```
line lambda 조정 (seed1 3.0 / seed2 1.0 을 건드리는 것 전부)
bbox margin · bbox weight · Huber scale · iteration 수 변경
line threshold sweep · DOPE padding 변경 · crop 비율 변경
seed cherry-pick, positive seed 만 headline 로 쓰기
learned line head, YOLO line head 신설
새 loss, RAFT, DiffPnP 변형, direct 3DoF 전환
새 student training · 새 self-training · 새 depth method · 새 temporal method
YOLO / DOPE architecture rewrite
REAL_FT_V1 실행 (라벨 소스에 106/402 좌우 순서 위반, 187/402 90도 순열)
PAPER_EVAL 을 보고 threshold·epoch·arm 을 고르는 모든 행위
결과를 본 뒤 더 나은 arm 을 Proposed 로 재지정하는 것
V1 artifact 수정 (읽기 전용)
```

이번 밤에 실제로 하지 않은 것: GPU 학습 0 · checkpoint 생성 0 · pseudo-label
생성 0 · threshold 조정 0 · 기존 실험 artifact 수정 0 (git 로 확인).


[NEXT ACTION]

USER_REVIEW_OVERNIGHT_RESULT
