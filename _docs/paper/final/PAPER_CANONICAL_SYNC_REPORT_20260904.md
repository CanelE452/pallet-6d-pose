# PAPER CANONICAL SYNC REPORT — 2026-09-04

새 학습 0 · 새 checkpoint 0 · 새 추론 0 · 새 metric 정의 0 · 기존 result 숫자 수정 0.
모든 숫자는 authoritative artifact 에서 읽었고 산문에서 복사하지 않았다.
출처는 `PAPER_CANONICAL_NUMBER_SOURCES.json` 에 파일·키 단위로 남겼다.

## [PAPER CANONICAL STATE]

```text
PAPER_EVAL              319 positive / 2,689 negative
                        plastic 194 · wood 125 · 13 recording groups
                        role DEV — repeatedly used development population
                        not held-out, not independently confirmed

POSE                    REPORTABLE
6D improvement claim     NO
```

## [6D MAIN]

`POSE_EVALUATION_*.json` · `paths.MAIN.ALL`

```text
arm                             PoseCov  AxisAcc   R med   Yaw med   t med cm     IoU3D  ADDsym AUC
───────────────────────────────────────────────────────────────────────────────────────────────────
R0  synthetic-only                1.000   0.7492  2.2625    1.2306      7.8969   0.60318     0.42847
R0_CONT source-only continuation  0.997   0.7264  2.4994    1.2612      8.5438   0.59432     0.40915
R1  naive                         1.000   0.7680  2.3723    1.2453      7.7955   0.58966     0.42045
R2  confidence                    1.000   0.7241  2.4765    1.3262      7.7847   0.59947     0.41580
R3  + reprojection                1.000   0.7492  2.3488    1.2070      7.7365   0.59979     0.41489
R4  + keypoint removal            1.000   0.7273  2.5242    1.2764      8.0549   0.59966     0.41205
R5  full consistency              1.000   0.7335  2.5345    1.2938      8.8265   0.58677     0.40010
```

R0_CONT solves 318 of 319 frames; every other arm solves all 319.  숨기지 않는다.

```text
comparisons                                                    6
metric blocks                                                 24
session-cluster intervals excluding zero (any direction)       0
session-cluster intervals excluding zero (improvement)         0
frame-level intervals excluding zero                           2   (both negative)
```

허용 문장:

> No self-training variant showed a session-cluster-resolved improvement in
> downstream 6D pose over the synthetic-only baseline.

금지: "No difference exists." · "Self-training statistically worsens all 6D
metrics." · "All 6D components degrade."  13 개 클러스터에서 구간이 0 을 포함한다는
것은 **가를 힘이 없다**는 뜻이지 같다는 뜻이 아니다.

## [RANKING]

```text
R0  AUROC   0.992131    frame CI95 [0.988118, 0.995351]
R5  AUROC   0.995311    frame CI95 [0.993409, 0.996950]
R0  FPR95   0.041651    frame CI95 [0.025660, 0.051320]
R5  FPR95   0.028263    frame CI95 [0.015247, 0.043882]

paired R5 - R0   AUROC  +0.003180   frame CI95 [+0.000092, +0.006898]   excludes zero
paired R5 - R0   FPR95  -0.013388   frame CI95 [-0.025660, +0.005578]   contains zero

session-cluster ranking CI    UNAVAILABLE
                              negative rows carry no session identifier
```

허용 문장:

> The full consistency variant achieved the highest observed AUROC.  Its paired
> frame-level AUROC difference relative to R0 was positive, while session-clustered
> uncertainty for the ranking metric could not be estimated for the negative
> population.

금지: "statistically confirmed ranking improvement" ·
"session-level significant ranking improvement".
이 구간은 결과를 본 뒤 계산됐다 — 점추정은 Tier A, 구간은 Tier B 다.

## [2D]

`arms/*.json` · `metrics.box_and_keypoint_2d`

```text
arm                 AP50-95    AP50   pooled kp median [px]   p90 [px]
──────────────────────────────────────────────────────────────────────
R0                   0.7688  0.9363                  6.6157      38.67
R4  best adapted     0.7578  0.9366                  6.9987      39.34
R5  full consistency 0.7585  0.9580                  7.2099      41.38
```

허용 문장:

> Self-training did not improve fine 2D keypoint localisation over the
> synthetic-only baseline.

frame-level 로는 방향이 갈리고 session-cluster 로는 갈리지 않는다는 단서를 함께
유지한다.  "self-training harms localisation" 처럼 무조건 표현은 쓰지 않는다.

## [SITE]

```text
A8 site-matched         ALREADY EVALUATED
population              SITE_A_EVAL_ELIGIBLE, N = 88, 7 recording clusters
                        (SITE_A_EVAL_POSITIVE 100 행 중 FT_OVERLAP 12 행 제거)

arm            kp med px    IoU3D    ADDsym AUC   AxisAcc
─────────────────────────────────────────────────────────
R0               11.7106  0.60919       0.39523    0.5909
A8_DAY_ONLY      12.3334  0.57177       0.36962    0.5455

A8 - R0        delta        cluster CI95            excludes zero
IoU3D         -0.03742   [-0.12289, +0.03140]            no
ADDsym        -0.02561   [-0.04558, +0.03268]            no
yaw           +0.53014   [-0.22477, +1.17664]            no
translation   +0.79105   [-0.79872, +10.26516]           no

result                  NO RESOLVED IMPROVEMENT
full-site training      NOT RUN / NOT PLANNED
```

허용 문장:

> A previously trained day-only site-matched arm showed no resolved improvement on
> the corrected site-specific evaluation population.

"유일하게 남은 실험" 표현은 폐기했다.  2,227 프레임 확대는 새 과학적 질문이 아니라
pseudo-label 수량 확대이고, method search 종료 이후 실행하지 않는다.

## [WOOD]

```text
pose included    YES
plastic          194
wood             125
ALL              319
contract         POSE_EVAL_OBJECT_CONTRACT.json
                 sha256 a4c2918b4b0e9c97f94332d2e7e35132a8cbe0e738db25d92ea55e0d81210dbd
registry rewrite NOT PERFORMED — OBJECT_GEOMETRY_REGISTRY.json 은 건드리지 않았다
```

객체 범주 표현: *constrained pallet category with rectangular load-bearing and
fork-access topology, while dimensions, material and local structure vary.*
arbitrary / unseen pallet generalisation 은 주장하지 않는다.

## [EXPLORATORY TRACKS]

```text
FAST6D V1 / V1B   NO PROMOTABLE SIGNAL — main quantitative table 에 넣지 않는다
                  bbox translation constraint 는 R0 를 개선하지 못했다
                  bbox-gated DOPE 는 peak 선택을 하나도 바꾸지 않았다
                  square-context DOPE 는 악화됐다
                  두 historical seed-specific line-fusion 구성 모두 강한 YOLO
                  기준선을 개선하지 못했고, 더 큰 악화는 seed-1 구성에서 나타났다
                  ★ seed 와 lambda 가 confounded 라 "lambda 탓" 이라고 쓰지 않는다
temporal          POPULATION_LIMITED — 적격 centre 0.  109 프레임 결과는
                  EXPLORATORY_DIAGNOSTIC_ONLY.  성능 개선/악화를 공식 주장하지 않는다
depth             main method 에서 제거.  Gate 1 은 실행되지 않았고
                  depth-assisted adaptation 을 제안 방법처럼 쓰지 않는다
```

## [CLAIMS UPDATED]

```text
claim C   ranking 에 frame-level paired 구간이 생겼다는 amendment 추가.
          session-cluster 는 여전히 계산 불가라는 사실도 함께 박았다
claim I   신설 — 6D 에서 session-cluster 로 해소된 개선이 없다
forbidden 추가: self-training improves 6D pose / confirmed 6D improvement /
          held-out 6D improvement / our line fusion improves pose /
          the YOLO bbox constraint improves translation / the temporal method
          failed / depth-assisted monocular adaptation
forbidden 제거: "we evaluate pose accuracy" — pose 층이 REPORTABLE 이므로
          평가 자체는 허용된다.  금지된 것은 개선 주장이다
newly permitted:
          "We evaluate downstream 6D pose using a geometry-reconstructed reference pose."
          "We report rotation, yaw, translation, oriented 3D IoU, and symmetry-aware ADD AUC."
```

## [STALE STATES FIXED]

`PAPER_CANONICAL_SYNC_AUDIT.json` 에 24 건이 기록돼 있다.  요약:

```text
POSE_METRICS_STATUS = BLOCKED        8 개 활성 문서에서 REPORTABLE 로
"pose 열을 제거했다"                  "6D 는 별도 pose 표에 있다" 로
"we evaluate pose accuracy" 금지      해제 (개선 주장 금지는 유지)
ranking 구간 없음                     frame-level 구간 있음 + 이유 있는 UNAVAILABLE 로
site-matched 승인 대기                이미 평가됨 + full-site 는 계획 없음
wood pose 결정 필요                   wood 125 포함 확정
"lambda 가 클수록 나빠진다"            seed 와 lambda confounded — per-seed 표현으로
"ground truth"                       geometry-reconstructed 6D reference pose 로
D1~D5 미결정                          전부 RESOLVED
```

historical / archive / result artifact 내부는 자동 수정하지 않았다.
`PAPER_CLAIM_LOCK.json` 의 first-pass BLOCKED 는 `pose_metrics.historical_first_pass`
에 **원문 그대로 보존**돼 있다.

## [FILES MODIFIED]

```text
새로 만든 것
  _docs/paper/final/PAPER_CANONICAL_SYNC_20260904.md
  _docs/paper/final/PAPER_CANONICAL_SYNC_REPORT_20260904.md
  _docs/paper/final/PAPER_CANONICAL_SYNC_AUDIT.json
  _docs/paper/final/PAPER_CANONICAL_SYNC_TESTS.json
  _docs/paper/final/PAPER_CANONICAL_NUMBER_SOURCES.json
  scripts/paper/framing_closure_v1/canonical_sync_audit.py

동기화한 문서
  _docs/paper/final/  PAPER_CLAIM_LOCK.json · PAPER_CLAIM_LOCK.md · LIMITATIONS.md
                      METRIC_NAMING_LOCK.md · DISCUSSION.md · ABSTRACT_DRAFT.md
                      FINAL_ABSTRACT_RESULT_SLOTS.md · INTRODUCTION_STORY.md
                      METHOD_OUTLINE.md · TITLE_CANDIDATES.md · RESULTS_STORY.md
                      CONTRIBUTIONS.md · FIGURE_PLAN.md
  _docs/paper/EXPERIMENTS.md
  data/pallet/results/paper_framing_closure_v1/  PAPER_MAIN_CLAIMS.md
                      PAPER_NO_CLAIMS.md · PAPER_EVIDENCE_TIER.md
                      PAPER_FRAMING_DECISION.md · PAPER_REVIEWER_GAP_AUDIT.md
                      PAPER_TABLE_PLAN.md · PAPER_FIGURE_PLAN.md

generator 수정 후 재생성
  scripts/paper/build_final_paper_summary.py        -> TABLE_FINAL_1.md
  scripts/paper/pose_metric_closure_v1/build_pose_tables.py
                                                    -> TABLE_FINAL_POSE*.md
                                                       POSE_AXIS_ORACLE_DIAGNOSTIC.md
  scripts/paper/build_experiment_tables.py          (출력이 _docs/archive 라 재생성 안 함)
```

재생성 검증: `build_final_paper_summary` 265 개 숫자 전부 동일, 바뀐 것은 캡션 한 문단뿐.
pose 표 세 개와 oracle 진단의 **숫자 행은 전부 동일**, 바뀐 것은 헤더·주석·타임스탬프.

## [TESTS]

`PAPER_CANONICAL_SYNC_TESTS.json` — 12 / 12 PASS.

```text
all_paper_json_parses                          PASS
claim_lock_pose_reportable                     PASS
claim_lock_no_6d_improvement                   PASS
claim_lock_history_preserved                   PASS
population_319_equals_194_plus_125             PASS
pose_table_matches_source_json                 PASS
zero_session_cluster_resolved_6d_improvements  PASS
ranking_auroc_delta_positive_and_excludes_zero PASS
ranking_fpr95_delta_contains_zero              PASS
site_matched_no_resolved_improvement           PASS
referenced_paths_exist                         PASS
no_active_stale_phrase                         PASS   (NEEDS_USER = 0)
```

## [SAFETY]

```text
training                        0
inference                       0
new model                       0
new metric definition           0
historical result modification  0
```

읽기 전용 namespace(paper_selftrain_v1~v5 · paper_pose_metric_closure_v1 ·
paper_fast6d_screen_v1 · v1b · paper_temporal_selftrain_v1 ·
paper_depth_selftrain_v1 · weights) 는 git 기준 무변경이다.

## [FINAL CANONICAL DECISION]

```text
EXPERIMENTATION_STATUS   STOPPED
POSE_METRICS_STATUS      REPORTABLE
CAN_CLAIM_6D_IMPROVEMENT FALSE
FULL_SITE_TRAINING       NO
NEW_METHOD_CANDIDATE     NONE
PAPER_FRAMING            CONTROLLED_EMPIRICAL_DIAGNOSTIC_STUDY
CENTRAL_FINDING          PSEUDO_LABEL_RELIABILITY_DOES_NOT_NECESSARILY_TRANSLATE_
                         TO_GEOMETRIC_ACCURACY
```

## [NEXT ACTION]

PAPER_WRITING

---

# FINAL TEXT CLOSURE — 2026-09-04

canonical sync(`63fd78d`) 이후 재감사에서 **활성 문서 네 곳**이 아직 어긋나 있었다.
canonical state 자체는 이미 맞았고, 남은 것은 산문이었다.  이번 작업은 그 네 개만
고친 **text-only closure** 다.  새 실험·새 통계·새 추론 0.

## 무엇이 어긋나 있었나, 그리고 왜 이전 감사가 못 잡았나

이전 감사는 문자열 검색이었다.  네 건 모두 **있어야 할 문장이 없는** 결함이라
검색으로는 잡히지 않았고, "6D pose" 라는 단어가 note 나 omission 표에만 있어도
있는 것처럼 보였다.  그래서 이번에는 본문 범위를 파싱하는 테스트를 넣었다.

```text
Abstract downstream 6D                       FIXED
  본문이 detection/ranking -> 2D 까지만 말하고 하류 6D 결과가 없었다.
  본문에 넣었다: 같은 arm 을 하류 pose 까지 끌고 갔고, geometry-reconstructed
  6D reference 기준으로 어떤 variant 도 session-cluster 로 해소된 개선을
  보이지 않는다.  결론 문장도 2D 와 6D 를 함께 말하도록 확장했다.
  중복을 압축해 297 -> 310 words (목표 280~310), 검증 숫자 10 개는 전부 유지.

Abstract ranking caveat                      FIXED
  "The ranking metrics carry no bootstrap interval in the artifacts" 는
  현재 사실이 아니다.  frame-level paired 구간은 존재하고(AUROC +0.00318,
  CI [+0.000092, +0.006898], 0 배제 / FPR95 는 0 포함), 없는 것은
  session-cluster 구간이다(negative 행에 session_id 없음).
  Tier B 라는 점과 함께 다시 썼고, 과거 상태는 "Historical:" 로 남겼다.

Contributions four-layer hierarchy           FIXED
  C1 이 detection / ranking / 2D localisation 세 축뿐이었다.
  네 층으로 바꿨다 — detection coverage · confidence ranking ·
  fine 2D keypoint localisation · downstream 6D pose(geometry-reconstructed
  6D reference pose 기준).  C3 결론도 label quality -> 2D -> 6D 를 잇도록 했다.
  contribution 개수는 3 개 그대로다 — 네 번째를 만들지 않았다.

Claim C internal contradiction               FIXED
  같은 object 안에서 note 는 "구간이 없다", amended_20260904 는 "구간이 있다"
  라고 말하고 있었다.  현재 note 를 최신 상태로 다시 쓰고, 원문은
  historical_note_before_20260904 에 보존했다.  prose twin 도 같은 의미로 맞췄다.

layer count 불일치 (부수)                     FIXED
  METHOD_OUTLINE 이 "three layers", CONTRIBUTIONS 가 "four layers" 였다.
  네 축으로 통일했다.
```

## 새 정적 테스트 — 단어 검색이 아니라 본문 범위 검사

```text
T13  abstract_mentions_downstream_6d          ## Abstract 의 blockquote 본문만 본다.
                                              note·omission 표에 있는 "6D" 는 세지 않는다
T14  abstract_ranking_note_not_stale          현재형 "no bootstrap interval" 금지,
                                              session-cluster 불가 언급을 요구
T15  contributions_has_four_layer_hierarchy   ## C1 본문 범위만 본다.
                                              "deliberately not claimed" 목록은 제외
T16  claim_c_current_note_matches_amendment   현재 note 와 amendment 가 모순되지 않을 것.
                                              historical note 는 허용
```

**음성 대조로 검증했다.**  `63fd78d` 시점 파일로 되돌리면 네 테스트가 모두 FAIL 하고
(11/16), 수정본에서는 모두 PASS 한다(16/16).  통과만 확인하고 넘어가지 않았다.

## 결과

```text
active stale issues     0        (NEEDS_USER = 0)
tests                   16/16 PASS
result numbers changed  0        generated/ 무변경, 재생성 없음
result artifacts        0        read-only namespace 12 곳 git 무변경
training / inference    0 / 0
```

## 남은 두 개는 데이터의 성질이지 미결정이 아니다

```text
BLOCKED_MISSING_ARTIFACT   ranking 의 완전한 session-cluster 구간
                           negative 2,689 행에 session_id 가 없다
BLOCKED_MISSING_ARTIFACT   필터 품질 지표의 신뢰구간
                           FILTER_SEPARABILITY.json 에 항목별 배열이 없다
```

## 두 문장을 혼동하지 않는다

```text
measured result   "No self-training variant showed a session-cluster-resolved
                   improvement in downstream 6D pose over the synthetic-only
                   baseline."
interpretation    "Pseudo-label reliability does not necessarily translate into
                   fine geometric localisation or downstream 6D pose."
```

첫째를 기제 주장처럼 쓰지 않고, 둘째를 인과 증명처럼 쓰지 않는다.

```text
PAPER_CANONICAL_SYNC_STATUS = COMPLETE
NEXT_ACTION                 = PAPER_WRITING
```
