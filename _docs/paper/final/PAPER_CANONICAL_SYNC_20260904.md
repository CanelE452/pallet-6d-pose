# Paper canonical state — 2026-09-04

이 문서가 **현재 정본**이다.  다른 paper-facing 문서와 충돌하면 여기가 이긴다.
숫자는 전부 authoritative artifact 에서 읽었고, 산문에서 복사하지 않았다.

새 학습 0 · 새 checkpoint 0 · 새 추론 0 · 새 metric 정의 0 · 기존 result 숫자 수정 0.

## Canonical state

```text
PAPER_EVAL role                      DEV — repeatedly used development population
PAPER_EVAL positives                 319          plastic 194   wood 125
PAPER_EVAL negatives                 2,689
sessions / recording groups          13

POSE_METRICS_STATUS                  REPORTABLE
6D reference                         geometry-reconstructed 6D reference pose
  constructed from                   manual 2D cuboid keypoints
                                     + calibrated intrinsics
                                     + registered pallet dimensions
  model prediction chose the GT?     NO
symmetry group                       {I, Ry(180 deg)}

can_claim_6d_improvement             FALSE

experiment search                    STOPPED
new method candidate                 NONE
full-site training                   NOT_RUN_AND_NOT_PLANNED
site-matched small-arm evaluation    ALREADY_COMPLETED
wood pose                            INCLUDED under POSE_EVAL_OBJECT_CONTRACT
independent final confirmation       NOT AVAILABLE
```

## REPORTABLE 과 improvement 를 혼동하지 않는다

이 둘은 서로 다른 진술이다.

```text
REPORTABLE          6D 지표를 측정할 수 있고 표로 낼 수 있다
                    -> 그렇다.  얼린 규칙으로 만든 geometry-resolved reference 가
                       selector 를 전혀 바꾸지 않고 그 자리를 열었다

improvement         adaptation arm 이 기준선보다 6D 에서 낫다
                    -> 아니다.  24 개 metric block 중 개선 방향으로 session-cluster
                       구간이 0 을 배제한 것은 **0 개**다
```

## 무엇이 바뀌었고 무엇이 바뀌지 않았나

```text
first pass (historical)    POSE_METRICS_STATUS = BLOCKED
                           blocker 를 selector 로 진단했다
second pass (current)      POSE_METRICS_STATUS = REPORTABLE
                           실제 blocker 는 selector 가 아니라 **GT 물리축의 부재**였다.
                           결과를 보기 전에 얼린 규칙(GT_AXIS_RESOLUTION_LOCK)으로
                           reference 를 만들면서 selector 는 그대로 두었다
바뀌지 않은 것             can_claim_6d_improvement = false
                           axis selector 자체는 여전히 약하다(실측 0.59~0.65,
                           gate 0.95) — 그건 진단 결과로 남는다
```

historical first-pass 상태를 삭제하지 않는다.  `PAPER_CLAIM_LOCK.json` 의
`pose_metrics.historical_first_pass` 에 보존돼 있다.

## Authoritative source hierarchy

```text
1순위  data/pallet/results/paper_pose_metric_closure_v1/
         POSE_CLOSURE_STATUS.json · POSE_EVALUATION_*.json · POSE_PAIRED_BOOTSTRAP.json
       _docs/paper/final/generated/TABLE_FINAL_POSE.md
       _docs/paper/pose_metric_closure_v1/
         POSE_AXIS_ORACLE_DIAGNOSTIC.md · SITE_A_ARM_EVALUATION.md
2순위  data/pallet/results/paper_fast6d_screen_v1b/
       data/pallet/results/paper_framing_closure_v1/
       data/pallet/results/OVERNIGHT_6D_DECISION_20260904.md
3순위  _docs/paper/final/  (이 sync 의 대상)
```

## 중심 문장

> Pseudo-label reliability does not necessarily translate into fine geometric
> localisation or downstream 6D pose.

abstract · introduction contribution · main claims · discussion · conclusion 에서
**같은 의미**로 쓴다.  "왜" 는 측정된 사실이 아니라 해석이다.

## 서사 계층

```text
Synthetic-only estimator
        v
pseudo-label self-training
        v
Detection / ranking            변화가 있다 (R5 가 관측된 AUROC 최고)
        v
Fine 2D keypoint localisation  개선 없음 (R0 6.616 px 를 아무도 못 넘음)
        v
Downstream 6D pose             개선 없음 (개선 방향 session-cluster 해소 0/24)
```

앞 단계의 변화가 뒤 단계로 단조롭게 전파되지 않는다는 것이 이 연구의 결과다.

## PAPER_FRAMING

```text
CONTROLLED_EMPIRICAL_DIAGNOSTIC_STUDY
CENTRAL_FINDING =
  PSEUDO_LABEL_RELIABILITY_DOES_NOT_NECESSARILY_TRANSLATE_TO_GEOMETRIC_ACCURACY
```

NEXT_ACTION = PAPER_WRITING
