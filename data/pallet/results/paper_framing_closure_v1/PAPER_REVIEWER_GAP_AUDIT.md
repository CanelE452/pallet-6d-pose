# 리뷰어 관점 공백 감사

`data/pallet/results/paper_framing_closure_v1/PAPER_STATIC_STAT_AUDIT.json` 의
G1~G5 를 사람이 읽을 형태로 옮긴 것.  새 추론 0 · 새 학습 0 · 새 threshold 0.

## 1. ★ 최우선 — pose 상태가 문서 사이에서 어긋나 있다

리뷰어가 파일 두 개만 열어봐도 바로 걸리는 모순이다.

```
_docs/paper/final/PAPER_CLAIM_LOCK.json
    pose_metrics.POSE_METRICS_STATUS = "BLOCKED"
    can_claim_6d_improvement          = false
    blocked_quantities                = [yaw, rotation, translation, ADD, ADD-S,
                                         3D IoU, 5cm5deg, 6D pose AUC]

_docs/paper/final/LIMITATIONS.md §3
    "No claim about rotation, translation, yaw, ADD, ADD-S, 3D IoU, 5cm5deg, or
     6D pose AUC appears anywhere in this paper. These columns are removed ..."

그런데

_docs/paper/final/generated/TABLE_FINAL_POSE.md
    7 개 arm 의 PoseCov · AxisAcc · R med · Yaw med · t med · IoU3D · ADDsym AUC
    전체 표가 이미 생성돼 있다

data/pallet/results/paper_pose_metric_closure_v1/POSE_CLOSURE_STATUS.json
    POSE_METRICS_STATUS = "REPORTABLE"   (second pass, 2026-09-03)
    second_pass.what_changed = "the block was the absence of a ground-truth physical
    axis, not the selector..."
```

**무엇이 실제로 맞는가.**  두 번째 pass 가 현재 상태다.  차단 사유는 selector 가
아니라 GT 축의 부재였고, 결과를 보기 전에 얼린 규칙으로 geometry-resolved GT 를
만들면서 selector 를 **전혀 건드리지 않고** 6D 지표를 보고할 수 있게 됐다.
`POSE_CLOSURE_STATUS.json` 안의 `metrics_still_blocked` 필드는 first pass 잔재이며
같은 파일의 `POSE_METRICS_STATUS = REPORTABLE` 과 모순된다.

**무엇이 바뀌지 않는가.**  `can_claim_6d_improvement = false` 는 그대로 옳다.
24 개 session-cluster 구간이 전부 0 을 포함한다.

**왜 내가 고치지 않았는가.**  claim lock 은 자율 작업이 편집하는 파일이 아니다.
`EXPERIMENT_STOP_LOCK.json` 의 `allowed` 는 "typo and bug fixes that do not change
any reported metric" 까지만 허용한다.  BLOCKED → REPORTABLE 은 그 범위가 아니다.

```
ACTION = REQUIRES_USER_DECISION
선택 A   claim lock 과 LIMITATIONS §3 을 갱신해 6D 표를 본문에 넣는다
         (측정은 되고 차이는 안 갈린다 — 서사에 오히려 잘 맞는다)
선택 B   6D 표를 부록으로 내리고 §3 을 "차이가 갈리지 않는다" 로 다시 쓴다
어느 쪽이든  표가 생성돼 있는데 문서가 "제거했다" 고 말하는 지금 상태로 두지 말 것
```

## 2. ranking 구간 공백 — 채웠다

`LIMITATIONS.md` §8 이 "no matching interval in the artifacts" 라고 적은 공백을
frozen per-frame 점수만으로 메웠다(정의는 `evaluate_arms.ranking` 그대로).

```
paired R5 - R0   AUROC  +0.00318  [+0.00009, +0.00690]   0 배제
                 FPR95  -0.01339  [-0.02566, +0.00558]   0 포함
```

§8 은 갱신이 필요하다 — 다만 "구간이 없다" 를 "구간이 있고 AUROC 는 갈린다" 로
바꾸는 것이 아니라, **frame-level 구간은 이제 있고 session-clustered 구간은 여전히
negative 의 session 라벨 부재로 계산 불가**라고 정확히 적어야 한다.

## 3. 계산 불가로 남는 것 (BLOCKED_MISSING_ARTIFACT)

```
ranking 의 완전한 session-cluster 구간   negative 2,689 행의 session_id 가 전부 빈 값
필터 품질 지표의 신뢰구간                FILTER_SEPARABILITY.json 은 요약값만 담고
                                       항목별 배열이 없다.  구간을 내려면 분리도
                                       측정을 다시 돌려야 하는데 그것은 새 계산이다
temporal 의 정식 결과                   적격 centre 0 개.  규칙을 완화해 N 을 만들지 않았다
6D pose 의 독립 확증                     그런 모집단이 존재하지 않는다
```

## 4. 정합성 점검 — 통과

```
G2  main 2D 표      arm JSON 과 ARM_RESULTS 롤업이 8 개 arm × 4 필드에서 일치
G3  모집단 수        per-frame CSV positives 319 = axis manifest 319 = contract 319,
                    plastic 194 + wood 125 = 319, negatives 2,689
G4  6D pose 표       arm JSON · by-session · bootstrap 이 서로 일치.
                    단 하나의 불일치는 R0_CONT 이고 **설명된다** — R0_CONT 는 318
                    프레임에서만 짝지어지므로 그 비교 안의 R0 기준값이
                    0.60341(318) 이고 표의 0.60318(319) 과 다르다.  결함이 아니다
```

## 5. 리뷰어가 물을 것 — 미리 답을 준비해 둔다

```
Q  "개발용 셋에서만 재는데 왜 믿나"
A  믿으라고 하지 않는다.  이 논문의 결론은 **개선 주장이 아니라 개선 부재**이고,
   개발 셋에서 유리하게 조율할 자유를 다 쓰고도 개선이 안 나온 것은 개발 셋의
   낙관 편향이 결론을 돕지 않는 방향이다.  단 confirmatory 라고는 부르지 않는다

Q  "왜 실제 GT 로 학습하지 않았나 — REALFT 가 훨씬 좋은데"
A  했다(REALFT_A/B).  그리고 그것이 논문의 논점을 강화한다.  다만 상한선이지
   통제된 비교가 아니고, 후속 REAL_FT_V1 은 라벨 규약 감사에서 106/402 좌우 순서
   위반과 187/402 90도 순열이 나와 학습 전에 중단됐다

Q  "필터가 라벨 품질을 올리는데 왜 student 가 안 좋아지나"
A  그것이 논문의 결과다.  해석(teacher 품질 병목)은 모든 진단과 일관되지만
   측정된 양이 아니라고 명시한다

Q  "6D pose 는 왜 없나 / 왜 있나"
A  §1 참조.  사용자 결정 대기 중이다

Q  "seed 하나로 낸 결론 아닌가"
A  line 트랙은 두 seed 를 독립 평가했고 둘 다 음성이다.  V1~V5 는 대체로 단일
   seed 이며 LIMITATIONS §12 에 그렇게 적혀 있다
```
