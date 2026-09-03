# 리뷰어 관점 공백 감사

`data/pallet/results/paper_framing_closure_v1/PAPER_STATIC_STAT_AUDIT.json` 의
G1~G5 를 사람이 읽을 형태로 옮긴 것.  새 추론 0 · 새 학습 0 · 새 threshold 0.

## 1. pose 상태 불일치 — RESOLVED (2026-09-04)

발견 당시 상태:

```
PAPER_CLAIM_LOCK.json      POSE_METRICS_STATUS = "BLOCKED"
LIMITATIONS.md §3          "pose 열을 표에서 제거했다"
그런데
generated/TABLE_FINAL_POSE.md   7 arm 6D 표가 이미 존재
POSE_CLOSURE_STATUS.json        POSE_METRICS_STATUS = "REPORTABLE" (second pass)
```

**조치.**  canonical sync 로 정리했다.  두 번째 pass 가 현재 상태다 — 차단 사유는
selector 가 아니라 **GT 물리축의 부재**였고, 결과를 보기 전에 얼린 규칙으로
geometry-reconstructed reference 를 만들면서 selector 를 전혀 건드리지 않고
6D 지표를 열었다.

```
PAPER_CLAIM_LOCK.json          pose_metrics 를 amendment.
                               POSE_METRICS_STATUS = REPORTABLE
                               can_claim_6d_improvement = false
                               historical_first_pass 에 BLOCKED 원문 보존 (삭제 아님)
PAPER_CLAIM_LOCK.md            prose twin 동기화
LIMITATIONS.md §3              "블록됨" -> "reference 는 재구성된 것이지 센서 GT 가
                               아니다" 로 재작성.  개선 부재는 그대로 유지
METRIC_NAMING_LOCK.md          6D 항목을 BLOCKED -> REPORTABLE, 이름 고정
DISCUSSION / ABSTRACT_DRAFT /
FINAL_ABSTRACT_RESULT_SLOTS /
INTRODUCTION_STORY / METHOD_OUTLINE /
TITLE_CANDIDATES / RESULTS_STORY /
CONTRIBUTIONS                  같은 취지로 동기화
TABLE_FINAL_1.md               generator 수정 후 재생성.  "pose 열은 BLOCKED 라
                               없다" -> "6D 는 별도 pose 표에 있다"
```

**바뀌지 않은 것.**  `can_claim_6d_improvement = false`.
개선 방향으로 session-cluster 구간이 0 을 배제한 metric block 은 **0 / 24** 다.

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

Q  "6D pose 는 왜 있나 — 전에는 블록이라고 하지 않았나"
A  §1 참조.  첫 진단이 blocker 를 selector 로 잘못 짚었고, 실제 blocker 는 GT 축의
   부재였다.  얼린 규칙으로 reference 를 만들면서 selector 를 건드리지 않고 열렸다.
   과거 BLOCKED 기록은 claim lock 안에 그대로 보존돼 있다.  그리고 표가 생겼다고
   개선을 주장하지는 않는다 — 0/24 다

Q  "site-matched 는 왜 안 했나"
A  했다.  A8_DAY_ONLY 를 site 정합 88 프레임(recording cluster 7)에서 평가했고
   네 지표 모두 구간이 0 을 포함한다.  2,227 프레임 수량 확대 학습은 새 과학적
   질문이 아니라 수량 확대라서 method search 종료 이후 하지 않기로 했다

Q  "seed 하나로 낸 결론 아닌가"
A  line 트랙은 두 seed 를 독립 평가했고 둘 다 음성이다.  V1~V5 는 대체로 단일
   seed 이며 LIMITATIONS §12 에 그렇게 적혀 있다
```
