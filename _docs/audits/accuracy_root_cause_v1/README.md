# accuracy_root_cause_v1 — 실제 정확도의 binding constraint 확정

착수 2026-09-06. 문서들의 HEAD 표기는 **`2e5ec0e`** 다(감사 착수 시점).

⚠️ 감사 진행 중 다른 세션이 `71ef40b` ("differentiable PnP loses to SQPnP on all three
sides") 을 커밋해 HEAD 가 한 칸 전진했다. 그 커밋의 내용은 감사 착수 시점에 **이미 작업
트리에 있던** `diffpnp_yolo_v1` · `solver_swap_v1` 작업이고, 이 감사는 그것을 읽고
`PNP_NOT_PRIMARY_LEVER` 판정에 반영했다. 따라서 **판정은 영향받지 않는다.**

## 목적 (한 줄)

«현재 실제 정확도를 막는 가장 큰 원인은 무엇이며, 다음 한 번의 실험 예산을 어디에 써야 하는가?»
에 답한다. 새 method 를 만드는 것이 목적이 아니다.

## 판정 지표

최종 선택지 A~I 중 하나를 **직접 증거와 함께** 지목한다.

```
A GT / annotation semantics 수정
B evaluation / physical geometry contract 수정
C 기존 point model 유지 + inference/PnP/selection 개선
D 모델 capacity/architecture 변경
E 실제 image evidence 를 쓰는 line/edge/voting representation
F synthetic 데이터의 특정 분포만 수정
G hard negative / distractor 데이터 수정
H real labeled supervision 또는 별도 adaptation 전략
I 개선 근거 부족 — 추가 실험 중단
```

"데이터를 더 만들자" 는 결론 금지. 어떤 데이터의 어떤 조건을 왜 바꾸는지까지
특정되지 않으면 데이터 변경 결론을 내지 않는다.

## 이 namespace 의 경계

새로 만드는 것 (3곳):

```
_docs/audits/accuracy_root_cause_v1/      보고서
data/pallet/results/accuracy_root_cause_v1/  machine-readable 산출물
scripts/research/accuracy_root_cause_v1/     새 코드
```

건드리지 않는 것: `_docs/paper/final/`, `_docs/paper/final/generated/`,
`data/pallet/results/paper_*`, `challenge/yolo_pose_one_model/` 기존 실험,
모든 기존 GT JSON, `challenge/config/` 의 geometry registry.

## 상태

| 단계 | 산출물 | 상태 |
|---|---|---|
| 과거 실험 taxonomy (논문 트랙) | PRIOR_EXPERIMENT_MAP_paper.md | 완료 (73실험) |
| 과거 실험 taxonomy (과제 트랙) | PRIOR_EXPERIMENT_MAP_challenge.md | 완료 |
| GT 신뢰 감사 | GT_TRUST_AUDIT.md | 완료 (메인 재현 완료) |
| artifact 재사용 인벤토리 | ARTIFACT_REUSE_INVENTORY.md | 완료 |
| capacity · real supervision | CAPACITY_AND_REAL_SUPERVISION_AUDIT.md | 완료 |
| Hough 구현 감사 | HOUGH_IMPLEMENTATION_AUDIT.md | 완료 |
| 배포 기하 계약 감사 | DEPLOYMENT_GEOMETRY_AUDIT.md | 완료 (부록 재검증) |
| 실패 계층 분해 | FAILURE_DECOMPOSITION.md | 완료 |
| 헤드룸 감사 | MODEL_HEADROOM_AUDIT.md | 완료 |
| selective risk | SELECTIVE_RISK_AUDIT.md | 완료 |
| line 관측성 | LINE_OBSERVABILITY_AUDIT.md | 완료 (NOT_RUN 사유 기록) |
| source-real gap | SOURCE_REAL_GAP_AUDIT.md | 완료 |
| GT semantics 리뷰 계획 | GT_SEMANTICS_REVIEW_PLAN.md | 완료 |
| GT 사람 리뷰 결과 1차 | GT_REVIEW_RESULT.md | 완료 (54/54 응답) |
| GT 사람 리뷰 결과 2·3차 | GT_REVIEW_RESULT_PHASE23.md | 완료 (코너 71 · 재리뷰 20) |
| 불확실성 (§29) | UNCERTAINTY.md | 완료 (주장 1건 철회) |
| real 라벨 감사 | REAL_LABEL_AUDIT.md | 완료 (설계 변경 유발) |
| 오검출 육안 검수 (§19) | HARD_NEGATIVE_REVIEW.md | 완료 (79셀 전수) |
| 6D 조건별 표 (가림·앙각) | POSE_BY_CONDITION.md | 완료 (MAIN 재현 확인) |
| 최종 판정 | FINAL_DECISION.md | **완료 (rev2)** |

## 규칙

- 모든 사실 문장에 `[확인]` / `[추정]` 태그. 새 gate·threshold 는 `[추정][미검증]`.
- 여기서 나온 모든 수치는 **DEVELOPMENT / POST-HOC DIAGNOSTIC** 이다.
  held-out / final / confirmed / SOTA 라고 부르지 않는다.
- 게이트를 통과하기 전에는 장시간 학습·synthetic 생성·self-training 재실행을 하지 않는다.

## 결론 (2026-09-06)

```
PRIMARY_ROOT_CAUSE = 저앙각(edge-on) 레짐에서 보이는 코너의 위치추정 실패
NEXT_EXPERIMENT    = real supervision 의 구성(composition) ablation
                     — <8도만 / 8-15도만 학습해 두 층에 전이시키는 2x2 (정사각 물체 단일)
BLOCKER            = 정본 keypoint 필드 확정. live_capture_gt 의 46.4% 에서
                     keypoint_annotations 와 manual_kps 가 정확히 90도 어긋난다
DO_NOT_RUN         = line·Hough 학습 / PnP solver 탐색 / self-training 재실행 /
                     hard negative 추가 / synthetic 장수 증가 / threshold sweep
```

상세와 근거는 `FINAL_DECISION.md`.
