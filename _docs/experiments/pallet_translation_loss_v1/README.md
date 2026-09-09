# pallet_translation_loss_v1 — Stage 0

목적 한 줄: R0 의 실제 translation 7.90 cm / depth 7.52 cm 를 줄이는 학습 신호를
찾되, **GPU 를 쓰기 전에** 그 신호가 실재하는지부터 검증한다.

현재 상태: **Stage 0 완료. GPU 학습 미착수. 사용자 승인 대기.**

## 문서

```
PURPOSE_TREE.md               목적 트리와 [소비처]/[문장]
PRIOR_WORK.md                 Linear-Covariance(ICCV23) · COPE(WACV23) 원문 정리
SYNTHETIC_DIMENSION_AUDIT.md  R0 가 실제로 학습한 치수 전수 (§6,§7)
LOSS_SYMMETRY_CONTRACT.json   asset 별 최대 대칭 차수 계약 (§8)
GEOMETRY_METADATA_AUDIT.md    K/pose/dims/corner 커버리지 (§10)
TRANSLATION_SURROGATE_AUDIT.md  surrogate 가 실제 depth 를 설명하는가 (§11-18)
LOSS_DESIGN.md                L_LC / L_TR 수식과 기존 연구와의 정확한 관계 (§14-17)
METHOD_LOCK_DRAFT.md          Stage A 사전등록 초안 (§26-35)
```

산출 JSON 은 `data/pallet/results/pallet_translation_loss_v1/`,
코드는 `scripts/research/pallet_translation_loss_v1/` 에 있다.

## 이 track 이 기존 판정과 충돌하는 지점

`_docs/audits/accuracy_root_cause_v1/FINAL_DECISION.md` 의 `DO_NOT_RUN` 목록에
**"새 loss 항"** 이 들어 있다 (2026-09-06, 같은 날). Stage 0 자체는 read-only 라
그 목록을 위반하지 않지만, Stage A 는 위반한다. 충돌은 봉합하지 않고 그대로
사용자 판단으로 올린다 — 근거는 `TRANSLATION_SURROGATE_AUDIT.md` 의 잔차-대조군이다.
