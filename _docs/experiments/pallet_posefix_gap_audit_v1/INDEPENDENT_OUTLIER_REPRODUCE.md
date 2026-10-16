# 독립 아웃라이어 검산 보충

[독립 CLI](../../../scripts/research/pallet_posefix_gap_audit_v1/outlier_independent.py)는 원래의 저장된 single/guarded/PoseFix 점수에서 직접 재집계한다. 주 집계 코드나 해당 CSV를 import하지 않고, 표준 라이브러리의 평균·분산과 직접 구현한 선형 분위수로 NumPy 결과를 대조한다. 7개 공통 집합의 제외·유지 ID·수치·승패1304항목에서 불일치0이며 최대 수치 차이는1.71e−13이다.

```bash
python -m scripts.research.pallet_posefix_gap_audit_v1.outlier_independent \
  --source-root "$ORIGINAL_SOURCE_ROOT" \
  --single-score "$SINGLE_ROOT/score/SCORED_ROWS.json" \
  --guarded-score "$GUARDED_ROOT/score/SCORED_ROWS.json" \
  --observed _docs/experiments/pallet_posefix_gap_audit_v1/outliers/OUTLIER_RESULTS.json \
  --prior _docs/experiments/pallet_posefix_gap_audit_v1/evidence/INDEPENDENT_OUTLIERS_V1.json \
  --output "$NEW_INDEPENDENT_OUTLIER_JSON"
```

최초 v1 독립 결과는 그대로 보존하고, 모든 경로에 공통 IQR을 적용한121장 보충을 v2로 추가했다. 71장 제외로 최소 B/I의 전체 손상14건 중6건도 함께 빠졌다. 조건부 통계가 전체 후단의 안전성이나 실제 배포용 gate를 검증하는 값은 아니다.
