# 재현 방법

공개된238행 저장 점수만으로 아웃라이어 분석을 재현할 수 있다. 새 학습·모델 추론·PnP·정답 fit은 수행하지 않는다. 모든 출력 경로는 아직 존재하지 않는 폴더로 지정한다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.research.pallet_posefix_gap_audit_v1.outlier \
  --summary _docs/experiments/pallet_posefix_gap_audit_v1/summary \
  --output "$NEW_OUTLIER_OUTPUT"
```

원본 캐시와 같은 score seal이 있는 로컬 자료에서는 전체 PoseFix 비교와 독립 검산도 재현한다.

```bash
python -m scripts.research.pallet_posefix_gap_audit_v1.audit \
  --source-root "$ORIGINAL_SOURCE_ROOT" \
  --single-score "$SINGLE_ROOT/score/SCORED_ROWS.json" \
  --guarded-score "$GUARDED_ROOT/score/SCORED_ROWS.json" \
  --output "$NEW_SUMMARY_OUTPUT"
python -m scripts.research.pallet_posefix_gap_audit_v1.independent \
  --source-root "$ORIGINAL_SOURCE_ROOT" \
  --minimal-root "$MINIMAL_ROOT" \
  --single-root "$SINGLE_ROOT" \
  --guarded-root "$GUARDED_ROOT" \
  --output "$NEW_INDEPENDENT_OUTPUT"
python -m scripts.research.pallet_posefix_gap_audit_v1.gallery \
  --source-root "$ORIGINAL_SOURCE_ROOT" \
  --private-root "$PRIVATE_ARTIFACT_PARENT" \
  --output "$NEW_GALLERY_OUTPUT"
```

갤러리는 기존 공개 RGB 바이트만 사용한다. 저장된6사례 선택은 점수를 확인한 뒤의 설명용이며 추론을 바꾸지 않는다. 원 e의 최종 자세만 표시하며, 저장되지 않은 optimizer 궤적이나 e 중간 보정점을 생성하지 않는다.

입력 수·경계 판정·분산·손상/회복·꼬리 기여 검사는 다음 명령으로 수행한다.

```bash
python -m unittest scripts.research.pallet_posefix_gap_audit_v1.test_audit -v
```

초기 시도의 seal 키 포장 오류와 수정 후 결과를 구분해 보존했다. 공개 점수나 기존 원본 봉인을 덮어쓰지 않았다. 이번 결과는 사후 아웃라이어 분석이며 전체192장 성능을 대체하지 않는다.
