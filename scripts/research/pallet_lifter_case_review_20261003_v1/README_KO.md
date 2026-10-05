# 리프터 고정 모델 평가 — 현재 상태

네 기록 영상의 저장 프레임 8,910장에 정확한 YOLO Base와 N3 seed1 가중치를 한 번씩 적용했다. 실행은 `VERIFIED_COMPLETE`이며 두 방법 모두 fresh 8,772, held 0, no-pose 138이다. 새 학습과 optimizer update는 0회다.

현재 수치는 출력 가용성과 인접 출력 변화다. 사람 코너 참조, 예측 객체↔사람 대상 대응, 확인된 정지 구간, 독립 물리 참조가 없으므로 정확도·정지 잡음·물리 오차는 `x`다. 상세 결과·그림·검증 receipt는 다음 두 위치에 있다.

- `combined_integration/output/ACTUAL_INFERENCE_REPORT_KO.md`
- `_docs/experiments/pallet_combined_closeout_20261003_v1/FINAL_REPORT_KO.md`

사람 직접 가시 코너 검수 화면은 저장소 루트에서 다음 명령으로 시작한다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 \
  scripts/research/pallet_lifter_case_review_20261003_v1/start_review.py
```

주소는 <http://127.0.0.1:8765>다. 고정 120장과 반복 24장을 검수하고 `LIFTER_REFERENCE_REVIEWED.json`을 export한다. 객체 대응 검수는 그 export가 생긴 뒤 `combined_integration/README_KO.md`의 포트 8766 절차를 실행한다.

사용자가 기억한 기존 어노테이션은 같은 네 세션에서 67개 확인했다. 원영상 픽셀은 8,910프레임 계획과 일치하지만 새 review-120과 겹치는 것은 2장뿐이고 reviewer/time·signed axis가 미확정이라 자동 사람 정답으로 승격하지 않았다. 근거는 `combined_integration/EXISTING_ANNOTATION_AUDIT.json`에 있다.

고정 추론의 원시 8,910행은 `data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl`이다. 원본은 덮어쓰지 않으며 mask와 사람 대응을 연결한 evaluator 입력은 별도 파생 파일로 만든다.
