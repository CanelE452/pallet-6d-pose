# 리프터 L4 평가 연결 점검

이 디렉터리는 frozen 소스·가중치·원예측을 바꾸지 않는 새 통합 adapter다. 모델 추론이나 사람 판정을 대신하지 않는다.

## 결론

고정 Base/N3 가중치는 작업복사본 `assets/`에서 기대 크기와 SHA-256에 정확히 일치했다. 최초 게이트 감사에서는 모델 추론을 실행하지 않았다. 이후 별도 고정 실행이 완료되어 8,910행 원예측과 파생 L4 예측을 이 디렉터리의 보고 도구로 검산했다.

키포인트 결측은 실사에서 이미 잘못 처리됐다고 단정할 수 없다. 정확한 YOLO 실행 환경은 Ultralytics 8.4.60이고, `Keypoints.xy`는 내부 좌표의 앞 두 성분을 그대로 반환한다. frozen extractor도 이 좌표와 confidence를 따로 저장하며 `point_valid`는 좌표 유한성으로 만든다. confidence threshold나 예측 sentinel은 고정 계약에 없다. 따라서 bridge는 `None`/`NaN`, 명시적으로 선언된 정확한 sentinel 쌍, 유한한 화면 밖 좌표를 분리하고 Base/N3 mask가 같음을 검사한다. 음수 좌표 전체를 결측으로 바꾸지 않고 confidence threshold도 새로 만들지 않는다.

객체 대응은 실제 공백이었다. 최초 코너 검수 JSON만으로는 평가기가 `object_match=None`을 거부한다. 새 두 번째 UI는 코너·오차·방법 이름을 숨기고 원본 영상과 frozen 선택 상자 하나만 보여 준다. 사람이 `same / different / undetermined`를 선택하면 실제 검수자·시간·노출 정보를 별도 sidecar에 저장한다. 이 sidecar는 원예측, 실행 identity, 코너 참조, manifest의 SHA-256에 묶인다. Base/N3에 같은 판정을 적용하며 후보를 다시 고르지 않는다. `different`는 PCK 분모의 실패로 남고 `undetermined`는 정확도 가지를 계속 대기시킨다.

`resume.py`의 단일 종료 상태도 그대로 사용하지 않는다. `bridge.py status`는 모델, 전체 추론, 코너 주석, 객체 대응, 정지 구간, 독립 물리 참조를 따로 보존한다.

## 기존 리프터 어노테이션 확인

사용자가 기억한 리프터 수동 어노테이션은 실제로 있다. 원 저장소의 `challenge/data/01_real/live_capture_gt/`에서 이번 네 세션과 정확히 같은 폴더를 확인했다.

| 세션 | JSON/객체 | `manual_click` 점 | PnP 투영점 | 자동 중심점 | 현재120장과 겹침 |
|---|---:|---:|---:|---:|---:|
| 173507 | 19 | 80 | 72 | 19 | 2 |
| 174126 | 4 | 20 | 12 | 4 | 0 |
| 174342 | 13 | 56 | 48 | 13 | 0 |
| 174925 | 31 | 135 | 113 | 31 | 0 |
| 합계 | 67 | 291 | 245 | 67 | 2 |

67개 PNG를 디코딩한 BGR 픽셀 SHA-256은 모두 현재 8,910프레임 계획의 같은 세션·저장 인덱스와 일치했다. 67개 JSON도 Git tracked 상태다. 전체 `live_capture_gt`에는 28개 폴더와 JSON 851개가 있으며, 나머지 784개는 다른 촬영 세션이다.

이 67개를 새 120장 참조로 자동 승격할 수는 없다. 겹치는 프레임은 `173507:2910`, `173507:3210` 두 장뿐이다. 모든 객체가 `MANUAL_REVIEW_REQUIRED`, `UNCONFIRMED_SIGNED_AXIS`, `occlusion_level=unknown`이고 reviewer ID와 실제 검수 시간이 없다. 67개 파일은 `population_role=DEV`이지만 객체는 `split=train`으로 되어 있어 독립 held-out 지위도 확정할 수 없다. 312개 점은 수동 클릭이 아니라 PnP 투영 또는 자동 중심이다. 원자료는 보존했고 새 `human_reviewed` 파일은 만들지 않았다. 상세 수치는 `EXISTING_ANNOTATION_AUDIT.json`에 있다.

원 저장소와 인계 작업복사본 전체에서 `annotations_in_progress.json`, `LIFTER_REFERENCE_REVIEWED*.json`, `LIFTER_OBJECT_MATCH_REVIEWED*.json`도 찾았지만 0개였다. 따라서 ZIP 작성 뒤 별도로 저장된 새 120장 검수 파일을 로컬에서 재사용할 근거는 현재 없다.

## 고정 추론 후 연결 명령

원예측을 덮어쓰지 않고 연속 출력 평가용 mask 검산본을 만든다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/bridge.py adapt-masks \
  --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl \
  --output data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES_L4.jsonl \
  --receipt data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/L4_MASK_RECEIPT.json
```

실제 코너 검수 export가 생긴 뒤 대응 queue를 만들고 두 번째 UI를 연다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/bridge.py prepare-object-review \
  --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl \
  --reviewed data/pallet/results/pallet_lifter_case_review_20261003_v1/review/LIFTER_REFERENCE_REVIEWED.json \
  --manifest data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json \
  --output data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json

/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/object_match_review.py serve \
  --queue data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json \
  --manifest data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json \
  --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl \
  --reviewed data/pallet/results/pallet_lifter_case_review_20261003_v1/review/LIFTER_REFERENCE_REVIEWED.json \
  --store data/pallet/results/pallet_lifter_case_review_20261003_v1/review/object_match_in_progress.json \
  --port 8766
```

사람이 UI에서 내보낸 `LIFTER_OBJECT_MATCH_REVIEWED.json`을 검증·import한 뒤 evaluator 파생본을 만든다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/object_match_review.py import \
  --queue data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json \
  --manifest data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json \
  --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl \
  --reviewed data/pallet/results/pallet_lifter_case_review_20261003_v1/review/LIFTER_REFERENCE_REVIEWED.json \
  --store data/pallet/results/pallet_lifter_case_review_20261003_v1/review/object_match_in_progress.json \
  --input LIFTER_OBJECT_MATCH_REVIEWED.json

/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/bridge.py apply-object-review \
  --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl \
  --queue data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json \
  --sidecar LIFTER_OBJECT_MATCH_REVIEWED.json \
  --output data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES_EVALUATOR.jsonl \
  --receipt data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/L4_EVALUATOR_DERIVATION.json
```

기존 `metrics/evaluate.py`에는 마지막 evaluator 파생본을 넘긴다. 원본 `ALL_STORED_FRAMES.jsonl`과 frozen 15개 binding은 수정하지 않는다.

## 완료된 고정 추론의 파생 보고

완료된 8,910행을 다시 추론하지 않고 다음 명령으로 연속성 수치, 그림, 해시 receipt를 재생성한다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 \
  scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/build_actual_outputs.py
```

출력은 모두 `combined_integration/output/` 아래에만 쓴다.

- `ACTUAL_INFERENCE_REPORT_KO.md`: 전체·세션별 가용성, 최장 결측, 인접 변화와 해석 제한
- `LIFTER_CONTINUITY_SUMMARY.json`: 정의·분모·짝지은 쌍·검열 상태를 보존한 전체 결과
- `LIFTER_CONTINUITY_SUMMARY.csv`: 원고 표로 옮길 수 있는 평면 표
- `figures/lifter_prediction_timeseries.png`: 네 세션의 실제 Base/N3 출력 시계열
- `figures/lifter_output_coverage.png`: 전체·세션별 fresh/held/no_pose coverage
- `figures/review120_prediction_overlay_contact_sheet.png`: 고정 120장 전부의 prediction-only overlay
- `EXECUTION_FILE_RECEIPT.json`과 `.sha256`: 원예측·L4예측·가중치·계획·코드·산출물의 크기와 SHA-256
- `VALIDATION_RECEIPT.json`과 `.sha256`: 실제 실행한 29개 시험, 197개 수치 회귀 대조, receipt 재검산 기록

전체 결과는 두 방법 각각 fresh 8,772장, held 0장, no_pose 138장이다. 같은 프레임 유한 pose 쌍은 8,772개이고, 세션 경계를 제외한 동일 인접 유효 쌍은 8,737개다. 이 값과 그림은 출력 연속성 자료다. 사람 코너 정확도, 객체 대응, 사람이 확인한 정지 구간, 독립 물리 참조는 계속 `x`다.

## 검증

- 새 L4 fixture 6개 통과: mask, Base/N3 동등성, 실제 localhost 페이지·이미지·저장·export, sidecar import, 기존 metrics 정상/실패 경로, 부분 상태 분리.
- 실제 파생 보고 시험 7개 통과: 선형 분위수, 결측 검열·짝 분모, 8,910장 실제 분모, 197개 evaluator 회귀 대조, 120장 전체 overlay, 입력·출력 해시, 주장 제한.
- 기존 metrics 회귀시험 16개 통과.
- Python compile과 JavaScript syntax 검사 통과.
- 합성 fixture의 reviewer 문자열은 시험 메모리·임시 폴더에서만 사용했고 실제 사람 파일로 내보내지 않았다.
