# 실제 가림 쌍·독립 참조 취득 패킷

[확인] 취득 패킷과 명령줄 인터페이스(Command Line Interface, CLI)는 구현·검증했어. 실제 독립 평가 상태는 `BLOCKED_DATA`야. 새 촬영·외부 계측·사람 검수·새 모델 추론·학습은 0회야. `test_measurement.py`의 작은 형식 fixture는 과학적 실험 결과가 아니야.

[확인] 기존 원영상·시간·독립참조 감사의 9,029관측/독립 참조 연결 0, 센서 19사본/27논리 출처 결과를 재사용했어. [기존 물리 참조 감사](../../pallet_remaining_evidence_connection_20261006_v1/time_physical/REPORT_KO.md)와 그 inventory의 바이트가 원 작업 트리와 최신 checkout에서 같아. 이번에는 좁은 신규/미연결 파일 후보만 확인했고 기존 센서 전체 파싱·12장·119장 작업은 반복하지 않았어. 새 자료가 없다는 판단은 [이번 확인 범위](LOCAL_DELTA_CHECK.json)에만 적용해.

[추정·미검증] 새 측정의 목적은 동일 팔레트·동일 카메라의 실제 상대 자세를 유지한 clean/실물 가림 쌍에서, 잠근 Base/N3/PoseFix 방식 및 실행 가능한 A 방법의 가림 민감도를 독립 참조로 비교하는 거야. 사진 이름이 clean/occluded이거나 CAN(Controller Area Network, 차량 통신망) 명령이 중립인 것만으로 같은 자세를 인정하지 않아.

- [실제 취득·교정·분리 절차](ACQUISITION_KO.md)
- [새로 정의한 manifest/예측/결과 스키마](SCHEMA_KO.md)
- [manifest JSON Schema](manifest.schema.json) · [예측 JSON Schema](prediction.schema.json) · [상대 자세 trace JSON Schema](stability.schema.json)
- [미취득 필드가 null인 작성 양식](manifest.template.json) · [불확실성 예산 TSV 양식](uncertainty_budget.template.tsv)
- [원문 확인 범위·버전·해시](PRIMARY_READING.json)
- 구현: `scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py`

[확인] 아래 세 명령은 실제 코드의 `--help`와 fixture 최소 실행으로 확인했어. `PAIR_DATA_ROOT`, `PAIR_SOURCE_MANIFEST`, `PAIR_OUTPUT_ROOT`, `PAIR_FINAL_POSES`는 미래 실제 자료를 받은 환경에서 설정할 경로야. 이번에 그 자료를 찾았다는 의미가 아니야. 출력은 개인정보가 없는 로컬/ignored 저장 위치를 선택해.

```bash
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py import \
  --manifest "$PAIR_SOURCE_MANIFEST" --data-root "$PAIR_DATA_ROOT" \
  --output "$PAIR_OUTPUT_ROOT/imported_manifest.json"

python3 scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py validate \
  --manifest "$PAIR_OUTPUT_ROOT/imported_manifest.json" --data-root "$PAIR_DATA_ROOT" \
  --output "$PAIR_OUTPUT_ROOT/VALIDATION.json"

python3 scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py evaluate \
  --manifest "$PAIR_OUTPUT_ROOT/imported_manifest.json" --data-root "$PAIR_DATA_ROOT" \
  --predictions "$PAIR_FINAL_POSES" --split independent_test \
  --output "$PAIR_OUTPUT_ROOT/evaluation"
```

[확인] `import`는 원자료를 복사하거나 수정하지 않고 모든 근거 파일의 존재·SHA-256과 계약을 검사해 manifest를 저장해. `evaluate`는 잠근 모델의 최종 정준 자세를 읽는 오프라인 평가야. 사진에서 모델을 실행하는 명령이 아니야. 실제 입력 예측 JSONL(JavaScript Object Notation Lines, 행별 JSON)은 해당 모델의 정상 inference 경로에서 만들고, [최종 PnP 연결 계약](SCHEMA_KO.md)의 필드로 내보내면 돼. 추론 생성과 독립 참조 채점은 서로 다른 작업이야. 예측을 만드는 모델 환경·가중치가 없으면 그 생성만 `NOT_RUN_DEPENDENCY`로 남겨.

[확인] 누락 예측은 `missing_prediction` 실패로 남아 전체 pair 분모를 유지해. 각 모델은 양쪽 성공/clean만 실패/occluded만 실패/양쪽 실패의 네 칸과 조건별 coverage를 보고해. 유한 clean→occluded 오차 변화는 모든 비교 모델이 쌍의 양쪽에서 성공한 공통 교집합에서 pair별로 계산해. 별도 성공 집합의 중앙값을 빼지 않아. `FRAME_ERRORS.csv`, `COMMON_PAIR_DELTAS.csv`, `SUMMARY.json`, `VALIDATION.json`이 생성돼. 실패의 오차 칸은 빈칸/JSON null이고 0이 아니야.

[확인] pair·녹화·실제 자세·인접 구간을 split 전체에 걸쳐 묶어. 모델의 학습/선택 노출과 잠금 시각도 검사해. 이 검사는 제공된 manifest 기록을 검사하므로 사람이 계측 출처와 노출 이력을 거짓 없이 기록해야 해. CLI가 장치 교정이나 검수 서명을 대신하지 않아.

```bash
python3 -m unittest discover \
  -s scripts/research/pallet_joint_action_handoff_20261006_v1 \
  -p 'test_measurement.py' -v
```

[확인] 테스트는 임시 디렉터리에 명시적인 비과학 fixture만 만들어. 실제 명령 세 개, 누출·누락·동기·상대 자세 보존·비독립 참조·잘못된 회전·모델 해시·공통 성공 교집합·불가능한 상관 공분산을 검증해. 미래 실제 데이터가 validation을 통과하더라도 독립 계측 불확실성과 사람 검수의 적합성을 확인한 뒤 과학적 평가로 보고해야 해.

[확인] 기존 개발 13세션의 변동 근거만 재집계하는 명령도 실행했어. 기존 receipt와 입력 해시가 일치해야 하고, 출력은 B 가림 분산으로 사용하지 않아. 아래 `EXISTING_YOLO_SCORES`도 현재 환경의 실제 로컬 경로로 설정해.

```bash
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py precision-source \
  --scores "$EXISTING_YOLO_SCORES" \
  --receipt _docs/experiments/pallet_n3_static_closeout_v1/PAIRED_POSE_ANALYSIS.json \
  --output "$PAIR_OUTPUT_ROOT/precision_source"
```
