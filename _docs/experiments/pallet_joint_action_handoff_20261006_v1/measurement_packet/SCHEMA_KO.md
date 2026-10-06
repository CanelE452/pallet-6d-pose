# 새로 정의한 스키마와 검증·평가 계약

[확인] 이 디렉터리의 JSON/CSV/TSV 필드는 이번에 새로 정의한 스키마야. 기존 결과에 같은 열이 있었다고 가정하지 않아. `manifest.template.json`은 미취득 값을 null로 남긴 양식이고 의도적으로 validation에 실패해. 파일이 있다는 이유만으로 실제 측정 manifest로 쓰지 않아.

## Manifest

| 필드 | 의미·검증 |
| --- | --- |
| schema / data_kind | `pallet_physical_pairs_20261006_v1`; 실제 측정 또는 명시적인 `format_fixture` |
| coordinate_contract | `opencv_optical__canonical_object_xyz_WHD__metre` 고정 |
| models / baseline_method | 비교 모델 전부를 먼저 등록; weights/code/pose contract/selection SHA-256과 잠금 시각·학습/선택 녹화 및 실제 자세 노출 ID를 기록 |
| pair_id / split | 고정 pair 분모; train/development/independent_test. 두 조건은 pair 행 내부에 있어 분리 불가 |
| recording_id / physical_pose_id / adjacent_interval_id | 각각 split 전체에 걸쳐 중복 소속을 금지하는 그룹 키 |
| pallet_id / camera_id | 같은 pair의 같은 물체·카메라를 표현; calibration 파일은 해당 장치와 연결해야 해 |
| dimensions_WDH_m / dimension_survey | 운영 치수 출처와 실측 근거; pose 평가 내부에서는 xyz=[W,H,D] |
| symmetry_order / symmetry_evidence | 1/2/4 proper Y 회전; 물리·외관상 허용 근거 파일을 요구 |
| occlusion_kind / occluder_description | `physical_external`과 실제 장애물 설명; 합성 masking 제외 |
| used_for_selection / first_model_exposure_ns | 독립 확인은 false와 모델 잠금 뒤의 첫 노출; 교정·촬영 timestamp와 다른 노출 시각이야 |
| camera | pixel K, image_wh, 왜곡 모델/계수, clock ID, intrinsic/time sync/reference extrinsic 교정 근거 |
| same_pose_tolerance / stability_record | 사전 고정 이동·회전·동기·trace gap 한계 및 근거; 전구간 상대 자세 JSON trace를 검증 |
| uncertainty | 6×6 clean/occluded/cross 공분산, tangent 단위/순서, 실제 예산 파일 |
| clean / occluded | 고유 frame ID, 실제 영상 경로·해시, 센서 시각·clock, 8개 관측 출처, 사람 검수 파일, 독립 reference |

[확인] 모든 근거 artifact는 `{path, sha256}`야. path는 `--data-root` 안의 상대 경로이고 절대 경로·상위 이동은 거부해. 이미지 경로·내용 해시, pair/frame ID의 중복도 거부해. SHA-256 검사는 파일 내용을 보존하는 검증이지 인간의 계측 적합성 심사 자체는 아니야. JSON Schema는 구조를 설명하고 실제 `measurement.py validate`는 파일/해시·물리 계약·누출·상관 등의 의미 조건까지 실행해.

[확인] reference의 method는 `optical_mocap`, `laser_tracker`, `surveyed_independent_tag_rig` 중 하나야. `independent_of_model_and_2d_labels=true`, `derived_from=[]`, 실제 source artifact와 원문 `source_record_id`, 정준 R_physical/centroid_m, 영상과 연결한 timestamp/clock을 요구해. 장치 원문에서 정준 좌표로 변환하는 과정을 교정 근거에 남겨야 해. 모델 입력 영상/동일 클릭 레이블의 PnP나 모델 기반 역산 참조는 이 independent manifest로 import하지 않아. 그런 자료는 기존 개발 일관성 진단에서 사용 가능한 별도 근거야.

[확인] 모델 잠금의 `locked_at_ns`와 자료 첫 모델 노출의 `first_model_exposure_ns`는 같은 UTC(Coordinated Universal Time, 협정 세계시) Unix nanoseconds 시간축으로 기록해. 이미지/reference/stability의 `timestamp_ns`는 교정으로 같은 `camera.clock_id`에 맞춘 센서 시각이야. 서로 다른 두 종류의 clock을 섞어 노출 순서를 판단하지 않아.

[확인] `stability_record`는 `stability.schema.json` 구조야. 독립 상대 자세의 strictly increasing timestamp와 실제 원문 record ID를 가진 두 개 이상의 기록이 clean/occluded 전 구간을 덮어야 해. 기록 간 gap과 각 상대 자세의 clean 참조 대비 drift를 검사해. 허용 공차는 operator가 근거 파일로 사전 지정하며 코드가 1mm/1° 같은 과학적 허용값을 새로 정하지 않아.

## 예측 JSONL과 실제 최종 자세

[확인] `prediction.schema.json`에 있는 다음 필드를 행마다 내보내면 평가를 바로 실행할 수 있어: schema, coordinate_contract, method, frame_id, image_sha256, state, reference_access=false, weights_sha256, code_sha256, pose_contract_sha256, selection_rule_sha256. 성공 state에는 R_physical(3×3 proper rotation), centroid_m(3-vector, camera 전방 양의 깊이)를 더해. state는 success/no_detection/pnp_failure/invalid_pose이고 행 자체가 없으면 evaluator가 missing_prediction으로 기록해. 실패 오차를 0으로 만들지 않아.

[확인] 초기 후보 생성의 Tk나 camera-facing R_cf를 이 최종 pose로 내보내면 안 돼. 각 정상 모델의 최종 9점 출력에 기존 prediction-only W/D+PnP wrapper F(q)를 실제 적용해 얻은 정준 `R_physical`/`centroid`를 내보내. 기존 API(Application Programming Interface, 프로그램 호출 규약)는 `pose.infer(points, K, xyz, source=False)`이고 성공 출력의 centroid를 새 필드 centroid_m으로 이름만 바꿔. source=False는 이번 실제 영상 계약이야. `a_common.py`가 제공하는 기존 POSE 로더도 같은 F를 읽어. 원영상→네트워크 affine, 왜곡 및 W/D 변환은 inference 생산자가 기존 계약과 일치시켜야 해. metric evaluator는 그 production 경로를 대신 실행하지 않아.

[추정·미검증] 실제 inference JSONL 생성에는 해당 모델의 원영상 preprocessing·검출/backbone·refiner·decoder·최종 F 실행 환경과 동결 가중치가 필요해. `reference_access=false`와 해시는 manifest/producer의 출처 기록을 검사하는 계약이야. 악의적인 허위 기록까지 실행 로그 없이 증명하는 검사는 아니야. prediction-only producer가 독립 참조 파일을 읽지 않았다는 실제 코드·명령·노출 기록을 함께 보관해. 독립 참조는 inference 종료 뒤 이 오프라인 evaluator에서만 읽어.

## 출력과 지표

| 출력 | 새 필드·분모 |
| --- | --- |
| FRAME_ERRORS.csv | pair_id, recording_id, method, condition, frame_id, state, translation_cm, rotation_deg, ADDsym_m; 등록된 pair×모델×2조건 전체 행 |
| COMMON_PAIR_DELTAS.csv | pair_id, recording_id, method와 세 지표의 occluded−clean 차이; 모든 모델/양쪽 조건 성공의 공통 pair만 |
| SUMMARY.json | 전체 pair, 네 실패 칸, 조건별/양쪽 coverage, 공통 성공 pair ID/개수, 같은 pair의 clean/occluded/차이 평균·중앙값·P90, 악화/개선 수, baseline 대비 실패 전환·오차 차이·악화 수 |
| bootstrap | 10,000회, seed=20260917, 녹화 전체 재표집을 모델 간 공유; 공통 pair 평균 paired delta와 baseline 대비 delta의 95% 구간 |
| reference_uncertainty | 실제 예산 파일과 상태 `recorded_not_propagated_to_metric_CI`; sampling CI를 교정 불확실성이라고 부르지 않아 |

[확인] translation은 정준 중심 위치의 Euclidean 차이 cm, rotation은 승인된 proper group 안의 최소 geodesic 회전 차이 degree야. ADDsym은 같은 최종 정준 pose가 변환한 대응 8코너 평균 거리의 proper group 최솟값 m이며 기존 `pose.py metric` 정의를 얇게 수학적으로 재현했어. 표면 최근접점의 ADD-S와 자동으로 같다고 하지 않아. 각 지표는 실제 한 prediction pose를 채점하므로 이동/회전 최솟값을 서로 다른 후보에서 가져오지 않아.

[확인] 공통 성공 교집합에 두 개 이상의 녹화가 없으면 bootstrap은 `NOT_ESTIMATED`야. 전체 분모에는 모든 실패가 남고 유한 오차 통계는 보조 conditional 지표야. 단일 seed를 독립 실제 sample로 늘리지 않아. 두/소수 녹화의 interval은 기술 통계이며 작은 cluster의 보장을 주장하지 않아. `scientific_evidence=false`/`evidence_role=format_fixture_not_scientific`는 fixture 평가에서 강제해. 실제 자료를 선택에 사용했다면 split을 development로 기록해.

[확인] 이 CLI는 2차원 직접 가시 정확도, uncertainty propagation, 계측장치 raw adapter, 모델 사진 추론을 추가로 구현한 것이 아니야. 필요한 의존성이 주어지면 완성된 최종 자세 평가 명령으로 B를 재개할 수 있고, 현재는 실제 독립 평가를 완료로 표시하지 않아.
