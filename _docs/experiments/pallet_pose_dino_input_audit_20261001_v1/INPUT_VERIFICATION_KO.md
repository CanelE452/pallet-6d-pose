# DINO TRAIN 입력 독립 검산

입력 검산 PASS입니다. 새 학습·T/R 성능 평가·실사 경로 실행은 모두 0회이며, 이 결과는 성능 개선의 증거가 아닙니다.

TRAIN 2,598행을 모두 보존하고 원래 실패 1행, 유효 후보 20,776개를 검산했습니다. 저장된 FP16 token을 한 프레임씩 읽어 독립 scalar 투영과 Torch CPU float64 grid_sample로 재계산했습니다. 픽셀 중심 변환, border 처리, 지원점 mask, 채널별 정렬 평균을 확인했으며 사전 고정 허용오차는 atol=rtol=2e-6입니다.

| 모델 | 후보 수 | descriptor 최대 절대차 | 서로 다른 두 가설 행 | raw 차이 L2 중앙값 | 정규화 차이 L2 중앙값 | 변동 채널 수 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 5194 | 1.51511843427e-06 | 2597 | 3.6293146 | 3.35564051 | 385 |
| DIVERSE251_s1 | 5194 | 1.33539335145e-06 | 2597 | 3.5521023 | 3.29350704 | 385 |
| DIVERSE251_s2 | 5194 | 1.62586728614e-06 | 2597 | 3.57225615 | 3.30701418 | 385 |
| DIVERSE251_s3 | 5194 | 1.70917929765e-06 | 2597 | 3.57452939 | 3.30314399 | 385 |

가설 간 차이와 채널 변동은 입력의 구별 가능성만 보여 줍니다. 실제 물리 T/R 방향이나 안전한 개선을 구별한다는 뜻은 아닙니다. 원래 후보 valid mask와 0지원 후보를 유지했으며, 지원점은 가시성 정답이 아닙니다.

NPY 파일 전체 SHA, 유효 프레임 2,597개의 token SHA, 실패 프레임의 0 token, 모든 NPZ 배열 hash와 R0 유효 후보 5,194개의 FP32 mean/std를 확인했습니다. 원본 RGB·모델 가중치·목표 오류는 읽지 않았고 backbone을 재실행하지 않았습니다. 따라서 이 검산은 저장된 token 이후의 입력 구성과 기록된 provenance를 검증하며, backbone 추론 자체를 독립 재현한 검산은 아닙니다.

공유 source feature/pose/metadata 캐시는 다른 split도 포함하지만 수치 검산에는 봉인된 eligible TRAIN 행만 사용했습니다.

[전체 영수증](INPUT_VERIFICATION.json) · [입력 설계 검토](FEATURE_DESIGN_REVIEW_KO.md) · [독립 검산 코드](../../../scripts/research/pallet_pose_dino_input_audit_20261001_v1/verify_train_inputs.py)
