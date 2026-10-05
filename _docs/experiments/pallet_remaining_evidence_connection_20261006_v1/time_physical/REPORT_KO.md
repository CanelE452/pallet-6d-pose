# 원영상·시각·기존 주석·독립 물리 참조 연결 검산

수신 PC 원자료를 실제로 읽고 재검산했습니다. 기존 모델 추론과 주석을 그대로 재사용했으며 새 학습·추론·PnP·장치 실행은 모두 0회입니다.

## 연결을 완료한 자료

네 세션 **8,910개** 저장 프레임을 인계 시각표, 고정 계획, 로컬 timing CSV, 원영상의 디코딩 픽셀, Base/N3 기존 예측과 연결했습니다. 원영상·meta·timing·control·state 파일 SHA-256도 고정 계약과 일치합니다. 중복 시각 **29개**를 지우지 않고 보존했습니다. 센서 시각을 사용하며 MP4의 nominal FPS로 바꾸지 않았습니다.

| 세션 | 저장 프레임 | 중복 센서 시각 | 최대 시각 간격(ms) |
|---|---:|---:|---:|
| 173507 | 3729 | 16 | 233.497070 |
| 174126 | 757 | 2 | 267.224121 |
| 174342 | 2501 | 9 | 300.170166 |
| 174925 | 1923 | 2 | 233.182861 |

기존 원시 예측의 겉쪽 pose 필드는 x/z/yaw지만 `raw_shared_prediction.methods.R0/N3_seed1.pose`에는 **3D 중심과 3×3 회전**이 이미 저장되어 있습니다. Base와 N3 각각 **8,772개**의 유효 full 6D 예측을 확인했습니다. 나머지 138개는 기존 no_pose이며 새 값을 만들지 않았습니다. 이 값은 모델의 예측이고 물리 정답이 아닙니다.

완료 추론의 RUN_IDENTITY에 고정된 모델 두 개·PnP/보정 코드·치수/대칭/선택/학습 receipt의 **15개 바인딩**은 실제 수신 파일의 크기와 SHA가 모두 일치합니다. 인계의 과거 annotation/report 소스와 현재 코드 버전이 다른 경우도 별도로 기록하고 과거 원본은 보존했습니다.

정사각형 **119장**은 원사진 바이트·디코딩 픽셀·원주석 SHA를 모두 대조했습니다. 인계 PC에서 찾지 못했던 기존 주석이 수신 PC에는 실제로 있습니다. 다시 주석할 필요가 없습니다. manual_declared **602점**, manual_in_frame **600점**을 분리해 보존합니다. 단일 촬영 세션의 DEV/train 자료라는 기존 제한도 유지합니다.

## 카메라·치수·코너 연결

카메라 좌표는 +X 오른쪽/+Y 아래/+Z 전방, 원점은 색상 광학 중심입니다. 팔레트 원점은 직육면체 중심입니다. W/D/H=[1.10,1.10,0.15] m를 PnP의 X/Y/Z=[1.10,0.15,1.10] m로 **한 번만** 변환합니다. 0–7 코너 순서와 C4의 네 회전·전체 객체 순열을 기존 로컬 계약과 검산했습니다. 네 meta의 K와 영 왜곡 계수도 일치합니다. `v4_extrinsics_measured=false`인 운영 설정을 실측 변환으로 바꾸지 않습니다.

## 물리 정확도가 남는 이유

동봉된 실제 센서/태그 원문 **19개 파일**을 직접 파싱했습니다. 이는 빈 원문 사본을 공유하는 **27개 논리 출처**이며, 같은 빈 파일을 여러 물리 파일처럼 세지 않았습니다. 다른 파일의 원문이 ZIP에 없으면 인계 inventory의 행 수와 실제 읽은 행 수를 구분했습니다. 8,910개 리프터와 119개 정사각형의 총 **9,029행**에서 대상 팔레트·촬영 시각·실측 좌표 변환까지 연결되는 독립 참조는 **0행**입니다.

레이저 CSV는 다른 촬영의 벽 거리와 벽 상대 각도이며, 태그 기록은 9월15일의 상대 회전입니다. 이 기록만으로 9월1일 리프터나 정사각형 119장의 팔레트 중심 위치·회전 정답을 만들 수 없습니다. 다른 트럭의 이미지 자세에 맞춰 역산한 태그 장착값, PnP 주석, 운영 추정과 합성 정답도 독립 물리 정답으로 사용하지 않았습니다. 따라서 독립 물리 T/R은 **x**입니다. 기존 코너 정확도와 연속성 결과는 유효한 별도 결과로 유지합니다.

## 근거와 실행

- [전체 검산 결과](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/VALIDATION.json)
- [8,910개 시각·픽셀·예측 연결](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/FRAME_TIME_PREDICTION_JOIN.csv)
- [정사각형 기존119장 원본 재사용](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/SQUARE119_EXISTING_SOURCE_REUSE.csv)
- [실제 센서 파싱과 inventory 구분](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/ACTUAL_SENSOR_RECORDS_AUDIT.csv)
- [9,029개 물리 참조 연결 상태](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/PHYSICAL_REFERENCE_JOIN_STATUS.csv)
- [읽은 입력 SHA-256](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/SOURCE_HASHES.json)

실행 명령:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_remaining_evidence_connection_20261006_v1/time_physical_connection.py --handoff-root /tmp/pallet-remaining-evidence-handoff-20261006 --receiver-root /home/minjae/Documents/github/pallet-pose --output-root /tmp/pallet-github-publication-20261006-v1
```

CPU 실행 10.430초. 원영상 전체 재디코딩과 119장 픽셀 검산을 포함합니다. GPU 추론 0회, optimizer update 0회, 원본 수정 0개.
