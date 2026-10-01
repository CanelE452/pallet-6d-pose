# IEEE Sensors: 세 기반 추정기 검증

현재 **진행 중**입니다. YOLO 기존 결과 + DOPE(VGG) + SimpleBaseline-derived ResNet-18을 각각 보정 전후로 비교합니다. 세 번째 모델은 기존 YOLO/DOPE의 head 이름만 바꾼 모델이 아니라 원영상의 독립 ResNet-18 추정기입니다.

2026-10-01 18:00 KST 기준 예상: **전체 추가 실행·결과 정리 약12–18시간**. 순조롭다면10/2 오전~낮 첫 전체 결과를 확인하는 일정입니다. R18 GPU 시험 학습은16장당0.106초로,60epoch 연산만 약6.18시간입니다. 파일 읽기·검증·추가 보정 학습·평가는 별도이므로 실제 epoch 속도로 갱신해야 합니다. [시간 근거](TIME_ESTIMATE.json).

| 기반 | 현재 공개 근거 | 추가 실행 |
|---|---|---|
| YOLO | 기존 3seed 보정 전후·큰 보정기와 속도 비교 | 기존 봉인 결과 재사용 |
| DOPE(VGG) | 6만 장 source 예측 및 해시 확인 완료, TRAIN44,063 usable | P/D 각각3seed 학습·같은319장 평가·속도·이미지 보고서 |
| SimpleBaseline-derived ResNet-18 | CPU 계약 검사·8회 폐기 GPU 시험 학습 통과 | 기본9점 추정기60epoch → 동결 → P/D 각각3seed → 동일 평가 |

[DOPE 상세 계획](../pallet_dope_refiner_20261001_v1/README_KO.md) · [ResNet-18 상세 계획](../pallet_resnet18_refiner_20261001_v1/README_KO.md) · [실행 순서](QUEUE_PLAN.json) · [상태 스냅샷](STATUS_KO.md) · [확장 원고 초안](../../paper/sensors_dope_extension_20261001_v1/README_KO.md)

완료 후에는 기반별 실제 RGB 이미지에 전후9점 좌표를 표시하고 원영상 해상도, 팔레트 치수(mm), 카메라 K를 함께 기록합니다. 2D 중앙값/P90, 전체GT PCK, 결측/실패, T/R 및 실제 latency를 남깁니다. 수치는 실제 완료 영수증이 있어야 표로 생성하며 결과가 악화된 경우도 유지합니다.

GitHub 문서는 게시 시점의 스냅샷입니다. raw 가중치·69.6MB SOURCE_MANIFEST·원 source/평가 영상은 데이터 의존성이며 GitHub에 새로 올리지 않습니다. 코드와 해시를 공개하는 것이 데이터까지 포함한 단독 재실행 패키지를 뜻하지 않습니다. 원고 PDF는 바로 열 수 있지만 저장소의 build.py는 로컬 Tectonic 실행파일/cache를 필요로 합니다.

세 기반 결과가 모두 생성되어도 원래 실사 T/R 안정적 공동 개선의 고정 판정 통과, 독립 TEST 확인, 논문 투고·채택을 의미하지 않습니다. 결과에 맞춰 원고 주장을 제한합니다.
