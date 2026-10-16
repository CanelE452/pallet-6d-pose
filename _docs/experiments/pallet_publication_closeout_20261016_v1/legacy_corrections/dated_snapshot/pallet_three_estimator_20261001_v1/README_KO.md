# IEEE Sensors: 세 기반 추정기 검증

2026-10-02 갱신: **DOPE 평가·속도 측정·보고서는 완료됐지만, 세 기반 전체 검증은 미완료입니다.** ResNet-18은 60epoch 학습을 종료한 뒤 고정 TRAIN16 기능 검사에서 실패해 후속 P/D 학습 전에 중단됐습니다. 별도 TRAIN-only 복구 진단이 진행 중이며, 성공이나 전체 재학습 완료를 뜻하지 않습니다.

사용자 요구가 **학습부터 RGB와 실제 치수를 함께 입력하는 방식**으로 구체화됐습니다. 아래 완료 실험은 학습기에 치수를 넣지 않았고 PnP에서만 치수를 사용한 기존 결과입니다. 새 치수 조건부 학습은 별도 미실행 후속이며, 현재 결과가 그 요구를 충족한다고 표시하지 않습니다.

YOLO 기존 결과 + DOPE(VGG) + SimpleBaseline-derived ResNet-18을 각각 보정 전후로 비교합니다. 세 번째 모델은 기존 YOLO/DOPE의 head 이름만 바꾼 모델이 아니라 원영상의 독립 ResNet-18 추정기입니다. 이전 [시간 예상](TIME_ESTIMATE.json)은 실패 전 추정이며 현재 완료 예정 시각으로 사용하지 않습니다.

| 기반 | 현재 공개 근거 | 추가 실행 |
|---|---|---|
| YOLO | 기존 3seed 보정 전후·큰 보정기와 속도 비교 | 기존 봉인 결과 재사용 |
| DOPE(VGG) | P/D 각각3seed 학습, 동일319장 평가, 속도 및 이미지 보고서 완료 | 실제 결과 원고 반영·게시 검토 |
| SimpleBaseline-derived ResNet-18 | 기본9점 추정기60epoch 완료, TRAIN16에서 유효점0/144·매칭0/16 | P/D 본학습0회, 실사 평가·속도 미실행; 별도 복구 진단 진행 |

[DOPE 상세 계획](../pallet_dope_refiner_20261001_v1/README_KO.md) · [ResNet-18 상세 계획](../pallet_resnet18_refiner_20261001_v1/README_KO.md) · [실행 순서](QUEUE_PLAN.json) · [상태 스냅샷](STATUS_KO.md) · [확장 원고 초안](../../paper/sensors_dope_extension_20261001_v1/README_KO.md)

DOPE의 조건부 2D 중앙값은 12.585→7.461px, 전체GT PCK10은 19.872→36.775%입니다(P의3seed 통계 평균). geometry 참조 T/R 중앙값은 10.046cm/3.441°→8.529cm/2.884°로 낮아졌지만, 2D P90은 50.609→50.713px로 악화했고 pose coverage는 210/319로 변하지 않았습니다. [실제 결과·RGB·치수·K·전체 지표](../pallet_dope_refiner_20261001_v1/REPORT_KO.md)를 함께 확인해야 합니다. P가 D보다 모든 지표에서 우수하다는 결과도 아닙니다.

[ResNet-18 실패 진단](../pallet_resnet18_refiner_20261001_v1/BASELINE_FAILURE_DIAGNOSTIC_KO.md)과 [복구 진단의 고정 조건](../pallet_resnet18_recovery_20261002_v1/DIAGNOSTIC_PROTOCOL.json)은 완료 성능표와 구분합니다. 복구 진단은 같은 TRAIN256·초기값·배치로 Adam weight decay만 비교하는 폐기용 두 fit이며, 새로운 실사/VAL 성능이나 배포 모델을 만들지 않습니다. 원래 실패 모델·로그·프로토콜은 보존합니다.

GitHub 문서는 게시 시점의 스냅샷입니다. raw 가중치·69.6MB SOURCE_MANIFEST·원 source/평가 영상은 데이터 의존성이며 GitHub에 새로 올리지 않습니다. 코드와 해시를 공개하는 것이 데이터까지 포함한 단독 재실행 패키지를 뜻하지 않습니다. 원고 PDF는 바로 열 수 있지만 저장소의 build.py는 로컬 Tectonic 실행파일/cache를 필요로 합니다.

세 기반 결과가 모두 생성되어도 원래 실사 T/R 안정적 공동 개선의 고정 판정 통과, 독립 TEST 확인, 논문 투고·채택을 의미하지 않습니다. 결과에 맞춰 원고 주장을 제한합니다.
