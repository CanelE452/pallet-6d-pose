# SimpleBaseline-derived ResNet-18 + P/D 추가 검증

2026-10-02 현재 상태: **기준모델60epoch 학습 종료 후 기능 검사 실패. P/D 본학습0회, 실사 평가·속도 미실행**.

새 요구는 학습부터 RGB+실제 치수를 함께 입력하는 것입니다. 여기 보존한 기존 기준모델은 RGB-only 학습이며 치수는 후속 PnP에만 사용하도록 설계됐습니다. 치수 조건부 후속은 별도 미실행 방법이며, 현재 모델이나 복구 진단이 그 요구를 충족한 결과는 아닙니다.

교수님이 요구한 세 기반(YOLO·DOPE·ResNet-18) 비교 중 하나다. 기존 DOPE 결과 완료와 이 실험의 실패를 구분한다. [기준모델 학습 완료 기록](BASELINE_TRAINING_COMPLETE.json)은 학습 실행 종료의 증거이며 모델 기능 성공을 뜻하지 않는다. [고정 TRAIN16 실패 진단](BASELINE_FAILURE_DIAGNOSTIC_KO.md)에서 원래 threshold0.1 기준 유효점0/144, 매칭0/16이었다. 최대 heatmap 값은0.005223으로 threshold보다 낮았다. 이 상태를 정상 실사 성능 결과로 제시하지 않는다.

독립 RGB 입력의 ResNet-18 + 3 deconvolution + 9 heatmap 기반 추정기를 ImageNet 공식 가중치에서 합성 TRAIN55,980장×60epoch 학습했다. batch16과 마지막12장 배치로209,940updates/3,358,800exposures를 마쳤고 최종 epoch60을 보존했다. calibration1,004장의 loss는 기록했지만 checkpoint 선택에 사용하지 않았다. YOLO crop/DOPE 출력/실사 GT를 입력으로 쓰지 않았다.

현재 별도 namespace에서 [TRAIN-only 복구 진단](../pallet_resnet18_recovery_20261002_v1/DIAGNOSTIC_PROTOCOL.json)이 진행 중이다. 동일한 fresh ImageNet 초기값과 TRAIN256·500updates×16 배치에서 Adam weight decay1e−4/0만 비교하는 폐기용 두 fit이다. 원래 threshold·실패 체크포인트·프로토콜을 바꾸지 않으며, 원인 확정·복구 성공·전체 재학습 완료를 아직 주장하지 않는다.

후속 계획은 기능이 확인된 기반을 동결한 뒤 P와 matched direct-control D를 각각3seed, 6,000updates×16으로 학습하는 것이다. 이번 실패 뒤 이 본학습은 실행하지 않았다. 준비된 합성 TRAIN/calibration/selection/heldout 분할과 입력 adapter 코드는 결과의 대체물이 아니다. 동일 YOLO head 가중치의 무학습 전이도 아니다.

향후 평가 계획은 같은 반복 DEV319장·13세션·주석2,818점이다. 조건부2D 중앙값/P90, 전체 GT PCK, 매칭·결측·pose 실패, geometry 참조 기준 T/R 중앙값·P90과 coverage를 함께 기록하도록 준비했으나 현재 ResNet-18의 해당 측정은 없다. 독립 TEST나 물리 측정 ground truth도 아니다. 속도26장×5반복×3방법 역시 미실행이다.

완료 결과가 생기면 실제 RGB 이미지 위의 전후 좌표, 개선·악화·결측 사례, 원영상 크기, 팔레트 long×short×height(mm), 카메라 K, 해시·프레임별 CSV를 함께 보고한다. 현재 DOPE 또는 과거 YOLO 값을 ResNet-18의 성능으로 채우지 않는다.

[실행 코드](../../../scripts/research/pallet_resnet18_refiner_20261001_v1/run.py) · [세 모델 실행 계획](../pallet_three_estimator_20261001_v1/QUEUE_PLAN.json) · [전체 상태 스냅샷](../pallet_three_estimator_20261001_v1/STATUS_KO.md)

GitHub 상태 문서는 게시 시점의 스냅샷이며 실시간 학습 로그가 아니다. 완료 영수증이 없는 단계는 완료로 간주하지 않는다.
