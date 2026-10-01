# SimpleBaseline-derived ResNet-18 + P/D 추가 검증

현재 상태: **실험 진행 중 — 추가 모델의 성능 결과 미완료**.

교수님이 요구한 세 기반(YOLO·DOPE·ResNet-18) 비교 중 하나다. 이 README나 CPU/GPU smoke 통과는 논문 결과가 아니다. 실제 결과는 실행 후 생성되는 `DEV_RESULTS.json`, `RUNTIME.json`, `REPORT_KO.md`로 확인한다. 원래 실사 T/R 안정적 공동 개선 목표는 아직 입증되지 않았다.

독립 RGB 입력의 ResNet-18 + 3 deconvolution + 9 heatmap 기반 추정기를 새로 학습한다. ImageNet 공식 가중치에서 합성 TRAIN55,980장 ×60epoch, batch16으로209,940updates/3,358,800exposures 후 최종 epoch60을 동결한다. calibration1,004장의 loss는 기록하고 checkpoint를 선택하지 않는다. YOLO crop/DOPE 출력/실사 GT를 입력으로 쓰지 않는다.

공통 보정 실험은 기반을 동결한 후 P와 matched direct-control D를 각각3seed, 6,000updates×16으로 학습한다. 합성 TRAIN/calibration/selection/heldout 분할을 사용하고 추가 실사 학습은0장이다. 실제 usable 행 수와 결측은 캐시 완료 후 공개한다. 기반별 내부 특징 채널·stride adapter와 source 오류분포가 다르므로 동일 YOLO head 가중치의 무학습 전이가 아니다.

평가는 같은 반복 DEV319장·13세션·주석2,818점이다. 조건부2D 중앙값/P90, 전체 GT PCK, 매칭·결측·pose 실패, geometry 참조 기준 T/R 중앙값·P90과 coverage를 함께 기록한다. 독립 TEST나 물리 측정 ground truth가 아니다. 속도는26장×5반복×3방법, 각20회 warmup 후 실제 측정하고 정확도3seed 평균과 속도대표 seed1을 구분한다.

완료 보고서에는 실제 RGB 이미지 위의 전후 좌표, 개선·악화·결측 사례, 원영상 크기, 팔레트 long×short×height(mm), 카메라 K, 해시·원시좌표/프레임별 CSV를 포함한다. 결과에 따라 불리한 값도 남긴다.

[실행 코드](../../../scripts/research/pallet_resnet18_refiner_20261001_v1/run.py) · [세 모델 실행 계획](../pallet_three_estimator_20261001_v1/QUEUE_PLAN.json) · [전체 상태 스냅샷](../pallet_three_estimator_20261001_v1/STATUS_KO.md)

GitHub 상태 문서는 게시 시점의 스냅샷이며 실시간 학습 로그가 아니다. 완료 영수증이 없는 단계는 완료로 간주하지 않는다.
