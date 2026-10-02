# DOPE(VGG) + P/D 추가 검증

2026-10-02 현재 상태: **DOPE의 학습·319장 DEV 평가·속도 측정·이미지 보고서 완료**. ResNet-18까지 포함한 결과는 [세 백본 최종 보고서](../pallet_three_backbone_closeout_20261002_v1/REPORT_KO.md)에 정리했습니다.

이 DOPE 실험의 학습 입력은 RGB에서 얻은 특징이며 실제 치수는 PnP에만 사용했습니다. RGB+실제 치수 공동 학습은 별도 ResNet-18 실험에서 검사했으므로, 이 DOPE 결과를 치수 조건부 학습의 성과로 해석하지 않습니다.

교수님이 요구한 세 기반(YOLO·DOPE·ResNet-18) 비교 중 DOPE 부분이다. [실제 결과·이미지](REPORT_KO.md), [집계 지표](DEV_RESULTS.json), [짝지은 비교](DEV_PAIRED_RESULTS.json), [실측 속도](RUNTIME.json)를 공개한다. 원래 실사 T/R 안정적 공동 개선 목표와 독립 TEST는 이번 결과로 통과시킨 것이 아니다.

| 지표 | DOPE | DOPE+P, 3seed 통계 평균 |
|---|---:|---:|
| 조건부 2D 중앙값 px | 12.585 | 7.461 |
| 조건부 2D P90 px | 50.609 | 50.713 |
| 전체GT PCK10 % | 19.872 | 36.775 |
| 유효 pose T 중앙값 cm | 10.046 | 8.529 |
| 유효 pose R 중앙값 ° | 3.441 | 2.884 |
| pose coverage | 210/319 | 210/319 |

중앙값 이득과 함께 2D P90의 악화와 결측 유지도 보고한다. 조건부 2D 분모는190/319장이고 전체GT PCK는2,818점의 실패까지 포함한다. P−D의 T/R paired 세션 구간은 모두0을 포함하므로 P의 pose 우월성을 주장하지 않는다. 대표seed1 전체 경로 중앙시간은 DOPE62.205ms/P64.924ms다.

기존 팔레트 DOPE 최종 epoch60 가중치를 동결한다. 9 belief / 16 affinity의 원 네트워크를 사용하지만 이번 평가는 semantic-channel peak decoder이며 affinity 기반 다중 객체 연결 평가가 아니다.

공통 보정 실험은 기반을 동결한 후 P와 matched direct-control D를 각각3seed, 6,000updates×16으로 완료했다. 합성 TRAIN/calibration/selection/heldout 분할을 사용했고 추가 실사 학습은0장이다. 실제 TRAIN 사용 행은44,063장이다. 기반별 내부 특징 채널·stride adapter와 source 오류분포가 다르므로 동일 YOLO head 가중치의 무학습 전이가 아니다.

평가는 같은 반복 DEV319장·13세션·주석2,818점이다. 조건부2D 중앙값/P90, 전체 GT PCK, 매칭·결측·pose 실패, geometry 참조 기준 T/R 중앙값·P90과 coverage를 함께 기록했다. 독립 TEST나 물리 측정 ground truth가 아니다. 속도는26장×5반복×3방법, 각20회 warmup 후390회 측정했고 정확도3seed 평균과 속도대표 seed1을 구분한다.

완료 보고서에는 실제 RGB 이미지 위의 전후 좌표, 개선·악화·결측 사례, 원영상 크기, 팔레트 long×short×height(mm), 카메라 K와 출처 hash를 포함한다. GitHub review bundle에서는 원시 전 프레임 좌표·CSV를 제외하고 불리한 선정 사례를 함께 유지한다.

[실행 코드](../../../scripts/research/pallet_dope_refiner_20261001_v1/run.py) · [세 모델 실행 계획](../pallet_three_estimator_20261001_v1/QUEUE_PLAN.json) · [당시 상태 스냅샷](../pallet_three_estimator_20261001_v1/STATUS_KO.md) · [세 백본 최종 보고서](../pallet_three_backbone_closeout_20261002_v1/REPORT_KO.md)

GitHub 상태 문서는 게시 시점의 스냅샷이며 실시간 학습 로그가 아니다. 완료 영수증이 없는 단계는 완료로 간주하지 않는다.
