# 주장 범위와 남은 증거

| 본문 주장 | 현재 근거 | 제한/다음 조건 |
|---|---|---|
| YOLO P의 중앙 keypoint 오차와 측정 비용의 관계 | 기존 Sensors 고정 DEV·paired·runtime 원본 | 재사용 319장/13세션이며 prior 대비 최저 오차 또는 D 대비 모든 지표 우월 아님 |
| 같은 local correction 원리를 DOPE에 구현 | 고정 adapter·P/D 공통 evidence·source-only protocol | 성능은 실제 DOPE 결과 전 미확정; YOLO head weight의 무학습 전이 아님 |
| 세 추정기에 적용 가능 | YOLO 기존 결과, DOPE/R18 구현 계약 | DOPE 결과와 R18의 baseline 학습·P/D 학습·before/after 평가가 필요 |
| 전체 시스템이 real-label-free | 주장하지 않음 | YOLO의 COCO-pose 초기화, 각 backbone의 이력, real DEV 평가를 구분 |
| DOPE/YOLO가 동일 학습 비용 | 주장하지 않음 | DOPE loop 0..60은 61 passes, YOLO best checkpoint; 동일한 것은 선언한 P/D head exposure |
| physical T/R 정확도 및 안정된 joint improvement | 주장하지 않음 | 기존 pose reference는 2D/K/치수에서 재구성한 값; 최근 별도 selector gate 미달성과도 구분 |
| 범용 model-agnostic 기법 | 주장하지 않음 | 각 추정기의 feature tap/channel/stride/좌표/missing 계약을 별도로 구현함 |
| 독립 TEST 일반화 | 현재 없음 | DEV 재사용/탐색 이력, 3 seeds/13 sessions의 범위를 유지 |

DOPE는 all-nine-finite 조건부 중앙값/P90, 전체 2818점 PCK, 319장 중 제외 프레임과 pose 실패/coverage를 함께 보고합니다. 조건부 GT 제외 수는 실제 nonfinite missing point 수와 다릅니다. 코너 일부가 유효한 프레임의 보조 PCK가 canonical primary endpoint를 대체하지 않습니다.

삽입할 DOPE 결과는 유리·불리한 방향과 관계없이 동일 표 구조에 들어갑니다. source calibration/selection과 고정 final checkpoint를 사용했다는 사실은 재사용 DEV를 독립 검증으로 만들지 않습니다. 세 번째 추정기는 실제 확인된 SimpleBaseline-derived ResNet18 구현이며, 이를 학습 완료나 세 번째 긍정 결과로 표현하지 않습니다. 그 baseline의 60-epoch 계획은 209,940 updates·3,358,800 exposures이고, P/D head의 별도 6000×16×3-seed 계약과 다릅니다.

현재 원고는 약 8쪽 본문+보충자료를 목표로 준비한 초안입니다. 실제 DOPE/세 번째 추정기 표가 들어온 뒤 prior 세부 구현과 per-seed 표를 보충자료로 옮기는 편집은 가능하지만, 불리한 수치·분모·실험 한계는 삭제하지 않습니다.
