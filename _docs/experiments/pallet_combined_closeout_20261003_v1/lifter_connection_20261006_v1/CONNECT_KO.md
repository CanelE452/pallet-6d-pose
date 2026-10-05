# 기존 리프터 자료를 정확도 평가에 연결하기

**새 촬영이나 전체 재어노테이션 없이, 기존 12장의 남은 5좌표를 먼저 채우고 기존 입력을 제출한 뒤 선택 박스만 확인합니다.**

## 1. 사진마다 남은 점 한 개만 클릭

아래 명령은 기존 `scripts/annotate/annotate.py`를 사용합니다. 기존 67좌표와 자체 가림 24상태를 유지하며 5장만 엽니다. 각 사진에서 선택된 실제 코너 한 번 클릭 → **S 저장·다음**입니다. 보임 여부는 이미 답변했으므로 다시 묻지 않습니다. 노란 빈 원은 본인 PnP 위치 참고이며 그 좌표를 정답으로 자동 저장하지 않습니다.

```bash
DISPLAY=:0 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python /home/minjae/Documents/github/pallet-pose/scripts/research/pallet_lifter_case_review_20261003_v1/open_existing_annotation.py --pass primary --visibility-only --revisit-saved --batch-plan /home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_lifter_case_review_20261003_v1/review/REMAINING_CORNERS_5_V1.json
```

| 프레임 | 추가할 코너 |
|---|---:|
| 174126:13 | 3 |
| 174126:419 | 5 |
| 174342:1190 | 5 |
| 174925:32 | 5 |
| 174925:1002 | 5 |

`--revisit-saved`가 필요합니다. 이 옵션이 없으면 기존 보임 분류가 완료된 것으로 인식해 창이 열리지 않습니다. 완료한 좌표를 다시 찍는 옵션이 아니라 남은 좌표 작업과 기존 입력 제출 화면을 열기 위한 옵션입니다.

## 2. 기존 12장 입력을 한 번 제출

다섯 좌표의 실제 저장을 확인한 다음 아래 명령을 실행합니다. 기존 점은 다시 찍지 않습니다. **B 기존 입력 제출**을 누르고 실제 작업 이력을 한 번 답한 뒤 기존 입력을 사용한다는 확인을 합니다. 아직 다섯 좌표가 없는데 B를 먼저 눌러 판단 보류로 바꾸지 않습니다. 제출 뒤 공식 참조 파일과 원시 입력 보존 아카이브가 생성됩니다.

```bash
DISPLAY=:0 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python /home/minjae/Documents/github/pallet-pose/scripts/research/pallet_lifter_case_review_20261003_v1/open_existing_annotation.py --pass primary --visibility-only --revisit-saved --batch-plan /home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_lifter_case_review_20261003_v1/review/SMALL_BATCH_12_V1.json
```

## 3. 모델이 선택한 박스만 12장 확인

참조를 제출한 다음 아래 명령을 실행합니다. 원이미지와 고정 선택 박스만 표시되며 모델 코너·방법명·오차는 표시하지 않습니다. **1 같은 파렛트 / 2 다른 대상 / 3 판단하기 어려움** 중 실제 판정을 선택하면 저장하고 다음으로 넘어갑니다. Base와 N3는 동일한 선택을 사용합니다.

```bash
DISPLAY=:0 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python /home/minjae/Documents/github/pallet-pose/scripts/research/pallet_lifter_case_review_20261003_v1/open_object_match_annotation.py --batch-plan /home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_lifter_case_review_20261003_v1/review/SMALL_BATCH_12_V1.json
```

## 평가와 범위

코너·대상 입력이 완료되면 기존 8,910프레임 예측을 그대로 재사용하여 12장의 가시 코너 오차 중앙값/P90과 PCK@10px를 계산합니다. 잘못 선택한 객체와 예측 결측은 평가 실패로 남깁니다. **부분 12장 평가**이며 전체 120장 검수 완료, 보류 103장 완료, 반복 23장 완료로 보고하지 않습니다. 독립 물리 T/R 참조는 없으므로 cm·각도 정확도는 x입니다. 정지 잡음은 별도로 실제 정지 구간을 사람이 확인해야 합니다.

사람 입력이 세 단계 모두 저장되면 작업자가 아래 명령으로 평가와 결과 생성을 이어갑니다. 승인·좌표·대응이 미완료면 해당 평가만 대기로 남기며 사람 답변을 자동 생성하지 않습니다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python /home/minjae/Documents/github/pallet-pose/scripts/research/pallet_combined_closeout_20261003_v1/evaluate_lifter_connection_20261006.py
```

이번 준비 검산은 새 학습·모델 추론·사람 승인·대상 대응을 만들지 않았습니다. 원시 JSON의 다섯 좌표는 여전히 null이고 공식 참조 승인 0장입니다. `--prepare-only` 두 번은 큐와 모듈 연결만 읽으며 창과 승인 파일을 만들지 않았습니다. 향후 실제 사용자 클릭/S/B는 해당 검수 저장소에 새 입력·승인 파일을 씁니다.

[실제 연결 검산·입력 해시·실행 비용](CONNECTION_AUDIT.json)
