# 일반 플라스틱 249장 — 학습 수도레이블 안정성·수동 검토

2026-09-20 KST. GPU 검사와 검토용 HTML 제작 완료. 자동 라벨 제외·학습 마스크 변경·재학습은 하지 않았다.

## 대상과 방법

- 현재 self-training 학습 후보: 일반 플라스틱 1,000장 → raw confidence·flip·LOO 259장 → frozen Replay 보정·all8 LOO 249장.
- 실제 학습에 이전에 샘플링된 217장뿐 아니라, 통과한 후보 249장 전부를 검사했다. 평가용 194장과 초록은 이번 검사 대상이 아니다.
- 원본, 밝기 ×0.85/1.15, 크기 ×0.9/1.1의 5개 입력에 각각 frozen R0 → frozen Replay를 실행했다. 추가 크롭·기하학적 탈락 필터는 없다.
- 확대·축소 좌표는 OpenCV pixel-center 식으로 원본에 역변환했다. 8개 native 코너 각각의 변형 간 RMS를 구하고, 그중 최대값을 원본 박스 대각선으로 나누어 순위를 매겼다. symmetry/GT 재배열은 하지 않았다.
- 저장된 원본 보정 좌표는 그대로 보존했다. 변형 결과를 평균내거나 원본 라벨로 대체하지 않는다.
- 불안정한 순서의 상위 50장을 우선 검토로 표시한다. 기존 80% retention의 보완 집합 크기를 검토 편의에만 사용하며, 50장 자동 탈락이나 나머지 199장 정확 판정이 아니다.

## 관측 결과

| 항목 | 값 |
|---|---:|
| 검사 이미지 | 249 |
| 입력 변형 총수 | 1,245 |
| 변형 검출/코너 결측 이미지 | 0 |
| 이미지별 최대 코너 RMS 중앙값 | 2.673px |
| 이미지별 최대 코너 RMS P90 | 6.506px |
| 이미지별 최대 코너 RMS 최대값 | 33.815px |
| 자동 제외 | 0 |
| 현재 확정된 수동 검토 | 0 |

RMS는 정답 오차가 아니라 변형 시 모델 예측의 변동이다. 높은 값은 검토 우선순위이지 오답의 증명이 아니다. 낮아도 일관되게 틀릴 수 있다. 이전 평가 집합에서 이 방법이 일관되게 좋은 라벨을 선별하지 못했던 한계도 유지한다. 따라서 이번 학습 후보에서 GT precision/PCK 향상이나 self-training 성능 향상을 주장하지 않는다.

## 사용자 검토

`outputs/pallet_type_selftrain_v1/stability_review/index.html`

각 카드: 원본 / R0 파랑 / 보정 라벨 노랑 / 변형 추론 자홍. 정답 오버레이는 없다. 클릭하면 크게 볼 수 있다.

1. `검토 우선 50장`부터 확인한다.
2. 이미지별 유지·제외·보류를 표시한다. 확실하지 않으면 보류한다.
3. 이미지의 일부 코너만 문제이면 해당 P0–P7을 체크하여 학습 제외를 제안한다. 이미 제외하는 이미지에 코너 표시를 해도 자동 학습 반영되지 않는다.
4. `검토 결과 JSON 내려받기`로 파일을 저장한다. 브라우저 임시 저장만 믿지 말고 JSON을 보관한다. 같은 페이지의 불러오기로 이어서 확인할 수 있다.
5. 이후 검토 JSON을 검사하고 별도 데이터 버전으로 반영하는 단계가 필요하다. 현재 train.txt·원본 수도레이블·가중치는 불변이다. 미검토/보류를 좋은 라벨로 취급하지 않는다.

사람이 선별하거나 코너 학습 제외를 표시한 경우 향후 논문에는 수동 검토를 명시해야 한다. 이를 완전 자동 unlabeled adaptation으로 보고하면 안 된다. 평가 annotation은 수정하지 않았다.

## 검증 및 재현

- 좌표 역변환·원본 불변·RMS 집계·순위 검사 13개 통과.
- 원본 재추론과 기존 수도레이블 parity: R0 최대차이 0, Replay 최대차이 0.000168px (허용치 0.001px 이내).
- 원본 249개 candidate payload와 `UNCHANGED_CANDIDATES.json` 완전 일치.
- 격리된 Chrome에서 249장 표시, 우선50 필터, 이미지 선택, 코너 제외, 임시 저장, 249개 레코드 JSON export 및 이미지 표시 검사 통과. 사용자 브라우저의 검토 선택은 건드리지 않았다.
- RTX 3080, 변형 추론 루프 약 38.3초, 최대 관측 62°C. 기존 RustDesk 프로세스 유지, 재부팅/드라이버 변경 없음.
- 원시 결과: `data/pallet/results/pallet_type_selftrain_v1/stability_review/INFERENCE.json`, `REVIEW_QUEUE.json`, `UNCHANGED_CANDIDATES.json`.
- 프로토콜·검증: 이 문서 폴더의 `PROTOCOL.json`, `VERIFICATION.json`, `BROWSER_CHECK.json`.

```bash
MPLCONFIGDIR=/tmp/pallet-selector-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_type_selftrain_v1.stability_review run
MPLCONFIGDIR=/tmp/pallet-selector-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_type_selftrain_v1.stability_review gallery
```
