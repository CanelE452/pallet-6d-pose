# 0918 초록 정사각형 팔레트 119장 감사

이 자료는 기존 GREEN150과 encoded SHA 및 decoded pixel SHA가 모두 0장 겹치는 별도 집단이다. 동결된 R0와 치수 없는 N0, 치수 입력 N2만 평가했으며 새 학습이나 checkpoint 선택은 하지 않았다.

## 2D 직접 클릭점 결과

| 모델 | median px | P90 px | PCK5 | PCK10 | PCK20 | matched/119 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 5.5257 | 11.4390 | 42.19% | 85.22% | 96.68% | 118/119 |
| N0_BASE_REPLAY_seed1 | 4.9620 | 10.0396 | 50.00% | 88.87% | 97.51% | 118/119 |
| N0_BASE_REPLAY_seed2 | 4.9664 | 9.9147 | 49.83% | 89.37% | 97.01% | 118/119 |
| N0_BASE_REPLAY_seed3 | 4.9612 | 9.8120 | 50.17% | 89.53% | 97.51% | 118/119 |
| N2_DIM_ONLY_seed1 | 5.0293 | 9.9862 | 49.17% | 89.20% | 97.01% | 118/119 |
| N2_DIM_ONLY_seed2 | 4.8158 | 9.8933 | 51.66% | 89.37% | 97.18% | 118/119 |
| N2_DIM_ONLY_seed3 | 5.0017 | 9.7672 | 49.50% | 90.20% | 97.67% | 118/119 |

주 표는 기존 GREEN 계약과 같이 declared-visible `manual_click` 602점을 모두 유지한다. 그중 이미지 밖 수동점 2개를 제외한 600점 민감도와 PnP 생성점을 포함한 proxy 결과도 `METRICS.json`에 함께 보존했다.

## 자료 상태

- 119장, 수동 클릭 602점(이미지 안 600점), PnP 생성 350점, 자동 중심 119점.
- 모든 파일은 `split=train`, `population_role=DEV`로 함께 표기되어 있다. 실제 학습 사용 여부는 이 플래그만으로 확정하지 않았다.
- 동결 N0/N2가 사용한 합성 TRAIN55,980장의 encoded image SHA와는 0장 겹친다.
- canonical pose 없음 119/119, signed axis 미확정 119/119, 수동 검토 필요 119/119.
- 저장 reprojection error 중앙값 0.9414px, 최대 8.1448px.
- 3D T/R 정답 완료 자료로 사용하지 않았다. 이번 결과는 재사용 개발 2D 감사다.

## 해석

119장 모두 같은 1.10×1.10×0.15m 치수를 가진다. 따라서 이 집단 안에서는 치수 벡터가 상수이고, 치수 정보 자체의 인과 효과를 식별할 수 없다. 같은 seed의 N2와 N0 차이는 학습된 전체 모델의 전이 결과로만 읽어야 한다.

## 주석 검토 예시

- [026500 overlay](../../../outputs/annotations/0918_dataset_square/_overlays/026500.png): 저장 reprojection error 7.7320px.
- [028924 overlay](../../../outputs/annotations/0918_dataset_square/_overlays/028924.png): 저장 reprojection error 8.1448px.
- [029710 annotation](../../../outputs/annotations/0918_dataset_square/029710.json), [029844 annotation](../../../outputs/annotations/0918_dataset_square/029844.json): visible 수동 클릭점이 이미지 밖이라 QA 민감도에서 분리했다.

[고정 데이터 스냅샷](DATASET_SNAPSHOT.json) · [전수 예측 gzip](../../../data/pallet/results/pallet_green0918_dimension_audit_v1/PREDICTIONS.json.gz) · [전수 지표 gzip](../../../data/pallet/results/pallet_green0918_dimension_audit_v1/METRICS.json.gz)
