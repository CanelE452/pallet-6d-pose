# V263 저장 current223 평가와 고정 규칙 12셀 시각화

4개 SOURCE fit 및 규칙 동결 뒤 current 예측 전체를 저장·봉인하고, 그 뒤 기존 정답·기준선에 연결했습니다. 새 학습이나 문턱·crop·수락 규칙 선택은 하지 않았습니다. 현재 평가셋은 이미 노출되어 반복 사용한 개발 평가이며 새로운 blind 검증이 아닙니다.

분류 PRIMARY는 같은 query D/E 1,237개(D 1,147 / E 90)입니다. 좌표오차나 입력 가용성을 추가로 필터하지 않았습니다. LOC-only와 ORIGINAL은 semantic VIS를 학습하지 않아 분류 N/A입니다. VIS-only의 위치는 학습되지 않았으며 원q를 위치 기준선으로 표시합니다.

| 모델 | D 정답 / 1147 | E 정답 / 90 | D recall | E recall | U |
|---|---:|---:|---:|---:|---:|
| v61 | 260 | 88 | 22.67% | 97.78% | 0 |
| CAL_LEGACY | 655 | 83 | 57.11% | 92.22% | 177 |
| V261_CONTROL | 964 | 68 | 84.05% | 75.56% | 0 |
| V261_TREAT | 959 | 68 | 83.61% | 75.56% | 0 |
| JOINT_SELECTED | 882 | 35 | 76.90% | 38.89% | 0 |
| VIS_ONLY_SELECTED | 1145 | 0 | 99.83% | 0.00% | 0 |

v61 및 CAL은 기존 저장 응답을 사용했습니다. CAL의 이전 개발/평가 노출 한계와 V261의 반복 평가 한계를 유지합니다. LOC-only의 분류 N/A를 U나 실패로 합치지 않았습니다. D/E×D/E/U와 각 precision/recall은 [CLASSIFICATION.csv](actual_evaluation_v1/CLASSIFICATION.csv)에 있습니다.

| 그룹 | 모델 | D 정답/분모 | E 정답/분모 |
|---|---|---:|---:|
| WOOD_PRIMARY105 | JOINT_SELECTED | 519/616 | 6/29 |
| WOOD_PRIMARY105 | VIS_ONLY_SELECTED | 614/616 | 0/29 |
| SQUARE118 | JOINT_SELECTED | 363/531 | 29/61 |
| SQUARE118 | VIS_ONLY_SELECTED | 531/531 | 0/61 |

위치 지표는 기존 manual-click D 전체 857개를 보존합니다. 원입력 U 3개는 unscorable로 별도 표시하여 실제 좌표 채점 분모는 854개입니다. 같은 점의 원q와 최종점으로 3px 회복·손상을 계산했으며, reject된 제안은 원q를 유지합니다. 저장된 초기 오차와 재계산 오차는 정확히 일치했습니다.

| 위치 방법 | ≤3px / 854 | 수락 수 | 수락 중 ≤3px | 회복 / 손상 | 최종 median / p90(px) |
|---|---:|---:|---:|---:|---:|
| ORIGINAL_QUERY | 376/854 | 0 | N/A | 0 / 0 | 3.562 / 10.333 |
| LOC_ONLY_INITIAL_GEOMETRY | 461/854 | 578 | 338/578 (58.48%) | 148 / 63 | 2.788 / 8.402 |
| JOINT_SELECTED | 457/854 | 520 | 303/520 (58.27%) | 140 / 59 | 2.781 / 9.090 |

원래 오차≤3px인 manual-click D 376개에서는 JOINT가 102개를 E로 분류했습니다(VIS-only는 1개). 위치 수정은 이 376개 중 LOC-only 63개, JOINT 59개를 >3px로 손상시켰습니다. 전체 manual D의 회복과 원래 정확했던 점의 손상을 함께 보아야 합니다. 수락된 점의 정확도와 전체 scorable 정확도는 서로 다른 분모입니다.

Human SELF 136개는 [HUMAN_SELF.csv](actual_evaluation_v1/HUMAN_SELF.csv)의 별도 binary 응답 분포입니다. SELF 정답률이나 물리 native gold로 재명명하지 않았습니다.

시각화는 JOINT를 principal로 WOOD/SQUARE 각각 D→D, manual D≤3→E, E→E, E→D, manual D 회복, manual D 손상 6셀을 사용했습니다. 각 셀 전체 census와 최소 원본 인덱스를 RGB 전에 봉인했고, 빈 셀은 대체하지 않는 규칙입니다. 이번에는 12셀 모두 nonempty였습니다. 결과 기반 진단 사례이며 모델·학습 설정 선택이나 인과적/통계적 일반화의 근거가 아닙니다.

[12셀 갤러리](current_fixed12_gallery_v1/INDEX_KO.html) · [overview](current_fixed12_gallery_v1/OVERVIEW.png) · [선택/전체 census](actual_evaluation_v1/CASE_IDS_BEFORE_PIXEL_READ.json)

실제 선택 crop은 C가 예측 전에 저장한 RAW96이며 원본 실사 RGB를 다시 열지 않았습니다. clean crop PNG 24개는 저장 tensor의 decoded RGB byte SHA와 일치합니다. 표시 heatmap은 셀별 min/max 색 정규화이며 calibration이나 가시성 정답이 아닙니다. q/correction/final/reference 표식은 별도 표시 패널에만 있습니다.

이 결과로 모델을 채택하거나 재학습·문턱 조정하지 않았습니다. 실물 native 의미/직접 관측성의 새로운 정답도 만들지 않았습니다.
