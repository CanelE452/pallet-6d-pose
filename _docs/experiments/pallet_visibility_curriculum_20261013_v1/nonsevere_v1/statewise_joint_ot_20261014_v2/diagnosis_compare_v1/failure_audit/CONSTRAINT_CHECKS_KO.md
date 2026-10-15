# 제약·reference target 교차 확인

봉인된 FAILURE_AUDIT.json을 유지하고 세부 집계를 추가했습니다. 98개 프레임의 모델 호환 판정은 기존 봉인 score와 모두 같으며, 호환 reference의 native projection은 고정 body reference/mapping 투영과 1e-9 px 이내로 일치합니다. 현재 source 19개 hash가 그대로입니다.

| 분할 | 모델 호환 | reference BI budget 통과 | reference BI geometry 통과 | reference BI canvas 통과 | 실제 DIRECT 좌표가 BI budget 안 |
|---|---:|---:|---:|---:|---:|
| CAL | 44 | 4 | 42 | 44 | 109/242 |
| TEST | 42 | 10 | 42 | 42 | 150/224 |

Reference pose의 배제는 주로 B/I의 작은 covariance budget에서 발생합니다. Source human DIRECT 주석도 B/I budget 밖인 경우가 많습니다. 다만 reference pose는 독립 물리 ground truth가 아니므로 이 표만으로 실제 물리 정답을 배제했다고 단정할 수 없습니다.

| 분할 | DIRECT 주석 / 정수 좌표 | 주석과 derived reference projection 평균 / 최대 px |
|---|---:|---:|
| CAL | 268 / 248 | 2.778 / 11.066 |
| TEST | 286 / 257 | 2.574 / 10.540 |

같은 이상적 투영을 가장 가까운 정수 픽셀로만 반올림한다면 이동 상한은 약 0.707 px입니다. 관측 불일치가 이보다 크다고 해도 annotation 실수, rigid cuboid의 실제 모서리 차이, pose fit 잔차, 카메라/등록/phase 문제 중 어느 원인인지는 이번 진단으로 확정하지 않았습니다. 하나의 rigid pose가 모든 주석에 정확히 맞지 않을 가능성을 구분해야 하며, 새 GT pose fit이나 재인덱싱을 하지 않았습니다.

| TEST subset | N | 원본 / current 평균 cm | 원본 / current 성공 |
|---|---:|---:|---:|
| 모델 대응 호환=True | 42 | 2.644 / 2.519 | 37 / 36 |
| 모델 대응 호환=False | 7 | 7.677 / 9.482 | 3 / 3 |

호환 subset에서는 평균 translation이 조금 줄었지만 success는 감소했습니다. 이 subset 분석은 사후 선택된 진단이며 full TEST의 열등 결과를 대체하는 superiority 근거가 아닙니다.

[세부 JSON](CONSTRAINT_DECOMPOSITION.json) · [검증](VALIDATION.json) · [원래 봉인 진단](README_KO.md)
