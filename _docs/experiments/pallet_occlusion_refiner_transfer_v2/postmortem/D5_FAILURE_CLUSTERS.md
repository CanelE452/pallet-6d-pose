# D5 — 실패 그룹의 기술 통계

[확인] selector/classifier 학습이나 threshold 탐색은 하지 않았다. 코너들은 프레임에 군집되어 있으므로 독립 표본의 유의성 검정으로 해석하지 않는다.

| [확인] 그룹 | n | bbox area med | aspect med | R0 conf med | A11 move med | A10/A11 disagreement med |
|---|---|---|---|---|---|---|
| GOOD->GOOD | 331 | 18097.351 | 6.074 | 0.997 | 3.787 | 1.036 |
| GOOD->BAD | 25 | 40842.635 | 5.337 | 0.999 | 5.653 | 3.734 |
| BAD->GOOD | 19 | 47018.897 | 4.693 | 0.998 | 10.413 | 5.343 |
| BAD->BAD | 338 | 15005.107 | 5.583 | 0.996 | 4.987 | 2.000 |

[확인] `GOOD->GOOD` canonical 코너 빈도 `{'0': 45, '1': 63, '2': 64, '4': 35, '6': 20, '7': 34, '5': 32, '3': 38}`; 세션 빈도 `{'eval_night08': 35, 'eval_night09': 41, 'eval_outside': 28, 'eval_pallet07': 94, 'eval_pallet09': 133}`.

[확인] `GOOD->BAD` canonical 코너 빈도 `{'5': 4, '6': 6, '4': 8, '0': 1, '3': 3, '7': 3}`; 세션 빈도 `{'eval_night08': 3, 'eval_night09': 5, 'eval_outside': 2, 'eval_pallet07': 11, 'eval_pallet09': 4}`.

[확인] `BAD->GOOD` canonical 코너 빈도 `{'4': 4, '7': 1, '0': 4, '3': 3, '5': 4, '6': 2, '1': 1}`; 세션 빈도 `{'eval_night08': 1, 'eval_night09': 4, 'eval_pallet07': 10, 'eval_pallet09': 4}`.

[확인] `BAD->BAD` canonical 코너 빈도 `{'3': 46, '0': 43, '4': 46, '7': 55, '5': 50, '6': 62, '1': 18, '2': 18}`; 세션 빈도 `{'eval_night08': 53, 'eval_night09': 72, 'eval_outside': 10, 'eval_pallet07': 82, 'eval_pallet09': 121}`.

## native_disagreement_px (고정 기술통계 구간)

| [확인] 구간 | n | 획득 | 손실 | good 손상/분모 |
|---|---|---|---|---|
| [0,5) | 608 | 9 | 15 | 0/113 |
| [5,10) | 76 | 8 | 5 | 1/3 |
| [10,20) | 26 | 2 | 4 | 0/0 |
| [20,inf) | 3 | 0 | 1 | 0/0 |

## A11_native_correction (고정 기술통계 구간)

| [확인] 구간 | n | 획득 | 손실 | good 손상/분모 |
|---|---|---|---|---|
| [0,5) | 393 | 6 | 11 | 0/96 |
| [5,10) | 205 | 3 | 7 | 0/19 |
| [10,20) | 99 | 7 | 7 | 1/1 |
| [20,inf) | 16 | 3 | 0 | 0/0 |

[확인] 매칭된 85프레임에서 disagreement와 A11−A10 평균오차의 Pearson 상관은 -0.032, 차이 절댓값과는 0.734다. 큰 불일치는 차이의 크기와 연관되지만 어느 쪽이 맞는지의 방향 신호로 검증된 것은 아니다.

[확인] R0 confidence는 모든 그룹에서 높고 획득/손실 그룹의 bbox 크기·aspect 분포도 겹친다. bbox area는 거리의 직접 측정값이 아니다. verified visible point count는 일부만 검토된 점의 수이지 실제 visible 수가 아니다. primary93에 PnP 가능 flag는 없으므로 UNKNOWN으로 남겼다.

[추정] 큰 보정량 하나로 손상을 거르기 어렵다. 가장 큰 이동 구간에도 획득이 있고, good-damage 분모는 작은 구간에서 극히 작다. 기존 RGB utility selector의 GREEN 실패를 재현하는 동일 feature 재학습을 제안할 근거는 부족하다.
