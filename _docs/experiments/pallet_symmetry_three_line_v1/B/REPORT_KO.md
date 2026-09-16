# B — 세 선 보조의 개발 성능 이득은 입증되지 않음

## 결과와 판단

**합성·실사 모두 `LINE_AUXILIARY_NOT_ESTABLISHED`.** Calibration256에서 B1/B2/B3 모두 eta=1이 선택됐다.
계수/seed/temperature/lambda/cap를 더 탐색하지 않았다. 원래 R0/P1–3/corrected DHT Hough1은 동결했고 새 학습은 0이다.

아래는 각 seed별 지표의 3-seed 평균이다. Pooled corner8 지표와 frame-mean 기반 primary는 다른 집계다.
이 표는 원래 논문의 matched 9-point 표를 대체하지 않는다.

| 평가 | 군 | E_sym ↓ | pooled median px | pooled P90 px |
|---|---|---:|---:|---:|
| 합성 | B0 P | 0.005342253 | 1.83783 | 6.82123 |
| 합성 | B1 한 선 | 0.005341702 | 1.83838 | 6.82924 |
| 합성 | B2 세 선 동일가중 | 0.005343325 | 1.83199 | 6.79168 |
| 합성 | B3 세 선 모호성가중 | 0.005341033 | 1.83844 | 6.78712 |
| 합성 | B4 다른 영상 선 | 0.005364700 | 1.85119 | 6.84265 |
| DEV | B0 P | 0.048683574 | 6.11942 | 61.63646 |
| DEV | B1 한 선 | 0.048677670 | 6.11874 | 61.67875 |
| DEV | B2 세 선 동일가중 | 0.048680530 | 6.08882 | 61.17100 |
| DEV | B3 세 선 모호성가중 | 0.048679214 | 6.10846 | 61.15016 |
| DEV | B4 다른 영상 선 | 0.048695681 | 6.12702 | 61.91829 |

B3−B0 primary 차이:

- 합성: −0.000001219, 95% CI [−0.000005159, +0.000002977]. 512장을 예측했고 corner 주석이 전혀 없는 2장을 명시적으로 제외해 평가 분모는 510장이다. 미검출을 제외한 것이 아니다.
- 실사: −0.000004359, 95% CI [−0.000021313, +0.000013411]. 319장·13세션 paired cluster bootstrap 10,000회, seed20260916. 8개 unmatched 프레임은 primary에서 diagonal penalty로 유지된다.
- 두 평가 모두 2/3 seed의 primary는 개선됐지만 CI가 0을 포함한다. 동등성/비열등성 증명도 아니다.
- B3−B1, B3−B2도 CI가 0을 포함한다. **세 선의 필수성 또는 entropy weighting이 개선 원인이라는 주장 모두 불가**.
- 합성의 B3−shuffle는 음의 CI지만 실사에서는 0을 포함한다. 사후 음성 대조의 탐색적 결과이며 B0 대비 이득을 대신 증명하지 않는다.

## 실제 실행과 누출 방지

원래 합성 256/512 observation cache에서 P 2,304회, DHT observation 768회 실행.
실사 전체 **3,008장** R0 추론 후 실제 검출이 있는 1,858장에서 DHT 1,858회, P 5,574회 실행했다.
실사 원래 R0의 모든 candidate/box/score/좌표는 cache와 bit-exact였고, positive의 원래 P 좌표도 기존 결과와 bit-exact였다.
각 군의 좌표를 먼저 직렬화한 다음 GT를 읽어 평가했다. GT로 inference orbit, eta, reliability, candidate를 선택하지 않았다.

Negative도 GT 양성 여부로 head를 끄지 않았다. 선택 후보의 코너는 이동할 수 있지만 candidate 수·순서·box·score·center8·미선택 후보는 보존된다.
따라서 음성 **검출 지표**는 그대로다. ‘negative의 모든 코너 좌표까지 그대로’라는 주장은 하지 않는다.

세 선은 실제 cuboid 12-edge graph의 코너별 degree3 모서리다. 하나의 선에서 top3 peak를 쓰지 않았다.
선 관측은 stem → line_head까지만 실행했으며 WLS/utility head 출력은 사용하지 않았다.
전체 2,340-bin posterior와 균일 lattice 배경 우도비를 사용했다. 점 생성은 원래 P expectation/lambda/cap뿐이다.
Null도 원래 점 위치에서 동일 scoring을 받았다. 가용 선이 없거나 평평한 posterior는 중립이며 분모는 3으로 고정한다.

## 선·모호성·손해

합성 예상 semantic edge 6,144개 중 GT로 평가 가능한 5,987개에서:

- DHT MAP 설명용 선의 endpoint normal-distance median 2.089px / P90 8.179px.
- full posterior 기대 normal-distance median 6.043px / P90 16.154px.
- P1 두 점 연결선의 같은 거리 median 1.008px / P90 3.776px.

이 선 정확도를 실행 gate로 사용하지 않았다. 최종 후보 재평가 결과로 판단했다.
Entropy는 선 분포의 집중도이며 실제 가림 확률이 아니다. visibility/night/viewpoint metadata가 검증되지 않은 분류는 만들지 않았다.
높이·폭·깊이, 투영 길이, ambiguity, asset/source/C1/C2와 GT-assisted point-error quartile 표는 별도 JSON/CSV에 있다.

좋았던 B0 점 <5px가 B3에서 >10px로 바뀐 점은 합성·실사 모든 seed에서 0개였다.
그러나 실사 frame 평균 오차가 악화한 프레임은 seed별 **148/146/150장**이다. 이 결과를 ‘overall safe’로 묶지 않는다.
실사 평균 raw frame-error 차이는 seed별 −0.00101/−0.01260/−0.00314px로 작다.
합성에서는 한 seed 한 프레임의 평가 symmetry branch가 바뀌었고, 실사는 바뀐 프레임이 0개다.

## 6D와 시간

기존 prediction-only axis selector + SQPnP/RefineLM을 유지했다. 모델 원점이 centroid여서 중심 이동은 t와 같다.
6D 표는 물리 축 대응을 명시한 C1/C2 orientation과 실제 선택된 cuboid extent로 계산한 보조 지표다.
실사 reference는 독립 계측이 아닌 geometry-reconstructed GT이며 원래 paper pose 표를 소급 수정하지 않는다.
Seed1 실사 centroid translation median은 B0 6.86470cm → B3 6.88538cm였다. 2D의 작은 감소가 6D 전반의 개선을 뜻하지 않는다.

RTX3080 동일 세션, 26영상 × 5반복, 군별20회 warmup, seed1의 **RGB부터 PnP까지 전체** median:

| 군 | 전체 median ms | 전체 P90 ms |
|---|---:|---:|
| B0 P | 12.727 | 13.996 |
| B1 한 선 | 36.554 | 38.505 |
| B2 세 선 동일가중 | 39.265 | 41.256 |
| B3 세 선 모호성가중 | 39.326 | 41.432 |

B3는 현재 reference 구현에서 B0의 약 3.09배 시간이다. 작은 eta나 작은 모델 parameter 수가 무료 추론을 의미하지 않는다.
B4는 donor-image 사후 대조로 배포 runtime을 주장하지 않는다. 메모리 값은 3개의 작은 P head가 상주한 shared wrapper의 peak이며 모델별 격리 메모리로 비교하지 않는다.

## 수치 감사와 범위

GPU/CPU bias 검산 최대 차이 2.70e-5로 사전 허용 오차 통과. Source 768건 순열을 실제 저장 3D 코너에서 다시 유도했다.
과거 DHT는 native batch32에서 512장 출력이 완전 재현됐지만 batch1에서는 cuDNN 산술이 달라 hard-MAP가 두 프레임에서 바뀌었다.
이것을 숨기지 않고 GT/성능 열람 전에 `NUMERIC_LOCK_AMENDMENT`를 남겼다. B는 full posterior를 쓰며 batch1-vs32 total-variation gate를 통과했다.
무주석 프레임 bootstrap 처리와 strict JSON의 invalid-target 표현 오류도 `SCORER_CORRECTION`에 보존했다. 재학습/seed 변경은 없었다.

다음 판단: **현재 고정 P/DHT·이 분포 수식·이 예산에서는 선 보조를 최종 추론에 추가할 근거가 부족하다.** Hough 원리의 불가능성이나 대칭 지도 A의 실패를 뜻하지 않는다. 자동 추가 탐색은 하지 않는다.
