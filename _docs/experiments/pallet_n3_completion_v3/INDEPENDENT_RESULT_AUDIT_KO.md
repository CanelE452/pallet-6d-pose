# N3 v3 독립 결과 감사

이 메모는 완료된 학습·선택·추론·평가·runtime·기록 리프터 산출물을 원시 행에서 다시 계산한 결과다. 숫자는 결과가 좋게 보이도록 다시 선택하지 않았다. DOPE와 ResNet-18의 `N3` 값은 별도 표기가 없으면 세 seed 통계의 산술평균이다.

## 무결성·노출 확인

- 관련 live JSON 16개에서 수집한 고유 `path/sha256/bytes` binding 45개를 다시 해시했고 불일치는 0개였다.
- DEV319의 네 방법별 319 frame row에서 pooled median/P90, full median/P90, PCK10, `E_sym`, pose median/P90을 다시 계산한 차이는 모두 정확히 0이었다. GREEN0918의 DOPE·ResNet-18 각 8개 방법/모드와 YOLO 20개 방법/모드도 요약값과 재계산값의 차이가 0이었다.
- synthetic source 60,000 image SHA와 DEV319 319장 및 GREEN0918 119장의 교집합은 0장이었다. DEV319와 GREEN0918 사이 교집합도 0장이었다.
- DOPE·ResNet-18의 DEV319/GREEN0918 예측 전체에서 base와 N3의 detection, box, score, center8, valid mask 불일치는 0건이었다. 원본 영상 대각선의 1% 이동 cap 위반도 0건이었다.
- 예측 receipt는 `GT_inputs=false`, `camera_inputs=false`, `base_receives_dimensions=false`, `n3_receives_dimensions=true`다. 이는 frozen base가 RGB만 받고, N3 head가 frozen feature·초기점·box와 함께 치수를 받았다는 뜻이다.

## 학습·선택 계약

- DOPE 3회와 ResNet-18 3회가 각각 6,000 update × batch 16으로 끝났다. 합계는 36,000 update와 576,000 exposure다.
- 모든 head는 23,331 parameter, FP32, AMP/TF32 off이며 base checkpoint는 재학습하지 않았다. 각 seed의 초기 상태, sample order, 최종 상태와 checkpoint hash는 서로 구분된다.
- synthetic calibration에서 여섯 fit 모두 temperature `1.0`이 선택됐다. 지원 frame은 DOPE 774, ResNet-18 989였고 real data 선택은 없었다. synthetic held-out는 temperature 고정 뒤 집계했다.
- visual evidence와 base logits를 고정하고 W,D,H만 `[1.1,1.3,0.11]`에서 `[0.8,0.59,0.14]`로 바꾼 CPU 감사에서 모든 head의 1,776 logits가 변했다. 최대 절대 logit 변화는 DOPE seed1/2/3 `6.1822/9.9286/12.8505`, ResNet-18 `8.3580/7.8720/5.5594`였다. 치수 경로가 실제 연결됐다는 기계적 증거이며, 치수의 정확도 증분을 분리한 인과 증거는 아니다.

## 대칭 target 실제 활성 감사

재현 가능한 결과는 `SYMMETRY_ACTIVATION_AUDIT.json`에 있다. 생성기는 exact `model.select_symmetric_target`를 CPU에서 호출한다. 입력은 각 base cache의 `points`, `point_valid`, `boxes`, `gt_points`, `gt_valid`, sidecar의 `permutations/group_valid`, 여섯 fit의 실제 order다. 각 whole-object permutation에 대해 frozen 초기점과 GT의 L2 거리를 predicted-box diagonal로 나눈 뒤 valid corner 0–7에서 평균한다. 초기점 결측은 정규화 penalty 1, invalid GT는 제외하고 identity가 첫 branch라 정확한 동률에서 먼저 선택된다.

- Sidecar 60,000행 중 valid permutation 수 1개는 20,281행, 2개는 39,719행이며 4개인 행은 0개다.
- DOPE usable 고유 44,063행 중 non-identity는 134행(0.3041%)이었다. 실제 seed별 96,000 exposure에서는 `290/290/289`회(약 0.301–0.302%)였다.
- ResNet-18 usable 고유 55,806행과 세 seed의 각 96,000 exposure에서 non-identity는 모두 0회였다.

따라서 두 기반에 동일한 symmetry-aware objective를 적용했다는 표현은 맞다. ResNet-18에서 비항등 대칭 target이 실제로 발동했다거나, synthetic 학습에서 square C4 target을 학습했다거나, 대칭 감독의 독립 증분 이득이 입증됐다는 표현은 결과가 지지하지 않는다.

## DEV319

| Backbone | 지표 | Base | N3 mean | 변화 |
|---|---|---:|---:|---:|
| DOPE | matched 2D median px | 12.570 | 7.469 | -5.101 (-40.58%) |
| DOPE | matched 2D P90 px | 51.282 | 53.570 | +2.289 (악화) |
| DOPE | full median / P90 px | 17.485 / 800.000 | 12.653 / 800.000 | median 개선, P90 무차이 |
| DOPE | PCK10 | 0.2417 | 0.4434 | +0.2017 |
| DOPE | T median / P90 cm | 10.046 / 68.341 | 8.356 / 63.602 | 둘 다 개선 |
| DOPE | R median / P90 deg | 3.530 / 80.583 | 3.051 / 82.173 | median 개선, P90 악화 |
| ResNet-18 CONSTANT-fold | matched 2D median px | 8.223 | 7.085 | -1.138 (-13.84%) |
| ResNet-18 CONSTANT-fold | matched 2D P90 px | 62.638 | 62.371 | -0.268 |
| ResNet-18 CONSTANT-fold | full median / P90 px | 9.216 / 248.418 | 7.921 / 249.615 | median 개선, P90 악화 |
| ResNet-18 CONSTANT-fold | PCK10 | 0.5226 | 0.5722 | +0.0496 |
| ResNet-18 CONSTANT-fold | T median / P90 cm | 9.739 / 97.545 | 9.134 / 100.304 | median 개선, P90 악화 |
| ResNet-18 CONSTANT-fold | R median / P90 deg | 4.342 / 87.412 | 3.832 / 87.473 | median 개선, P90 근소 악화 |

각 seed의 13-session paired bootstrap에서 두 기반 모두 matched median, full median, PCK10, `E_sym`의 95% CI가 0을 배제했다. matched/full P90 CI는 0을 포함하거나 DOPE full P90처럼 변화가 0이었다. 다중비교 보정은 없고 pose CI도 없다. DEV pose reference는 annotation과 등록 geometry에서 재구성한 값이며 독립 physical 6D 측정이 아니다.

DOPE의 detection/match/pose coverage는 `297/233/210` frames, ResNet-18은 `319/292/319`다. N3가 이를 보존하므로 coverage 개선을 주장할 수 없다. 중앙 2D 오차와 pose median 개선은 두 기반에서 반복됐지만 tail T·R의 안정적 동시 개선은 달성되지 않았다.

## GREEN0918 정사각형 2D

primary 해석에는 영상 안 수동점 600개인 `manual_in_frame`을 사용한다.

| Backbone | matched median px | matched P90 px | PCK10 |
|---|---:|---:|---:|
| DOPE Base → N3 | 11.748 → 7.037 | 30.226 → 28.244 | 0.3683 → 0.6483 |
| ResNet-18 Base → N3 | 6.595 → 5.898 | 17.078 → 15.891 | 0.7233 → 0.7828 |
| YOLO R0 → N3 | 5.521 → 4.980 | 11.473 → 9.919 | 0.8533 → 0.8933 |

`manual_declared=602`와 `manual_in_frame=600`의 차이는 임의 누락이 아니다. frame `029710`의 corner0 `(-42,289)`와 frame `029844`의 corner4 `(-7,310)`가 수동 선언됐지만 640×480 영상의 왼쪽 밖에 있다. 전자는 두 점을 포함하고 후자는 `0 <= x < 640`, `0 <= y < 480`인 점만 포함한다. 검출 match box는 두 모드 모두 all-known이면서 영상 안인 점으로 만든다.

119장이 한 capture session이라 session bootstrap은 같은 한 unit만 반복해 점추정과 동일한 퇴화 CI가 된다. 통계적 불확실성 추정으로 해석하면 안 되며 YOLO receipt처럼 CI는 사실상 `x`다. 모든 frame의 치수가 `[1.1,1.1,0.15] m` 하나라 치수 변화의 인과효과도 식별할 수 없다. 독립 canonical pose가 없어 T/R은 `x`다.

YOLO에서 N3는 R0보다 개선됐지만 N2 대비 matched median `+0.0398 px`, P90 `+0.0514 px`, full median `+0.0527 px`, full P90 `+0.1308 px`, PCK10 `-0.00333`으로 대부분 근소 악화했다. 이 결과는 square에서 대칭 감독의 증분 이득을 지지하지 않는다.

## Runtime

RTX 3080, 고정 26 frames, path별 warm-up 20회, 5 repeats, 130 measured calls, file I/O 제외 결과다.

| Backbone | Base E2E median | N3 E2E median | 증가 | N3-only median | parameter 증가 |
|---|---:|---:|---:|---:|---:|
| DOPE | 62.736 ms | 65.807 ms | +3.071 ms | 2.838 ms | 23,331 (0.0464%) |
| ResNet-18 | 8.890 ms | 11.592 ms | +2.702 ms | 2.632 ms | 23,331 (0.1518%) |

E2E FPS는 DOPE `15.940→15.196`, ResNet-18 `112.487→86.269`다. E2E peak allocated memory는 base peak가 지배해 각각 동일했으며 N3-only maximum incremental allocation은 DOPE 32.063 MB, ResNet-18 19.816 MB다.

## 기록 리프터

실제 제어는 호출하지 않았다. 네 session의 고정 1,000 ms sensor-time grid 846 samples 중 R0와 N3 seed1 모두 fresh output 832개(98.345%), 결측 14개였다. coverage가 같은 이유는 N3가 base detection을 보존하기 때문이다.

`median_abs_step`은 같은 session 안의 사용 가능한 연속 예측 쌍에서 yaw를 `[-180,180)` 최단각으로 wrap한 뒤 `|Δyaw|`의 median이다. 단위는 degree다. `median_abs_rate_per_s`는 각 쌍의 같은 `|Δyaw|`를 실제 `camera_sensor_timestamp_ms` 간격으로 나눈 값의 median이며 단위는 degree/s다. 사용된 820개 쌍의 간격은 0.8009–1.2004 s, median 1.00070 s라 step과 rate가 비슷하지만 같지는 않다. 또한 `median(step/dt)`는 `median(step)/median(dt)`가 아니다.

- R0: median step `0.20824°`, median rate `0.21202°/s`, P90 step `0.82860°`, P90 rate `0.80660°/s`.
- N3: median step `0.22947°`, median rate `0.23127°/s`, P90 step `0.78798°`, P90 rate `0.79837°/s`.

N3는 median variation은 약 9–10% 악화하고 P90은 약 1–5% 개선하는 혼합 결과다. 기록에 독립 GT와 visibility label이 없어 위치 오차, yaw 오차, 실제 정확도는 모두 `x`다. 이 variation은 실제 장면 운동도 포함하므로 순수 센서 noise나 제어 안정성으로 바꿔 쓰면 안 된다.

## 주장 경계와 문서 주의점

- 지지되는 표현: 동일한 치수 입력·대칭-aware N3 절차를 세 frozen estimator에 각각 학습해, DOPE와 ResNet-18에서도 중앙 2D 오차 감소를 재현했고 square 2D 집단으로 전이를 확인했다.
- 지지되지 않는 표현: weight 공유 plug-and-play, 모든 backbone·모든 지표 robust, symmetry의 독립 효과, 치수 입력의 독립 인과효과, tail T·R 동시 개선, 리프터 정확도 개선.
- ResNet-18 결과는 10-epoch synthetic `CONSTANT` checkpoint를 zero-context에서 RGB-only로 접은 기반이다. 60-epoch pure-RGB ResNet 결과가 아니다. `PROTOCOL.json`의 nested `input.checkpoint_selection="fixed final epoch60"` 문구는 같은 객체의 실제 checkpoint/deviation과 모순되는 잔여 문구이므로 원고에는 쓰지 않는다.
- protocol은 seed mean과 spread를 요구하지만 evaluation의 `seed_aggregate`는 mean만 명시한다. per-seed 값 또는 range를 함께 보여야 한다.
