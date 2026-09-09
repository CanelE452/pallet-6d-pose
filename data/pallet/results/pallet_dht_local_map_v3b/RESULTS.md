# v3b fixed-weight MAP prior probe — 완료

기존 v3 Hough seed1 최종 가중치를 그대로 사용하고, Hough likelihood logits에서 고정된 점 위치 prior를 뺐다. 원본 입력픽셀 기준 rho sigma=2, angle sigma=atan(2√2/edge length)이며 사후 튜닝은 없다. 코드 변경은 local_hough logits 한 곳이다.

합성 calibration256에서 기존 scale grid와 조건으로 scale1을 선택한 뒤 synth_val512(509개 감독 프레임, 4021코너)를 평가했다. 원래8코너 ID와 감독 마스크를 유지했고 centroid는 복사했다.

| 합성 검증 | mean px | median px | P90 px |
|---|---:|---:|---:|
| baseline | 4.957429 | 2.117257 | 7.799472 |
| 기존 v3 Hough | 4.965718 | 2.136389 | 7.861490 |
| v3b MAP prior | 4.943098 | 2.095173 | 7.809183 |

**실행·수치 검증 PASS, 사전 진입 기준 미통과(advance=false).** 평균은 baseline보다 0.2891% 줄었고 중앙값도 줄었지만, P90이 증가했고 요구한 평균1% 감소에 못 미쳤다. baseline≤10px인3743코너에서 새로10px를 넘긴 점은0개다.

실제 새 합성 이미지 forward768회, optimizer0회, 실사 forward0회. 가중치 차이 최대값과 분산은0이며 checkpoint/source SHA 보존을 확인했다. 이는 추론 변경의 메커니즘 진단이며 학습된 최종 모델 성공 또는 실사 개선 증거가 아니다. Prior는 선 posterior의 위치와 entropy 기반 precision을 함께 바꾸므로 둘의 기여를 따로 증명하지 않는다.

[RESULTS.json](RESULTS.json) · [COMPLETION.json](COMPLETION.json) · [INDEPENDENT_REPLAY.json](INDEPENDENT_REPLAY.json) · [PROTOCOL.json](PROTOCOL.json) · [MODEL_CHANGE.json](MODEL_CHANGE.json)
