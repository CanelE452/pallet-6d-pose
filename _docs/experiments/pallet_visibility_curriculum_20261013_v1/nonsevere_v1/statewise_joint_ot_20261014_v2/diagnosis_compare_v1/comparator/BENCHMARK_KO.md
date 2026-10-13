# 동일 CAL46/TEST49의 봉인 자세 비교

모든 방법의 저장된 pose를 다시 읽고 **현재 scorer의 동일 reference, physical R, 고정 대칭, 위치/회전/ADDsym/5cm·5° 기준**으로 재계산했다. 이전 표 숫자를 합치지 않았다. prediction/protocol seal과 exact IDs, 원 입력, 등록 모델을 먼저 검증한 뒤 격리된 CAL/TEST target을 열었다. 새 영상 추론·자세 fitting·모델 학습·튜닝은 0회다. crop과 severe는 제외하며 fallback을 포함한 전체49장을 유지했다.

| TEST49 방법 | 위치 mean cm | 회전 mean ° | ADDsym mean cm | 성공 /49 |
|---|---:|---:|---:|---:|
| 원 N3+SubPix PnP | 3.3630 | 6.5271 | 9.8309 | 40 |
| 현재 보정 B/I만 PnP | 4.6924 | 8.2853 | 10.5721 | 38 |
| 현재 모든 O/T 공동 v3 | 3.5139 | 6.5675 | 9.7564 | 39 |
| 기존 RGB 신뢰도·보정, family 유지 | 2.6186 | 6.5923 | 9.2492 | 42 |
| 기존 경계 융합·진행 보정 | 3.0040 | 6.5243 | 9.2919 | 43 |
| 원 YOLO+F | 4.1397 | 6.6253 | 10.6559 | 30 |
| 별도 PoseFix RAW+F | 3.9306 | 6.3392 | 10.5977 | 34 |
| PoseFix→SubPix, PF cap | 3.2161 | 6.3495 | 9.8658 | 39 |
| PoseFix→SubPix, YOLO cap | 3.2432 | 6.3553 | 9.8945 | 38 |
| N3 count rule, raw | 3.3354 | 6.5659 | 9.8157 | 40 |
| N3 count rule, fused | 3.0655 | 6.4980 | 9.5447 | 40 |
| 이전 보호 completion BASE | 3.0384 | 6.4876 | 9.5266 | 40 |
| 이전 유한 visible 후보 | 3.0384 | 6.4876 | 9.5266 | 40 |
| 이전 유한 region 후보 | 3.0384 | 6.4876 | 9.5266 | 40 |
| 기존 RGB 보정+W/D branch | 3.1923 | 11.8512 | 16.2753 | 40 |
| 기존 RGB visible together | 3.3216 | 13.5283 | 18.4534 | 40 |
| 기존 segmented RGB curriculum | 3.3124 | 6.5620 | 9.6131 | 42 |
| PoseFix-SubPix count raw | 3.2477 | 6.3638 | 9.8937 | 40 |
| PoseFix-SubPix count fused | 2.7970 | 6.3166 | 9.4419 | 42 |

## 같은 입력과 다른 파이프라인

원 N3, 기존 RGB family 유지, 경계 progressive/count, 이전 finite 및 현재 v3는 동일 N3_SUBPIX native 입력을 사용한다. K·원 이미지·canvas와 초기 pose가 모두 같은95장임을 확인했다. same-family arm들은 cf_extents 및 physical registration Q도 유지한다. W/D branch arm은 같은 원 N3에서 family를 바꾸므로 별도로 표시했다.

PoseFix RAW/SubPix와 YOLO arm은 같은 사진·물체·평가 target을 공유하지만 원 image head 좌표가 다르다. PoseFix는 별도 RGB ResNet152 head이며 N3와 같지 않다. 자체 q로 F의 W/D family를 다시 고르므로 전체 pipeline 비교다. R_cf를 직접 비교하지 않고 R_physical의 고정 대칭 지표를 썼다.

CAL46도 전부 재계산한 수치가 JSON에 있다. 기존 RGB confidence의 CAL은 full-CAL fit의 in-sample 결과다. TEST49는 이미 반복 관찰한 탐색 평가이며 독립 일반화 검증은 아니다.

## 현재−비교법의 짝비교

- 현재−원 N3+SubPix PnP: 위치 +0.1509cm, 회전 +0.0404°. 위치 세션 bootstrap95% [-0.6647, +1.8198]cm. 성공 복구 1 / 손실 2.
- 현재−현재 보정 B/I만 PnP: 위치 -1.1784cm, 회전 -1.7178°. 위치 세션 bootstrap95% [-10.0968, +0.1641]cm. 성공 복구 1 / 손실 0.
- 현재−기존 RGB 신뢰도·보정, family 유지: 위치 +0.8953cm, 회전 -0.0248°. 위치 세션 bootstrap95% [+0.4103, +2.3089]cm. 성공 복구 1 / 손실 4.
- 현재−기존 경계 융합·진행 보정: 위치 +0.5099cm, 회전 +0.0432°. 위치 세션 bootstrap95% [+0.2550, +1.7411]cm. 성공 복구 0 / 손실 4.
- 현재−N3 count rule, raw: 위치 +0.1786cm, 회전 +0.0016°. 위치 세션 bootstrap95% [-0.4947, +1.6805]cm. 성공 복구 1 / 손실 2.
- 현재−N3 count rule, fused: 위치 +0.4484cm, 회전 +0.0695°. 위치 세션 bootstrap95% [-0.0041, +1.8439]cm. 성공 복구 1 / 손실 2.
- 현재−PoseFix→SubPix, PF cap: 위치 +0.2979cm, 회전 +0.2180°. 위치 세션 bootstrap95% [-1.0476, +3.0265]cm. 성공 복구 4 / 손실 4.
- 현재−PoseFix→SubPix, YOLO cap: 위치 +0.2707cm, 회전 +0.2122°. 위치 세션 bootstrap95% [-1.0781, +2.9745]cm. 성공 복구 4 / 손실 3.

모든 comparator에 대해 프레임별 승/패/동률, 성공 flip ID, 위치·회전 동시 개선/악화, 큰 손실/개선 10건과 세션별 평균·총 기여도를 `BENCHMARK.json`에 저장했다. 부호는 현재−비교법이며 오차 지표의 양수는 현재가 나쁘다는 뜻이다. TEST는 4개 세션 cluster를 10,000회 bootstrap했고 모든 프레임을 유지했다. 표본 세션 수가 작고 TEST를 재사용했으므로 통계적 우수성·새 일반화의 증명이 아니다.

현재3조건을 기존 v3 postseal scorer의 개별95장 값과 대조했으며 모든 metric 차이가 정확히0이었다. 물리 pose가 같은 arm도 JSON에 표시했으므로 여러 이름을 독립적인 근거 수로 세면 안 된다.

`COMPARATOR_INVENTORY.json`: 방법별 출처·seal·exactIDs·원 KP/K/model 호환성. `PER_FRAME_METRICS.jsonl.gz`: 전체 개별 점수. `BENCHMARK.json`: 전체 CAL/TEST 분포와 현재 기준 짝비교. 현재 source19개·모델·예측과 보호 사용자파일6개는 바뀌지 않았다. 실제 DOPE/SOTA의 동일사진 예측 artifact는 없으므로 해당 비교를 만들지 않았다.
