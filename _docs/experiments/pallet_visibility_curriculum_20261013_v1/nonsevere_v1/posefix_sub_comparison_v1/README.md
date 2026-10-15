# PoseFix + SubPix 동일 코호트 비교

동일한 clean/moderate TEST49에서 경계 융합 방법은 새로 계산한 PoseFix→SubPix 두 조건보다 평균 위치 오차와 성공 수가 좋아졌지만, 평균 회전 오차는 커졌다. 위치·회전 모두 개선됐다는 결론은 성립하지 않는다.

기존 초기값은 **N3_DIM_SYM seed1 → SubPix → 원 YOLO 기준 1% 총 이동 cap → F**다. 별도 RGB ResNet152 PoseFix 모델과 구분된다. `N3_SUBPIX_NO_MASK_STANDARD` 캐시의 `initial_pose`를 쓴다는 사실과 원 F 재계산은 95장 모두 오차0으로 확인했다. 이후 `actual_pose` 재피팅 결과를 초기값으로 사용하지 않는다.

문자 그대로의 과거 PoseFix→SubPix 결합 캐시·레시피는 찾지 못했다. 그래서 기존 synthetic-only PRIOR1 seed1의 저장된 RAW 첫 추론 좌표에 같은 고정 SubPix를 적용했다. PF 좌표 기준 추가 이동 cap과 원 YOLO 좌표 기준 총 이동 cap을 사전에 두 조건으로 선언했으며, 결과에 따라 한 조건을 선택하지 않았다. 새 학습·모델 재추론은 없다.

| TEST49 방법 | 평균 위치 cm | 평균 회전 ° | 성공 /49 |
|---|---:|---:|---:|
| 원 YOLO + F | 4.1397 | 6.6253 | 30 |
| N3 + SubPix (기존 초기값) | 3.3630 | 6.5271 | 40 |
| PoseFix RAW + F | 3.9306 | 6.3392 | 34 |
| PoseFix → SubPix · PF 기준 cap | 3.2161 | 6.3495 | 39 |
| PoseFix → SubPix · YOLO 기준 cap | 3.2432 | 6.3553 | 38 |
| 경계 융합 + 단계적 보정 | 3.0040 | 6.5243 | 43 |
| 고정 축 joint 보정 | 2.6186 | 6.5923 | 42 |

성공은 위치<5cm 및 회전<5°를 동시에 만족하는 경우다. 모든49장에 유효 포즈가 있고, 실패·가려진 사진을 사후 제외하지 않았다.

경계 융합−PF 기준 PoseFix+SubPix 차이는 위치 −0.2121cm [4세션 bootstrap95%구간 −1.3208, +2.0204], 회전 +0.1748° [−0.1522, +0.3610]다. 성공은39→43이며 복구4·손실0; 복구4장은 모두 eval_noapril이다. 원 YOLO 총 cap 조건에서는 성공38→43, 복구5·손실0이다.

고정 축 joint 방법은 PF 기준 PoseFix+SubPix보다 위치 −0.5974cm, 회전 +0.2428°; 성공39→42, 복구5·손실2다. 비교별 프레임 목록, 위치·회전 동시 개선/악화 목록, 모든 짝비교 구간과 median/P90은 `COMPARISON.json`에 있다.

CAL46도 같은7조건을 평가했다. CAL joint는 전체 CAL로 학습한 고정 모델의 in-sample 재생이며 OOF 결과와 구분한다. 저장된 확률·보정량·RGB/Tiny 증거를 현재 생산 wrapper에 주입한46장 QA에서 사용점·채택 여부가 일치했고, 포즈 차이는 최대4.44e-16이다.

기존 F는 PoseFix/Sub의 자체 좌표에서 예측만으로 W/D 축을 고르고 SQPnP+LM을 실행한다. 경계 방법은 N3 초기 축/H와 초기 prior를 유지한다. 따라서 이 비교는 전체 파이프라인 비교이며 후처리 하나만의 인과 효과가 아니다. 현재 TEST49는 이미 반복 관찰한 개발 평가 자료이고, 독립 검증을 새로 확보한 결과는 아니다.

7조건665개 예측을 봉인한 뒤 격리된 CAL46·TEST49 정답만 읽었다. severe/FINAL 정답은 사용하지 않았다. 새 F475회, 원 영상 SubPix95회; 코드·모델·파라미터·축 cap 선택의 결과 기반 변경은 없다. 직접 원 함수 정의를 실행하는 namespace에서 상수 하나가 빠진 오류는 SubPix/F0 상태에서 별도 수리 기록을 남겼다. 원 사전 선언과 소스 스냅샷은 보존했다.

[레시피·출처](BASELINE_PROVENANCE.json) · [비교](COMPARISON.json) · [사전 선언](PROTOCOL.json) · [예측 봉인](PREDICTION_SEAL.json) · [CAL joint QA](JOINT_CAL_CONTROL_PARITY.json)
