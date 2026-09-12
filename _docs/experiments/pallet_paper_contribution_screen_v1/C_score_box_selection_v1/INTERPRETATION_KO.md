# 무학습 score–box–selection 분해: 결론

새 학습 없이 R0와 C2 seed 1/2/3을 모두 비교했다. 기존 C 판정과 원본 결과는 유지한다.

## 확인된 결과

동일 DEV319 + NEG2689에서 AP50–95의 세 seed 평균:

| box 출처 | score 출처 | AP50–95 | R0 대비 %p |
|---|---|---:|---:|
| R0 | R0 | 0.768767 | 기준 |
| C2 | C2 | 0.744155 | −2.4611 |
| R0 | C2 | 0.751695 | −1.7071 |
| C2 | R0 | 0.761184 | −0.7583 |

모든 seed에서 두 hybrid 모두 R0보다 낮았다. C2 box를 R0 box로 바꾸면
C2 대비 약 0.7540%p를 회복하지만 원본 R0에는 못 미친다.
따라서 **score는 좋아졌고 box만 나빠졌다는 설명은 지지되지 않는다.**
고정 체크포인트의 출력 교환에서 score와 box 변경 모두 AP 손해를 보였다.
이는 두 손해가 일반적으로 가산적이라는 주장이나 학습 원인의 단독 분해는 아니다.

C2 score는 IoU50 match를 0.974922에서 평균 0.981191로 높였지만,
AUROC는 0.992131에서 0.987231로 낮아지고 FPR95는 0.041651에서 0.067807로 높아졌다.
기존 0.001 추론 floor에서 후보가 있는 negative frame은 R0 1539/2689,
C2 score의 seed별 2321/2359/2402장이다. 이 floor는 실제 배포 운용점으로 새로 선택한 값이 아니다.

## 기하 보존과 선택 변경

전체 이미지에서 입력, backbone/neck feature, grid/stride, dense pose는 R0/C2 간 bit-exact였다.
원본 후보 전체도 이전 저장 캐시와 정확히 재현됐다. 교환은 같은 grid index에서만 했다.

양성 319장 중 C2 score가 top 후보를 바꾼 수는 seed별 48/37/32장이다.
네 구성 공통 검출 프레임은 311/311/310장, 그중 같은 top grid index인 프레임은 266/276/282장이다.
같은 top 후보를 선택한 부분집합의 기하 통계는 네 구성에서 완전히 같았다.

공통 프레임 kp median/P90 평균은 R0 score에서 6.6285/38.6978px,
C2 score에서 6.6373/38.7709px다. R0 대비 안정적 기하 개선을 주장할 근거는 없다.
기존 C0 대비 보존 이득을 원본 R0 대비 개선으로 바꾸어 말하지 않는다.

box만 바꾸면 최종 keypoints와 MAIN pose가 원본 R0와 같았고,
score를 C2로 바꾸면 C2와 같았다. 319장 전체 pose coverage는 모든 구성에서 1.0이다.
translation 평균은 R0 score 7.8969cm, C2 score 8.0003cm;
yaw는 각각 1.2306°와 1.2232°다. 전 지표 개선이나 통계적 유의성은 주장하지 않는다.

## 다음 의사결정

- 이번 hybrid를 새 방법으로 채택하지 않는다. 기존 C FAIL을 PASS로 바꾸지 않는다.
- 이 결과만으로 frozen feature 부족이 증명된 것은 아니므로 adapter를 자동 구현하지 않는다.
- 논문 마감 경로는 기존 baseline·통제된 adaptation/선택 분석을 유지하고, A의 고정 후보는 새 독립 확인 자료를 준비한다.
- D의 명시적 local appearance feature 누락은 인정한다. 기존 scalar-trust 결과를 full visual trust 또는 학생 성능 실패로 확대하지 않는다.
- B/D/E 재학습, threshold 완화, 추가 seed·architecture 탐색은 수행하지 않았다.

이 결과는 사후 개발셋 진단이다. 독립 일반화, 신규성, 센서 ground truth에 대한 주장이 아니다.
상세 수치·IoU별 AP·분모는 RESULTS.json, 검증은 INFERENCE_AUDIT.json 및 INDEPENDENT_RECOMPUTATION.json에 있다.
