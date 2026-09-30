# 평가 수학·실패 처리·입력 잠금 독립 검토

2026-10-01. 검토자는 새 모델의 실제 예측·T/R 결과·평가 GT를 열지 않고 평가 코드, 기존 metric 구현, 사전 프로토콜과 인공 배열만 사용했다. GPU 학습과는 독립적으로 수행했다.

**수학과 판정식은 effective251 프로토콜에 부합하며, 발견한 세 보완점의 코드 수정과 인공 배열 재검증도 확인했다.** 아래 이슈의 완료 상태는 마지막 검증 문단으로 구분한다. 평가 자체검사 통과를 실제 모델 개선으로 해석하지 않는다.

## 비교하는 양

각 seed에서 모집단의 T·R 중앙값을 먼저 구하고 세 seed의 중앙값을 평균한다. 비교량은 `mean_seed(median_after) − mean_seed(median_before)`다. 모든 seed·프레임을 합쳐 한 번 중앙값을 계산하거나, 프레임별 차이의 중앙값과 혼동하지 않는다. 기존 `M.paired` 결과는 후자의 별도 정보도 보존한다.

bootstrap은 recording과 학습 seed를 각각 복원 추출한다. 선택된 recording의 모든 프레임을 유지하므로 frame을 독립 반복으로 세지 않는다. 같은 draw에서 전후 모델에 같은 recording·seed index를 사용하고, 비교마다 RNG를 같은 seed로 초기화하므로 같은 모집단의 모든 모델 비교가 같은 draw를 공유한다. 기존 R0·PRIOR1·FULL125는 세 seed 축에 같은 값으로 반복해, 새로운 학습 난수 변동을 가진 척하지 않는다.

각 recording은 cluster로 균등 추출하지만 cluster 안 모든 프레임을 유지한다. 따라서 추정 대상은 원래 frame 가중치의 모집단 중앙값이며 recording별 중앙값의 평균이 아니다. recording 크기가 다르다는 사실을 버그로 오해하지 않는다.

95% 구간은2,000개 paired crossed-bootstrap draw의 percentile이다. finite하지 않은 draw는 별도 집계하고, 한 개라도 존재하면 CI gate를 통과하지 못하게 한다. 세 seed·여섯 natural recording과 반복 사용 DEV라는 한계가 명시되어 있다. 이 구간을 새 촬영 일반화의 증명 또는 측정 noise floor로 해석하지 않는다.

## 프로토콜과 게이트 대조

| 사전 조건 | 구현 확인 |
|---|---|
| 세 seed 모두 T/R 개선 | DIVERSE의 각 seed가 paired SINGLE과 고정 R0/PRIOR1/FULL125 각각보다 두 중앙값이 작아야 한다. best seed 선택이 없다. |
| 공동 불확실성 | DIVERSE−SINGLE, DIVERSE−R0에서 T·R 모두95% 상한<0이어야 한다. |
| recording 민감도 | 여섯 recording을 각각 제외해도 두 비교의 mean-seed 중앙값 차이가 T·R 모두 음수여야 한다. |
| 자연 꼬리 | mean-seed P90 T·R이 SINGLE과 R0 대비 각각1.05배 이하여야 한다. |
| clean 보존 | clean29 mean-seed 중앙값·P90 T·R이 R0 대비 각각1.05배 이하여야 한다. |
| 실패 수 | natural/clean의 각 seed 실패 수가 비교 기준보다 늘면 실패한다. 평균 실패 수로 나쁜 seed를 감추지 않는다. |
| 판정 범위 | 성공해도 `strong_generalization=False`, `goal_complete=False`로 남겨 데이터 범위와 전체 목표를 구별한다. |

P90/clean의5%는 잠긴 공학적 허용치이며 실측 동등성 한계나 MDE가 아니다. wood45는 별도 모양 전이 stress 결과로 유지되며 plastic99와 섞지 않는다.

## 실패 분모와 참조 접근

명시적인 pose unavailable은 tensor의 T·R 모두+∞로 남긴다. CSV에도 원래 frame 행과 실패 상태가 있고, conditional summary와 전체 분모의 extended-real summary를 따로 출력한다. conditional 계산에서 없는 pose가 제외되어도 실패 증가 gate가 별도로 막는다. 기존 R0 natural99/clean29에 실패가 없다는 사실을 새 실패 삭제의 근거로 사용하지 않는다.

`locked_inputs()`는 모든9모델·173개 ID·prediction hash·metadata를 확인해야 다음 단계로 진행한다. `freeze()`에서 prediction-only GEO와 최종 corner8 PnP 후보를 먼저 저장한다. 평가 참조를 사용하는 `Pose.metadata('REAL_DEV')` 호출은 prediction/pose lock 뒤에 있다. `C.D.metadata()`는 기존 추론 metadata JSON을 읽으며 평가 좌표를 생성하는 함수가 아니다. baseline full128의3모델×128=384개 metric parity 검사도 있다.

참조 loader의 실제 경로는 `GEOMETRY_RESOLVED_POSE_GT.json`의 R/t, `AXIS_REVIEW_MANIFEST.json`의 mapping, annotation의 intrinsics, object registry의 physical dimensions/symmetry, paper membership이다. 따라서 annotation 파일만 연결하는 것으로 전체 GT 동일성이 증명되지는 않는다.

## 검토에서 발견하여 전달한 보완점

1. **예상하지 못한 NaN 값의 누락:** 최초 구현에서 `guard()`는 conditional summary의 NaN frame을 제외하면서 실패 수는 infinity만 세었다. 인공 배열에서 이 상태로 PASS가 나오는 것을 재현했다. available T/R의 finite assertion과 NaN guard 거부를 요청했다. 이는 실제 결과에서 NaN을 발견했다는 뜻이 아니다.
2. **GEO 계보:** 최초 pose freeze는 코드를 binding했지만 GEO_LINEAR의 선택 lock·checkpoint binding을 새 pose lock에 직접 넣지 않았다. 과거 선택 lock과 checkpoint를 함께 연결하도록 요청했다.
3. **참조 동일성:** 최초 score는 현재 참조를 읽고 과거384개 scalar parity를 비교했으나, 원래 참조 파일 binding을 직접 검증하지 않았다. 실제 R/t 파일·axis mapping·registry·intrinsics 파일의 사전 binding/검증을 추가하도록 요청했다. 새 checkpoint·threshold·GT 값을 변경하라는 요청이 아니다.

## 실제 실행한 인공 배열 검사

기존 `self_check()`가 통과했다. 여기에 다음 추가 검사를 실행했다.

- 크기가1/3/1인 recording과 서로 다른 오차를 가진 배열에서 모든 seed의 변화가 T−2/R−1이면 CI가 정확히[−2,−2]/[−1,−1]이다.
- 세 seed의 변화가 각각−1/−2/−3이고 recording 효과가 없으면 점 추정−2, 고정2,000회 bootstrap CI[−3,−1]이다. seed pairing이 유지된다.
- 동일 출력의0차이, 나쁜 seed 하나, 정확한5% 경계,5% 초과, 명시적∞ 실패, 전부 실패로 undefined draw가 되는 기존 fixture가 통과했다.
- 최초 NaN fault fixture: `before=10`, `after=9`인(3,5,2) 배열에서 after의 첫 frame을 모두NaN으로 바꾸면 guard가 True였다. 구현 담당자에게 전달했으며 수정 뒤 같은 fixture를 다시 검증한다.

실제 평가 좌표나 새 모델 결과는 이 검사에 사용하지 않았다.

## 보완 후 독립 재검증

검토한 수정본 `evaluate.py`의 SHA256은 `be6451a4cdfcc0759f59d238a3a5615eb48dfaf7bdfda79c61479e296f779150`이다.

- `error_tensor`가 available=True인 비유한 또는 음수 T/R을 거부한다. `validate_tensor`가 NaN·음의∞·T/R 한 축만 실패한 배열을 거부한다. 같은 NaN fault fixture를 다시 실행했으며 이제 assertion으로 거부된다. available=True의 NaN metric fixture도 거부됐다.
- pose freeze에 과거 publication→RUN_MANIFEST→GEO 선택 lock→checkpoint 연결이 추가됐다. registry와 symmetry 계약, PAPER319 membership 및 깊은 geometry/metric 코드도 연결한다.
- prediction·pose lock 뒤 참조 파싱 직전에 기존 expected hash로 plastic128+wood45의 annotation173/image173, 실제 geometry R/t와 axis mapping을 검증하고 `REFERENCE_BINDINGS.json`을 남기는 코드를 확인했다. 단순히 현재 파일을 새로 hash하여 과거와 같다고 주장하는 방식이 아니다.
- 기존 self-check와 추가 seed-변화 fixture를 수정 뒤 다시 실행해 모두 통과했다. 새 CI·threshold·모델을 선택하거나 실제 성능을 열어본 검사는 아니다.

이 검토 범위에서 채점 실행을 막는 미해결 결함은 없다. 실제 예측과 pose lock 생성, 참조 binding 실물 검증,384개 baseline parity 및 최종 판정 결과는 평가 실행 영수증으로 따로 확인해야 한다.
