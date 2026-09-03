# POSE_METRIC_CLOSURE_V1

**[소비처]** 논문의 6D pose 결과절.  현재 `POSE_METRICS_STATUS = BLOCKED` 라
`R med · yaw med · t med · IoU3D · ADD · ADD-S · pose AUC` 열이 전부 비어 있다.
이 트랙이 닫히면 그 열이 열리고, 안 닫히면 **왜 못 여는지가 논문의 한계 절**이 된다.

**[문장]** *"prediction 만으로 W/D 축 가설을 고를 수 있으면 6D pose metric 을
누수 없이 계산할 수 있고, 없으면 그 지표는 이 데이터에서 원리적으로 열리지 않는다."*

## 이 트랙이 아닌 것

self-training 성능 개선이 아니다.  `EXPERIMENT_STOP_LOCK.json` 은 그대로 유효하고
이 트랙은 그 예외가 아니라 **다른 종류의 작업**이다.

```text
POSE_EVALUATION_CLOSURE_ONLY
```

금지: V6 self-training · pseudo-label 필터 수정 · threshold 재튜닝 ·
reliability weighting 수정 · R0~R5 재학습 · 더 좋은 student 선택 ·
PAPER_EVAL 을 보고 architecture 반복 수정.

## 판단 지표 (결과 보기 전에 고정)

```text
selector overall accuracy   >= 0.95
selector night accuracy     >= 0.90
selector coverage           >= 0.95      어려운 프레임을 버려서 만든 정확도는 PASS 아님
```

세 개를 **동시에** 넘어야 한다.  하나라도 못 넘으면 `POSE_METRICS_STATUS` 는
BLOCKED 로 두고 끝낸다.  학습을 늘려서 뚫지 않는다.

## 가장 중요한 금지

> GT pose / GT yaw / GT axis_assignment 를 보고 prediction 의 W/D 가설을 고르지 않는다.

그렇게 하면 evaluator 가 모델 대신 90° 모호성을 풀어주는 leakage 다.
실제로 그 누수가 있었을 때 `5cm5deg` 가 30.4% 였고, 배포 가능한 정보만 쓰자
19.3% 로 내려갔다.  그 11.1%p 가 이 트랙이 정직하게 풀어야 할 몫이다.

## 1차 범위 (§24)

```text
포함   blocker 재현 · data inventory · selector 정보 분석
       symmetry/canonical 감사 · split 설계 · method lock · evaluator unit test
제외   장시간 학습.  auxiliary selector 학습이 필요하다는 결론이 나오면
       비용과 데이터 계약을 보고하고 **중단**한다.  사용자 승인 없이 학습하지 않는다.
```
