# TRAIN 입력 표현·잔차 방향 감사

**이 결과는 학습 성능이 아닌 입력 충돌·중복 진단이다.** 기존 TRAIN2598행·실패1행과253입력/타깃6해시를 유지했다. 모델 weight·objective·새argmin·VAL/실사 오류는 읽거나 평가하지 않았다.

코너별 signed residual18은 기존 R0/PoseFix 예측과 고정 pose/K의 투영에서 얻었다. R0의 유효TRAIN5194후보만으로 float32 mean/std를 만들고 같은 연산으로 정규화한 뒤 float64에서 후보−anchor 차이를 계산했다. 이미지 forward·PnP·새 참조 오류 계산은 없다.

## 입력 열공간 중복

SVD에는 유효 nonanchor 입력만 전달했다. 타깃은 사용하지 않았으며 절편·추가 중심화·타깃 회귀를 하지 않았다. rank tolerance는 각 행렬에 `max(shape)*eps64*smax`로 고정했다.

| 모델 | nonanchor 행 | 기존 rank | 확장 rank | 잔차 Frobenius 비율 | 비율≤1e−4 |
|---|---:|---:|---:|---:|---|
| R0_ONLY | 2597 | 200 | 218 | 0.189748594 | False |
| UNION_s1 | 7791 | 203 | 221 | 0.422550821 | False |
| UNION_s2 | 7791 | 203 | 221 | 0.433444381 | False |
| UNION_s3 | 7791 | 203 | 221 | 0.436207306 | False |

일부 모델에서 입력 비중복이 관측되지만 학습·성능·전이의 증거는 아니며 fit을 승인하지 않는다.

1e−4는 새 입력이 기존 열공간에서 벗어나는 정도의 사전 기준이며 pose 선택·T/R 성능 기준이 아니다. 수치 rank는 선택된 정밀도/허용오차의 판정이다. 비중복은 정답 정보·개선·전이 가능성을 증명하지 않는다.

## 반올림 없는 입력 충돌

+0/−0만 같은 +0으로 통일하고 float64 실제 byte로 묶었다. 그 외 반올림·근접 임계값은 없다. sign-conflict는−/0/+ 중 두 부호 이상, strict-opposite는 min<0<max다. anchor의 입력/타깃0은 강제 구성이라 별도로 보존했다.

| 모델 | 기존 반복 그룹 | anchor/nonanchor 혼합 그룹 | T strict 충돌 기존→확장 | R strict 충돌 기존→확장 |
|---|---:|---:|---:|---:|
| R0_ONLY | 1 | 0 | 0 → 0 | 0 → 0 |
| UNION_s1 | 1 | 0 | 0 → 0 | 0 → 0 |
| UNION_s2 | 1 | 0 | 0 → 0 | 0 → 0 |
| UNION_s3 | 1 | 0 | 0 → 0 | 0 → 0 |

동일 입력의 상반된 타깃은 그 행들을 완벽하게 예측할 수 없다는 제한이다. 충돌이 없다는 사실은 표현 충분성이나 공동 T/R 목표 달성 가능성의 증명이 아니다. 추가18개가 그룹을 나누더라도 새 모델을 학습하거나 선택기를 평가한 결과가 아니다.

NPZ는 IDs·source index·anchor·정규화·새18개·원타깃·valid·기존/확장 그룹 번호를 보존한다. 전체 singular spectrum과 열별 잔차 비율, 반복 그룹의 타깃 범위·분할/미해결 여부는 JSON에 있다.

source feature NPZ 컨테이너는 VAL 입력도 포함하지만 적격TRAIN행만 선택했다. 라벨은 이미 고정된 TRAIN 전용 NPZ만 사용했다. 이는 TRAIN 내부의 진단이며 일반화/신뢰도 보정·새 gate·실험 성공을 주장하지 않는다.

[감사 JSON](REPRESENTATION_AUDIT.json) · [방향 특징 감사](FEATURE_AUDIT.json) · [입력 진단 프로토콜](REPRESENTATION_PROTOCOL.json) · [이전 입력·타깃 해시](../pallet_pose_signed_axes_sign_20261001_v1/PREFIT_REVIEW.json)
