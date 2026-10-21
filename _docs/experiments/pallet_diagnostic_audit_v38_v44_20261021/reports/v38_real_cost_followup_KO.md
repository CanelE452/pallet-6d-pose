> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.



# v38 실사 저장 비용 보완 감사

새 head/영상 forward/SAM/PnP/학습/전체 성능평가 없이 기존 E_RUN·BOUNDARY_BUNDLE·고정 CONFIG와 두 저장 pose의 비용 함수만 계산했다. CPU1개, 1.205553초, 5분 제한 안에서 완료. 두 후보의 context 및 첫26개 특징/validity가 저장값과 정확히 일치했다. bank gap은 다시 만들지 않았으며 원본 pose/사진/관측을 변경하지 않았다.

대상은 PRIMARY 반복 개발 사례 frame_704b3381eea10171f7c8이며 원 OCPR의 N0 대 E0_REGENERATED 비교다. 합성 v32/v33 사례와 별개다.

## 같은 family 내부의 결과

| Family | 비용 | N0 | 선택 오답 | 제약 |
|---|---|---:|---:|---|
| ORIGINAL_WD | fixed | 37.753927 | 654.318936 | N0 feasible, 오답 infeasible |
| ORIGINAL_WD | A | 40.258363 | 18.180630 | 둘 다 feasible |
| ORIGINAL_WD | B | 184.251354 | 7935.186604 | N0 feasible, 오답 infeasible |
| ORIGINAL_WD | AB | 187.023912 | 84.719527 | 둘 다 feasible |
| ORIGINAL_WD | edge | 425.197409 | 8051.190093 | N0 feasible, 오답 infeasible |
| SWAPPED_WD | fixed | 1175.928734 | 15.412891 | N0 infeasible, 오답 feasible |
| SWAPPED_WD | A | 40.258363 | 18.180630 | 둘 다 feasible |
| SWAPPED_WD | B | 12873.195132 | 81.947905 | N0 infeasible, 오답 feasible |
| SWAPPED_WD | AB | 187.023912 | 84.719527 | 둘 다 feasible |
| SWAPPED_WD | edge | 13226.087016 | 406.317847 | N0 infeasible, 오답 feasible |

A는 global owner assignment marginalization, B는 boundary region/profile replacement라는 모델 이름이다. 관측 역할 B/I와 같은 뜻이 아니다. 서로 다른 family의 비용을 직접 빼지 않는다. 비용이 유효한지와 제약 feasible 여부도 따로 본다.

두 family 모두 A와 AB는 저장 참조상 오답을 지지한다. 따라서 이 실사 사례를 ‘관측은 모두 N0가 맞다고 하는데 H1만 무시했다’고 설명할 수 없다. 한편 고정 대응의 ORIGINAL_WD 제약과 비용은 N0를 지지하므로 모든 근거가 오답을 지지하는 것도 아니다. global assignment를 허용하면 어떤 대응을 통해 오답의 낮은 비용이 가능한지 추적하는 문제가 구체화됐다.

## 어느 관측이 지지하는가

A audit에 저장된 global owner posterior에 따른 group energy 기대값을 계산했다. 이는 실제 owner 정답이나 정확히 가산되는 marginal 인과기여가 아니다. 동일 ORIGINAL_WD·A에서 N0의 argmax h=0, 오답의 argmax h=3이다.

| A 관측 group | 역할 | N0 기대 energy | 선택 오답 기대 energy |
|---|---|---:|---:|
| full:2 | B | 0.505804 | 2.530117 |
| full:5 | B | 0.179346 | 1.045321 |
| partial:0 | I | 0.148166 | 2.259687 |
| partial:1 | B | 1.374634 | 0.140614 |
| partial:3 | B | 11.598186 | 3.834032 |
| partial:4 | B | 0.292123 | 0.234009 |
| partial:7 | B | 10.470416 | 0.869003 |

I0 및 full B2/B5는 N0 쪽의 작은 잔차를 지지한다. 반면 B 부분선3/7은 owner 재배정 후 오답 쪽 비용이 크게 낮다. A의 outline 비용도 동일 ORIGINAL_WD 안에서 N0 12.917099, 오답4.495260으로 오답을 지지한다. 전체 marginal/outline/feasibility를 포함한 A 비용과 group 기대값을 구분해야 한다. AB는 여기에 기존 region/profile 교체가 들어가며 여전히 오답 지지다. 이 결과는 관측/대응 의미의 병목을 직접 보여 주지만, 물리적 모서리 정답이 검증됐다는 뜻은 아니다.

## 원 v38 설명과 그림의 단계 보정

OBSERVATION.jsonl의 frozen_BI_fit에는 I0가 2D backbone fallback으로 남아 있다. 그러나 실제 D1 solver packet에서는 I0의 complete 2D likelihood를 명시적으로 revoke하고, native edge0–4의 관측 RGB 부분선으로 대체했다. 사유는 no_unique_face_supported_junction이다. 따라서 ‘관측 파일에 2D 점이 있다’와 ‘최종 solver가 그 2D likelihood를 쓴다’는 다르다.

실제 사용 부분선 native ID는 0(I),1/3/4/7(B)다. 원 그림은 OBSERVATION 단계의 I0 점을 보여 줬고, 이를 반영한 새 그림 CACHED_MASK_OBSERVATION_POSE_SOLVER_V2.png는 solver packet의 I0 청록선을 보여 준다. 두 그림 모두 원본을 보존했다. 현재 전달 시에는 V2를 우선 읽는다.

full/tight mask가 상판 왼쪽의 넓은 검은 영역까지 포함하는 시각 단서는 유지된다. 그러나 I0의 긴 내부/측면 선과 B 부분선·외곽 마스크는 서로 다른 관측이다. 그림자/바닥 혼동과 실제 owner 대응 오류를 구분하려면 해당 경계의 물리적 의미를 독립 판독해야 한다. 이번 비용 결과만으로 SAM이 원인이라고 확정하거나 해당 선을 제거하는 정책을 만들지 않았다.

## 상태
