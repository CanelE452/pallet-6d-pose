# 후속 분석 독립 검토

2026-10-01. 검토 범위는 새 selector attribution·candidate bounds 코드와 산출 JSON/CSV, 보고서, 그림2개다. 기존 모델·예측·평가 참조·분석 코드는 수정하지 않았다. 검토자는 이 단계에서 실사 GT를 새로 읽거나 pose를 새로 채점하지 않았고, 학습·이미지 신경망 추론도 수행하지 않았다.

**현재 저장 결과의 수학·집계에 결론을 뒤집는 오류를 찾지 못했다.** 이것은 성능 개선, 새 데이터 일반화, 새로운 선택기의 실효성에 대한 PASS가 아니다. 전체 목표의 안정적 T/R 공동 개선은 여전히 미달성이다.

## 독립 재계산

- 결과가 직접 연결한 파일12개의 SHA256/크기를 확인했다.
- [후보 CSV](CANDIDATE_BOUND_ROWS.csv)의3,114행이 model×frame×oracle 기준으로 중복 없이 존재하며 모두 완전한 pose 이름과 finite T/R를 갖는지 검사했다.
- CSV에서9모델×3집단×2oracle×2지표×2통계=216개 중앙값/P90을 직접 다시 계산했다. [JSON](CANDIDATE_BOUNDS.json)의 conditional 및 full-population 값과 일치했다.
- 개별모델 및3seed 평균의 하한, 원래1.05 허용값과 비교한 판정을 다시 계산했다.
- 저장 feature tensor와 동결 Linear94 weight/std로 contribution 배열을 다시 계산했다. R0 대비 명시적 center/나머지 residual 그룹 통계54개도 별도 NumPy 식으로 재계산해 [점수 분해 JSON](SELECTOR_ATTRIBUTION.json)과 일치했다.
- 그림2개를 실제로 열어 모델·3seed 평균 표기, 단위, ceiling, log축 및 인과적 중요도가 아니라는 문구를 확인했다.

여기서216개 재계산은 공개 CSV의 집계 검산이다. 원래 평가 참조와 모든 pose의 물리적 정당성을 별도 독립 GT로 다시 확인했다는 뜻이 아니다. 기존3모델과의48개 oracle parity는 원 실행 코드·결과에서 확인했으며 이 감사의216개 검산과 구분한다.

## 하한 논리와 실패 처리

동일 프레임의 고정 후보에서 `b_i=min_c T(i,c)`이면 임의의 선택 결과는 `b_i≤T_i`다. 원소별 순서를 보존하는 동일 분모의 경험적 median/P90, 그리고 그 seed별 통계의 평균에도 이 부등식이 유지된다. 따라서 하한부터 허용값을 넘으면 해당 후보 집합의 선택 변경만으로 그 기준을 만족시킬 수 없다.

코드는 T-best와 R-best에서 각각 **하나의 전체 pose**를 선택한다. 별개 후보의 T최솟값과 R최솟값을 합쳐 가짜 joint pose를 만들지 않는다. 두 oracle은 배포 입력 또는 모델 선택 결과로 내보내지 않는다.

현재9모델의 natural99/clean29/wood45 모든 oracle에서 실패0을 확인했다. 따라서 이번 결과의 conditional/full-population 통계는 동일하다. 원래 실패 수 비증가 gate는 seed마다 적용되므로, 어려운 프레임을 의도적으로 실패 처리해 조건부 분모에서 삭제하는 것도 통과 방법이 아니다. 코드의 `bound()`가 conditional 값을 사용한다는 사실을 숨기지 않아야 하며, 실패가 있는 다른 데이터에 그대로 재사용할 때는 이 동등성을 새로 검증해야 한다.

| 판단 대상 | 검토 결론 |
|---|---|
| FULL125 clean | T 하한3.432615cm > 허용3.076355cm. 고정 후보 선택만으로 이 기준 통과 불가 |
| DIVERSE3seed natural tail | T P90 하한129.502732cm > 허용126.494939cm. 독립적으로 기준 초과 |
| DIVERSE3seed clean | 초과량0.000016957cm는 수학적으로 엄격한 기준 밖이지만 물리적으로 의미 있는 붕괴로 과장하지 않음 |
| R0 및 SINGLE3seed 평균 | 이 두 하한으로 제외되지 않음. 실제 구현 가능한 선택기, joint T/R 가능성, 다른 gate 통과 또는 일반화의 증거는 아님 |

개별 seed의 불가능 판정과3seed 평균 gate를 혼동하면 안 된다. 이 결론의 범위는 고정 좌표·선택 검출박스·solver·기존 W/D 후보이며, 모델 간 후보 합집합이나 detector/좌표 변경까지 불가능하다고 확대하지 않는다.

## 점수 기여와 해석

동일 Linear94를 두 후보에 적용하면 bias 및 정규화 mean이 점수 차이에서 상쇄된다. `Σ_j w_j(x_long,j−x_short,j)/std_j`는 결정 margin의 올바른 분해다. float32 원점수와 float64 재구성 차이 최대약2.9e−6은 기록돼 있으며, 실제 선택 이름1,557건은 원 결과와 일치했다.

그룹 값은 **특징별 절댓값의 합이 아니라, 그룹 내부의 부호 있는 변화 합에 절댓값을 취한 뒤 프레임 평균한 값**이다. 그룹 내부와 그룹 사이 모두 상쇄가 가능하다. 따라서 값을 합쳐 전체 변화의 백분율로 만들거나 원인 기여율로 해석할 수 없다. 보고서와 그림은 이 점을 명시한다.

공통 confidence/bbox 직접항의 margin 기여0은 대수적으로 맞다. confidence 가중 residual, bbox 정규화 등 간접 영향까지0이라는 주장은 아니다. 명시적 center4항 이외의 mean/P90 residual 등에도 center가 들어간다. 현재 분해는 center 전체를 제거한 ablation이 아니며, “중심이 무관하다” 또는 “residual feature를 지우면 개선된다”를 증명하지 않는다.

추가 문서의 solver 표현도 수정 후 확인했다. `pnp_selector.py`는 corner8로 SQPnP/LM을 풀고 중심을 포함한9개 점을 투영해 residual 특징을 만든다. feature에9점이 포함된다는 사실을9점 PnP fitting으로 부르면 안 된다. 최종 `SELECTOR_FEASIBILITY_KO.md`는 이 차이를 정확히 구분한다.

## 보고 문구와 게시 범위

검토 중 원 보고서의 “모든 seed·recording에서 공동 개선하는 조건”은 각 recording 자체의 개선이 사전 gate였던 것처럼 읽힐 수 있어 수정 의견을 전달했다. 최종 보고서에서 실제 조건인 **세 seed 공동 개선·recording 재표본/제외 민감도·꼬리 보존**으로 수정됐음을 확인했다. 다른 주 수치와 주요 결론도 JSON에 부합했다.

[연속 pose 감사](CONTINUOUS_POSE_AUDIT_KO.md)의 상대 링크는 게시 checkout의 base `51907cf35173ade213eec154abb2e9bcd307adf3`와 새 phase 파일을 합친 경로 기준으로 모두 존재했다. 링크가 있다고 private checkpoint·원본 cache까지 게시된다는 뜻은 아니다. 실제 게시·push SHA 및 remote 일치 검증은 별도 게시 영수증의 범위이며 이 검토가 대신하지 않는다.

새로운 실험이 성능 향상을 입증한 것처럼 보고하지 않고, 이미 실패한6회 학습을 근거로 다음 변경의 가능 범위를 좁힌 탐색 분석으로 게시하는 것이 현재 증거에 맞다.
