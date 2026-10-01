# 고정 RBF64 특징의 학습 전 독립 검산

**입력·수학 검산 PASS. 새 학습이나 성능 평가는 실행하지 않았다.** 변경은 기존189 context 뒤에 고정 RBF64를 추가한253 특징 하나다. 기존94 FP32 정규화→FP64 context, target·risk·margin·valid·2,598행 분모·λ=1e−4와 solver 예산은 유지한다.

센터64개는 참조와 TRAIN label을 열기 전에 metadata로 고정된 eligible TRAIN 및 기존 feature cache만으로 독립 재현했다. `[frameID,expert,hypothesis]`의 UTF-8 canonical JSON을 SHA256 순서로 정렬하고, 동일 identity를 중복 제거한 다음 float64 context189의 bytes가 다른 최초64개를 선택했다. 양의 센터 쌍별 제곱거리 median을 bandwidth²로 고정했다. 실제 센터·identity·normalization·bandwidth는 잠금 artifact와 정확히 같았다. 해당 단계에서 source 오류·target·VAL 품질·실사 참조를 읽지 않았으며, 단계별 접근 경계와 해시를 JSON에 기록했다.

후보와 센터를 하나씩 순회한 별도 RBF 구현이 실제253 map과 정확히 일치했다. all-invalid 실패1행은 anchor=-1과 zero253 특징을 유지한다. 유효 후보가 있는데 anchor가 없으면 중단하며, 원래 invalid mask를 바꾸지 않는다. TRAIN label 접근을 허용한 이후에는 기존 scalar target·risk·margin으로 직전 risk PREFIT의 모든 supervision hash가 그대로임을 대조했다.

실제 네 이전189 checkpoint에0을64개 붙여 전체 TRAIN의 동일 입력 score가 유지되는지 확인했다. 허용 절대차는 1e-10이며 모델별 최대차는 JSON에 기록했다. 이 계산은 표현 경로 보존 확인뿐이며 실제 label-dependent objective·선택률·T/R 성능을 계산하지 않았다. 253 reduction 순서 차이 때문에 수학적 동등성과 부동소수 허용오차를 구분한다.

목적식·margin·target·Hessian·certificate의 AST를 직전 구현과 대조했다. 합성 fixture에서 zero64 embedding의 목적값과 앞189 gradient가 유지됨을 확인했으며, 새64 gradient가0이라고 주장하지 않는다. 독립 scalar objective/covariance Hessian 및253방향 중앙차분 최대 오차는 gradient 6.13e-11, Hessian 1.25e-10였다. 최소 Hessian 고유값은 0.0001로 λ-strong convexity 계약을 지킨다. runtime에서 margin 생성함수를 오류로 바꿔도 score가 동작하며, runtime에 GT 가산항이나 안전 mask를 넣지 않는다.

이 검산은 새로운 비선형 특징이 T/R를 개선한다는 증거가 아니다. 후속 학습과 고정 source45 및 실사 두 비교 묶음의5개 AND 조건은 별도 실행·판정이 필요하다. 새 fit·optimizer step·VAL/실사 품질 조회·forward·PnP·threshold sweep은0회다.

[독립 검산 JSON](PREFIT_REVIEW.json) · [고정 basis](RBF_BASIS.json)
