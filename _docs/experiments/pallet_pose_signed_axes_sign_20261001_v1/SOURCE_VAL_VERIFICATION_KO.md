# Sign-logistic Source VAL 독립 검산

**검산 PASS. 학습된 방법의 source gate는 FAIL (43/45)**이다. 검산 통과와 방법 성공은 별개다.

GT 접근을 막은 상태에서 최종4개 checkpoint의 **4,096개 선택**을 원래 cached feature로 재현했다. 입력-only R0 GEO index를 GEO 이름과 완전한 pose dictionary에 대조했다. float32 정규화→float64 z94, abs(z−z_anchor)94, identity1과 고정 Gaussian64의 phi253을 독립 계산한 뒤 같은 행 anchor의 phi253을 뺐다. 원래189·phi253·anchor 차분 입력 모두 실제 구현과 bit 단위로 같다. 별도 후보·축 반복의 내적으로 두 signed-axis 예측을 계산하고 max 점수와 기존 동률 규칙으로 선택했다. 저장된 모든 두 축 예측값은 공유 scorer와 정확히 같고 독립 계산 최대 차이는 3.02e-14, score 차이는 3.02e-14이다. 모든 index·후보 이름·fallback·원래 geometric-valid mask가 일치했다.

anchor 입력과 두 예측은 정확히0이다. 이는 예측상 선택 점수≤0일 뿐 실제 T/R 비악화 보장은 아니다. 추론에서 TRAIN target 생성기를 차단한 합성 검산도 통과했으며 margin·참조 기반 safe mask·새 GT target을 사용하지 않는다. signed target 등6개 TRAIN 해시는 PREFIT·checkpoint·START·FIT·TRACE·COMPLETE에서 provenance로만 대조했다.

center 선정 자체는 기존 RBF PREFIT에 근거하며 이번 signed PREFIT는 해당 phi253의 정확한 재사용과 anchor 차분을 확인했다. checkpoint embedded basis와 원본 RBF_BASIS 바인딩, normalization SHA, 고정 bandwidth를 확인했다. VAL 입력이나 품질로 center·폭·정규화를 다시 만들거나 맞추지 않았다. invalid 후보는253 입력과 두 예측을 모두0으로 유지한다.

Huber+Sign_logistic+L2 합이 objective와 같은지 확인했다. 계수1과 nonzero-target sign 규칙은 메타데이터에서 대조했으며 source 추론에는 sign label/참조오차를 전달하지 않았다. 모든 objective trial을 포함한 Newton trace의 초기점·채택점 연결, Armijo 값/alpha 순서, gradient L∞≤1e-8 및 gap≤1e-6, 최종 채택 call을 추가 확인했다. 거부 trial 뒤 이전 채택점을 보존하는 계약이며 초기평가와 모든 거부 trial도2,000회 예산에 포함한다. 이 검산에서 TRAIN 최적화나 타깃 계산은 하지 않았다.

전체 routing lock과 모든 최종 fit 바인딩 확인 후에만 기존 source 참조의 **VAL1,024개 index**를 읽었다. renderer R에 `diag(1,-1,-1)`를 적용한 물리 기준과 C2 대칭 회전, 중심 이동 cm를 직접 계산했다. 평가 helper의 metric/gate 함수는 import하지 않았다. **8모델×1,024 pose = 8,192개**, T/R **16,384개 오류 값**이 저장 배열과 정확히 일치한다. 동일 회전각의 Frobenius 식도 확인했으며 최대 차이는 3.1e-11°다.

직전 signed_axes_newton 결과의 R0 GEO 및 DIVERSE3 GEO **8,192개 오류 값**은 현재 fixed baseline과 byte 단위로 같다. 기존 결과를 새 모델 성능처럼 다시 생성하거나 baseline을 바꾸지 않았다.

전체1,024분모·실패/+∞·median/P90을 독립 정렬 보간으로 확인하고, 각 UNION을 R0_ONLY/R0_GEO/paired DIVERSE_GEO와 비교하는 **45개 Boolean**을 모두 재현했다. 임계값·seed·checkpoint를 바꾸지 않았다. 실패 목록은 JSON에 전부 남긴다.

원본 geometry 컨테이너에는 다른 source 행도 들어 있지만 검산 대상은 잠금된 VAL1,024 index뿐이다. TRAIN/TEST/실사 참조 오류 계산, 새 fit, 이미지 forward, PnP 및 새 routing 산출물은0회다. 이 VAL은 반복 사용 개발 자료이며 실제 실사 개선이나 전체 목표 달성의 증거가 아니다.

[검산 JSON](SOURCE_VAL_VERIFICATION.json), [source gate](SOURCE_VAL_GATE.json), [선택 동결](SOURCE_VAL_ROUTING_LOCK.json)
