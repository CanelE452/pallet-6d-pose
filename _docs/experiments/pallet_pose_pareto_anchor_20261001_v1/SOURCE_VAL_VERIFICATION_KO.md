# Source VAL 독립 검산

**검산 PASS. 학습된 방법의 source gate는 FAIL (38/45)**이다. 검산 통과와 방법 성공은 별개다.

GT 접근을 막은 상태에서 최종4개 checkpoint의 **4,096개 선택**을 원래 cached feature로 재현했다. float32 정규화→float64 scorer와 별도 곱셈·합산/동률 선택을 비교했으며 모든 index·후보 이름·fallback·원래 geometric-valid mask가 일치했다. runtime에서 참조 기반 safe mask를 사용하지 않았다. 독립 score 최대 차이는 8.53e-14이다.

전체 routing lock과 모든 최종 fit 바인딩 확인 후에만 기존 source 참조의 **VAL1,024개 index**를 읽었다. renderer R에 `diag(1,-1,-1)`를 적용한 물리 기준과 C2 대칭 회전, 중심 이동 cm를 직접 계산했다. 평가 helper의 metric/gate 함수는 import하지 않았다. **8모델×1,024 pose = 8,192개**, T/R **16,384개 오류 값**이 저장 배열과 정확히 일치한다. 동일 회전각의 Frobenius 식도 확인했으며 최대 차이는 3.1e-11°다.

이전 convergence 결과의 R0 GEO 및 DIVERSE3 GEO **8,192개 오류 값**은 현재 fixed baseline과 byte 단위로 같다. 기존 결과를 새 모델 성능처럼 다시 생성하거나 baseline을 바꾸지 않았다.

전체1,024분모·실패/+∞·median/P90을 독립 정렬 보간으로 확인하고, 각 UNION을 R0_ONLY/R0_GEO/paired DIVERSE_GEO와 비교하는 **45개 Boolean**을 모두 재현했다. 임계값·seed·checkpoint를 바꾸지 않았다. 실패 목록은 JSON에 전부 남긴다.

원본 geometry 컨테이너에는 다른 source 행도 들어 있지만 검산 대상은 잠금된 VAL1,024 index뿐이다. TRAIN/TEST/실사 참조 오류 계산, 새 fit, 이미지 forward, PnP 및 새 routing 산출물은0회다. 이 VAL은 반복 사용 개발 자료이며 실제 실사 개선이나 전체 목표 달성의 증거가 아니다.

[검산 JSON](SOURCE_VAL_VERIFICATION.json), [source gate](SOURCE_VAL_GATE.json), [선택 동결](SOURCE_VAL_ROUTING_LOCK.json)
