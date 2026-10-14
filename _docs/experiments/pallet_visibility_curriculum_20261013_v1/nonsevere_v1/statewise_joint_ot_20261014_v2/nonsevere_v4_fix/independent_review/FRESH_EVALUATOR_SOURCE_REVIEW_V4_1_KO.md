새 evaluator 소스는 predict-only 실행을 승인할 수 있는 상태다. 실제 currentAPI.refine를 98개 모두 호출하며 Tiny/SAM 모델·실제 RGB·native/K/Q를 바인딩한다. O/T 제거와 prior 제거 대조군은 같은 관측·fit 공분산·scale을 유지한다. ALL arm은 항상 실제 솔버 호출이고 대조군만 같은 수학 입력을 재사용한다.

이전 cache는 당시 실행 source23 snapshot에, 현재 실행은 현재 source에 각각 묶인다. 실제 98개 관측 parity가 실패하면 score 가능한 seal을 내지 않는다. source/model/input/user6를 전후 확인하고 ordered98·fallback·모든 distinct feasible samples를 보존한다.

scorer의 좌표 label 접근은 complete prediction seal 검증 뒤에 시작한다. old RGB/boundary/v3의 같은95 입력·Q·봉인 pose를 다시 동일 C2로 평가한다. GT native 호환성이 낮은 사례도 TEST49 점수에서 빼지 않는다.

현재 소스 material blocker는 0개다. 실행 결과와 성능 완료를 인증한 것은 아니다. root 지시에 따라 TEST scoring은 추가 CAL β 선택 봉인까지 보류하고, 비영 β가 채택되면 해당 최종 방법의 새 fullfresh API 통합을 별도로 검증해야 한다.
