# 고정 첫 CAL의 새 PnP 후보와 native 식 재현

`eval_cad:1778653003088339968`은 원래 CAL 목록의 첫 사례다. 오차나 GT를 보고 고르지 않았다. 현재 positive worker는 초기 독립 coarse와 관측 B/I만 받았다. 실제 fresh SQPnP32회, 두 W/D별23후보(22fresh+자체coarse1), 기존 evaluator/refiner wrapper·연속 optimizer 호출0을 기록했다.

저장된 원 B/I q·source·covariance·weight로 fresh input digest를 다시 만들고, 모든 seed의 출처/상태 행 보존·native 시작점 hash/원천·이름에 대응하는 R/native_t/p/scale을 확인했다. 전체 seed의 acceptance와 비용을 독립 재계산한 것은 아니다. 자체 coarse와 선택된 시작점 총16개에 대해서는 독립17 guard·29 scalar 식으로 native 비용·factor 정체·support 제약·중앙차분 gradient를 재현했다.

**1,179개 검사 / 실패0**이다. 최대 native scalar 차이는2.27374e−13, 최대 gradient 성분 차이는5.02340e−7이었다. 16개 probe 중8개가 현 제약을 만족했으며 이것은 pose 정확도 성공 수가 아니다.

[결과](ACTUAL_AUDIT_V1.json), [봉인](ACTUAL_AUDIT_SEAL_V1.json), [실제 smoke](../clean_fresh_BI_actual_runtime_smoke_v4_32/SEAL.json), [driver 독립 검토](../clean_native_runtime_equation_driver_peer_review_v4_32/SOURCE_REVIEW_V1.json)를 보존했다. 최초 driver의 probe 이름/자세 결합 공백은 initial_driver_source_attempt_v1에 남긴 뒤 수정하고 개정4개 회귀 검사를 통과했다. 이전3과 개정4는 합산하지 않는다.

이 검사는 native 비용과 시작점 출처 연결의 범위다. constraint Jacobian·O/S/C·selector의 독립 검산, 실제 물리 edge 대응·전역 최적·전체 pose 정확도를 증명하지 않는다. 전체238 관측은 별도 봉인됐고, 전체 pose 입력 준비의 ndarray 저장 hash 오류는 원 v4_32 실패 산출물을 보존해 별도 r2 운영본에서 수정한다. 여기의 원 numerical smoke/source/모델은 바꾸지 않는다.
