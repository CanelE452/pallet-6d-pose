# 상속 proposal bank의 e 출처 확인

의도한 no-numeric-e candidate/start 기준에서는 **source gate FAIL**입니다. 기존 입력·runtime·모델을 변경하지 않았고 새 V31 예측은 평가하지 않았습니다.

고정 첫 CAL 한 행의 proposal 300개 중 e 출처가 명시된 distinct pose는 50개(출처 참조112개)입니다. CACHED_0은 보존 e의 physical R와 t가 정확히 같고, 두 단계의 SAME_FROZEN_E/published 출처가 있습니다. 동시에 D/ORIGINAL_WD/published 등과 dedup된 복수 출처입니다.

v17의 모든 arm 수집 규칙이 e solution을 직접 수집하고 V29/V31은 그 목록 전부를 현재 guard로 평가하는 cached contender로 사용합니다. 같은 목록은 refiner로 넘어갑니다. 실제 guard 통과·6개 start 채택·최종 winner는 이 감사에서 계산하지 않았습니다. fallback/prior가 e 숫자를 새로 쓰는 코드라는 뜻도 아닙니다.

E tag만 지우면 원 posteror pose의 출처 문제가 해결되지 않습니다. 앞으로 엄격한 first-input PnP bank를 쓰려면 별도 정책과 선언이 필요합니다. 현재 모든 원본과 봉인은 그대로 보존합니다.

세부 근거: [고정 첫 CAL inventory](FIRST_CAL_E_ORIGIN_INVENTORY_V1.json). 원 목적함수 차이: [source audit](REPORT_KO.md).
