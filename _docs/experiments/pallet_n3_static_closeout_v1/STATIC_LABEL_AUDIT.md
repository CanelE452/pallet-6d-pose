# 정지 이미지 주석 감사

현재 DEV319: {'clean': 29, 'moderate': 20, 'severe': 79, 'unclassified': 191}. 과거 128장과 ID 집합은 같다.

- `eval_pallet09:1778653661653195264`: MODERATE_OCCLUSION → SEVERE_OCCLUSION. 현재 직접 검수 manifest/잠금 응답과 이전 SPLIT_LOCK의 복사본이 다르다. 변경 시각·사유는 미확인. 개수를 맞추는 변경은 하지 않았다.

승인된 FINAL_V2 원자료에서 직접 가시 66점과 외부 가림 5점의 상태를 회수했다. 기존 DEV 좌표는 수정하지 않았다. 수동 재클릭 좌표와 기존 참조 좌표는 다르므로 이 표는 기존 참조 오차의 가시성별 분석이다. PnP 보조 검수 이력과 사전 예측 노출 미확인을 명시한다. 나머지 점은 UNKNOWN이며 유효 좌표를 가시성으로 간주하지 않는다.

미분류191장의 직접 검수 근거는 찾지 못했다. 그중 66장은 SPLIT_LOCK에 등급이 복사되어 있지만 원 직접 검수 manifest에 없는 ID여서 확정 레이블로 승격하지 않았다. review/index.html과 REVIEW_TEMPLATE.json은 검수 대기 자료이며, 모델·오차·순위가 없다.
