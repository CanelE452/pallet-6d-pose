# 최신 확인 사항 — 2026-10-15 V9

[전체238장 평가·실제 과정 이미지 보고서](publication_followup_20261015_v9/REPORT_KO_FINAL_V1.md)와 [초기 자세 guard·비용 추가 검증](publication_followup_20261015_v9/INITIAL_GUARD_ADDENDUM_V1.md)을 공개했다. 고정 SOURCE+전경 포함 비용(C)은 주 평가192장에서 위치≤5cm 120장/동시117장(기존 e115/114)이지만 TEST49의 동시49→42, 평균 위치오차14.1121→15.0995cm로 악화됐다. 최대 위치오차191.31cm다. **기본값 SAME_FROZEN_E 유지, 목표 미달.**

전체예측238×5·수치검산294,819개·점수독립재집계16,511개 모두 누락/검사실패0이다. 이는 계산과 기록의 일치이며 정확도 인증이 아니다. 세 실제 사례/15과정그림과 모든 점수를 보존했다. [V8 코드·CAL 모델](publication_followup_20261015_v8/REPORT_KO_V3.md), [V5 출처 정정](publication_followup_20261015_v5/REPORT_KO.md)을 함께 읽는다.

현재는 분리된 mask 성분을 하나의 convex hull로 묶는 영향을 별도 버전에서 검토하며, 기존 결과나 기본값을 바꾸지 않는다.
