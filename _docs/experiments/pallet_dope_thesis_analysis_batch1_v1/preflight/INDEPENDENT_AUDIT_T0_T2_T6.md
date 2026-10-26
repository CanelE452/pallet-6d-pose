# T0 T1 T2 T3 T4 T5 T6 최종 독립 감사

상태: **PASS_AFTER_EXPLICIT_DERIVED_SCORES_AUTHORIZATION_WITH_LIMITATIONS**. 사용자 답변 “기존 파생 SCORES 사용만 허용하고 push”를 `GATE5_RESOLUTION.json`의 SHA로 연결했다. 보호 원본 FINAL RGB와 annotation 본문은 계속 금지다.

## 중지 이력과 승인 범위

초기에 T2 및 독립 감사가 FINAL-role56장의 파생 reference.xy를 소비한 사실을 유지한다. assistant의 자체 cache 허용 해석으로는 부족했으므로 작업과 push를 중지했고, 당시 STOP JSON/MD를 `_BEFORE_AUTHORIZATION`으로 그대로 보관했다. 직접 사용자 승인 뒤 기존 파생 SCORES 범위만 재개했다. 18개 동결 수치CSV의 SHA는 승인 기록과 모두 같고, 결과를 다시 계산하거나 바꾸지 않았다.

## T3 필터 검증

876 teacher-frame 쌍, unique DEV292, main/no-flip 각7884개 채널(합15768)을 cached original mask와 final mask로 대조해 불일치0을 확인했다. 각모드1395개 등록가설 비교에 대한 original-function 동치 receipt도 source SHA와 일치한다. s0의 native bounds·finite·confidence≥0.3, 두 view 교집합, 두 confidence-only 비교를 별도 재확인했다. 단계는 원래 최종 선택가설과 고정 consensus/median에 조건부인 누적 진단이며, 단계별 가설 재선택이나 flip의 독립 인과효과를 뜻하지 않는다.

132개 집계행 모두 selected eligible reference=selected observed임을 확인했다. 선택/탈락 partition과5/10/20px 정답 수·각 분모의 비율이 일치한다. 주석 없는 선택점의 정확도를 만들어내지 않는다. 이 감사에서 PnP·LOO·단계 정밀도를 새로 실행하지 않았다.

## T4 가능성

15모델×3종류45그룹과4380행의 cached flag를 확인했다. 주비교21/교차보조24이며 모든45그룹에서 검출6점 이상 수와 positive-depth PnP가능 수가 같다. 이는 제공K·고정기하 아래 수치해가 가능한 비율이고 독립 자세 정확도가 아니다. 새 PnP·모델 forward를 실행하지 않았다.

## T5 고정 사례

기존292행 CSV의 정확한 유리수 차이를 사용해 종류별 max/모집단 median에 가장 가까운 값/min과 모든 동률ID오름차순 규칙을 대조했고9사례가 정확히 일치했다. PNG9/PDF9 및 PDF렌더SHA와 별도18개 원해상도 visualQA receipt를 확인했다. 이 감사자는 원본 RGB·annotation을 열지 않았고 시각검사를 새로 했다고 주장하지 않는다.

## 남는 한계

- T0 recording groups are declared SHA-overlap/union-find provenance, not independently authenticated continuous camera recordings.
- PLASTIC has5 ambiguous recording IDs; GREEN119 has no declared recording ID; WOOD has only2. Every requested recording CI is correctly unavailable.
- All computed bootstrap CIs are frame-level and may be narrow under within-recording dependence. There is no multiple-comparison adjustment.
- GREEN manual means recorded source=manual_click, not independent manual verification; reference source mix and nonphysical6D limits remain.
- PLASTIC184 is a separate derived-score supplemental panel, includes FINAL-role56, and is not added to the core population.

이 감사는 source/receipt/수치 정의 및 불변성 확인이며 새로운 독립 TEST나 물리6D 검증이 아니다.
