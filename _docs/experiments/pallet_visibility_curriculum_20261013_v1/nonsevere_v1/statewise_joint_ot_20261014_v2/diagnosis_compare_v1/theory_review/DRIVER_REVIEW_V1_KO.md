# 고정 counterfactual driver 독립 검토

입력 통제·phase·fallback·GT 표기와 실행 봉인을 확인했으며 미해결 material finding은 없다. 내부 검토 후 3CPU 진단 실행이 시작되었다. 실험 결과나 원인이 이미 입증되었다는 뜻은 아니다.

95건×11arm의 1,045개 pure input을 새 PnP fitting이나 RGB 읽기 없이 검산했다. 5,958개 확인은 실패0이며 compatible CAL44/TEST42, distinct782/cache100/new682가 실행 protocol과 같다. sameΣ raw/corrected, 별도 B2px/I3px rawstandard control, knownfalse→U, 고정O scale, reference-center의 BI/T 유지, reference R/Q/symmetry, fullhuman+budget6의 eligible86-only 및 MODEL_IDEAL95 construction feasibility를 확인했다.

ORACLE_BINARY_RAW_STANDARD_COV는 초기 S/T를 유지하면서 human-known DIRECT/EXTERNAL을 바꾸고 나머지 SELF/UNANNOTATED는 U로 보류한다. 따라서 **S/T gate를 유지한 human-known D/E 교정+나머지 label abstention의 controlled support-selection oracle**이며 pure D/E classifier만 고친 인과 효과로 해석하지 않는다. FULL_HUMAN은 초기 S/T 지원점 구조까지 바꾸는 다른 raw-query oracle이다.

‘같은 starts’는 같은 supplied coarse와 설정·검색 전략을 뜻한다. q가 달라지면 P3P roots와 global depth bootstrap도 달라지는 원 알고리즘을 유지한다. reference-init은 BI seed/fallback에도 영향을 줄 수 있으므로 accepted joint recovery와 구분한다. closestcoarse selector는 같은 저장 feasible pool만 사용한다. oracle-best는 minimum translation의 **하나의** 후보를 선택하고 그 pose의 모든 metric을 보고하며, metric별 attainable floor는 별도로 기록한다.

현재 source/실행 snapshot21개(기존 core19 + 새 driver/test), protected user6 및 evidence binding이 모두 일치한다. score는 source·evidence·label·protected SHA를 다시 검사한다. oracle record는 nested diagnostic GT arguments를 명시해 pure solver의 내부 file read0와 외부 GT argument를 구분한다. 모든 진단은 이미 재사용된 CAL/TEST의 posthoc 원인 연구이며 deployment나 새 unseen 우월성 평가가 아니다.

최초 검토 source는 `9669ac350c5ab298392df50b90615a3619126a34e37d9d626d7f6ac51d733f3b`이다. 실행 직전 checkpoint I/O만 추가한 실제 source는 `9572f383d3947dffdef7ae0554346e9038e1fa2a81fe9ab90d88a885226e10d3`이며 실행 protocol/snapshot에 고정됐다. gzip/json import와 checkpoint context, completed-solution serialize/flush를 제거한 byte reconstruction SHA가 최초 검토 SHA와 **정확히 같다**. 수학·변환·설정·입력 변화는0이다. 중간 checkpoint는 final prediction/score seal을 대신하지 않는다. 실행 중 source를 수정하지 않았다.

상세 근거는 [DRIVER_REVIEW_V1.json](DRIVER_REVIEW_V1.json), [PURE_INPUT_AUDIT_V1.json](PURE_INPUT_AUDIT_V1.json), [I/O 보충 기록](DRIVER_CHECKPOINT_IO_AMENDMENT_V1.json)에 있다. 모델 oracle의 feasible reference나 perfect observations는 실제 image state/location 성공을 증명하지 않으며, any-C2 가능성과 fixed-native-phase 가능성을 나누어 보고한다. 여러 제약이 동시에 reference를 배제하면 단일 intervention의 음성 결과만으로 원인을 배제할 수 없다.
