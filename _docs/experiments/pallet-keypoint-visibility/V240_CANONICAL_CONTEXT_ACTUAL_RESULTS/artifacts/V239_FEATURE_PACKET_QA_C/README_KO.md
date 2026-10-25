# V239 feature packet QA / SOURCE report 준비

실제 payload 읽기·실행은 아직 0입니다. 원 published C 폴더는 변경하지 않았습니다.

Root 별도 exact entry 뒤 순서:

1. `python -B verify_feature_packet.py --entry ROOT_ACTUAL_FEATURE_QA_ENTRY.json`
2. 양팔 final1000 동결 및 global QA 후 `python -B summarize_saved_pair.py --entry ROOT_ACTUAL_REPORT_ENTRY.json`
3. `python -B summarize_saved_pair.py --verify-saved results` (stdlib 저장 산술 검산)

7개 특징만 숫자 검산하고 원 RGB/valid·DINO·가중치·head를 열거나 forward하지 않습니다. 원 samples와 geometry는 nonpixel member에만 한정합니다. 원 all15040 map과 실제7617 sample 관계를 검산합니다.

SOURCE 보고서에는 family별 CM/P/R/U·balanced recall·DEV 회복/악화·고정 게이트를 보존합니다. q-r 거리 추가 sidecar는 읽지 않으므로 UNKNOWN입니다. masked7423은 모델 U가 아닙니다. SYN MAIN과 YCB AUX의 참조 정의가 다르며, 노출 SOURCE DEV 결과는 새 실사 성공이나 모델 채택을 뜻하지 않습니다.

첫 준비 단계의 f-string syntax 오류는 FIRST_*에 보존했고 실제 payload 접근 전 교정했습니다.
