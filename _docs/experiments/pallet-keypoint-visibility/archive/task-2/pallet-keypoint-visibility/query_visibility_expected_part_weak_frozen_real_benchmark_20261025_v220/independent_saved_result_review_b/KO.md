# V220 고정 실사 비교 독립 수치·보고서 QA

**PASS.** 주요6개 모델(v61/legacy/C55/C63/C64/C65)의 원래 current1784 원장과 기존 truth CSV를 id/native로 조인하고 Counter로 재집계했습니다. primary1237(D1147/E90), 목재/정사각형, native8, SELF136/원flag21, 기존 reference≤3px humanD461, 회복·손실이 저장 CSV 및 root 파생5표와 일치합니다. 모델·원 입력·NPZ/PT·RGB·3D GT를 열거나 실행하지 않았습니다.

C65 대 C63의 D 정답은843→901(+58), E 정답은19→14(−5)입니다. C64→C65는 D 회복377/악화58, E 회복2/악화23입니다. 보임 보존 증가를 외부 가림 해결이나 모델 채택으로 해석하지 않았습니다. legacy 주 평가 U177을 유지했고, 새 모델 U0은 강제 이진 응답의 결과라는 한계도 유지합니다.

전체 SELF136의 기존 primary 응답과 paired SELF21의 과제별 응답은 별도입니다. legacy geometry S는 전자에서 미해결·과제 외이며, D_vs_S 과제에서는 binary hidden으로 취급됩니다. SELF21 flag가 원장에 없어 승인된 원 truth CSV를 조인했으며 D_vs_E flag로 추정하지 않았습니다. 수동 근접 D376도461과 별도이고, 저장 표의 C63 오삭제90→C65 69(목재18→17/정사각형72→52)가 맞습니다. manual376의 원 membership 자체를 새로 재구성한 검산은 아닙니다.

첫 검사(d151ba)는 truth에 없는 roi_side를 읽으려 한 B parser 가정 오류로 집계/출력 이전 중단했습니다. 원 코드·entry·실패를 보존하고 root 승인으로 이 비교만 제거한 V2를 실행했습니다. L 보존은 predictor의 원 RAW10/raw8 fingerprint와 evaluator sameqL guard에 의존합니다. 수정 actual V2는2aed3b exit0 PASS입니다.

최종 보고서 Markdown30행(주요6·재질12·SELF6·reference6), HTML6행, 상대 링크37개가 일치했습니다(4bf82b exit0). 독립 수치 집계를 반복하지 않았습니다. root 실제12 image/gallery QA는 제공된 정확 binding에 근거하며 B는 이미지나 gallery body를 재판독하지 않았습니다. 브라우저 렌더 QA는 없습니다.

공개 held360 수치 본문은 B가 독립 재집계하지 않았습니다. 보고서의 weak center-owner204/UNKNOWN156/inputU108 및 권위 구분이 producer 계약과 맞는 범위만 확인했습니다. 반복 노출·legacy CAL 이력·GT 유래 SOURCE 학습 query·유한 부재 약라벨과 human native 가시성의 차이·미채택·미해결을 보존합니다.

[종합 바인딩](REVIEW.json) · [실제 수치 QA](actual_review_v2/REVIEW.json) · [최종 문구·링크 QA](FINAL_REPORT_TEXT_AND_LINK_QA.json)
