# 실행·결과 검토표

지시문23항목의 실제 근거 요약이다. 측정 성능의 승리, 모델 시각 검토, 독립 물리 정답, 원격 게시는 별개다.

|항목|상태|근거|
|---|---|---|
|버전|확인|base870cb9e; 원본status/HEAD/index·6보호파일·25봉인artifact 보존|
|D1 재사용|확인|238/1003; 기존producer 재실행0|
|실제 행동|확인|모든 신규선택 원bank의 wholepose; alias/eligibility/순서 보존|
|F0|독립PASS|188차원 bitwise 및 기존9checkpoint×238 재현|
|회귀 감독|독립PASS|log두축 SmoothL1·minimax; 결측/비교식 유지|
|순위 감독|독립PASS|성공우선·실패Pareto; tradeoff무감독,23314검사|
|전부 실패|보존|train416중215Pareto/201no-info; realfull41/pair59 별도|
|F1|독립PASS|두family144+144,1241장/42285행동,259350검사|
|F1 대조|독립PASS|동일34753params/init/공통epoch순서·쌍;408검사|
|B 진단|완료/공백보존|합성1003; raw/backend분리; originalstrictFAIL660유지; owner미검증|
|V gate|실제통과|합성valmiss12/입력1241/예산; PRIMARY점수사용0|
|V 비교|독립PASS|동일57977params/query/mask;RGB3만차이,363검사|
|새 fit 수|실제18/18완료|optimizer실패0; CLI기술실패별도보존|
|모델 선택|확인|각자원syntheticval loss strictmin; 대표20261018|
|예측 분리|확인|32군×238출력봉인 후 realref receipt|
|전량 평가|독립PASS|192/49/143/CAL46,모든seed/실패;2645768검사|
|연속 손상|전량보존|공동실패 및 full41/pair59 ΔT/ΔR·꼬리·최대|
|실제 회복 출처|전량보존|N0/E0/OTHER source와정확8칸 THREE_WAY_SUCCESS|
|독립 검산|수준별기록|입력/감독/기하/숫자 별도,표현차이7741와strictFAIL구별|
|이미지|모델실제검토|232래스터생성,34개view_image;나머지개별modelreview아님|
|안전/개인정보|기술검사별도|PRIMARY파생만허용;원RGB/weights/bank미게시;PUBLIC_PACK_AUDIT 참조|
|원격|별도실제검증|PUBLICATION_RECEIPT.json과최종실제원격SHA 참조|
|목표|교환관계확인|회귀위험감소/일부회복손실; F1/RGB채택근거없음;새독립실사0|

[전체검산](VERIFICATION.json) · [모델pixel검토](PIXEL_REVIEW.json) · [공개기술검사](PUBLIC_PACK_AUDIT.json) · 원격 증거 파일: `PUBLICATION_RECEIPT.json`
