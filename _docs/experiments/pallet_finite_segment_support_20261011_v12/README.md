# V12: 저장된 유한 선분 지원의 읽기 전용 진단

**목표 미해결이며 새 자세·GT 채점0이다.** 기존 V11 C2의 무한선 법선 잔차가 실제 유한 segment의 관측 구간 지원을 보장하지 않는 조건을 확인했다. 전체245장 중 선택 후보 radius 초과64장, 비동일 branch가 같은 전체8px 합의를 만족한106장이 있었지만 어느 물리 edge/pose가 옳은지는 판정하지 않았다.

[결과와 분모·한계](RESULT_KO.md), [실행량](BUILD_LEDGER.json), [실제 진단 receipt](DIAGNOSTIC_RECEIPT.json), [독립80,320 PASS](SEGMENT_CHECKS.json)를 함께 확인한다. [245개 진단 원행](SEGMENT_ROWS.jsonl.gz)은 모든 candidate·실제 query·원 radius·segment 거리·원 운용 상태를 보존한다. FILTER97NEW/148fallback과 SAFE94NEW/151fallback을 바꾸지 않았다.

[진단 계약](DIAGNOSTIC_CONTRACT_KO.md)과 [검사 계약](CHECKER_CONTRACT_KO.md)은 별도로 실행 전 동결됐으며 radius는 불확실성 surrogate다. 새 모델·영상·학습·GT·fit·projection·J·SVD·정책 gate0이다. V11의 H 초기 좌표 fit 제외와 승인 NEW 후 재투영 계약은 유지했다. 공개 검토·정상 전용 브랜치 게시의 완료는 후속 실제 receipt가 권위다.
