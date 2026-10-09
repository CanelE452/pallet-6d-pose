# Observation refiner / robust PnP — 2026-10-09 protocol

이번 파일·key·CLI는 새 실험 스키마다. 과거 파일에 이 key가 있다고 가정하지 않는다. [RESULT_KO.md](RESULT_KO.md)가 결과 보고서다.

- INPUTS/OBSERVATIONS/LEARNED_OBSERVATIONS는 추론 입력·선택 봉인. GT는 봉인 후 별도 평가에만 사용한다.
- PREDICTIONS와 LEARNED_PREDICTIONS는 319개 ID를 유지하는 수치 원행이다.
- METRICS/PAIRED_COMPARISONS는 신규 자세·공통집합·fallback 전체 운용을 구분한다.
- GEOMETRY_STRESS는 렌더0인 수학적 진단이며 실사 증거가 아니다.
- REAL_CORRESPONDENCE는 참조상 좌표 정확성과 사람 가시성을 구별하는 사후 oracle 진단이다.
- RUNTIME는 detector와 초기/후단 PnP를 포함한 실제 전체 실행이다.
- SOURCE_FAMILY_SPLIT/TRAIN_LOGS/TRAINING_COMPLETION은 source-only 동일 학습 예산의 근거다.
- 실사 RGB·대형 cache·checkpoint는 공개 git에 넣지 않는다. 로컬 binding은 환경변수로 제공한다.

재현은 [REPRODUCE.md](REPRODUCE.md), 검산은 [VERIFICATION.json](VERIFICATION.json)을 참고한다.
