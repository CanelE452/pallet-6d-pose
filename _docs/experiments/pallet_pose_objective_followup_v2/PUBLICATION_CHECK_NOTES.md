# 공개 전 검수 정정

최종 공개 검수 첫 실행에서 `public_figures`가 `Unregistered public image file`로 실패했다. 새 그림 목록은 13개였지만 이전에 공개한 A/B 예시 4개도 삭제하지 않고 보존되어 있었기 때문이다. 수치·기존 파일 보존·학생 점수·학습 대조·예산·보고서 표 검사는 같은 실행에서 통과했다.

이미지를 삭제하거나 결과를 다시 선택하지 않았다. 기존 공개 commit `329425a3b328f1f3a87e8c8dca648c7dd453ab69`의 승인 목록에서 해당 4개를 가져와 retained historical 이미지로 등록하고, generator가 재실행해도 등록을 유지하도록 수정했다. 현재 실험의 개선 사례로 승격하지 않는다.

이 수정은 보고서 공개 목록의 수정이며 추가 학습이나 수치 변경이 아니다. 첫 실패 감사 JSON은 비공개 verification history에 보존한다. 재검사 결과는 [최종 감사](AUDIT.json)에 기록한다. 학습 난수 반복의 설계 누락은 별개의 문제이며 [반복 유효성 정정](REPLICATION_VALIDITY_CORRECTION.md)을 따른다.

목록 정정 뒤 보고서를 재생성하기 전에 테스트를 실행한 시도에서는 이전 manifest SHA를 참조한 수치 추적 검사 1개가 실패했다. 실패 TEST_RESULTS도 보존하고, 순서를 draft 재생성 → tests → 원장 정산 → final 재생성 → audit로 고정했다. tests 진행 중에는 이전 receipt의 원장 참조를 바꾸지 않도록 실행기를 수정했다. model fit 재시도는 없었다.
