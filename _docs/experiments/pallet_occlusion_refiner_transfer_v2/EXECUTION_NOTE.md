# 실행 순서 기록

- [확인] 첫 pytest는 test_contracts.py의 상대 import로 인해 수집 단계에서 실패했다. 모델/데이터 계약 assertion 실패가 아니라 Python module import 문제였다.
- [확인] 도구 묶음의 다음 호출이 pytest 종료코드를 조건으로 삼지 않아 학습 프로세스가 이미 시작됐다. A00 step1 출력이 관측된 뒤 PID1943733에 SIGSTOP을 보내 잠시 정지했다. 엄밀히 모든 자동검사 완료 이전에 학습이 시작됐다는 실행순서 일탈을 기록한다.
- [확인] 테스트 파일의 import를 절대 module 경로로만 수정했다. training/pseudo-label 코드·고정 입력·checkpoint·hyperparameter는 변경하지 않았다. 재실행10개 검사 모두 통과(6.68s).
- [확인] 동일 프로세스에 SIGCONT로 재개했다. seed/order/optimizer state를 재시작하거나 결과로 선택하지 않았다. elapsed wall time에는 정지시간이 포함된다.
- [확인] 이 실행순서 일탈은 보고에서 숨기지 않는다. 이후 모든 arm의 actual exposure/step/target/BN/checkpoint 무결성을 재검사한다. 검사 실패가 발견되면 해당 결과를 유효 실험으로 승격하지 않는다.
