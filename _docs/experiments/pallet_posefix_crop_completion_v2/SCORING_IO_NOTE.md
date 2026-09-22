# 첫 집계 시 빈 band 형식 처리

네 조건 예측을 모두 동결한 뒤 첫 집계에서 코너가0개인 보조집합 오차band의 bool 배열이 float dtype으로 추론되어 bitwise 연산이 중단됐다. 모델추론/GTmatching/좌표계 오류가 아니라 빈 집합 dtype 처리 문제다.

최초 SCORING_START와 code hash는 ATTEMPT1 이름으로 보존하고, 전이 bool dtype을 명시했다. 빈/비어있지않은 band 회귀검사2개를 추가했다. 모델·예측·분모·임계값·프로토콜은 변경하지 않았고 추가학습/추론도 없다. 동일 frozen출력만 재집계한다.
