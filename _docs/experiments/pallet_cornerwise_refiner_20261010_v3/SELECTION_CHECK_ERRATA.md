SELECTION_CHECKS.json의 `projection_calls_after_final_PnP=0` 표기는 투영 함수 호출을 계수한 값이 아니다. 해당 검사가 실제 OpenCV 진입 카운터로 확인한 것은 **조립 단계의 추가 PnP 호출이 0**이라는 조건이다. 검사 코드에서 출력 좌표를 독립 확인하기 위한 `project()` 계산은 추가로 한 번 실행한다. 재투영 수학까지 0회라고 해석하지 않는다.

검사 코드와 원래 receipt의 SHA를 보존한다. 이 설명은 감독·선택 정책·수치 결과를 변경하거나 검사를 다시 실행하지 않는다. 실제 CPU 검사13개가 통과했고 solvePnPGeneric1687회, solvePnPRefineLM56회, solvePnP0회였다. 실사 모델이나 자세 개선을 이 검사만으로 주장하지 않는다.
