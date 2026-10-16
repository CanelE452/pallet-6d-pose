# 진단 경로의 격리 범위와 기록 제한

이 문서는 전체 정확도 결과를 열기 전 수행한 구현 검토 결과다. 원 solver를 바꾸기 위한 새 정책이나 임계값을 추가하지 않았다.

- REOBS 쌍의 측정·공분산·soft loss·가시 기하/화면/측정 예산 제약은 동일하다. SINGLE은 E0 한 시작의 SLSQP만 실행하고 별도 least-squares seed fit과 전역 후보 생성을 생략한다. 일반 원 코드에는 corrected-seed fallback이 있으나 기존 MULTI238의 거부77장은 모두 자기 E0로 복귀했다. 따라서 이번 원 MULTI 자료에서 그 fallback 분기로 달라진 출력은0이다. 이 관측을 모든 자료의 정책 동등성으로 확대하지 않는다.
- TYPED MULTI는 원 G10 family 선택 후 G13의 같은 family 최종 결과를 사용한다. SINGLE은 이미 선택된 family 안에서 원 backend의 목적함수·제약을 보존한다. family 선택과 내부 목적함수가 다른 이 외부 정책까지 시작점 하나의 순수 효과라고 주장하지 않는다. `NOT_ISOLATED`의 차이 목록을 유지한다.
- TYPED에서 outline/점 수 admission에 걸리면 원 함수의 delegation을 보존한다. 최적화를 하지 않은 행을 ‘국소 보정 성공’으로 세지 않는다. SINGLE의 거부 시 실제 출력이 E0인지 별도 수치 검사하고 다르면 해당 행의 엄격한 자기 초기값 복귀 해석을 보류한다. 임의 새 fallback을 발명하지 않는다.
- SINGLE의 ambiguity는 국소 rank 및 상속된 family 불확실성의 제한된 기록이다. 다른 시작을 검색하지 않아 대안이 없다는 뜻이 아니다. MULTI의 대안 검출 수와 동일 검출 범위로 비교하지 않는다. 어떤 경로도 전역 유일성을 증명하지 않는다.
- 원 `optimizer_calls`는 후보 결과 수에 근거한 값이며 REOBS의 seed least-squares를 포함하지 않는다. 비용 보고서는 seed 수·SLSQP 수·endpoint 수를 분리하고 저장되지 않은 실제 함수 평가 횟수는 unknown으로 남긴다. 저장된 과거 bank의 계산은 이번 새 호출로 세지 않는다.
- 원 저널은 탈락/중복 후보 대부분에 R/t가 없었다. 이번 계측은 원 outline consider 및 실제 partial-fit endpoint를 보완한다. 과거 WD/P1의 모든 내부 생성점이나 optimizer 내부 iterate를 전부 재생성한 것은 아니다. 범위 밖의 전체 generated oracle는 `CANDIDATE_COVERAGE_UNKNOWN`이며 후보 부재로 단정하지 않는다.
- after_support는 원 guard predicate를 저장 후보에 다시 적용한 사후 분해다. 원 코드가 그 순서대로 별도 filter를 실행했다는 뜻이 아니다. 새로운 guard나 threshold를 만들지 않는다.
- 자원 장부의 정확한 계측 범위와 최초 실패 추출의 미계측 CPU를 구분한다. 총 wall에는 읽기·코드 작성·입력 구성도 포함한다. 전체 작업 wall을 solver 시간으로 부르지 않는다.
- 모든 결과는 반복 개발 자료의 기존 기하 재구성 참조 기준이다. 의미 corner/가시성의 독립 정답과 새로운 독립 test는 없다. C2 동등성과 잘못된 W/D를 구분하고 90° 대칭을 추가하지 않는다.
