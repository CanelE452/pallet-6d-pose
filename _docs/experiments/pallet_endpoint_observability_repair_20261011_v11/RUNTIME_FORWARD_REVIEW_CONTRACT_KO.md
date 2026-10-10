# 첫 호출 내부 준비 연산의 계수 보완

원 runtime는 메타데이터 중복으로 첫 호출 반환 전에 실패했다. 별도 metadata review는 반환을 고쳤지만 첫 warmup의 실제 detector forward 2회/N3 1회를 detector 1회/N3 1회 조건으로 검사해 중단됐다. 두 프로토콜·원행 prefix·실패 receipt와 cleanup을 그대로 보존한다. 두 실패 모두 공식 시간 통계는 없다.

새 별도 `runtime_forward_review` 출력에서 동일 20 warmup + 26장 × 5회 = 150회 경로를 측정한다. 고정 원 runtime.py SHA를 확인한 뒤 measure 함수의 detector/N3 계수 조건 **한 곳만** AST로 바꾼다. 첫 warmup(index 0)만 detector 2회/N3 1회, 나머지 149회는 detector 1회/N3 1회여야 한다. 실제 초기 모델 준비 forward는 측정 구간 안에 남고 첫 warmup 시간에 포함된다. 첫 호출을 실행해 버리거나 시간을 빼지 않는다. 초기 모델 로딩 후 카운터는 0/0이고 총합은 detector151/N3150, detector 내부 준비1, 실제 프레임 detector 호출150이다. 별도 검산기가 이를 확인한다.

메타데이터 보완은 앞 review와 동일하다. 보정 모델·관측·PnP·LOCAL solve·임계값·R,t·프레임 집합·시작점·마스크·성능 원행은 변경하지 않는다. 모든 정상 프레임의 실제 detector 호출은 1회다. 이 계수 보완은 성능 향상 실험이 아니다. 새 학습·RGB·seed·GT 사용은 0이다. 원 source와 기존 두 실행 기록은 바이트를 바꾸지 않는다.

새 프로토콜은 원 모든 입력 closure에 `forward_review:review_code`, `forward_review:review_contract`와 두 번째 실패의 7개 artifact를 추가한다. 기존 metadata review 입력 10개도 유지한다. 다른 assertion·예외·시간 수식·스케줄 변경은 0이며 실패 자동 재시도는 하지 않는다. 출력은 새 폴더에 배타적으로 생성하고 wrapper 글로벌 및 원 measure 함수 참조를 복구한다.
