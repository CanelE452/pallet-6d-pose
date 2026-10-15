# V32 clean adapter 범위와 추가 계약

원 설계 V1은 보존합니다. 새 adapter는 clean 기하·B/I 입력, 새 가족과 fresh PnP seed, 최대6 native diverse+자기 coarse 시작점 및 고정 시작점 refiner 연결까지만 구현합니다. 전체 evaluator·관측 AUTH·hardguard factory·e 비교 병합 구현은 포함하지 않습니다.

독립 초기 coarse의 R/t/정회전/양의 깊이 등이 잘못되면 행을 제거하지 않습니다. `ready=False`, `accepted=False`, `pose_available=False`인 명시 abstention을 반환하며 다른 방법/e 자세를 대신 넣지 않습니다. provenance receipt가 없거나 거짓인 경우에는 입력 준비 자체를 failclosed 합니다.

새 native start selector는 전달받은 현재32 core를 직접 평가합니다. 과거 `reference_start_problem`/상속된 `_REFINER`/`V29._run`은 호출하지 않습니다. 같은 policy native/C는 같은 fresh start packet을 사용합니다. evaluator의 rotvec 왕복으로 생기는 부동소수 오차만 R 2e-12, t 2e-12×bodydiagonal 이내로 허용합니다.

root caller는 core가 새32 AUTH/잡음/선택집단으로 구성됐음을 검증하고 원 모든 hardguard·O/T/U 영역·bounded O/S 및 선택 C를 명시적으로 붙여야 합니다. adapter의 xyz/K/Q/state 일치 검사는 임의 core의 출처를 인증하지 않습니다. 초기 source receipt의 실제 파일·입력 대응도 root reader가 확인해야 합니다.

실제32 입력·영상·라벨·모델·평가 읽기는 0입니다. 기존31 및 과거 소스/모델/관측/결과는 바꾸지 않았습니다.
