# T0 T1 T2 T6 감사 중지 기록

상태: **STOP_GATE5_DERIVED_REFERENCE_ACCESS_REQUIRES_USER_CLARIFICATION**. 게시 준비 완료 또는 전체 gate 통과로 해석하면 안 된다. Root가 사용자에게 이 사실을 알리고 확인을 요청했으며, 회신 전 의존 작업과 push를 중지했다.

## 확인된 경계 초과

사용자의 gate5는 “보호 FINAL의 영상·정답 내용을 열지 않는다(메타데이터 집계만)”이고, 확인된 사용자 예외는 gate3의 고정 T5 사례 선정뿐이다.

T2의 `code/t2_bootstrap.py:load_panels`는 이미 계산된 `PLASTIC184_SCORES.json`을 읽고 `code/t1_cached_bias.py:frame_signature`로 `reference.xy`를 비교했다. 이 필드는 `models/<model>/frames/<frame_index>/all_annotated_reference/<point_index>/xy`이다. PLASTIC184의 기존 DEV128 외 FINAL-role56장도 포함됐다. 알려진 패널 분모는 전체1413점과 DEV985점이다. 중지 후 원본을 다시 열거나 좌표 수를 새로 집계하지 않았다.

실제 bootstrap 비율은 기존 `error_px`에서 정답·관측·전체 기준점 수를 집계해 계산했다. 좌표는 입력 파싱과 모델 간 기준점 정체성 확인 과정에서 소비했다. 따라서 “FINAL 정답 내용을 전혀 열지 않았다”라고 보고하면 부정확하다.

**보호 원본 RGB 열람0, 보호 원본 annotation 본문 열람0**과 **파생 SCORES의 FINAL-role56 reference.xy 소비**는 서로 다른 사실이다. 독립 감사 역시 엄격한 중지 판정 전에 같은 파생 점수 payload를 재열람해 기존 CI를 검증했다. 이를 숨기지 않는다.

`FIXED_PLAN.json`의 already-derived cache 허용 문구는 assistant가 작성한 해석이며 사용자 허가를 대신하지 않는다. 감사자가 처음 이를 허용 근거로 취급했던 판단을 철회했다.

## 앞서 완료된 수치 검증의 의미

T1의21행 및 관측14252점은 별도 계산과 일치했고, T2의150개 frame CI도 기존 고정 표본에서 재현됐다. T0 namespace와 T6의72행도 기존 cache와 일치했다. 이들은 중지 이전의 산술·출처 검증 사실이며, gate5 충족이나 게시 권한을 보장하지 않는다. 자세한 기존 수치 검증은 JSON의 checks에 보존했다.

T3/T4/T5 확장 감사는 코드·receipt 확인 도중 중단됐고 최종 인증을 내리지 않았다. 사용자 확인 뒤 root가 정한 범위에서만 후속 작업을 재개해야 한다. 이번 갱신에서는 이 감사 JSON/MD 두 파일 외 데이터·코드·통계를 수정하지 않았다.
