# V10 seal 이후의 대응·conditioning 진단 집계 계약

이 supplement는 전체245의 새 endpoint audit가 완료되고, 독립 scalar checker가 PASS한 **이후** 별도 protocol을 고정한다. 지금은 코드 작성 계약이며 실제 산술·fit·모델·GT score 실행 증거가 아니다. core `AUDIT_PROTOCOL`이나 V7/V8/V9/V10 관측·pose·NEW/fallback·mask·점수는 변경하지 않는다.

`diagnostic_summary.py freeze|run`은 기본 V10 DOC의 완료 raw/seal/receipt/독립 protocol/checker/PASS, cohort와 immutable V9 full geometry bytes를 bind한다. 별도 `DIAGNOSTIC_SUMMARY_PROTOCOL.json`/`DIAGNOSTIC_SUMMARY_STARTED.json`/`DIAGNOSTIC_SUMMARY.json`을 exclusive 생성한다. `DIAGNOSTIC_FRAME_ROWS.jsonl.gz`에는 모든245 ID와 개별 진단을 남기고 오류가 나면 interrupted prefix와 incomplete receipt를 보존한다. 기존 output을 덮어쓰거나 자동 재실행하지 않는다.

새 endpoint의 SUPPORTED/CONTRADICTED/UNKNOWN, 이유와 RMS를 원행에서 복사한다. V9의 used point/unused line/H, status, best/all candidate의 observed/modelled rank·singular spectrum·condition·centroid·softcost, original alternatives의 saved gap·tie·physical equivalence, start cache를 복사한다. Jacobian·SVD·condition·gap을 새로 만들지 않는다. 실제 선의 물리적 소유권은 UNKNOWN이다.

weak flag는 기존 [V1 solver.py:208](../../../scripts/research/pallet_observation_refiner_20261009_v1/solver.py#L208)의 `s_min < s_max * 1e-4`를 그대로 **저장된 singular values에 대한 진단**으로 계산한다. 원 AST와 code SHA를 bind해 새로운 threshold로 바꾸지 못하게 한다. 이 flag는 선택·성공 gate나 관측 제거 조건이 아니다. rank6이지만 기존 기준에서 weak인 경우도 별도로 기록한다. 비어 있거나 없는 candidate는 weak=False의 성공 증거가 아니라 별도의 unavailable/NO_BEST_CANDIDATE 집합이다.

고정 집합은 ALL, 기존 numeric NEW, 기존 fallback, line 있음/없음, point<4+line 있음, checkable 있음, contradiction 있음, 모두 supported, UNKNOWN 있음, 모두 UNKNOWN, best observed weak, best modeled weak, best observed rank6+weak, best 없음이다. combined/easy/medium마다 frame 수와 **전체 ID 목록**, 원 status, endpoint 결정 수를 계산한다. 새로운 평균·분산·CI·draw·score는 만들지 않는다. 중첩 집합은 서로 배타적인 population으로 표현하지 않는다.

`--include-saved-outcomes`는 선택 사항이며 own freeze 전 같은 설정으로 고정한다. 활성화하면 immutable V9 `POSTHOC_ROWS.jsonl.gz`, 원 producer `statistics.py`, 기존 `VERIFICATION.json`을 추가 bind하고 audit seal+independent PASS 이후에만 읽는다. producer의 `compute()`와 기록 루프는 LOCAL 245행만 posthoc에 쓴다. 정확도 원행 980개는 별도 LOCAL245·point245·fixed490 세 파일이며, posthoc 980개로 오인하거나 다른 방법 원행을 이 스트림에 넣지 않는다. 저장 METRICS의 이 population 계약도 확인하고 optional visited/copied 245개와 초과 행 없음을 기록한다. 같은 ID·같은 used/line/status가 맞는 `actual_pose_vs_N3`와 `actual_pose_vs_point`의 **이미 저장된 delta와 category**만 복사한다. source GT/scorer·재fit·후보 선택에는 전달하지 않는다. 이 옵션은 새 정확도 평가가 아니다.

이 결과는 관측 모순·검증 불가·약한 conditioning이 저장된 실패와 같이 나타나는지 찾는 자료다. 공통 factor-pool을 변경한 인과 실험이 아니므로 선 효과나 새 보정 개선을 인증하지 않는다. 이후 최소 repair는 별도의 방법 계약과 검증으로 진행해야 한다. 여기서는 새로운 threshold·seed·학습·RGB·filtered LOCAL 경로를 추가하지 않는다.
