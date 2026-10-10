# V18: point-only 합의 결정 의미의 최소 복원

이 실험은 알려진 DEV Clean153·Moderate92의 동일245장, 봉인된 V7 `IMAGE_ROLE_CORNERWISE_HYBRID` q와 같은 H/K/등록 치수를 유지한다. 새 감독·학습·RGB·좌표·LOO·seed·손실·재적합·기하 문턱은 0회다. parent V16 `HYBRID_FACTOR_SUBSET_POINT_ONLY`의 실제 후보와 적합을 그대로 사용한다. 모든 기존 결과와 소스는 변경하지 않는다.

같은 원 packet에서 `HYBRID_POINT_CONSENSUS_ORIGINAL`과 `HYBRID_POINT_CONSENSUS_RESTORED`를 프레임별 두 행으로 보존한다. ORIGINAL은 method 이름과 재생 출처만 추가한 완전 복사이며 원79 NEW·161 비용차 거절·1 진짜 동점·4 관측 부족의 원 표시를 유지한다. RESTORED는 같은 후보 pool의 결정만 복원한다. 두 출력490행은 새 PnP 두 번이 아니라 새 solver 호출0인 결정/표시 두 종류다.

원 첨부 §5.4는 동일 U의 inlier 수 우선, `Σmin(r²,64)` 우선과 대안·점수차 기록을 정했다. V4 `pose.py:295–314`는 물리적으로 비동등하며 동일count·objective차≤1e−8인 수치 동점만 미해결 다중해로 거절한다. S15 `solver.py:309–327`은 같은 최종 inlier ID이면 비용차와 무관하게 거절하는 추가 보수정책을 넣었다. V18은 이 추가 거절만 제거한다. 원 sorted best와 모든 대안·다중해 존재·점수차는 유지한다. 다른 fit-pool의 soft_l1 cost는 비교하지 않는다. 진짜 비동등 수치 동점, 관측 부족, 실제 inlier rank·weak·depth·표준4점·refit 실패는 그대로 반환한다. global uniqueness나 물리 ownership을 인증하지 않는다.

`solver.restore_solver`는 기존(-count, 동일전체U truncatedSSE, dimension, start) best를 검증하며 재선택하지 않는다. 원 `AMBIGUOUS_FINITE_CONSENSUS`가 오직 different-cost same-support veto이고 true non-equivalent numerical tie가 없을 때만 같은 best R,t의 출력을 복원한다. 실제 후보를 생성·fit한 것은 원 V16이다. 저장 결정 재생은 새 solver/optimizer/projection/J/SVD 호출0, 새 자세 적합0이다. NEW 표시는 승인한 자세의 산출이며 정확도 성공이 아니다. 재생 경과시간을 배포 latency로 주장하지 않는다.

복사한 solver의 원 `policy`는 frozen 후보 생성/적합의 역사적 출처로 보존한다. 현재 출력 결정은 `operative_output_decision_policy`와 `decision_restoration_policy`에 명시하며 원 any-cost veto가 현재 승인 규칙이라고 해석하지 않는다.

V16 전체 scalar는198134007검사/564FAIL이며 이 결과를 PASS로 바꾸지 않는다. V18 실행은 원 실패100개가 아니라 모든564를 보존한 logging-only TRACE가 완전하고 원 predicates/문턱/첫100/검사수·실패수가 동일하며, **모든 실패가 full-line arm에 속하고 POINT_ONLY 실패가0**임을 입증한 경우에만 허용한다. 이를 independently executed full paired audit의 point-only 범위에 대한 부분 권위로 명시한다. 이 증거가 없으면 fail closed한다. 저장 accepted flag만으로 독립 scalar PASS를 대신하지 않는다. 후단 GT/점수는 V18 출력 봉인·cleanup·독립 결정/출력 검산 이후에만 접근한다.

H∪{k}의 기존 native-versus-q LOO는 그대로다. H 초기XY와 center8은 후보 생성·fit·합의점수에서 제외한다. 승인 R,t 뒤 H는 기존 최종 projection으로 한 번 display에 대체하고 refit하지 않는다. 다른 실제 q와 metadata/center/conf/box를 보존한다. 거부하면 원 전체8점 N3/초기 자세의 기본 반환 또는 완전 실패를 구분한다.

복원된 자세의 hidden_after 표시는 변경 없는 V1 `hidden_mask`의 convex cuboid·2° margin 산술을 새 stdlib 함수로 계산한다. 이 scalar pose-to-mask 진단은 별도 호출계수로 기록하며 fit·합의·후보 순위·승인에는 쓰지 않는다. 원 V1 inference/solver 소스 바이트를 묶는다. 물리 mesh visibility나 GT로 인증하지 않는다. H 좌표는 이미 저장된 projection에서만 복사하므로 새 projection 호출은0이다.

Fresh path는 `fresh_bank_class()`의 새 subclass가 변경 없는 S15 POINT_ONLY.solve를 실제 한 번 실행한 뒤 동일 pure 결정을 적용한다. frozen 모듈 전역을 바꾸지 않는다. 복원으로 새 H display 대입이 생긴 만큼 실제 `hidden_reprojection_assignments`와 operation/local-bank counters를 갱신하며 두 번째 최적화는0이다. 실제 inference·H∪k LOO·최초 pointbank 준비·강건 생성/fit·최종 H 처리까지 포함한 별도 전체 경로 측정은 이후 자체 protocol로 잠그며 이번 saved 실행으로 대체하지 않는다.

이것은 원래 합의 점수 계약의 복원 한 가지이며, 저장 점수로 더 좋은 설정을 고르는 탐색이 아니다. 실제5377/1156 same-support 비용차 대안과 true tie1의 진단은 정확도 개선의 증명이 아니다. 후단 평가 전에는 개선 여부를 주장하지 않는다.
