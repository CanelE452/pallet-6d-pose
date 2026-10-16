# 사전 고정 목적과 실행 계약

기준 main은 `96360c61418c435602875f5f34ca8ea4c694fbdc`다. 원래 작업공간의 사용자 변경과 기존 실험을 보존하고 별도 연구 worktree와 브랜치에서 실행한다. 이 문서는 새로운 결과를 계산하기 전에 작성했다. 결과와 계산 불가 범위는 최종 보고서에 별도로 기록한다.

| 우선순위 | 확인하는 것 | 다음 판단 |
|---|---|---|
| 관측 전달 | 238장의 실제 소비된 점·부분 선·영역과 owner | 후단 형식 변환·예측 재주입 수정의 필요성 |
| 같은 관측의 탐색 정책 | MULTI와 E0 한 시작의 SINGLE | 재선택 손상과 관측 패키지 차이를 구분 |
| 원래 e 후보 | 생성·검사·family·refinement·선택의 저장 범위와 상한 | 관측/후보 생성과 scoring 중 남은 문제 |
| 전체 비교 | PRIMARY192, TEST49, 추가143, 별도 CAL46 | 회복과 손상을 함께 보고 기여의 지지 범위를 결정 |

경쟁 설명은 다른 pose basin 선택, 잘못된 의미 대응, 상태·마스크·fallback 오차, 물리 변과 선의 잘못된 대응, 관측 정보 중복, 등록/참조 공백, 큰 실패의 평균 영향이다. 코드 경로의 가능성, 사례 연관, 동일 조건 개입의 증거를 구분한다. B/I-only의 실패를 전체 상태별 방법의 실패로 취급하지 않는다.

진단은 REOBS_MULTI, REOBS_SINGLE, TYPED_MULTI, TYPED_SINGLE 네 개만 실행한다. REOBS는 현재 E 이후 새 관측, TYPED는 원래 e 이전의 실제 활성 관측을 운반한다. 두 패키지 차이는 재번호 하나의 효과가 아니다. 각 쌍은 같은 잔차·가중치·제약을 유지하고 SINGLE은 E0에서 한 SLSQP와 선택된 native/physical assignment를 사용한다. 원 coarse prior는 보존하며 E0로 prior를 교체하지 않는다. 임의 이동/회전 상한은 넣지 않는다. 원 outer family 정책과 inner objective가 다르면 `NOT_ISOLATED`로 보고한다. SINGLE은 전역 유일성이나 가까운 basin을 보장하지 않는다.

실제 원 e 소비자는 typed 및 legacy partial의 native ID를 모두 full point 손실에서 제외한다. 원 요약의 라벨과 다른 경우 소비자 구성을 우선한다. CAL/TEST의 원 관측은 evaluation_v3 이후 추가 복구를 수행한 shared_edge_partial_b_eval_v4_7 저널에 연결한다. 기존 e_trace와 원 기록은 수정하지 않는다. 모든 238장의 실제 T domain은 0이고 union branch 수는 1이다. 따라서 이번 데이터에서 T의 추가 효과는 확인할 수 없다.

새 학습·seed·자료·GT 변경·임계값/손실/guard 탐색은 0이다. S 모델 투영과 O/T/U 영역 중심을 정밀 점으로 만들지 않는다. inherited predictor fallback은 실제 영상 교점과 구분한다. 유일 모델 vertex 등록이 영상 의미 대응을 검증하지는 않는다.

CAL 세션 정렬 후 세션 안 SHA256(ID) 순서로 round robin 6장을 선택하며 참조/성공을 보지 않는다. 이 ID와 모든 입력·소스 해시는 REPLAY_PROTOCOL 및 SOURCE_AND_INPUT_MANIFEST에 봉인한다. 허용 오차는 기존 1e-9, rtol 0이며 결과를 보고 늘리지 않는다. 동일 파일럿 결과는 전체 실행에 재사용한다. 최대 새 top-level 슬롯은 4×238=952, 원 e 로깅 replay는 1×238이며 TYPED_MULTI와 공유한다. 원 후보 예산을 줄이지 않는다.

GT/성공/오차는 관측·추론·후보 정렬에서 읽지 않는다. 예측과 후보의 해시 봉인 이후 별도 프로세스에서 평가한다. 성공은 위치≤5cm AND physical C2 회전≤5°이며 90° 대칭은 추가하지 않는다. unavailable은 전체 이진 분모의 실패, 연속 오차는 NA이며 available N과 공통 available paired N을 따로 쓴다. 평균·표본 SD·분산·최대·>50/>100cm·4분할 전이를 보존한다. 군집 bootstrap은 실제 recording cluster ID로 10,000회, 기존 seed20260917을 사용한다. 세 주대비는 TS−E0, TS−TM, TS−RS이며 결과를 보고 교체하지 않는다.

GPU 프로세스 경과시간 누적 3600초, CPU solver/분석 단계 경과시간 누적 7200초, 무거운 CPU worker 최대 3개다. worker CPU 시간과 작업 총 wall time도 별도 보고한다. 초과가 예상되면 후보/영상 예산을 줄이지 않고 미완료 경로와 필요한 추가 시간을 보고하며 읽기 전용 감사와 기존 결과 분석을 계속한다. 운영 기본값과 논문은 자동으로 바꾸지 않는다.

PoseFix RAW는 같은 238 고정 결과를 재집계한다. 기존 +SubPix의 두 cap 사이 단일 전체192 규칙이 확정되지 않았으므로 `POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED`로 분리한다. 새 PoseFix 학습은 하지 않는다. 선행 원문·공식 구현·기존 실패는 별도 감사 문서로 연결한다.
