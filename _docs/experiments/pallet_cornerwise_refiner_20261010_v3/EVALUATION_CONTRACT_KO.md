이 문서는 새 코너별 관측 선택 경로의 평가 계약을 실행 전에 정리한 것이다. 실제 성공이나 개선 결과를 보고하는 문서가 아니다. 기존 데이터·코드·가중치·결과·사용자 checkout은 수정하지 않는다.

평가 모집단은 기존 [COHORT](../pallet_kp_corrected_supervision_20261010_v1/COHORT.json)의 쉬움153장과 중간92장, 총245장이다. 어려움74장은 새 추론·채점·시간 측정에서 제외한다. 중간 라벨에 부분 가림이 있을 수 있으므로 가림 없는 모집단이라고 표현하지 않는다. COHORT의 ID 순서, 영상 SHA, session, K, 실제 치수와 원래319장의 입력 메타데이터 연결을 유지한다. 새로운 라벨이나 성능 기준으로 영상·코너를 선별하지 않는다.

고정 비교 방법은 다음 네 가지다.

| 방법 | 관측 좌표와 마스크 | 비교 목적 |
| --- | --- | --- |
| N3_CORNERWISE_ROLE | 새 코너별 선택 규칙, 고정 초기 N3 자기 가림 제외 | 주경로 |
| N3_VALIDATED_ROLE | 기존 v2 경계 교체 규칙, 같은 초기 N3 자기 가림 제외 | 선택 규칙의 효과 분리 |
| N3_BASIN_ROBUST | 초기 N3 관측, 자기 가림 제외, 같은 자세 정책 | 경계 교체 없는 대조 |
| N3_BASIN_NO_MASK_ROBUST | 초기 N3 관측, 마스크 없음, 같은 자세 정책 | 가림 판단 없는 강건 PnP 대조 |

네 방법 모두245장에 대해 새 전체 경로로 실행한다. 예상 방법 원행은980행이며, 새 고정 BASE·N3_SUBPIX 대조는490행이다. 주 비교는 N3_CORNERWISE_ROLE 대 고정 N3_SUBPIX의 전체 운용245장이다. 같은 자세 정책의 두 N3 대조와 기존 v2 교체 대조도 별도로 비교한다. 일부 가림 오판이나 재추정 후 마스크 변화 자체를 자세 실패로 취급하지 않는다.

입력과 구현 연결

고정 Base, N3, 수정 감독으로 이미 학습한 마지막 IMAGE_ROLE checkpoint와 기존 v2 CALIBRATION을 그대로 사용한다. 새 학습·seed·RGB 생성·특징 재학습·source 임계값 조정은 없다. 새 프로토콜은 현재 코드, checkpoint·학습 완료 receipt, calibration, COHORT, registry, 원래 입력 메타데이터, 초기 자세 대조의 SHA·bytes를 고정한다. 이 연결 확인을 GT를 읽는 허가로 취급하지 않는다.

원래 v2 common과 evaluate는 방법5개·1225행을 고정해서 검증하므로 새980행에 원래 evaluate.run을 그대로 호출하지 않는다. 새 namespace의 명시적인 드라이버가4개 방법의 ID·분모·봉인을 검증하고, 기존 score와 references 함수의 채점 수학만 재사용한다. 기존 모듈의 METHOD·DOC·프로토콜을 새 실험 값으로 덮어쓰지 않는다. 원래 deployment context가 checkpoint 위치와 source module namespace를 잠시 연결하는 기존 처리는 종료 시 복원한다.

GT 차단과 봉인

1. 모델 생성·새 capture·코너 선택·자세 계산 중 기존 GT 차단과 새 넓은 IO canary를 활성화한다. TARGETS, visibility, resolved pose, axis reference와 함께 PREDICTIONS, POSTHOC, METRICS 등 채점 결과도 추론에서 읽지 않는다. 프로토콜의 보호 해시 검사는 추론 입력 선택과 분리한다. 생성자도 같은 차단 범위에서 실행한다.
2. 한 번의 실제 RGB capture에서 나온 Base·N3 좌표와 관측 후보를 방법별로 공유할 수 있다. 후보 선택 index·box·confidence·center·Base/N3 좌표를 원래 고정 대조와 허용오차1e-7, rtol0으로 비교한다. 초기 자세는 새로 계산해 봉인하며, 완전한 봉인 뒤 고정 대조의 자세·코너 채점 parity를 확인한다. 이전 자세 캐시를 추론에서 읽지 않으며 이전 경계 좌표나 자세 캐시로 새 경로를 대신하지 않는다.
3. 새 코너별 선택은 해당 코너를 뺀 다른 유효 N3 관측으로 실제 LOO 자세를 구하고, 고정된 후보 대 native 잔차 규칙을 적용한다. 자기 코너를 그 검증 fit에 넣지 않는다. 계산 불능·관측 부족·수치 실패·다중해는 관측 선택 근거에서 구분하며, GT나 실사 오차로 선택·예외 정책을 바꾸지 않는다. 상세 규칙과 solver 호출은 별도 선택 프로토콜·원행의 책임이다.
4. 최종 fit에서 초기 자기 가림 좌표는 제외한다. 최종 자세가 실제로 구해지면 해당 숨은 코너 출력은 재투영으로 바꾸고, 재투영점은 독립 관측으로 다시 fit하지 않는다. 부분 선을 교점과 중복 관측으로 추가하지 않는다.
5. GT를 열기 전에245개 observation, 네 방법별245개 geometry, 두 고정 대조별245개 geometry, parity 기록을 모두 gzip 원행으로 저장하고 각 SHA를 봉인한다. 중복·누락 ID, 잘못된 방법 수, 이미 포함된 pose/corner/mask_audit/evaluation_reference 등 채점 필드가 있으면 채점하지 않는다. 중단·실패 기록은 보존하며 자동 재실행하거나 행을 빼지 않는다.

채점과 참조 위상

완전한 봉인과 현재 입력 SHA를 확인한 뒤에만 기존 references와 score를 호출한다. references는 원래 axis의 session__stem ID를 기존 공개 image path와 session으로 정확하게 연결한다.319장 authority 메타데이터를 검증할 수 있으나 실제 참조 frame·채점 모집단은 고정245장만 허용한다. resolved pose·TARGETS·visibility는 기존 자료를 사용하며 새로운 참조나 라벨을 만들지 않는다.

기존 score의 위치·회전·ADDsym 계산을 변경하지 않는다. 원행 ADDsym_m과 보고서 cm 변환을 구분한다. 고정 BASE/N3 채점은 이전 같은 영상 대조와 availability, 자세 지표, corner branch와 corner error를1e-7로 비교해 참조 경로·위상 차이가 없는지 확인한다. 새 방법의 코너·사람 상태·손상 진단 참조는 새 고정 N3_SUBPIX의 native permutation을 기준으로 삼는다. 이 GT 위상은 후향 진단에만 쓰며 관측 선택·fit 입력에 전달하지 않는다.

scoring 단계에서는 cv2 PnP, LM, scipy least_squares와 기존 pose.refine 호출을 금지한다. 봉인된 R,t를 다시 구하지 않는다. proxy 참조를 독립 물리 실측 GT라고 표현하지 않는다. 코드 실행 성공과 실제 위치·회전 개선을 구분한다.

원행과 통계

새 자세 산출, 고정 N3 fallback, 완전 실패를 서로 다른 상태로 유지한다. 모든245장 운용 결과, 두 방법 공통 산출 집합, 후보 새 자세 집합의 분모·ID를 함께 보고한다. 평균·표본분산(ddof1)·SD·중앙값·P90·최댓값은 저장 원행에서 계산한다. 기존10,000회×13session bootstrap multiplicity를 그대로 재사용하며 새 draw나 설정 선택은 하지 않는다. 쉬움·중간·합계의 결과를 분리한다. 직접 가시점 손상, 실제 제외한 숨은 코너 재투영 오차, mask 오판과 최종 자세의 좋고 나쁨을 구분한다.

기존 v2 통계 구현의 방법 목록과 contrast는 그대로 새4방법에 맞지 않으므로 새 통계 드라이버가 방법과 분모를 명시한다. 기존 mean/variance/paired 계산 수학을 재사용해도 입력은 새 봉인·채점 원행이며, 이전 METRICS의 수치를 새 결과로 옮기지 않는다.

보호와 실행량

새 protocol·코드·원행·보고서는 이 namespace에만 쓴다. 기존 tracked/sparse16,584항목과 원본 사용자 checkout 상태를 보존한다. 아직 미게시인 bootstrap/source-neck 두 완료 namespace는 git ls-files 보호에서 빠질 수 있으므로 전체 파일의 별도 SHA·bytes 목록을 실행 전후 보호에 포함한다. 기존 frozen protocol과 실패·재개 receipt는 바꾸지 않는다.

실제 LOO·초기 자세·최종 PnP·LM 호출과 detector/N3/head 노출을 계수한다. 시간 측정은 별도 조용한 자원 창에서 BASE, N3_SUBPIX, N3_BASIN_NO_MASK_ROBUST, 새 주경로를 각150번, 총600번 새 전체 경로로 실행한다. 새 주경로의 LOO와 최종 강건 PnP까지 포함한다. 캐시 좌표 재생이나 기존 구간의 단순 합산을 처리시간으로 제시하지 않는다. 해당 runtime 실행이 완료되기 전에는 예정량과 실제 실행량을 구분한다.

이 계약 검토 자체에서는 모델·head·PnP·ray·훈련·실사 채점을 실행하지 않았다. 새 선택 방법이 개선됐는지는 사전 고정한 한 번의245장 평가와 그 원행으로만 판단한다.
