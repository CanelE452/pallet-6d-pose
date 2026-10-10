이 문서는 초기 자세 정보가 새 검증·최종 fit에 들어가지 않는 v4의 실행 전 평가 계약이다. 실행 완료나 성능 개선을 주장하는 보고서가 아니다. 이미 게시된 v3와 과거 실험·사용자 checkout·가중치는 보존한다.

평가 범위는 기존 [COHORT](../pallet_kp_corrected_supervision_20261010_v1/COHORT.json)의 쉬움153장·중간92장, 총245장·13세션이다. severe74장은 새 추론·채점·시간 측정에서 제외한다. 중간 라벨에는 부분 가림이 있을 수 있다. 원래 영상 ID·session·순서·K·실제 치수·영상 SHA와 기존 참조 연결을 유지한다. 새 라벨이나 성능 기준으로 영상·코너를 다시 선택하지 않는다.

|방법|관측과 자세 계산|목적|
|---|---|---|
|N3_INDEPENDENT_CORNERWISE_ROLE|후보 코너 k와 초기 H를 뺀 다른 N3 점으로 pose prior 없는 검증; 실제 채택 후보+남은 N3 점으로 pose prior 없는 최종 강건 PnP|주경로|
|N3_INDEPENDENT_ROBUST_H|원래 N3 관측; 초기 H 제외; pose prior 없는 강건 PnP|경계 교체 없는 대조|
|N3_INDEPENDENT_ROBUST_NO_MASK|원래 N3 관측; 마스크 없음; pose prior 없는 같은 강건 PnP|가림 판단 없이 강건 PnP가 충분한지 비교|
|N3_CORNERWISE_ROLE|게시된 v3 선택과 초기 N3 차원·재투영 prior, 원래 v2 PoseBank 그대로|구조 변경의 대조|

네 방법980행, 같은 새 capture의 고정 BASE/N3_SUBPIX490행, observation245행을 모두 GT 접근 전에 봉인한다. 네 방법에 같은 새 RGB capture를 공유할 수 있으나 solver bank는 독립 v4와 기존 v3를 섞지 않는다. 주 비교는 새 주경로와 고정 N3의 전체 운용245장이다. 추가 비교는 주경로 대 기존 v3·native H·무마스크·BASE, native H 대 무마스크, 무마스크 대 고정 N3다. 세 층·세 paired scope·세 자세 지표에 총189개 CI slot을 요청한다.

독립성의 범위

새 validation/final solver는 초기 R,t·초기 차원 선택·초기 8코너 재투영을 API 입력으로 받지 않는다. 4점 부분집합 생성·branch 점수·refit은 남은 실제 관측 U만 사용한다. 동일 좌표·K·치수·4점 ID 가설은 재사용한다. 실제 치수가 있다는 이유로 native camera-facing W/D 분기를 임의로 dim0에 고정하지 않는다. 서로 다른 두 registry 차원 가설을 유지하며 같은 U의 합의 점수로 비교한다. 수치적으로 동점인 서로 다른 자세는 명시적인 다중해로 채택하지 않는다. 로컬 Jacobian rank6은 전역 유일성 증명이 아니다.

독립적으로 알려진 native W/D identity를 사용하려면 초기 pose에서 복사하지 않은 명시적인 외부 provenance가 필요하다. 현재 평가에는 그런 witness가 없어 강제 차원 고정을 사용하지 않는다. 실제 물리 치수, native 코너 ID, physical R 변환 convention은 기존 정의를 유지한다. 성능 결과를 보고 차원 분기 규칙·임계값·예외를 바꾸지 않는다.

초기 자기 가림 H는 기존 N3 자세에서 오고, 후보의 영상 특징·role은 고정 Base 예측에서 온다. 이 단계에서 수리하는 것은 H/k 좌표의 **수치 pose prior 영향**이다. H·제안 특징·영상의 통계적 의존까지 없어진 물리적 독립 검증이라고 부르지 않는다. 검증 자세의 재투영은 후보를 검사하는 예측이며 새로운 관측 좌표로 들어가지 않는다.

관측과 출력

고정 Base/N3, 수정 감독의 기존 IMAGE_ROLE 마지막 checkpoint, 기존 source-only calibration과 선/코너 admission을 유지한다. 새 학습·seed·RGB 생성·feature 재학습·source threshold scan은 없다. 같은 고정 candidate residual8px·native basin8px·strict squared-residual improvement 규칙을 사용한다. 검증 불능·부족·동점·다중해는 native N3를 유지하고 사유를 남긴다. 마스크 오판이나 재추정 후 H 변화 자체로 프레임을 실패시키지 않는다.

최종 fit에서 H 초기 좌표를 제외한다. 새 최종 R,t가 있으면 H 출력 좌표를 해당 자세의 재투영으로 대체하고 재fit하지 않는다. 임시 heldout k는 새 자기 가림 라벨이 아니며 최종 선택 이후에 실제 관측으로 사용할 수 있다. 새 자세가 없을 때는 전체 고정 N3 좌표·자세를 기본 출력으로 반환하며 NEW_POSE와 구분한다. 관측 부족·수치 실패·다중해·기본 반환·완전 실패를 별도 기록한다. 부분 선을 교점과 중복 관측으로 추가하지 않는다.

GT 차단과 이전 v3 대조 검산

생성자·capture·후보 선택·PnP를 기존 GT canary와 새 IO canary 아래에서 실행한다. 기존 GT·visibility·axis와 함께 PREDICTIONS·POSTHOC·METRICS·저장 geometry/observation도 추론에서 읽지 않는다. 프로토콜 입력 보호용 해시와 추론 데이터 접근은 별개다. 모든245 observation,980 geometry,490 fixed geometry와 parity 기록이 완전해야 references를 열 수 있다. 중복·누락 ID, 다른 방법 집합, pre-seal의 pose/corner/mask_audit/evaluation_reference 점수 필드는 채점을 막는다.

봉인 뒤 **GT 참조를 열기 전** 새 N3_CORNERWISE_ROLE 대조와 게시된 v3 GT-free GEOMETRY_SEALED245행을 비교한다. 이전 geometry는 검산 입력만 되고 새 pose/selection에는 사용되지 않는다. 좌표·R,t·치수·projection은 atol1e-7/rtol0, 상태·H·선택 ID·fit/inlier/generator ID·선택 사유는 정확 일치를 요구한다. 허용오차 안의 0이 아닌 수치 차이도 원행으로 기록하고 bit-exact라고 하지 않는다. 범위를 넘은 차이는 V3_CONTROL_PARITY 실패 receipt를 먼저 남기고 GT 채점을 중지한다. 과거 점수·코너 참조를 이 비교에 쓰지 않는다.

채점은 기존 references/score 수학을 그대로 사용하고 새 namespace의 명시적인245/980/490 봉인 검사를 적용한다. 기존 모듈의 전역 METHOD·DOC·reference scope를 새 값으로 덮어쓰지 않는다. cv2 PnP/LM와 scipy optimizer fitting을 채점 중 금지한다. 봉인 R,t를 다시 구하지 않는다. 원래 source context의 일시 경로 연결은 종료 시 복원한다.

위치·회전·ADDsym과 코너 진단

고정 BASE/N3 availability·자세 지표·corner phase/error는 이전 같은 입력과1e-7로 비교한다. 새 방법의 사람 상태·직접 가시점·SELF·실제로 재투영한 H는 새 고정 N3의 native permutation으로 정렬한다. 그 참조 위상은 후단 감사에만 쓰고 observation/fit에는 전달하지 않는다. 기존 geometric proxy를 독립 실측 pose GT라고 표현하지 않는다.

평균·표본분산(ddof1)·SD·중앙값·P90·최댓값은 새 저장 원행에서 계산한다. 전체 운용245장, 공통 산출 집합, candidate NEW 집합, 양쪽 NEW 집합의 ID·분모를 명시한다. 기본 반환·큰 오차 영상을 전체 평균에서 제거하지 않는다. 기존10000×13 session multiplicity를 그대로 재사용하고 새 draw를 만들지 않는다. 빈 paired 집합은 NA이며0오차가 아니다. 쉬움·중간·합계를 분리한다.

후단 진단은 참조가 알려진 점의 input8px 정확/부정확과 참조 미상, 실제 fit/inlier, 직접 가시점5→10px 손상, 사람 SELF 오차, 실제 H 재투영 오차를 따로 기록한다. mask가 틀렸지만 위치·회전 모두 좋아진 영상과 mask가 일치했지만 둘 다 나빠진 영상을 분리한다. 상대 참조에서 채택 코너가 나빠졌다는 사실로 물리 경계 소유권 오류나 공유 prior의 인과 비율을 단정하지 않는다.

보호·실행량·시간

새 코드·프로토콜·원행·검산·보고서는 v4 namespace에만 쓴다. 원본·사용자 변경·기존 게시 실험·완료된 source-neck/bootstrap 감사와 frozen 실패/재개 기록은 해시로 보호한다. fresh detector/N3/head와 초기·LOO·최종 PnP/LM의 실제 호출량을 기록한다. 시간 측정은 별도 조용한 창에서 Base/N3/새 무마스크 강건/새 주경로를 각각150회, 총600회 실제 전체 경로로 실행하며 초기 자세·LOO·최종 PnP·H 재투영을 포함한다. 캐시 재생 시간이나 기존 시간 합산을 쓰지 않는다.

이 평가 계약과 코드 준비 자체의 모델·PnP·ray·훈련·실사 채점은0회다. 실행 완료와 개선 여부는 사전 고정한 한 번의245장 fresh 평가, 원행, 별도 실제 실행 receipt로 판단한다.
