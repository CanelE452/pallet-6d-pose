# 同一192장 정확도 비교와 E0 정보 계약 교정: 실행 계약

최상위 목표는 기존 전체 상태별 E0를 중심으로 최종6D 자세 공동성공률이 N3+SubPix와 두 고정 PoseFix+SubPix 경로보다 높은지 검증하는 것이다. 경량화·FPS·새 논문 작성으로 목표를 바꾸지 않는다. 원격 main fetch 결과는 `b843726ee2a78d7d29b79dafecdfc9d982b266fd`이며 지시문 작성 기준 이후 관련 diff는 없다. 지시문 전체 SHA256은 `c99493d8b3d17024f95290d4579e53ecbf18b488d1ec87ef7a477e2624477cce`다.

## 모집단과 원결과 합격 조건

PRIMARY192=원 TEST49+REMAINDER143, CAL46은 별도다. 모든 영상·실패·자세 불가용을 분모에 남긴다. 원 중심 위치≤5cm AND 같은 자세의 physical/C2 회전≤5°이며 직사각형90° 오류를 지우지 않는다. 원 R0=66, N0=108, E0=114, POSEFIX_RAW=80 성공과 원 평균을 seal로 연결한 저장 자세·참조에서 재현한다. E0/N0 교차 BOTH_SUCCESS89/E_ONLY25/N0_ONLY19/BOTH_FAILURE59가 어긋나면 이후 실행을 멈추고 계약 오류부터 복원한다. 이후 단계의 소스 읽기·준비는 병렬로 할 수 있으나 새로운192장 선택/가설 통계는 해당 gate 통과 후 수행한다.

COMMON95=CAL46+TEST49의 기존 PF_RAW66/PF-anchor71/YOLO-anchor70/N074/E086을 재현한다. 숫자만 맞추지 않고 원95 입력·원 PoseFix tensor·중간 점·최종F pose와 봉인 출력을 비교한다. 허용치와 실제 기술 차이를 기록한다.95장 규칙을 바꿔 유리한 결과를 고르지 않는다.

## PoseFix 고정 두 비교 경로

원 synthetic-only PRIOR1 seed1/last6000 checkpoint와 `seed1_REAL_DEV.npz`의 `points[selected_rows,0,1]` 첫 RAW uncapped 출력을 재사용한다. 저장319ID 중 필요한 기존238ID를 연결하고 다른 seed/245장 결과로 누락을 채우지 않는다. 새 NN forward는 필요한 캐시가 정말 없는 경우에만 계약을 명시한 뒤 허용되며, 현재 확인한 cache 범위에서는 필요 없다. 새 학습·모델·데이터는 도입하지 않는다.

PF-anchor는 uncapped PF출력→원 SubPix→PF점 기준 추가 이동만 원 RGB 대각선1% 제한, YOLO-anchor는 같은 PF+SubPix→원 YOLO점 기준 총 이동1% 제한이다. 원 window(5,5), zeroZone(-1,-1),40회/.001, 최소15px 및 원 finite/support/중심 처리와 연산 순서를 소스·원프로토콜 그대로 복원한다. 원 F의 prediction-only W/D selector→SQPnP+RefineLM→자기 물리 Q 변환을 사용한다. 본 비교에서는 이 원 F 실행이 허용되며 이전6장 시험의 no-PnP 제한을 새 과제로 옮겨 적용하지 않는다. E0의 고정후보 진단과 새 PF F 계산의 비용/범위를 구분한다.

REMAINDER native 자료의 `raw_points`를 이름만 보고 YOLO anchor로 쓰지 않는다. 원 PF 입력의 YOLO q0와 실제 RGB/K/치수/Q를 원95에서 검증한 뒤 같은 정의로143장에 적용한다. 참조/성공/과거오차를 제외한 neutral 입력만 selector에 주고 모든 예측·pose를 먼저 봉인한다. 별도 evaluator가 그 후 원 fixed reference를 읽는다. 두 경로를 모두 유지하며 `TWO_FIXED_COMPARATORS_NO_GLOBAL_WINNER`로 기록한다. 최강의 관측 성공수와의 차이는 설명하되 배치할 대표를 사후 선발하지 않는다.

## E0 원인 감사와 단일 변경 조건

원 full_BI/partial/S/O/T/U의 native owner·Q/phase, 좌표·원점, measured/fallback/model-projected 출처, 원 공분산·CAL override·whitening·단위·reliability, full 철회·grouped partial loss·RGB 중복을 코드와 실제 저장 관측으로 연결한다. CAL46의 실제 세션·상태·출처별 표본 수와 분포를 보고한다. 파생 참조와 실물 semantic owner는 구분한다.

수정 전에 `MECHANISM_AND_EDIT_PROPOSAL.md`에 코드 위치→관측 불일치→원인→적용 장면→한 변경→반례/부작용→판정 기준을 고정한다. 명확한 결함이 있으면 새 opt-in 한 경로만 구현하고 전체192장에 한번 적용한다. 원 E0/N0/PF·관측·가중치·논문·운영 기본값을 덮어쓰지 않는다. 다른 frame/family의 Q 혼합, full/partial 중복 또는 철회 복구, covariance 단위/소유자 전달 오류 같은 계약 결함을 우선 검증한다. 큰 참조오차만으로 결함을 확정하지 않는다.

단일 교정이 근거 없이 필요해 보인다는 이유로 숫자를 정하거나 global weight·threshold·A/B guard·재료/시간/frame ID 조건을 탐색하지 않는다. 콘 부모 전체 제외나 모델 보조 contour 반복·실패했던 replay 학습도 반복하지 않는다. 저렴한 소스/고정후보 검사가 가설을 지지하지 않으면 `NO_SUPPORTED_CHANGE`로 기록하고 성능개선을 만들지 않는다. 이는 정확도 목표의 축소가 아니라 실제 결함 여부의 실험 결과다.

## 검산·보고·게시

동일192장 E0/N0/PF_RAW/두PF+SubPix를 전량 비교한다. 원49/추가143·세션·별도CAL46, 평균±표본SD/중앙값/P90/P95/max·>50/>100cm·>45°·오류/불가용/fallback·paired회복/손상/양쪽성공/공동실패·연속차를 유지한다. bootstrap은 기존 recording cluster 단위를 확인하고 사전 고정된 재표본 규칙으로만 기술적95% CI를 산출한다. 적은 군집·반복 실사 개발·PF synthetic-only와 E0 실사CAL 사용·서로 다른 전체 시스템 경로의 비대칭을 명시한다. 통계적 유의성/동등성/독립 일반화를 혼동하지 않는다.

새 코드와 원자료·출력·참조 SHA,95점/pose parity, 동일192ID·분모, selector I/O 분리, 자기nativeQ/좌표단위, 독립 산술 재집계, 실제 wall/CPU/peakRSS와 실패 이력을 남긴다. 새 namespace만 stage하고 별도 연구 브랜치에 commit·push하며 허용된 원격 main에도 결과 보존 목적의 연구 코드/보고서만 반영한다. 사용자 dirty 원본을 stash/reset/clean하지 않고 이전 브랜치·가중치·참조·운영 기본값·논문을 보존한다. 원격 SHA와 실제 파일을 다시 읽는다. 최종 판정은 ADOPT_FOR_FURTHER_VALIDATION / REJECT / BLOCKED 중 근거에 맞게 고른다.
