# 상태별 관측 기여: 선행 근거와 이번 검증 범위

유지할 기여는 **가시 상태에 따라 영상에서 측정한 점·부분 선과 비관측 영역·모델 투영의 역할을 구분하고, 그 물리 대응을 하나의 강체 자세 계산까지 운반하는 팔레트 보정 방법**이다. 이 문서는 기여의 기술 범위를 고정한다. 이번 네 진단과 잔차 분석은 예측·후보 봉인 후 완료했다. 아래 결과는 원 e의 typed 관측 복원을 지지하지만 e보다 향상된 방법의 우월성을 입증하지 않는다. 코드 감사의 PASS는 정확도 개선의 PASS가 아니다.

## 직접 확인한 선행 원리

| 읽은 primary source | 이미 알려진 원리 | 현재 방법과의 차이·검증 경계 |
|---|---|---|
| [PoseFix, CVPR2019 저자 공개본](https://arxiv.org/pdf/1812.03595) §4·5.1–5.2, [공식 구현](https://github.com/mks0601/PoseFix_RELEASE) | 이미지와 완만한 초기 pose 표현으로 오류를 보정하고 오류합성·good 입력을 학습에 포함 | 사람 관절 오류 prior는 팔레트 W/D·선 owner와 다르다. 현재 port에는 Gaussian·좌표 loss가 이미 있다. S/O/T의 회귀 좌표를 실제 가시 B/I 관측으로 쓰는 근거는 아니다 |
| [EPOS, CVPR2020 저자 공개본](https://arxiv.org/pdf/2004.00605) §3.1–3.3, [프로젝트](https://cmp.felk.cvut.cz/epos/) | 한 영상 위치에 복수 3D 대응을 두고 강건한 가설 추정으로 모호성을 다룬다 | dense fragment 대응과 학습·mesh를 사용한다. 현재 sparse 코너·B/I 선의 운반과 동일 방법은 아니다. **대응 모호성 유지 자체**는 새 원리가 아니다 |
| [Uncertainty-Aware Camera Pose Estimation From Points and Lines, CVPR2021 저자 원문](https://alexandervakhitov.github.io/scripts/publications/files/vakhitov2021uncertain.pdf) §3.1–3.2, [프로젝트](https://alexandervakhitov.github.io/uncertain-pnp/) | 점·선의 2D/3D 불확실성을 residual covariance에 반영하여 혼합 pose를 구한다 | 기존 대응과 Gaussian noise 모델을 전제한다. covariance가 작다는 사실은 해당 물리 변의 의미가 맞음을 인증하지 않는다. 현재 원 residual·partial 관측 보존과 전제 확인의 근거이며 새 solver를 도입하지 않는다 |
| [Object Pose Estimation With Statistical Guarantees, CVPR2023 저자 원문](https://arxiv.org/html/2303.12246v1) §3–5, [저자 프로젝트](https://research.nvidia.com/labs/aspire/publication/yang.pavone.cvpr2023/) | 별도 calibration의 conformal keypoint 집합을 pose uncertainty 집합으로 전파하고 보장된 범위를 계산한다 | 현재 최대오차 타원·Gaussian은 conformal coverage가 아니다. 기존 영역을 pose에 전달한다는 아이디어만으로 최초성을 주장하지 않는다. 새 conformal fit·교환가능성 보장·worst-case 인증을 추가하지 않는다 |

위 논문들은 인접 원리의 근거이며 전체 관련연구를 망라한 검색이나 최초성 증명이 아니다. 원문 접근 실패와 대체 저자 공개본·직접 읽은 구간·파일 SHA는 [LITERATURE_SOURCE_AUDIT.json](LITERATURE_SOURCE_AUDIT.json)에 남겼다. [PoseFix 전달 감사](POSEFIX_TRANSFER_AUDIT.md)는 현재 port·과거 실패를 더 자세히 구분한다.

## 이번에 검사할 세부 주장

| 주장 | 필요한 증거 | 현재 판정 |
|---|---|---|
| S를 국소 영상점으로 질의하지 않고 모델 투영으로 분리한다 | 실제 query/prompts/solver 인자를 원 호출부까지 추적. display 좌표 변경이 비활성 residual에 영향을 주지 않는 검사 | 238 계보·실제 consumer와 strict parity를 연결. S를 활성 정밀점으로 만들지 않음. 전체 CNN 수용영역과의 무관성은 주장하지 않음 |
| B/I에 서로 맞는 영상 관측을 쓴다 | B 외곽·I 접합의 실제 line/intersection 출처와 fallback 구분, 가시판정의 근거·분모 | 역할 정의와 실제 보임/가림을 별도 변수로 취급. 분류 정확도·인과효과는 미확정 |
| partial을 full point로 잘못 재활성화하지 않는다 | 원 e에서 철회한 complete likelihood와 실제 활성 residual의 전량238 계보; 직렬화 동등성 | 원 e complete/partial 활성 분리와 typed 동등성 전238을 확인. source 문자열이나 필드 존재만으로 full point라 판단하지 않음 |
| 모델 재번호와 영상 corner 의미를 구분한다 | Q·물리 꼭짓점 bijection 검산과 원 owner 운반; 의미 정답 부재를 별도 flag | 모델 등록 검산이 semantic owner 정답 보장은 아니라는 구분은 유지. 영상 의미 미검증은 임의 정답 배정하지 않음 |
| 같은 관측의 재선택이 손상에 기여한다 | 동일 좌표·종류·가중치·활성 항·제약에서 SINGLE↔MULTI 비교; 시작점·후보량 기록 | typed SINGLE/MULTI는 e와 같은114/192·TEST49/49. 같은 typed에서 단일 시작의 추가 정확도 이득 없음; 가까운 basin·전역유일해 보장도 없음 |
| 좋은 e 후보를 검사·선택이 잃는다 | 전체 생성·support·feasible·선택·refinement bank 완전성, 봉인 후 같은 한 후보의 joint oracle | 원 consumer bank 전238 완전, PRIMARY joint feasible oracle146/192·최종114. 최종실패78 중 selection32/consumer proposal·observation limit46, pruning0. 역사적 consumer 밖 생성 전량은 별도 불완전 |
| 기존 영역 O/T/U가 추가 효과를 낸다 | 원 e에서 실제 활성인 영역·기존 가중치만 운반한 비교와 전체 실패 보존 | 비활성인 항은 `해당 없음: 유효 제약 또는 추가 효과 미확인`. 새 영역·중심점·계수를 만들지 않음 |
| 기존 성공과 큰 실패를 함께 개선한다 | PRIMARY192·TEST49·REMAINDER143·CAL46, 공동성공·회복/손상·mean/SD/max·양쪽 실패의 오차 기여 | typed는 e를 복원했고 성공·tail을 새로 개선하지 못함. reobs PRIMARY 회복10/11·손상15, TEST손상7. 기본값 교체·우월성 미입증 |

## 결과 해석의 한계

TYPED_MULTI가 원래 e의 관측·후보·선택을 그대로 재현한다면 E0와 같은 것이 정상이며 **동등한 복원 대조군**이다. 잘못된 후단을 제거해 E0로 돌아간 것을 e보다 개선된 방법으로 세지 않는다. REOBS↔TYPED는 관측 패키지 전체의 비교여서 재번호 하나의 순수 효과라고 부르지 않는다. 목적함수·제약을 같게 유지하지 못한 대비는 `NOT_ISOLATED`다.

실제 봉인된 TYPED_SINGLE audit는 시작정책 비교가 `NOT_ISOLATED`임을 명시한다. TYPED_MULTI는 원 e의 G10 full/legacy family 선택 뒤 G13을 재현하고 E0를 시작점으로 쓰지 않는다. TYPED_SINGLE은 이미 선택된 family를 고정하고 별도 B/I seed fit을 생략해 E0 한 초기값으로 시작한다. REOBS 두 경로와 TYPED_SINGLE의 E0 초기화 사용 여부, SINGLE의 local ambiguity 범위는 원 audit에서 파생해 수치와 별도로 기록한다. 따라서 두 typed의 결과가 같아도 이를 순수 초기값 개수의 인과효과라고 주장하지 않는다.

현재 상태는 초기 pose에 조건화된다. 코너의 기하 B/I 역할과 실제 외부 가림 여부를 혼합하지 않는다. 관측 정밀도·소유한 물리 대응·관측 종류도 각각 다르다. 한 영상선과 그 선으로 만든 교점을 둘 다 쓰는 정보 의존성을 먼저 기록하며, 이번에는 새 covariance나 loss로 바꾸지 않는다.

GT는 봉인 후 오차·oracle·실패 분류에만 사용한다. 생성·정렬·선택에 역류시키지 않는다. geometric reconstruction 참조에 대한 oracle은 물리적 참자세의 절대 상한이 아니다. C2 대칭을 인정하되 직사각형 W/D 오류를 없애는 임의 90° 동등성을 추가하지 않는다.

TEST49·추가143·CAL46은 이미 개발에 노출됐다. 이번 봉인은 정보 유입을 통제하기 위한 것으로 새 독립 test를 만들지 않는다. 작은 covariance, 작은 reprojection, 수락, ambiguity, 물리 pose 성공을 구분한다. 조사에서 같은 약자를 찾지 못한 사실을 최초성으로, 비유의성을 동등성으로 바꾸지 않는다.

## 현재 의사결정

상태별 처리의 목적은 유지한다. 새 알고리즘의 채택·기본값 교체·논문 수정은 하지 않았다. 우선 네 진단이 어디까지 같은 관측과 선택 조건을 유지했는지, 전체 실패를 어떤 단계까지 좁혔는지를 보고한다. 현재 typed 전달을 복원한 결과와 남은 관측·제안/선택 병목을 근거로 범위를 좁혔다. 학습지원은 미확인이므로 새 학습 필요성은 결론내리지 않는다. 향후 효과 주장은 봉인된 실제 결과와 반사실 비교가 지지한 부분에만 한정한다.

## 봉인 후 직접 얻은 근거

| 모집단 | 원 e | REOBS_MULTI | REOBS_SINGLE | TYPED_MULTI | TYPED_SINGLE |
|---|---:|---:|---:|---:|---:|
| CAL | 37/46 | 36/46 | 37/46 | 37/46 | 37/46 |
| TEST | 49/49 | 42/49 | 42/49 | 49/49 | 49/49 |
| REMAINDER | 65/143 | 67/143 | 68/143 | 65/143 | 65/143 |
| PRIMARY | 114/192 | 109/192 | 110/192 | 114/192 | 114/192 |

사전고정 TYPED_SINGLE−REOBS_SINGLE의 PRIMARY 회복 15장·손상 11장, 위치 평균 짝차 -0.586207cm(95% CI -4.272008, 1.821466), 회전 -2.584853°(95% CI -7.574677, -0.279055)다. 실제 원녹화 whole-cluster 10,000회/seed20260917이며 두 경우 공통 available192다. 관측 종류·활성 항·prior 등 패키지 전체의 차이여서 재번호 하나의 순수효과로 부르지 않는다.

TYPED_SINGLE−E0와 TYPED_SINGLE−TYPED_MULTI는 PRIMARY 회복0/손상0, 위치 짝차0이며 회전은 부동소수 연산 수준 차이다. 원 e로 돌아온 복원 결과를 새로운 정확도 향상으로 세지 않는다. PRIMARY 위치오차는192장 모두 정확히 같고, 회전 원시 부호 수는 개선20/악화7/동률165이지만 최대차 3.04256619899e-12°로 기존1e-9 replay 허용치 안이다. 원시 숫자는 그대로 보존하며 실질 개선으로 승격하거나 새 성공 임곗값을 만들지 않았다. 원49 보존을 회복했지만 나머지143의 실패78과 tail을 해결한 결과가 아니다.

원 e가 실제 소비한 범위의 joint oracle은 generated/support/feasible/family/refinement/selected 순서로 146/146/146/136/114/114장이다. 원 consumer 범위의 완전성은192/192이고 최종실패78은 관측·제안 제한46과 선택손실32로 분류된다. 선택손실의 처음 손실은 family10/동일 family 내부22이며 pruning손실0이다. 이는 고정 consumer·기하 재구성 참조 안의 도달성이고 전역적인 불가능성 증명이 아니다. G10/G13은 관측 패키지가 달라 별도 원패키지 사다리도 보존했다.

REFINEMENT_LOSS candidate flag는 전체124장 중 최종실패31장·최종성공93장에 존재한다. 이를 최종실패78의 서로 배타적인 추가 원인124건으로 더하지 않는다. [oracle 범위](ORACLE_LIMITS.md), [전량 상한·분모](oracle/ORACLE_SUMMARY.json), [전19경로·모든 실패](metrics/RESULTS.json), [CAL·좋은 입력 진단](diagnostics/DIAGNOSTICS.json)을 연결한다.

현재의 구체적인 결론은 관측 전달 계약을 보존해야 한다는 것이다. 선택손실32가 존재하므로 향후 개선의 검증 대상은 남지만, 사후 oracle 정답으로 선택기를 만들거나 이번 자료에서 영상별 routing·새 margin을 고르지 않았다. 독립 검산·비용·최종 게시 상태는 최종 보고서와 별도 영수증을 우선한다.

최대 위치오차는 e/typed 260.411707cm, REOBS_MULTI 177.714661cm, REOBS_SINGLE 177.714615cm다. 반면 >100cm 영상은 각각 3/7/4장이다. typed가 모든 지표에서 우월하지 않으며 최대값·빈도·평균·성공을 함께 보존한다. 선택손실32를 모두 큰 W/D 뒤집힘으로 부르지 않고 작은 임계값 실패도 별도로 다룬다.
