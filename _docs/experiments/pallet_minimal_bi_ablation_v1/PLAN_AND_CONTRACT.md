# 최소 B/I 직접 연결 실험 — 사전 계약

사용자가 지정한 지시문 전체를 읽고 실행한다. 지시문 SHA256은 `182856627cdd61ac58a31d40b7185de6ad86f3adad15eaa00fb612d8add2bca9`이며 기준 커밋은 `f1a30071382f6304fe13d6c546b8f8033c6b9f49`다. 이 문서는 실행 전 계약이다. 최종 결과는 RESULTS_KO.md에 별도로 남긴다.

## 질문과 고정 네 경로

| 경로 | 자기 초기 입력 | 후단 |
|---|---|---|
| R0 | 기존 동일 YOLO seed42 RAW native 좌표 | 기존 공통 F |
| R1 | R0 좌표와 R0의 자기 F 자세 | 최소 상태별 B/I + 공통 B/I-only PnP |
| N0 | 위와 동일 YOLO 실행 → 기존 N3 seed1 → 기존 SubPix+상한 | 기존 공통 F |
| N1 | N0 좌표와 N0의 자기 F 자세 | R1과 동일한 최소 상태별 B/I + 공통 B/I-only PnP |

N3/SubPix 묶음의 필요성을 비교한다. 두 성분의 개별 기여를 분리하는 실험은 아니다. 초기 검출기와 B/I 상태 모델의 고정 ImageNet ResNet 특징 추출기를 구분한다. 후자는 제거하지 않는다.

## 입력, 상태와 보호 범위

CAL46, TEST49, REMAINDER143의 고정 ID를 사용한다. 주평가는 TEST49+REMAINDER143=192장이고 전체 원영상은238장, 네 경로 결과는952행이다. severe는0장이다. ID·동일 영상 중복·카메라·전체영상 native9 좌표·선택 검출 candidate·원 XYZ·F의 CF치수/물리 등록 Q·가중치 SHA를 입력 계약에서 검사한다. 오래된 `dimensions_whd_m`라는 키 이름으로 실제 XYZ 순서를 다시 바꾸지 않는다.

RAW/N3 좌표 캐시는238장 모두 있다. RAW ownF는102개 캐시와136개 동일 F 계산이며, 기존102개의 재계산 결과가 모두 정확히 일치했다. N0 ownF는238개 캐시다. 새 학습·optimizer update·학습 seed·검출기 forward·데이터 생성·정답 변경은0이다. 입력 추출은 predictor 필드 whitelist를 사용하고 수치 참조를 읽지 않는다.

S는 자기 초기 F의 등록 모델로 정해지는 self-occlusion이다. S의 예측 좌표는 마스크 프롬프트·상태 국소 RGB·보정/PnP 2D 대응에 사용하지 않는다. 초기 F 자체의 예측 채널 영향까지 없앴다고 주장하지 않는다. B/I의 기하 역할과 실제 보임 판정은 구분한다. O/T/U는 2D 대응에서 제외하고, B/I도 실제 새 교점과 원좌표 fallback을 별도 기록한다. RAW/N3의 상태·프롬프트·마스크·관측은 각각 독립 실행한다. 마스크 여집합은 외부 가림 정답이 아니다.

기존 과학적 정본은 `new_gain10_paired_BI_v2.observe_branch/common_BI_pose`다. 상태 모델·임계값·RGB/ResNet 특징·Tiny/MobileSAM·마스크 선택·B/I 선/교점·지지/거부·공분산·보호 상한·solver starts/config/guards/fallback은 그대로 둔다. 사용하지 않는 O/T domain 생성과 U 표시 모델/영역 및 보고서/과거 E-bank 의존만 제거한다. 상태 모델은 CAL46 및 기존 DIRECT263/EXTERNAL7 실사 감독 이력이 있으며 학습 없는 방법이라고 부르지 않는다. RAW 출처 교정 범위는 이번 실행 중 재학습하지 않는다.

사용자 원작업 디렉터리의6개 dirty 파일, 기존 실패/실험·사용자 수정·논문을 보존한다. 별도 sparse 연구 worktree/브랜치에서 새 파일만 게시한다. main·운영 기본 e는 변경하지 않는다.

## 동등성 게이트와 한 구성 고정

기존10장×두 초기화의 native 입력/F/XYZ/Q/카메라가 정확히 일치해야 한다. 각자 새 실제 영상 관측을 실행해 상태·ID순서·마스크 배열·프롬프트·선/지지/거부·관측/공분산·solver inputs/config·최종 분기/수락/fallback을 대조한다. 마스크 배열은 bitwise 일치, 이산값/가설은 정확한 일치를 요구한다.

새 최소 경로 실수 비교 전에 원본 solver를 같은 입력으로 두 번 실행했다. R_cf/R_physical/중심의 반복 차이 및 기존 저장 결과 차이가 모두0이었다. 이에 pose atol=1e−9, rtol=0을 먼저 고정했다. 기존 관측 audit 부동소수 scalar는 atol=1e−8, rtol=1e−8이며 결과를 보고 완화하지 않는다. 사용하지 않는 `ot_domains`, `u_region_display`만 관측 비교에서 제외하고 모델 투영점은 새 별도 필드에 둔다. 모든 parity 시도와 기술적 실패를 보존한다.

CAL 파일럿6장은 세션 이름 순환과 각 세션 내 SHA256(ID) 순서로 고르고 정답/성공을 사용하지 않는다. 파일럿은 구현·실제 자원 확인이며 임계값 탐색이 아니다. PROTOCOL.json에 단 하나의 구성·소스/모델/입력 해시와 정확한6개 ID를 봉인하고 전량 실행한다. 입력에 schema/환경 오류가 있으면 기술적 미완료를 명시하고 수리한다. 이를 알고리즘 fallback으로 바꾸지 않는다.

## 지표와 통계

주 성공은 동일 등록 모델 중심 위치≤5cm AND 같은 물리 등록/C2 대칭 회전≤5°다. 과거 strict <5/<5와 정확 경계 건수는 별도다. 모든 실패/no-pose를 전체 성공 분모에 보존한다. 큰 유한 오차를 자르지 않는다. 연속 오차는 가용 n, 표본 SD(ddof=1), 최대를 내고 공통 가용 영상 통계도 낸다. accepted/fallback/pose available/정답 성공은 별개이며 초기 기준선의 없는 acceptance는 NA다.

네 대비는 R1−R0, N1−N0, R1−N1, R1−N0다. 모든 대비에서 회복·손상·양쪽 성공·양쪽 실패의 수와 전체 ID를 저장한다. 원 녹화 군집으로 짝지은10,000회 bootstrap, 기존 통계 seed20260917을 재사용한다. PRIMARY는 원녹화6개(프레임192개)이며 이름이 나뉜 같은 녹화는 묶는다. 추정대상은 전체 영상 가중 성공률 차이다. draw를 이 모집단에서 새로 만들고 모든 경로의 짝을 보존한다. 기존 성공률 검정 정의가 검증되지 않아 p 및 Holm p는 NA다. CI에서 p를 역산하거나 동등성/비열등성을 선언하지 않는다. 허용 손실폭을 발명하지 않는다. 반복 개발자료와 소수 군집의 제한을 보고한다.

## 고정 진단, 비용과 상한

상태/역할·ownF 치수·마스크/프롬프트·교점/원좌표·이동·선과 모델변 지지·solver 시작/수락/fallback/최종 가설을 저장한다. 기존 사람 상태는 예측 봉인 뒤 진단에만 사용하며 없으면 NA다. 새 주석/oracle을 만들지 않는다.

PRIMARY192에서 `R1_PRECORRECTION_XY_FIXED_OBSERVATION`, `N1_PRECORRECTION_XY_FIXED_OBSERVATION` 두 진단을 고정한다. 자기 보정 후 B/I ID·상태·지지·공분산·solver를 유지하고 xy만 자기 보정 전 좌표로 바꾼다. 최대384회 pose-only solve이며 새 영상/마스크/상태 계산은0이다. 조건부 진단이며 자연스러운 무보정 기준선·인과 증명·주검정 후보가 아니다.

비용은 파일럿과 같은6장, batch1, 각 경로2회 예열, 영상당5회, 균형 경로 순서를 기존 seed20260917로 고정한다. 메모리 RGB+자기 좌표/자세→상태/마스크/B/I/최종pose의 보정 블록을 실측한다. CUDA를 동기화하고 precision/thread/peak allocated/reserved, 실제 모델/보정/pose 호출을 기록한다. RGB부터 YOLO·N3/SubPix·F까지 모두 실제 연결해 실행하지 않으면 전체 시간은NA이며 캐시 시간 합산/Jetson FPS 주장은 하지 않는다. 시간 실행의 최종 pose·가설·판정을 정확도 실행과 대조한다.

GPU 관련 프로세스는 동시1개, CPU pose worker≤3 및 BLAS/OpenMP1이다. GPU 프로세스 누적 경과60분, CPU solver/analysis 단계 누적120분 상한에 실패·재시도·계측도 포함한다. worker 합산 CPU시간과 코드/복원/검산을 포함한 총 경과를 따로 기록한다. 자료/권한/가중치/방법 정의의 실질적 변경만 사용자에게 보고하고 중단한다. 낮은 성능은 중단/행 삭제 이유가 아니다.

## 게시와 판단

952행·예측/관측 seal·모든 전환·통계·진단·계측·독립 재집계·실패 이력을 보존한다. 갤러리는 사전 부류(R1 회복, R1 손상, N1만 회복, 양쪽 실패)마다 고정 ID 첫 사례를 고르고 없는 부류도 명시한다. 실제 관측과 모델 투영은 다른 기호를 사용한다. 원영상/가중치/원주석은 기존 공개 권한 manifest를 확인하고 비공개 원자료는 push하지 않는다.

성능/오차/손상/fallback/실제 의존성과 비용을 함께 보고 고정 레이블 중 하나로 권고한다. `SIMPLER_CANDIDATE_SUPPORTED`, `UPSTREAM_STILL_USEFUL`, `TRADEOFF_OR_UNRESOLVED`, `NO_CLEAR_ADDED_VALUE`, `BLOCKED_OR_INCOMPLETE`의 의미는 사용자 지시문을 따른다. 운영 채택이나 논문 수정은 실행하지 않는다. 결과가 나쁘더라도 연구 브랜치에 동일하게 commit/push하고 SHA·전체192 표·유지/제거 권고·보고서/갤러리를 알려준다.
