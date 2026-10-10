# 코너 관측·LOO·자기 가림·PnP의 독립 방법 감사

가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**이 문서는 그 질문의 새 성능 답을 계산하지 않았다.** 두 첨부 문서 전체와 V2–V7의 관련 실제 코드·기존 보고서를 읽어, 구현한 블록과 이미 검증한 비교, 현재 관측에 남은 원인 분리 공백을 구분했다. V7의 현재 성능 판정과 실제 실행 상태는 [RESULT_KO.md](RESULT_KO.md), [BUILD_LEDGER.json](BUILD_LEDGER.json)의 해당 단계가 기록한다. 이 감사는 별도의 성능 실험·새 배포 정책이 아니다.

현재 확인되는 방법 문제는 **self-occ를 제외하지 않았거나 LOO를 구현하지 않은 것**이 아니다. 학습기가 선택한 경계와 그 교점이 해당 native 코너에 대한 올바른 관측인지 판단하는 단계에 한계가 남아 있다. 수치적으로 일관된 오대응은 작은 선 잔차나 PnP 합의만으로 제거되지 않을 수 있다. 이를 모든 방법이 불가능하다는 결론으로 확대하지 않는다.

## 읽은 지시와 사용자 요청의 구분

전체 읽은 첨부는 `pallet_cli_observation_refiner_robust_pnp_20261009.md`와 `pallet_branch_cli_error_audit_20261010.md`다. 첫 문서는 원래 A/B/C 실험의 명세이고, 두 번째는 당시 snapshot의 오류 인수인계·읽기 전용 감사 명세다. 첨부의 지시를 이후 사용자 요청보다 높은 승인 근거로 취급하지 않았다.

두 번째 첨부의 “수정 감독으로 재학습한 결과는 아직 없음”은 그 문서 시점의 사실이다. 이후 별도 사용자 승인으로 실제 수정 감독 학습과 245장 평가가 완료된 기록이 존재한다. 이 문서가 그 승인을 새로 발급하거나 재학습을 승인하지 않는다. 사용자의 후속 “쉬움·중간만” 지시에 따른 새 실사 범위는 Clean153+Moderate92=245이며, 기존 전체319장 결과는 보존한다. 첨부의 전체319 지시를 근거로 Severe74를 새 경로에 추가하지 않는다.

이번 독립 감사에 직접 부여된 범위는 **코드·문서 읽기와 이 새 문서 작성**이다. V7 core·이미지·가중치·원행·GT·기존 결과를 수정하지 않았고, 새 실험을 구현하거나 실행하지 않았다. 외부 사용자 시간 측정 작업이 확인된 뒤에도 얇은 정적 읽기와 이 문서 작성만 수행했다.

## 코너 특성을 정의하고 보정하는 실제 단계

| 단계 | 실제 코드 | 구현된 조건과 남은 한계 |
| --- | --- | --- |
| native 코너와 변 | [model.py:35](../../../scripts/research/pallet_observation_refiner_20261009_v1/model.py#L35) | 8개 native 코너와 12개 의미 변, 변마다 7개 query, 법선 방향 65개 후보다. 중심점8은 코너 대응 수에 포함하지 않는다. |
| 외곽·내부 역할 | [model.py:47](../../../scripts/research/pallet_observation_refiner_20261009_v1/model.py#L47) | 고정 Base 초기 자세로 투영한 cuboid의 convex hull에서 외곽/내부/불가를 만든다. 실제 목재 판재·홈·구멍 경계의 소유권 인증은 아니다. 실제 팔레트 마스크를 이 hull로 대체해 감독한 것도 아니다. |
| 대응 위치 선택 | [observations.py:134](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L134) | MAP 후보와 NONE의 score 및 source CAL 조건을 사용한다. source confidence는 실사 물리적 대응 진위를 보장하지 않는다. unsupported는 모델 CAL coverage 부족이며 물리 경계 부재 판정이 아니다. |
| 선 합의 | [observations.py:76](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L76) | 최소 3개 query, 지지 구간·잔차 합의, weighted TLS와 정해진 한 번의 재검사를 사용한다. 다른 경계 위의 잘못된 query들도 직선 합의를 만들 수 있다. |
| 선→코너 교점 | [observations.py:116](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L116) | 서로 다른 두 incident edge의 교점, 불확실성·외삽을 검사한다. 같은 선에서 만든 여러 교점의 공유 근거를 독립적인 새 코너 증거로 확대하지 않는다. |
| 최종 관측 admission | [pipeline.py:35](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L35) | 최종 선 지지 잔차와 8px corner uncertainty cap을 검사한다. sparse 공급은 없는 좌표를 NaN으로 유지한다. 작은 반경만으로 N3보다 더 정확한 좌표임을 증명하지 않는다. |
| LOO 관측 선택 | [selection.py:135](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py#L135) | H∪{k}를 제외한 나머지 N3 관측만으로 k의 검사 자세를 구한다. 후보가 그 예측과 일치하고 native 좌표보다 제곱 잔차가 낮을 때 실제 경계 교점을 채택한다. LOO 투영 좌표를 새 관측으로 채우지 않는다. |
| 현재 두 관측 공급 | [V7 pipeline.py:284](../../../scripts/research/pallet_three_head_observation_20261010_v7/pipeline.py#L284) | BOUNDARY_ONLY는 native 거리·LOO gate 없이 admitted sparse 경계만 사용한다. CORNERWISE_HYBRID는 V4의 정해진 LOO 선택을 유지한다. 세 head는 각각 별도 절제이며 ensemble이 아니다. |
| 최종 PnP | [pose.py:134](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L134) | 유한 4점 부분집합, SQPnP/IPPE의 반환해, 두 registry 치수 가설, 같은 U의 inlier 수/잘린 SSE, 최대 3개 재적합을 사용한다. 4점 미만·퇴화·수치 실패·합의 부족·물리적으로 다른 동점 해를 구분한다. |
| 숨은 코너 출력 | [pose.py:315](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L315), [assemble:109](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L109) | 새 자세가 나오면 H 초기 좌표를 최종 R,t 투영으로 교체하고 다시 fit하지 않는다. 전후 H 변화는 기록이며 프레임 veto가 아니다. 새 자세가 없으면 별도 기본 출력 반환이다. |

LOO의 k는 **검사 중 임시 제외**이고 H는 **초기 예측 자기 가림 집합**이다. 두 집합 모두 검사 fit에서는 빠지지만, k를 새로운 self-occ 라벨로 만들거나 k를 자동 재투영 출력으로 교체하지 않는다. H는 초기 N3에서 오므로 완전한 통계적 독립성은 없다. 경계 proposal의 Base feature·role 의존도 남아 있다. V4가 확보한 것은 이 의존까지 모두 제거한 독립성이 아니라, 제외된 좌표가 초기 전체점 수치 자세 prior를 통해 검사와 final fit에 다시 들어가지 않는 계약이다.

## 이미 수정했거나 실행한 것을 다시 미완료로 부르지 않음

과거 [training.py:91](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py#L91)의 exact-edge ray miss를 `close=False`로 만든 뒤 NONE으로 감독한 오류는 별도 새 경계 선택기 학습에 있었다. main의 N3→cornerSubPix 오류가 아니다. 기존 source-test의 대응 **2,164개**는 실제 wire·투영·마스크·깊이 증거로 복구됐고, POSITIVE/NONE/IGNORE 준비 및 검산 이후 수정 감독으로 실제 추가 **3×3,000=9,000 업데이트**가 완료됐다. [수리 보고서](../pallet_kp_supervision_repair_20261010_v1/RESULT_KO.md), [실제 학습 완료](../pallet_kp_corrected_supervision_20261010_v1/TRAINING_COMPLETION.json)가 그 이력을 분리한다. 수정 전 IMAGE_ROLE **0/319 NEW**를 수정 후 결과로 재사용하지 않는다.

V3 LOO는 fit에서 k/H를 제외했지만 [selection.py:135](../../../scripts/research/pallet_cornerwise_refiner_20261010_v3/selection.py#L135)에서 초기 전체점 자세·치수 prior를 사용하는 V2 solver를 호출했다. V4는 [pose.py:134](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L134)의 초기 R,t·투영·치수 선택 입력이 없는 solver로 이를 제거했다. V7도 그 solver를 사용한다. 이 수리는 이미 완료됐으며 같은 수리를 다시 제안하지 않는다.

수정 감독 v1은 원래 66-way decoder의 `IMAGE_ROLE_NO_MASK_ROBUST`, `IMAGE_ROLE_STANDARD`, `IMAGE_ROLE_POINT_LINE`을 **이미 실제 실행**했다. [subset_downstream.py:29](../../../scripts/research/pallet_kp_corrected_supervision_20261010_v1/subset_downstream.py#L29)와 [기존 결과표](../pallet_kp_corrected_supervision_20261010_v1/RESULT_KO.md)가 연결된다. 그 보고서의 두 대조는 각각 **T15.63916cm/R14.64982°**, **T14.57928cm/R13.24067°**다. 이 문서는 그 수치를 새로 재집계하지 않았다. 따라서 “수정 모델의 mask/standard 절제를 한 번도 안 했다”는 설명은 틀리다.

점이 부족한 local C2도 원래와 수정 감독 v1에서 실행된 적이 있다. [point_line.py:19](../../../scripts/research/pallet_observation_refiner_20261009_v1/point_line.py#L19)는 초기 자세를 optimizer 시작점으로 사용하고 numerical rank6을 검사했으며, 이를 독립 초기화 PnP라고 부르지 않았다. V5/V6은 calibrated ROLE의 V4 hybrid 공급에 unused line을 추가한 별도 경로를 실제 실행했다. V5 [solver.py:30](../../../scripts/research/pallet_partial_line_independent_20261010_v5/solver.py#L30)는 4개 실제 point inlier를 유지하므로, sparse <4점 local rescue와 같은 경로가 아니다. 이 차이가 과거 local C2 미실행을 뜻하지는 않는다.

## 확인된 한계와 아직 증명하지 못한 원인

다음은 [기존 방법 단계 감사](../pallet_corner_mechanism_audit_20261010_v1/RESULT_KO.md)에 이미 기록된 **사후 기술 통계**다. 새 수치 실행 없이 그 출처를 인용한다.

| 기존 선택 교점의 구분 | N3보다 개선186개 | N3보다 악화237개 |
| --- | ---: | ---: |
| 더 큰 지지선 RMS 중앙값(px) | 0.51668 | 0.52041 |
| 참조 코너까지 교점 오차 중앙값(px) | 2.51279 | 4.32812 |

악화237개 중 **233개가 기존 DIRECT_VISIBLE**였다. source CAL 반경 안의 참조는 **331/348**, 실사 교체 후보에서는 **188/423**이었다. 이 근거는 “선택한 query들에 잘 맞는 선”과 “해당 코너에 정확한 선”이 다른 조건임을 보여 준다. 자기 가림 분류 오류만으로 교체 좌표의 악화를 설명할 수 없다. 작은 잔차·source 반경을 실사 정확도 인증으로 쓰거나, 이 수치에 맞춰 cap을 조정할 근거도 아니다.

기존 source의 bin/axis/query/wire 계약과 실제 wire 교점↔native 정의 감사는 큰 좌표 swap/index 결함을 발견하지 않았다. source CAL348의 두 참조 차이 평균 약 **0.000031px**, 최대 **0.000246px**도 같은 보고서에서 인용한 기존 결과다. 실제 경계 소유권 오류, appearance 전이, 선 편향, reference 불확실성 각각의 인과 비중은 아직 증명하지 못했다.

source feature 초기 R,t·projected witness가 없는 role3의 독립 semantic replay와 같은 장면 네 통제 변형의 coverage는 별도 미검증 제한이다. source의 정확한 감독이 기존 RGB에서 확보됐기 때문에 새 RGB를 만들지 않았다는 우선순위도 유지한다. 이 두 제한을 확인된 새 구현 버그나 추가 학습 필요성으로 단정하지 않는다.

## 현재 관측에 남은 원래 C3 원인 분리 공백

원지시문 C3의 항목5/6은 **IMAGE_ROLE 동일 관측의 NO_MASK_ROBUST와 STANDARD**를 요구한다. 과거 수정 감독 v1의 실행 이력과, 현재 V2 CAL decoder/V4 solver 조합에서의 동일 관측 비교는 구분해야 한다.

| 현재 관련 경로 | 코드 근거 | 실제 비교 범위 |
| --- | --- | --- |
| V2 sparse ROLE_ONLY | [pipeline.py:304](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L304) | sparse ROLE_ONLY는 masked robust다. |
| V2 no-mask | [pipeline.py:311](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L311), [:316](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L316) | `N3_VALIDATED_ROLE_NO_MASK`는 N3+경계 hybrid 좌표다. sparse ROLE_ONLY의 mask만 뺀 대조가 아니다. |
| V2 standard | [pipeline.py:318](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L318) | `N3_BASIN_STANDARD`는 native N3 좌표다. sparse ROLE_ONLY의 solver만 바꾼 대조가 아니다. |
| V7 no-mask | [pipeline.py:278](../../../scripts/research/pallet_three_head_observation_20261010_v7/pipeline.py#L278) | H를 쓰지 않는 방법은 native `N3_INDEPENDENT_ROBUST_NO_MASK`다. |
| V7 learned supplies | [pipeline.py:304](../../../scripts/research/pallet_three_head_observation_20261010_v7/pipeline.py#L304), [METHODS:19](../../../scripts/research/pallet_three_head_observation_20261010_v7/common.py#L19) | 3head×2supplies는 모두 H+robust다. 현재 sparse IMAGE_ROLE 관측의 ordinary/no-mask 행은 method 목록에 없다. |

따라서 현재 방법의 원인 분리에 필요한 가장 작은 다음 블록은, **봉인된 V7 IMAGE_ROLE_BOUNDARY_ONLY의 같은 sparse q·K·치수·head·CAL을 고정한 3-way 비교**다.

1. H 제외 + 기존 동일 robust: 현재 행 재사용.
2. H 제외 + 동일 V4 solver의 standard 분기: 같은 U의 solver 대조.
3. H 없음 + 동일 V4 robust: 같은 sparse 좌표의 mask 대조.

이는 이 문서에서 구현·실행한 것이 아니다. 다음 블록을 수행한다면 별도 protocol에서 코드·관측 입력·방법·출력 상태·검산·실행량을 먼저 잠가야 한다. source confidence/선 지원/반경/8px residual/치수 분기/가중치/학습량을 성능에 맞춰 바꾸는 정책 탐색으로 만들지 않아야 한다. 모든 방법의 대안 해·rank·실제 final inlier와 NEW/fallback/완전 실패, 공통 산출집합 및 전체245장 결과를 유지해야 한다.

sparse 좌표에는 admitted 경계 교점만 들어 있고 결측은 NaN이다. H의 제외 여부를 바꿔도 초기 N3 좌표로 결측을 채워 점 수를 맞추지 않는다. 동일 bank의 exact 좌표/K/치수/4ID 가설은 재사용할 수 있으나 timing을 이 재생 비용으로 주장하지 않는다. no-mask 행은 명시적 진단 대조이고, 배포 주경로의 **H 초기 좌표 final fit 제외 → NEW R,t 후 H 재투영 교체 → 재투영 재fit 금지** 조건은 그대로 둔다.

이 비교는 현재 관측에서 H가 유효 대응을 과도하게 제거하는지, 같은 관측의 robust가 ordinary보다 오대응을 흡수하는지 분리한다. 그 자체가 성능 개선이나 실제 물리 경계 소유권을 증명할 것이라고 미리 약속하지 않는다. 기존 native no-mask 결과로 새 selector의 필요성을 대신 주장하거나, 기존 66-way 절제 결과로 현재 CAL decoder의 효과를 대신 인증하지 않는다.

## 이번 감사의 실행량과 보호 범위

| 항목 | 이번 감사의 실제 실행량 |
| --- | ---: |
| 두 첨부 전체 읽기 | 2개 문서 |
| 관련 코드·기존 보고서의 정적 읽기 | 수행; 재현 수치 실행으로 세지 않음 |
| detector / N3 / head forward | 0 |
| 이미지·GT·대형 원행 수치 읽기 | 0 |
| PnP / optimizer / ray / bootstrap / 통계 재집계 | 0 |
| 학습 업데이트 / 새 RGB / 새 실사·수동 주석 | 0 |
| 기존 코드·core·protocol·원행·가중치 수정 | 0 |
| 신규 문서 작성 | 이 파일1개 |
| commit / push / main 변경 | 0 |

위의 기존 수치는 연결한 이전 보고서에서 인용했으며 이번 감사가 독립 재실행한 숫자가 아니다. 코드상 경로의 구분은 정적 해석이다. 최신 V7 실제 성능·resource 충돌·게시 상태는 root의 실행 receipt와 주 보고서가 별도로 판정한다.
