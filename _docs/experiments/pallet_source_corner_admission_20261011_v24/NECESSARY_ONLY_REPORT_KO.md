가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**현재 저장된 실사 평가에서는 방법 전체의 개선을 확인하지 못했다.** LOO의 자기 가림 제외와 최종 재투영 계약은 지켰지만, 최신 강건 경로는 단순 대조보다 위치·회전 평균이 나쁘다. 이번 추가 검산은 합성 query 성능, 중간 코너 후보, 최종 관측 채택을 혼동하지 않도록 원인을 더 좁힌 것이다. 새 학습·영상·추론·선 fit·PnP·자세 성능 재실험은 모두 0회다.

**이번에 실제로 확인한 것:** 저장된 세 모델의 source CAL128 중 조건부 actual-wire 교점 후보 573행을 읽고, 이미 쓰던 영상 범위와 불확실성 반경 조건만 산술 검산했다. IMAGE_ROLE의 >8px 후보 33개 중 **29개는 이 선쌍 후보에 대해 확실히 거절된다.** 가장 큰 258px 오차를 최종 채택된 코너 오류로 읽으면 잘못이다. 남은 4개는 필요한 두 조건만 통과하며, 최종 채택 여부는 저장 행에 없다.

아래 값은 **중간 후보 행**이다. 8px는 기존 진단 구간으로 집계할 때만 사용했다. admission threshold를 8px로 바꾸거나 재보정하지 않았다. 같은 코너의 다른 선쌍이 선택될 가능성도 별도다. 세 arm마다 조건부 후보 모집단이 달라 이 표로 모델의 자세 우열을 판단할 수 없다.

| 모델 | 조건부 후보 | 필수조건 위반으로 확실히 거절 | 두 필수조건 통과·최종 UNKNOWN | >8px 후보 | >8px 중 확실히 거절 | >8px 중 최종 UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| GEOMETRY_ONLY | 125 | 44 | 81 | 8 | 8 | 0 |
| IMAGE_NO_ROLE | 100 | 57 | 43 | 12 | 12 | 0 |
| IMAGE_ROLE | 348 | 134 | 214 | 33 | 29 | 4 |

![저장된 모든 조건부 source 후보 573개](ALL573_SAVED_CANDIDATES.png)

세 패널은 모든 573행을 표시한다. 가로축은 저장 반경/초기 query 간격, 세로축은 실제 경계 support에서 만든 참조 교점과의 오차다. x=1은 기존 uncertainty gate이며 y=8px는 설명용이다. 파란색도 최종 accepted를 뜻하지 않는다. 그래프에는 새 실사 성능이나 미실행 추론을 넣지 않았다.

**코너 정의와 코너 구성 오차를 분리했다.** 기존 CAL348 계약에서 actual-wire 가상 교점과 annotation native projected-cuboid 코너의 차이는 평균 0.0000306744px, 최대 0.0002457794px다. 이 지원 선쌍에서는 코너 ID·정의가 어긋났다는 증거가 없다. 반면 head 위치로 만든 중간 교점 오차는 평균 5.253406px, 최대 258.055898px다. query 위치 오차는 선의 방향·외삽·교차각을 거치며 코너 오차로 증폭될 수 있다. 이번 집계에서 **모든 support query가 2px 이내인데 교점이 >8px인 후보**도 GEOMETRY_ONLY 5개, IMAGE_NO_ROLE 11개, IMAGE_ROLE 22개 있었다. IMAGE_ROLE 22개 중 20개는 기존 반경 조건을 위반한다. 이것은 physical mesh vertex 존재나 실사 경계 ownership을 입증한 결과가 아니다.

**최종 승인 여부를 아직 모르는 IMAGE_ROLE 4행:** 실제 gate가 사용하는 예측 교점의 외삽률과 코너별 후보 선택 결과는 저장되지 않았다. 저장된 `true_corner_support_extrapolation_ratio`는 참조 교점 기준이며 이를 예측 교점의 gate 값으로 대신하지 않았다.

| source ID | corner | edges | 교점 오차 px | support query 평균오차 px | support 최대오차 px | 반경 px | 초기 query 간격 px |
|---|---:|---|---:|---:|---:|---:|---:|
| P0__shard_06_f0585 | 1 | [0, 9] | 8.661354 | 0.466641 | 0.885302 | 8.661354 | 9.347328 |
| P0__shard_06_f0700 | 5 | [4, 9] | 19.295842 | 2.411036 | 4.559754 | 13.138760 | 28.884742 |
| P0__shard_03_f0494 | 0 | [0, 8] | 11.449058 | 0.483471 | 1.114131 | 24.964100 | 26.120333 |
| P0__shard_07_f0942 | 1 | [0, 9] | 8.144162 | 2.165358 | 4.404142 | 2.660038 | 4.639386 |

8.661px와 11.449px 두 행은 support query 최대오차가 각각 0.885px와 1.114px인데도 큰 교점 오차가 있다. 그러나 두 행이 최종 decoder나 실사 LOO에서 채택됐다고 주장하지 않는다. `recorded_radius_px` 안에 참조점이 들어간다는 사후 비교도 최종 admission과 다르다. 기존 최종 CAL aggregate에는 accepted_corners=231이 있으나, 그 수와 조건부 geometry348의 행별 교차 집합은 공개되지 않았다.

**수정된 학습의 query 성능을 자세 성능으로 바꾸어 읽지 않았다.** 기존 수정 감독의 고정 3×3000 업데이트는 이미 완료됐고, 이번에 추가 학습한 것은 없다. 아래 표는 저장된 마지막 step3000 source TEST128 probe다. 공통 POS4497/NONE1966/IGNORE4289 query 중 raw66-way argmax를 평가했다. accepted POS의 조건부 평균 bin 오차에는 TLS·코너 교점·LOO·PnP가 포함되지 않는다. TEST는 P0 내부 분할이고 감독 오류 진단에도 이미 열람된 집합이므로 독립 실사 일반화 증거가 아니다.

| 모델 | 채택된 POS | POS 채택률 | 채택 POS 평균 query 오차 | NONE 오채택률 |
|---|---:|---:|---:|---:|
| GEOMETRY_ONLY | 3987/4497 | 88.659% | 1.036899px | 3.967% |
| IMAGE_NO_ROLE | 4271/4497 | 94.974% | 0.884172px | 3.103% |
| IMAGE_ROLE | 4370/4497 | 97.176% | 0.806732px | 1.628% |

GEOMETRY_ONLY는 기존 구현상 image 채널을 없애고 geometry와 ROLE 채널을 유지한다. IMAGE_NO_ROLE은 ROLE 채널만 없애고 IMAGE_ROLE은 28채널 전부를 사용한다. 이 역사적 입력 정의를 변경하거나 새 모델로 교체하지 않았다. ROLE은 hull의 BOUNDARY/INTERNAL/UNAVAILABLE 입력이고 모델의 출력은 가림 분류나 코너 좌표가 아니라 query 후보 위치 또는 NONE이다.

**실사에서 여전히 해결해야 할 곳은 관측 선택과 최종 PnP 가설 선택이다.** 기존 쉬움153+중간92=245장의 선택 원행에서 실제 채택한 경계 코너는 85프레임/133점이며, 80점은 기존 N3보다 reference 오차가 줄고 53점은 커졌다. 133점 중 기존 8px point proxy 기준 GOOD120/BAD13이다. H를 제외한 LOO에서 q가 native보다 잔차가 작았다는 사실만으로 관측 위치가 옳다고 보장할 수 없다. 이것이 잘못된 점을 더 정확한 점으로 바꾸는 데 실패한 실제 단계다.

동시에 나쁜 자세 사례 대부분을 head 코너 오류만으로 설명할 수 없다. 기존 whole-U 기본 가설이 모두 거절된 59프레임 중 known-GOOD 대응이 4개 이상인 24프레임은 모두 그 4점 조합 요청과 유효 root를 가졌다. 22프레임은 그 root에서 승인된 refit도 있었다. 위치·회전이 함께 악화된 8프레임 중 **7프레임은 경계 보정점 채택이 0개**였다. 예를 들어 `eval_pallet07:1778652138515809024`는 보정점 0개인데 선택된 해가 BAD2,4를 포함한 6 inlier를 얻었다. GOOD4에서 출발한 가설의 5 inlier보다 개수/SSE 순위가 앞섰고, N3 2.188809cm/0.454581°에서 새 출력 10.716346cm/86.942401°로 악화됐다. 다른 root의 실제 T/R를 새로 계산해 더 좋다고 주장하지는 않는다.

**LOO와 self-occ 계약은 유지한다.** corner LOO는 고정된 초기 predicted-H와 검사 코너 k를 함께 제외하는 `H∪{k}`이고, endpoint LOO는 `H∪{a,b}`다. 최종 새 pose의 fit에는 자기 가림 초기 좌표와 center8을 넣지 않는다. 다른 남은 실제 관측으로 강건 PnP를 구한 뒤 새 R,t로 H 코너를 한 번 재투영해 교체하고, 재투영점을 독립 관측으로 다시 fit하지 않는다. H와 사람 마스크가 다르다는 이유로 프레임을 실패시키지 않으며, GOOD4/rank6도 실제 정확성이나 다중해 해소의 충분조건으로 부르지 않는다. 새 pose가 없을 때 반환하는 기존 N3는 별도 fallback 상태로 기록한다.

기존 전체 운용 결과는 최신 경로 240 NEW/5 fallback/0 완전 실패다. 전체245 mean(T cm/R deg/ADDsym cm)은 N3 9.754548/10.914842/17.028647, 단순 HYBRID ordinary 9.207154/10.613464/16.439686, 최신 강건 경로 9.745696/12.622492/18.580216이다. 단순 ordinary 대조는 이 최종 self-occ 제외 계약과 같지 않으므로 그대로 요구 방법의 성공으로 대체할 수 없다. 모든 실사 자세 값은 기존 annotation/K/cuboid에 의한 **DEV GEOMETRIC_PROXY**이며 독립적으로 측정한 physical R,t는 아니다. 평균·분산·표준편차·중앙값·P90·공통 산출집합·운용 결과 및 실제 RGB 사례는 기존 [전체245 보고서](../pallet_whole_u_closure_20261011_v21/RESULT_KO.md)와 [가설/LOO 원행 감사](../pallet_consensus_cause_audit_20261011_v23/RESULT_KO.md)에 보존했다. Severe74는 이번245에서 제외했고 역사적319 결과는 삭제하지 않았다.

**실제로 잘못됐던 것 / 이미 고친 것 / 아직 미검증인 것을 구분한다.** 수정 전 `training.py.targets()`는 수치적인 ray miss를 physical no_match로 감독했고, 실제 대응 2164 query를 잘못 NONE 처리했다. 실제 mesh/depth/wire의 POS/NONE/IGNORE 감독으로 수정하고 고정 재학습을 완료했다. 기존 IMAGE_ROLE 0/319는 수정 전 모델의 역사적 실패 결과이며 수정 후 결과로 재표기하지 않는다. 이번 검토에서 새 edge/bin/axis/코너 정의 구현 오류는 확정하지 못했다. P0의 외부 가림·비가림 방해물·저대비 네 통제 변형은 여전히 별도 검증되지 않았다. unknown을 NONE이나 BAD 정답으로 바꾸지도 않았다.

**남은 문제를 확인할 최소 작업:** 첫째, 동일 CAL128의 이미 저장된 logits를 고정 decoder로 검산해 위 4행의 예측 외삽률·최종 후보 선택을 행별로 기록하면 source 코너 승인 여부를 확정할 수 있다. 이는 기존 출력의 decoder 검산이고 새 학습/RGB/GT가 필요하지 않다. 이번에는 line fit도 0회인 순수 저장행 감사까지만 실행했다. 둘째, 실사 가설 선택을 바꾸기 전에 같은 좌표·K·치수·H·모든 finite4 roots를 고정해 선택 기준을 사전에 명시하고, 별도의 허용된 평가로 실제 위치·회전을 검증해야 한다. 테스트 오차를 보고 threshold를 바꿔 반복하거나 GOOD proxy를 배포 선택기에 넣는 것은 해결로 처리하지 않는다. 독립 physical GT 부재도 그대로 남는다. 목표는 **아직 완료 아님**이다.

코드의 근거는 [query/선/교점/최종 gate](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L117), [조건부 geometry 참조 생성](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/calibration.py#L140), [native 코너 계약과 반경 기록](../../../scripts/research/pallet_corner_mechanism_audit_20261010_v1/cal_corner_contract.py#L138), [마지막 source TEST probe](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py#L288), [고정 H의 LOO 선택](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py#L135)이다. 이번 산술 감사는 [saved_admission_audit.py](../../../scripts/research/pallet_source_corner_admission_20261011_v24/saved_admission_audit.py), 원행은 [ROWS.jsonl.gz](ROWS.jsonl.gz), 통계는 [SUMMARY.json](SUMMARY.json), 입력 결속은 [PROTOCOL.json](PROTOCOL.json), 검산은 [CHECKS.json](CHECKS.json)에서 확인할 수 있다.

게시된 파생행 검산은 다음과 같다. 573행의 ID·support 오차·필수조건·카운트·이미지 hash를 확인하며 original supervision/실사 ownership/상류 전체를 재실행했다고 주장하지 않는다.

```bash
python3 -I -S -B scripts/research/pallet_source_corner_admission_20261011_v24/check_saved_public.py _docs/experiments/pallet_source_corner_admission_20261011_v24
```

상류 파일이 준비된 checkout에서는 `saved_admission_audit.py --source CHECKOUT --output NEW_OUTPUT`으로 원행 join 자체를 재실행할 수 있다. protocol은 사용한 16개 원본 입력의 bytes/SHA와 실행 코드 SHA를 기록한다. gzip 원행은 압축 전과 byte 단위로 동일하게 복원됐고, 모든 입력 hash는 실행 전후 같았다. 이번 추가 실행량은 저장 geometry 비교573행과 기존 query support 산술뿐이며 detector/head/line-fit/PnP/pose-score/업데이트/RGB는 모두0이다. 기존 실제 경로 timing을 이 감사 시간으로 대신하지 않았다.
