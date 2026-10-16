“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”

**최신 완료된 V15에서는 아직 아니다.** 전체 쉬움153·중간92의245장을 유지한 위치·회전 평균은 **10.065714713805312cm·11.51370306863193°**이고, 같은 N3→cornerSubPix 대조는 **9.754548028679153cm·10.914841863166323°**다. 새 자세50장과 기존 N3 반환195장을 모두 포함했다. V16은 실제 실행 중이며 아직 최종 score가 없으므로 개선 여부를 판정하지 않는다.

# 팔레트 관측 보정 브랜치 검토 시작점

이 문서는 여러 실험의 확인된 문제·수정·미검증 영역을 연결하는 색인이다. 새로운 성능 실험이나 수치 재계산은 하지 않았다. 기존 RESULT/README/BUILD snapshot은 수정하지 않았다. 아래 수치는 저장 METRICS/실행 영수증의 전사이며, V16 완료 후 이 새 색인의 상태만 갱신한다.

## 코너 정의와 보정기의 어느 부분이 문제였나

|부분|확인된 사실|현재 해석|
|---|---|---|
|합성 감독의 대응 존재 판정|별도 경계 head의 [old training.py:91](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py#L91)는 실제 경계 ray 교차의 수치 실패를 no_match로 처리했다. source-test에서 실제 대응이 있던 **2164 query**가 해당한다.|main의 N3→cornerSubPix 오류가 아니다. 대응이 없다는 감독과 계산에 실패했다는 상태를 혼동한 실제 오류다.|
|감독 수정|[실제 owned-wire·depth 검토](../pallet_kp_supervision_gate_20261010_v1/RESULT_KO.md)로 대응 위치/POSITIVE, 증명된 부재/NONE, 증명 불가/IGNORE를 나눴다. 초기 준비·CPU gradient 이후 [수정 감독의 3×3000 업데이트](../pallet_kp_corrected_supervision_20261010_v1/RESULT_KO.md)가 별도 승인으로 실제 완료됐다.|초기 읽기 전용 감사 때 수정 후 학습 결과가 없었다는 말과 이후 실제9000회 학습은 서로 다른 시점이다. 원 잘못된9000회와 수정된9000회는 별도다. 현재 V10–16의 추가 학습·새 RGB·seed는0이다.|
|head target에서 corner q로의 조립|target은 query normal과 의도한 semantic physical edge의 실제 wire 교차다. [decoder](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L147)는 서로 다른 incident edge의 TLS line 두 개를 교차시켜 corner q를 만든다.|임의 ray 교점 하나를 바로 3D corner에 대응시킨 조립 버그는 확인되지 않았다. source owned-wire 검증을 실사 edge ID의 물리 ownership 증명으로 확대할 수 없다.|
|관측 공급과 실제 대응의 정합|마스크가 맞아도 q의 위치·서로 다른 선의 의미·branch가 틀릴 수 있다. [V14 저장 대응 진단](../pallet_stable_incidence_guard_20261011_v14/CORRESPONDENCE_DIAGNOSTIC_CHECKS.json)에서 fit q가 모두8px 이내인46 NEW 중22장은 N3보다 T/R이 함께 나빴다.|8px는 같은 proxy/번호 정렬에 조건부인 품질 진단이다. q가 물리 경계 위인지와 그 경계가 의도한 3D edge인지, pose branch가 맞는지는 별개다. 이 자료만으로 ownership 오류가 확정되지는 않는다.|
|부분 선의 기하 모델|[V12 진단](../pallet_finite_segment_support_20261011_v12/RESULT_KO.md)은 선택 candidate의 지원이 원 radius 밖인 **64/245 frame**, 같은 전체8px 합의를 만족하는 비동일 대안이 있는 **106/245 frame**을 확인했다.|무한선의 작은 법선 잔차만으로 유한 물리 edge의 관측 구간을 설명했다고 할 수 없었다. 두 수치는 frame 집합이며 NEW94장의 분모로 바꾸지 않는다.|
|관측 충분성과 후보 선택|V13은 finite support·실제 inlier refit/합의·branch 처리를 추가했다. V14는 자유로운 선 길이 방향의 hinge를 안정적 관측 등식으로 세던 두 candidate를 거절했다. V15는 factor 부분집합/root 후보를 보존해 전체 합의로 검사했다.|rank6·약함 검사·작은 residual·NEW는 정확한 물리 대응이나 전역 유일해의 인증이 아니다. V15도 N3를 이기지 못했다.|

구감독 IMAGE_ROLE의 **0/319 새 자세 산출은 수정 전 모델의 실패 결과**다. 수정 후에도0/319라는 주장은 하지 않는다. 수정 학습 후 IMAGE_ROLE은245장에서220 NEW를 산출했지만 T/R 평균15.20239cm·14.56122°로 N3보다 나빴다. 감독 버그 수정과 방법의 정확도 성공은 구분한다. [학습 완료 영수증](../pallet_kp_corrected_supervision_20261010_v1/TRAINING_COMPLETION.json)을 함께 확인할 수 있다.

## 자기 가림과 LOO는 어떻게 처리했나

가림을 완벽히 분류하는 것을 성공 조건으로 쓰지 않는다. 일부 올바른 점을 잘못 제외해도 남은 실제 대응의 수·배치가 충분하면 계속 풀고, 잘못 남은 대응은 유한4점 부분집합 강건 합의에서 제외할 수 있는지 검사한다. 동일 좌표·K·등록 치수·4-ID 가설은 마스크 사이에서 공유한다. 마스크 차이만으로 전체 frame을 실패시키지 않는다.

이미 실행한 corner LOO와 native endpoint LOO는 검증할 점/선의 endpoint를 **H∪{검사 ID}** 또는 **H∪{두 endpoint}**로 함께 제외한다. [V10 감사](../pallet_unused_line_endpoint_audit_20261011_v10/RESULT_KO.md)는245장의370 unused edge에서 SUPPORTED265·CONTRADICTED67·UNVERIFIED38을 기록했고 [독립441191검사](../pallet_unused_line_endpoint_audit_20261011_v10/VALIDATION_CHECKS.json)를 PASS했다. 이후 경로는 CONTRADICTED 선만 제외하고 UNKNOWN은 유지했다. native validator도 틀릴 수 있으므로 이 label은 실제 물리 ownership 정답이 아니다.

**승인 NEW의 실험 fit·합의 점수에서 자기 가림 H의 초기2D를 제외하고, 최종 R,t가 있으면 H 좌표를 재투영으로 대체한다.** 재투영점을 독립 관측으로 다시 PnP/LOCAL에 넣지 않는다. 같은 선에서 만든 점과 선도 중복 factor로 세지 않는다. 관측 부족·수치 실패·다중해·약함을 typed 원행에 남기며, 새 자세를 못 구하면 기존 all8 N3의 좌표/pose를 그대로 반환한다. 이 fallback은 새 H-excluded fit나 NEW가 아니다.

가림 없는 강건 PnP 대조는 [V7](../pallet_three_head_observation_20261010_v7/RESULT_KO.md), 같은 sparse 관측의 일반/강건·H/무마스크 대조는 [V8](../pallet_same_observation_controls_20261010_v8/RESULT_KO.md)에 이미 있다. 기존 [245장 무학습 대조](../pallet_kp_corrected_supervision_20261010_v1/RESULT_KO.md)는 N3 좌표의 예측 H 제외+일반 PnP가 T/R 평균8.95157cm·10.61764°로 소폭 개선됐다고 보고했다. 따라서 새 학습기·복잡한 강건 경로가 필요하다고 단정하지 않는다. 그 단순 대조의 개선을 아래 새 경계 방법의 성공으로 옮겨 쓰지도 않는다.

## 실제 완료된 자세 평가

모든 행은 같은245장을 유지한 **전체 운용 평균**이다. 기본 반환 때문에 큰 오차 frame이 빠지지 않는다. BASE/N3는 고정 대조이고 NEW0을 실패로 세지 않는다. 전체/공통 산출집합, NEW/fallback 별6 moment, paired CI, 직접 가시점 손상과 실제 H 재투영 오차는 각 링크의 전체 보고서에 있다.

|방법|NEW / 기본 반환 / pose 없음|위치 평균 cm|회전 평균 °|ADDsym 평균 cm|원행·상세 근거|
|---|---:|---:|---:|---:|---|
|N3→cornerSubPix 고정 대조|고정245 / 0 / 0|9.754548028679153|10.914841863166323|17.028646667097725|[V15의 같은 N3 원행 집계](../pallet_factor_subset_consensus_20261011_v15/METRICS.json)|
|기존 sparse LOCAL V9|115 / 130 / 0|17.87901676799424|13.415258792985208|26.64545714508252|[V9](../pallet_sparse_local_line_20261010_v9/RESULT_KO.md)|
|V11 필터+weak SAFE|94 / 151 / 0|12.611230982495721|11.787036654409325|20.23206129946827|[V11](../pallet_endpoint_observability_repair_20261011_v11/RESULT_KO.md)|
|V13 finite 합의|75 / 170 / 0|11.926597046320317|12.556904694659371|20.112381623955297|[V13](../pallet_finite_edge_consensus_repair_20261011_v13/RESULT_KO.md)|
|V14 stable incidence|73 / 172 / 0|10.529132872230429|11.934935026928532|18.68400203211673|[V14](../pallet_stable_incidence_guard_20261011_v14/RESULT_KO.md)|
|V15 factor 부분집합 합의|50 / 195 / 0|10.065714713805312|11.51370306863193|17.820328530896433|[V15](../pallet_factor_subset_consensus_20261011_v15/RESULT_KO.md)|

V11의 관측 필터와 weak 거절은 기존 LOCAL보다 전체 T/R을 줄였다. V14는 기존75 NEW 중2개를 기존 N3 반환으로 바꿨으며73개의 R,t를 더 보정한 실험이 아니다. V15는 STABLE/POINT보다 T/R 평균은 낮지만 N3보다 둘 다 높다. 방법 간 차이를 가림 분류·선 추가·강건 PnP·재투영 각각의 독립 인과 효과라고 분해하지 않는다.

마스크 오류와 실제 pose 결과도 분리했다. V15에서 mask가 틀렸는데 T/R이 함께 좋아진3 NEW가 있고, mask가 맞았는데 함께 나빠진17 NEW가 있다. 실제 교체된 H63점은 pixel 중앙값이 낮아져도 평균은7.274476032915819→8.187343258748255px로 나빠졌다. 직접 가시1503점 중 기존≤5px에서 출력>10px로 손상된3점도 남겼다. 숨은 점 재투영이나 mask 분류만으로 최종 pose 개선을 주장할 수 없다.

## 검산·비용·그림을 어디서 확인하나

|완료 항목|실제 권위|해석 제한|
|---|---|---|
|V10/V12 원인 진단|[V10 VALIDATION](../pallet_unused_line_endpoint_audit_20261011_v10/VALIDATION_CHECKS.json):441191 PASS; [V12 SEGMENT_CHECKS](../pallet_finite_segment_support_20261011_v12/SEGMENT_CHECKS.json):80320 PASS|새 최종 자세 성능이나 배포 latency 실험이 아니다.|
|V11 자세·전체 경로|[보고서](../pallet_endpoint_observability_repair_20261011_v11/RESULT_KO.md), [actual forward runtime](../pallet_endpoint_observability_repair_20261011_v11/runtime_forward_review/RUNTIME_RECEIPT.json)|150 fresh·공식130회, 전체 경로 평균149.6915392153846ms/P90229.74461220000003ms. 원 검산/두 runtime 실패는 별도로 보존했다.|
|V13 통계·시간·그림|[독립 통계](../pallet_finite_edge_consensus_repair_20261011_v13/REVIEW_VERIFICATION.json), [runtime](../pallet_finite_edge_consensus_repair_20261011_v13/RUNTIME_RECEIPT.json), [실제8이미지 검토](../pallet_finite_edge_consensus_repair_20261011_v13/ROOT_VISUAL_REVIEW.json)|1225행/810 moment/81CI 검산 PASS. fresh150·공식130회 평균136.75885975384614ms/P90169.69911470000002ms. 원74 cusp FAIL·통계 metadata FAIL·renderer 끝 binding FAIL과 별도 완료 검토를 보존했다.|
|V14 통계·시간·그림|[VERIFICATION](../pallet_stable_incidence_guard_20261011_v14/VERIFICATION.json), [RUNTIME_CHECKS](../pallet_stable_incidence_guard_20261011_v14/RUNTIME_CHECKS.json), [실제8이미지 검토](../pallet_stable_incidence_guard_20261011_v14/ROOT_VISUAL_REVIEW.json)|1470행/972 moment/81CI,150/130/56 timing 검산 PASS. 실제 전체 경로 평균277.66035302307694ms/P90400.4057953000001ms.|
|V15 통계·시간·그림|[ENVELOPE_VERIFICATION](../pallet_factor_subset_consensus_20261011_v15/ENVELOPE_VERIFICATION.json), [RUNTIME_CHECKS](../pallet_factor_subset_consensus_20261011_v15/RUNTIME_CHECKS.json), [실제8이미지 검토](../pallet_factor_subset_consensus_20261011_v15/ROOT_VISUAL_REVIEW.json)|1225행/810 moment/81CI,56311 timing 검사 PASS. 실제150 fresh·공식130회 평균179.83371015384614ms/P90315.9298425ms. 통계 setup 실패 두 건과 public metadata 실패는 그대로 남았다.|

runtime은 초기 N3/pose·Base 특징·head·point PnP·native endpoint LOO·최종 solver·H 재투영/metadata까지 포함한 실제 호출의 clock이다. 모델 로딩·원본26장 decode·logging·반환 후 parity·자원 probe는 밖에 둔다. 캐시 재생 시간이나 이전 시간의 합산은 쓰지 않았다. 저장 timing 검산은 하드웨어를 다시 측정한 것이 아니다. 각 실행의 quiet snapshot/cleanup과 이력이 원 영수증에 있다.

![V15 전체245장 운용 결과](../pallet_factor_subset_consensus_20261011_v15/figures/01_all_operational.png)

![V15 같은 frame paired 비교](../pallet_factor_subset_consensus_20261011_v15/figures/02_same_frame_pairs.png)

위 두 그림과 보고서의 고정6사례 그림은 실제 생성·검토됐다. 사례는 V15 오차로 새로 고르지 않았으며6사례는 모두 V15 fallback이다. 가독성과 bytes 검토는 물리 대응 정확도 인증이 아니다.

## 기준·공개 재현·아직 미완료인 것

평가 범위는 **Clean153+Moderate92=245**, Severe74는 최신 사용자 지시에 따라 제외했고 기존 전체319 기록은 보존했다. Moderate 일부에는 부분 가림이 있을 수 있어245장을 모두 무가림이라고 부르지 않는다. 같은 장면 변형/분할·원영상·가중치·사용자 변경은 보존한다.

reference R,t는 N3 예측을 정답으로 사용한 값이 아니다. 기존 annotation의 유한8 corner XY·K·등록 W/D 양가설을 SQPnP+LM으로 적합한 이상적 cuboid **GEOMETRIC_PROXY**다. 일부 annotation XY는 PnP projection으로 채웠을 수 있고 source/direct visibility 태그가 XY 추출에서 걸러지지 않아 독립 물리 실측·전점 독립 사람 클릭은 보장되지 않는다. N3-phase는 corner permutation과 사람 mask/native XY 진단의 고정 정렬이며 pose truth 생성과 별개다. 모든 결과는 알려진 DEV에 대한 값이고 unseen test 성공이 아니다.

[V11 공개 검토](../pallet_endpoint_observability_repair_20261011_v11/PUBLIC_REVIEW_KO.md), [V13 실제 fresh 공개 검토](../pallet_finite_edge_consensus_repair_20261011_v13/PUBLIC_REVIEW_KO.md), [V14 실제 fresh 공개 검토](../pallet_stable_incidence_guard_20261011_v14/PUBLIC_REVIEW_KO.md)를 통해 private weights/RGB/GT 없이 저장 통계와 timing을 검산하는 절차·실패 보존·복사 provenance를 볼 수 있다. fresh byte bundle은 Git clone이나 모델 재실행이 아니다. V15 공개 closure/fresh 검토는 진행 중이며 최종 게시 SHA는 아직 이 색인에 기록하지 않았다.

**남은 핵심은 실제 대응의 의미·배치와 잘못된 합의/pose branch를 구분하는 것이다.** source 역할 특징의 재현, 정확한 mask, rank 통과 어느 하나만으로 이 문제가 해결되지 않는다. 현재 V16의 실제 실행 결과와 독립 검산·최종 score를 기다리고 있다. 결과가 없는데 성공·채택·전체 완료를 보고하지 않는다. 정상 research branch push와 원격 SHA 일치는 별도 실제 게시 영수증으로 확정하며 main 자동 병합/force push는 하지 않는다.
