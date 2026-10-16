# V15 고정 방법 계약: 후보를 버리지 않는 factor 부분집합 합의

이 조건은 알려진 DEV 쉬움153·중간92의 같은245장, 같은 IMAGE_ROLE sparse q/H 및 실제 unused physical/CAL line에 대한 구조 절제다. V13/V14의 전체 pool LOCAL 시작만으로 만들어지지 않던 대응 부분집합 후보를 생성한다. 실제 실행·성능 수치는 이 코드 전용 계약의 내용이 아니다. 기존 감독/모델/threshold·seed·영상·원행은 수정하지 않는다.

`FactorSubsetConsensusBank(points,K,xyz,image_size,point_bank=None).solve(observation,adopted_boundary_corner_ids,hidden,initial_pose=…,point_result=…)`는 V13 호환 solver 원행을 반환한다. `point_result`의 available/selected verdict는 가설 제거 또는 추가 LOCAL 시작으로 사용하지 않는다. 같은 q/K/치수/4-ID 문제를 기존 V4 numeric cache에서 계산하고 SQPNP/IPPE가 반환한 유효한 모든 raw root를 직접 proposal로 보존한다. fresh point-only 경로가 이미 계산한 in-memory `PoseBank`는 digest(q/K/xyz/image size)·타입·외부 dimension prior 없음이 같을 때 `point_bank`로 공유한다. Generic 문제를 다시 풀지 않는다. legacy 빈 cache의 원인 witness가 없을 때는 기존 layout rank 기준만 재검산하고 Generic을 재실행하지 않는다. 모든4-ID에서 H를 제외한다. masked/AMBIGUOUS point-only 결과에도 유효 root가 있으면 선택 결과에 종속하지 않고 쓴다. 새로운 P3P·GT point·외부 pose prior는 넣지 않는다.

실제 point 한 개 또는 실제 unused edge 한 개가 factor block 하나다. line이 적어도 하나인 모든3-block 부분집합을 정해진 ID 순서로 열거한다. 세 point만 있는 부분집합은 쓰지 않으며, point-only는 기존4-point 표준해법이다. mixed 부분집합은 같은 available initial N3 physical pose를 두 등록 치수로 remap한 시작값만 이용한다. 이6개의 normal scalar는 필요조건이다. LOCAL에서 모든 수학적 root를 찾았다고 주장하지 않는다. line support query의 edge 방향 위치를 새3D point ID로 만들지 않는다.

각 proposal을 같은 전체 U+retained unused edge pool에서 기존8px point 거리·finite edge block norm으로 평가한다. edge는 query 수와 관계없이1표다. point+edge inlier 수, cap64 truncated factor SSE, 차원/고정 열거 순으로 순위를 정한다. proposal 부분집합의 cost를 전체 합의 점수로 대신 쓰지 않는다. 각 proposal에서 얻은 실제 inlier pool에 단 한 번 refit하며, line inlier가 없는 point-only refit은 실제4점 이상이어야 한다. line이 있는 mixed refit은 실제6 normal scalar 이상 및 최종 관측성 검사를 요구한다. exact dimension/pose bytes/fit IDs가 같은 refit만 재사용한다. 다른 물리 branch를 같은 ID만으로 버리지 않는다. 별개 branch의 수치 실패·관측 부족·weak·unstable refit은 모두 기록하고 나머지 branch는 계속한다.

기존 V13 finite hinge/J/soft_l1=8/max_nfev50·positive depth·actual-fit stable8px·column-L2 rank1e−10/weak1e−4는 바꾸지 않는다. 그 gate를 통과한 **각** candidate의 같은 실제 fit에 V14 normal-only incidence guard를 적용한 뒤 선택한다. hinge는 fit/score에 남지만 관측성 등식으로 세지 않는다. 고정 consensus를 설명하는 물리 비동일 accepted branch 또는 기존 numerical tie는 AMBIGUOUS로 반환한다. 전역 유일해·실사 physical ownership은 증명하지 않는다.

관측 수 F≤20, 치수 D≤2, point 수 p≤8이면 mixed 부분집합 M=C(F,3)−C(p,3)≤1140, LOCAL proposal 호출≤DM≤2280이다. 4-ID Generic 요청≤D·C(p,4)≤140이다. R은 실제 반환된 유효 raw root 전부의 수이며 LM/refit alias가 아니다. 실제 inlier refit≤DM+R이고 전체 LOCAL optimizer 호출≤2DM+R≤4560+R이다. 반환 root 수를 임의로 cap하여 branch를 삭제하지 않는다. 실제 요청/호출/반환/exception, source bank cache 및 proposal/refit 중복 cache를 원행에 보존한다. 이 upper bound와 replay wall을 배포 latency로 주장하지 않는다.

CONTRADICTED line만 원래 V10 endpoint LOO 기준으로 제외하고 UNKNOWN은 유지한다. 실제 q에서 소비한 edge는 모든 부분집합/score/refit에서 제외하며, 제거한 point의 edge를 다시 관측으로 되살리지 않는다. H 초기 XY와 center는 모든 가설/score/fit에서 제외한다. NEW R,t가 승인되면 H를 한 번 재투영으로 교체하고 그 점을 다시 fit하지 않는다. 실패는 상위 조립기의 full original N3 fallback과 별도로 typed state로 기록한다. 숫자 NEW는 위치·회전 개선의 증명이 아니다.
