# A: 전역 물리 owner 대안과 관측별 none

## 변경과 중복 확인

A는 같은 영상점·부분 선 관측을 다른 **물리 native 꼭짓점/모서리**에 연결하는 전역 대안을 평가한다. 물리적 corner owner는 독립 확인되지 않았으므로 네 대안을 정답 후보가 아니라 조건부 가설로 취급한다. 원 상태·관측·covariance·reliability를 보존하고 기존 CAL adapter를 다시 적용하지 않는다.

기존 `mixture_native_edge_factor_v4_18.py`는 모든 component의 `(native_corner,native_neighbor)` 집합이 동일해야 한다고 직접 검사한다. 서로 다른 영상 TLS 선 대안이 같은 native edge에 속하는 혼합이다. `joint_geometry_guided_lines_v4_28.py`의 node는 원 native IDs로 생성·grouping되며 graph는 그 이름이 고정된 edge마다 영상선을 고른다. 후속 `observed_bi_line_association_v4_30.py`, `observed_anchor_native_ray_association_v4_31.py`, `initial_only_observed_ray_association_v4_32.py`는 같은 고정 native IDs의 관측 admission/방향/출처를 변경한다. v4_31/v4_32는 해당 edge와 q-anchor의 원 ancestry까지 고정한다. 이 경로들에는 한 관측을 다른 native owner에 연결하는 공통 전역 permutation이 없다. 기존 W/D 두 family가 있다는 사실도 같은 family 안에서 네 전역 owner를 혼합하는 A와 구분한다. 기존 기능을 새 이름으로 재실행하지 않는다.

## 정확한 전역 가설

원8개 `xyz`를 중심/반치수로 정규화한 부호 좌표에서 y를 보존하는 회전 `C=(x,y,z)→(z,y,−x)`의 네 거듭제곱만 허용한다. `p_h[k]`는 **원 관측 채널 k의 owner가 원 좌표 배열의 어느 실제 model vertex인가**를 뜻한다. 좌표·치수·Q를 회전/확대하거나 GT의 허용 대칭을 늘리지 않는다. 직사각형90° 대응은 여전히 평가에서 회전 오류다.

모든 p는 일대일이며 원 cuboid adjacency/면 순서/높이 방향과 오른손 orientation을 보존한다. 반사, 독립적인 점별 swap, 위아래를 뒤집는 permutation은 거부한다. 한 h가 full point, partial normal, incident edge, O/T/U domain과 state vector를 **동시에** 운반한다. 가설마다 상태를 S로 바꿔 관측을 삭제하거나 서로 불가능한 점 owner들을 독립 선택하지 않는다.

**가설 공간의 실제 중복:** 참조를 읽지 않고 CAL 한 장의 등록 `xyz@Q`를 per-index 비교했다. 두 W/D family×네 h의8조합은 정확한 물리 owner 배열4개이며 기존 C2를 묶으면 **2개**다. 예를 들어 ORIGINAL/h0=SWAPPED/h1이고 ORIGINAL/h0와h2는 C2 관계다. [기하 동등성 장부](A_OWNER_EQUIVALENCE_AUDIT.json)에 전 조합을 남겼다. 따라서 새로운 독립 물리 가설8개를 만들었다거나 기존 W/D+heading/C2보다 넓은 기하 공간을 발명했다고 주장하지 않는다. 실제 변경은 같은 전역 대응 공간에서의 normalized mixture+none 및 공통 G13 재채점이다. 같은 edge 안의 영상선 혼합과도, 기존 hard hierarchical family 선택과도 계산이 다르지만 새 물리 대칭 자체가 아니다. 이는 기하 검산이며 성능을 본 뒤 모델을 고친 것이 아니다.

## 원 손실과 정규화

원 E0의 `huber_scale=3`은 코드상 radial **pseudo-Huber/soft-L1**다. piecewise Huber로 바꾸지 않았다.

`ρδ(r)=2δ²(√(1+r²/δ²)−1)`

원 point의 유효 covariance `Σ`와 partial의 normal variance를 그대로 사용한다. 같은 partial corner의 법선들은 하나의 radial group이다. 실제 선 표본 수나 동일 line 복제로 group 정보량을 늘리지 않는다. partial tangent interval은 원 hard constraint로 남으며, partial을 정밀2D point로 승격하지 않는다.

d차 whitened radial density의 정규화는 다음과 같다.

`Z_d=2(2π)^((d−1)/2) δ exp(δ²) K_((d+1)/2)(δ²)`

2차에서는 `Z_2=2π(1+1/δ²)`다. 1차/고차는 `scipy.special.kve`로 안정적으로 계산한다. 단위는 full의 px² covariance, partial normal의 px² variance이며 density의 log-volume을 빠뜨리지 않는다. 원의 동일 RGB 상관을 해결한 calibrated likelihood라는 뜻은 아니다.

관측 g는 다음 inlier/none 혼합을 갖는다.

`f_g(r) = exp(−ρδ(||Σ_g^−1/2 r||)/2) / (Z_d √detΣ_g)`

`m_g = (1−η) f_g + η/V_g`

`E_g = −2 w_g [log m_g − log m_g(r=0)]`

`w_g`는 원 reliability이며 visibility confidence를 owner probability로 복사하지 않는다. `V_g`는 후보와 h에 독립인 고정 pixel-domain 부피다. full은 원 canvas `[0,W]×[0,H]` 및 해당 고정 관측을 포함하는 사각형, partial은 같은 canvas/관측의 법선좌표 지원 범위 곱이다. 이 none branch는 고정 관측이 어느 model owner에도 속하지 않을 가능성을 표현한다. `η`는 API의 필수 인자이며 숨은 기본값이 없다. 허용 합성 validation에서 사전 목록 `.01/.05/.1`만 검증하고 PRIMARY 계산 전에 고정한다. 시험의 `.05`는 기술 검산 값이지 실사 확률 교정 결과가 아니다.

영점 density를 뺀 항은 h/pose에 독립이므로 비교를 바꾸지 않는다. η→0이면 원 `w_g ρδ(r)`와 같다. 큰 residual에서 none이 지배하더라도 비용은 양의 고정 상수로 남는다. 관측이 공짜로 사라지는0점 해가 아니다. 모두 none이면 pose 정보가 부족하다는 진단을 남기며 안전한 pose라는 뜻이 아니다.

## 전역 합성과 A+B

중복 제거된 네 h에 균등 prior를 주고 **전체 관측을 설명한 후** 혼합한다.

`E_A(T)=−2 log Σ_h π_h exp(−Σ_g E_g(T,p_h)/2)`

관측마다 h를 따로 고르는 `Σ_g min_h E_g`가 아니다. 동일 h 복제로 prior가 증가하지 않으며 관측 duplicate도 단일 group으로 유지한다. hard constraint를 위반한 h는 `+∞`를 넣고 나머지 prior를 다시 정규화하지 않는다. h를 더 삭제하면 무료 likelihood 이득을 얻을 수 없다.

AB에서 B가 대체한 original group IDs는 arm 전체에서 한 번 고정한다. 동일한 제외 집합을 모든 h에 적용하고, `B(T,p_h)`를 해당 h의 합에 더한 다음 위 global log-mixture를 계산한다. B와 original full/partial/outline을 아무 설명 없이 중복 합산하지 않는다. h별로 다른 image support를 빼는 API는 지원하지 않는다.

## API와 검산

`owner_model.compile_owner_groups(ctx,none_prior=η)` 또는 `xyz,K,measurements,boundary,image_hw` primitive 입력으로 compile한다. group ID는 `full:k`, `partial:k`다. 원 Context에 남은 revoked full archive는 partial owner 집합으로 한 번 철회한다.

- `energies_per_global_h(model,projected8xy,return_audit=True)` → h별4개 energy, 4×8×2 projection gradient, group/none audit.
- `energy(model,projected8xy,return_gradient=True,return_audit=True)` → global energy, 8×2 gradient, owner responsibilities.
- `exclude_group_ids=...`, `extra_h_energies=...`는 AB replacement와 owner별 hard constraints를 global mix 전에 연결한다.
- `map_owner_records(...)`는 domains/partial tangent/neighbor IDs와 원 state vector를 같이 운반한다. 준비된 `object_xyz`/`incident_normals`도 새 owner에서 다시 계산한다. immutable producer provenance는 고치지 않는다.

14개 합성 geometry 테스트가 실제 다른 global owner의 선택, proper adjacency/반사 금지, duplicate hypothesis/measurement 불변, analytic normalizer의 수치적분 일치, none 양의 비용, 공통 관측 수, partial 철회/접선 불확실성, full/partial projection gradient finite difference, AB의 전역 혼합 순서, no-input 항등성, 해상도 변환, domain/normal 운반을 검사한다. 이 시험은 RGB renderer/실사 owner 정답/새192 성능 검증이 아니다. CAL46 compile-only 점검은 원 활성 관측 집합의 보존을 확인하며 참조/새 pose score를 읽지 않는다. 실제 fixed-bank/full-method 결과는 campaign의 봉인 평가에서 별도 보고한다.

## 선행 원리와 실제 차이·라이선스

[EPOS 저자 공개본](https://arxiv.org/pdf/2004.00605) §3.2는 object 존재와 fragment 대응을 분리하고, §3.3은 한 pixel의 대안들을 별개의 지지 수로 합산하지 않는다. 여기서는 이 분리 원리만 사용했다. sparse cuboid/기존 상태/네 전역 graph permutation이며 EPOS의 dense fragment network·Progressive-X를 재현하지 않는다. 공식 [EPOS](https://github.com/thodan/epos/tree/d7c19eb5168f9d47bc7976cca12bfc2a7344d7b8) head는 확인 시 `d7c19eb5168f9d47bc7976cca12bfc2a7344d7b8`, [라이선스](https://raw.githubusercontent.com/thodan/epos/d7c19eb5168f9d47bc7976cca12bfc2a7344d7b8/LICENSE)는 MIT다. 외부 코드는 복사·설치하지 않았다. CVF 페이지/PDF는403으로 접근하지 못해 저자 arXiv v1과 공식 project를 읽었다.

[Olson–Agarwal RSS2012](https://roboticsproceedings.org/rss08/p40.pdf) §III의 mixture와 §IV-A의 null 가설이 대응 실패를 별도 모드로 두는 근거다. 원문은 효율적 max-mixture/SLAM이고 여기서는 작은 전역 sum-mixture와 uniform none을 사용한다. 원문의 null 수치나 결과를 가져오지 않았고 원 solver를 이식하지 않았다. 해당 논문으로부터 구현 코드를 가져오지 않았으므로 제3자 source license를 임의로 추정하지 않는다. 논문의 방법 재현·수치적 보장·새 원리의 최초 발명이라고 주장하지 않는다.
