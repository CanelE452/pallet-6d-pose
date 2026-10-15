# Frozen V29 finite·directed support guard 소스 검토

현재 V29/V30 구현·모델·PLAN은 바꾸지 않았다. 소스만 읽었으며 실제 e 자세·새 점수·GT·관측 저널·영상·새 실험은 읽거나 실행하지 않았다.

**결론:** 읽은 수식에서 명백한 국소 부호·단위·Jacobian 오류는 찾지 못했다. runtime과 독립 scalar kernel이 같은 조건을 구현한다. 다만 이 조건은 **선택 체인의 모든 inlier가 하나의 올바른 named finite physical edge에 속한다**는 가정을 hard constraint로 쓴다. 실제 픽셀/TLS provenance가 진짜라는 사실은 그 가정의 참을 보장하지 않는다. V30의 법선 43px 대응 검사는 선 방향 대응과 끝점 순서를 보장하지 않는다.

## 정확한 식과 지지 단위

TLS 단위 법선 n, 접선 u=(−n_y,n_x), 실제 inlier 집합 A에 대해 a=min(u·A), b=max(u·A)다. 모델 투영 끝점의 접선 좌표를 s_min=min(u·q_k,u·q_m), s_max=max(…)라 하면 다음 두 slack을 모두 ≥−feasibility_tolerance로 제한한다.

```text
finite_left  = a − s_min + 11
finite_right = s_max − b + 11
```

이는 관측 구간 전체 [a,b]가 모델 선분의 접선 투영을 양쪽 11px 확장한 안에 들어가야 한다는 조건이다. 단순 overlap도, 모델 선분 전체가 영상에서 보여야 한다는 조건도 아니다. 관측 내부가 가려져도 min/max만 쓰며, 극단의 잘못된 inlier 하나가 조건을 바꿀 수 있다. min/max 사이에 가짜 실제 픽셀을 만들지는 않는다.

각 활성 B/I owner h의 고정 sign ε에 대해 z_i=εu·A_i, m=ceil(0.6N−1e−12), c를 m번째 큰 z 값이라 하면 다음도 hard 조건이다.

```text
directed_support = c + 1 − εu·q_h
projected_order  = εu·(q_other − q_h)
```

첫 식은 실제점 최소 60%가 h→mate 쪽에 있고 1px 역방향을 허용한다는 뜻이다. sign=None이면 이 두 방향 조건만 생략하고 finite 조건은 남는다. 현재 pose를 보고 유리한 sign을 고르지 않는다.

구현은 `geometry_guided_edge_problem_v4_29.py:224–267`, 독립 식은 `independent_geometry_edge_energy_v4_29.py:67–98`이다. finite Jacobian은 선택된 min/max 끝점의 ∓uᵀJ, directed는 −εuᵀJ_h와 εuᵀ(J_other−J_h)다. runtime 정렬과 독립 quantile은 동치다. min/max 끝점이 바뀌는 곳은 piecewise 미분의 분기점이다. 기존 FD/끝점 뒤집힘 fixture는 소스로 확인했으며 이번에 재실행하지 않았다.

11px은 CAL 확률·pose 오차 분위수가 아니라 기존 `along_start=4`와 `boundary_context=7`을 합친 engineering 여유다(`visible_bi_correction_v2.py:19–24`). 같은 숫자가 component 연결 간격, 최소 실제 span, 교점 extrapolation, rounded 진단의 점 제외 반경에도 쓰이지만 역할은 다르다.

체인은 실제 NMS의 8연결 component 또는 endpoint 거리≤11px·공동 TLS/RMS/각도 조건을 통과한 component 경로다. bridge 점은 만들지 않는다. 이후 IRLS inlier 배열이 AUTH의 `actual_inlier_xy`가 된다. 원 component IDs와 gap 장부는 raw chain에 남지만 consumer는 그 분할·연결 관계를 다시 판단하지 않고 한 배열의 min/max와 quantile을 쓴다(`image_supported_edge_chains_v4_17c.py:95–165,250–307`; AUTH `:22–105`).

FULL/partial 모두 같은 pose guard를 받는다. fresh partial은 span 비율 .55만 미달할 수 있고 count6/fraction.6/RMS1.25/생성 guide 각12°·실제 span≥11 및 RGB/lattice provenance는 유지한다. partial이라고 endpoint uncertainty를 넓히지는 않는다.

현재 consumer는 **원래 전체 실제 지지점**을 쓴다. rounded junction에서 가까운 11px 점을 빼고 재적합한 진단 구간은 factor 입력이 아니다(`image_edge_observations_v4_29.py:264–284`). 점 제거 후의 gap을 원 지지 부재로 읽는 문제와 현재 pose guard를 혼동하면 안 된다.

## 이름·owner·순서의 가정

AUTH는 graph winner 한 선/physical edge만 쓴다. 다른 실제 대안은 보관되지만 현재 자세 문제의 latent 선택지가 아니다. 다른 native edge가 같은 lattice 지지를 재사용하면 양쪽 요인을 보류한다. 양끝 B/I가 owner일 수 있지만 반대 끝은 `MODEL_CONSTRAINT_ONLY`이며 새 영상 점이 아니다. 새 owner의 옛 partial locality guard를 제거하고 위 조건을 넣는다. O/T/U·depth·canvas·기존 visible geometry는 원 primitive를 유지한다.

sign은 양끝이 B/I이면 보관 q_m−q_h의 접선 hemisphere, 그렇지 않으면 **선택된 generating guide 방향**에서 온다(AUTH `:108–123`). 후자는 모든 다른 guide와의 일치나 유일성 증명이 아니다. TLS 각도≤12° 결합 때문에 정상 unknown-mate 선에서는 |u·guide|≈1이므로 sign=None의 수치 조건 |dot|≤1e−12가 거의 작동하지 않는다. graph는 B/I 사이 거리가 10px 이상일 때 측정 hemisphere를 쓰지만 AUTH는 별도 10px 기준 없이 dot 부호를 고정한다(`ordered_geometry_support_v4_28b.py:41–77`). 이는 바로 산술 bug라는 증거가 아니라, 불확실한 q/guide 순서를 hard 사실로 올리는 가정이다.

rank2 q 제거는 두 실제 선의 sine≥.25, 원 지지 밖 교점 gap≤11, canvas와 두 선의 directed fraction≥.6을 요구한다. 한 선이면 실제 선택선 접선의 archived q 비용을 남긴다. 이 사전 pair 검증과 현재 pose guard는 다른 단계다.

두 WD는 각 family의 같은 ID xyz[k],xyz[m]를 투영한다. Q/물리 box만 같게 운반해도 native channel이 달라질 수 있어 named-edge·순서 guard가 같은 unordered box에 대해 불변일 필요는 없다. 그 family의 대응을 영상이 지지한다는 근거가 필요하다. 이번 소스 검토로 어느 실제 frame의 대응이 정답인지 판단하지 않는다.

## edge_visibility가 실제로 제한하는 것

`classify_expected_edge`는 초기 모델 선분을 샘플링하고 실제 chain과의 거리≤기존 context7px 및 target-mask 문맥으로 `OBSERVED_SUPPORT / OCCLUDER_SUPPORTED_GAP / UNSUPPORTED_UNKNOWN`을 기록한다. native identity 인증이 아니고 target mask 부재만으로 가림을 확정하지 않는다(`edge_visibility_v4_28.py:196–272`).

이 분류는 **graph winner 선정 뒤** 호출되어 `edge_visibility`에 저장된다(`image_edge_observations_v4_29.py:361–381`). V29/V30 AUTH·graph score·pose factor는 그 counts/classes를 읽지 않는다. V30 재순위도 새 winner를 정한 뒤 분류를 갱신한다. 따라서 이 메타데이터의 OBSERVED_SUPPORT가 0이어도 그것만으로 선정·가중치·finite guard가 바뀌지 않는다.

반면 같은 context의 `query_allowed`는 실제 prequery/historical-support gate다. S/non-BI, canvas, U/O/T mate 주변 14px+probe footprint와 독립 occluder 팽창을 제외한다. 이 predicate 자체는 target-mask 값을 보지 않는다(`edge_visibility_v4_28.py:141–194`, graph `:494–500`). **분류 메타데이터와 query 허용 gate는 구분해야 한다.** 초기 pose가 틀리면 초기 선분 coverage를 바로 hard gate로 바꾸는 것도 맞는 선을 제외할 수 있어, 현재 추가하지 않는다.

## 산술 오류와 모델링 문제를 구별하는 근거

법선 부호·고정 owner sign·끝점 순서·quantile 식은 runtime/독립 kernel이 일치하고 runtime은 scalar/vector guard 일치도 검사한다. 현재 읽기로 명백한 국소 산술 bug는 특정하지 못했다. 이 일치는 실제 physical association의 정확성 증명이 아니다.

CAL σ_normal/σ_sine·접선 γ·reliability·pseudo-Huber는 **hard support slack에 들어가지 않는다**. covariance가 커도 한 slack 위반이면 infeasible이다. 현재 CAL 모델은 endpoint coverage/order 실패나 named-edge outlier 확률을 적합하지 않았다. normal σ를 11px 접선 여유의 통계 근거로 쓸 수 없다.

분석적 예로 같은 무한선에서 모델 접선 구간 [0,10], 잘못 연결한 지지 구간 [100,120]이면 법선 거리는 모두 0이지만 finite slack은 111,−99다. 축 부호를 바꿔도 결과는 같다. 이는 새로운 실험이나 특정 frame 재평가가 아니다. Root가 별도 저장 PROCESS만으로 보고한 noapril 변0–3 사례도 작은 법선 거리와 큰 along-gap이 공존한다는 참고 근거다. 해당 원 row·V30 현재 선택 여부는 여기서 읽거나 판단하지 않았다.

graph에는 이미 `None` 미선택 후보가 있다(`joint_geometry_guided_lines_v4_28.py:529–545`). 그러나 winner 이후 pose optimizer에는 null/다른 chain으로 돌아가 원 q 비용을 복원하는 단계가 없다. ‘null 도입’은 graph에 None을 처음 넣는 것이 아니라 pose/association 계층과 관측 복원 장부를 함께 다루는 제안이다.

## 현재 적용하지 않는 후속안

| 순서 | 구체 수정 위치·내용 | CAL·검증 조건 |
|---|---|---|
| 1. 지지 분할·연결 검증 | NMS `_collinear_component_hypotheses`→AUTH `_component`에 component IDs, IRLS 후 gap/연결·foreground/unknown/occluder 구간을 바인딩. 서로 다른 collinear 구간을 하나의 finite edge 극값으로 강제하지 않는 명시적 정책. bridge 점 생성 금지. | 픽셀 재사용이어도 선정 모집단이 바뀌면 CAL46·정책별 새 noise 필요. 올바른 부분/가림 선을 보존하는 CAL 사례와 병렬 다른 경계·내부 opening·분절 outlier·zero-support를 함께 검증. TEST/e 오차로 임계값 적합 금지. |
| 2. null/outlier 잠재 대응 | AUTH→`plan_geometry_guided_replacement`→objective/constraints에서 후보 또는 null을 일관되게 평가. null이면 단순 새 선 삭제가 아니라 v26 nonfallback q2D 또는 fallback legacy normal+T4와 옛 guard를 복원. shared owner/edge 1회, rank2는 같은 assignment의 두 선에서만 q 제거. | winner-conditioned 기존 σ/γ 재사용을 검증된 latent 모델로 표현하지 않음. 새 CAL population/NLL·normalizer·null mass·4LOGO 필요. 공짜 전체-null 방지, empty-new-evidence exact identity·중복0·전체238 fallback 보존 검증. 이전 V18 mixture와 구분하고 성공 가정 금지. |
| 3. 불확실성에 맞춘 soft support | `_actual_support_constraints`의 max(0,−slack)/σ_support를 별도 robust/bounded 항으로 두는 새 control. orientation hard 여부도 order 신뢰도와 분리. O/T/U/depth/canvas 물리·범위 guard는 유지. | σ_support는 접선 endpoint/ordered-quantile 오차로 CAL-only 적합하거나 명시적 engineering control. normal σ와 구분·q/guide 상관 명시. CAL 부족은 미입증/보류. FD·분기·부호/끝점/Q 대응·장부1회·0weight exact baseline·full238/원49 보존 검증. |

V30 전체 결과가 나빠도 자동 적용하지 않는다. 새 source·사전 계획·CAL 근거·동일 전체 평가가 별도로 필요하며 e나 GT를 runtime 후보·분기·임계값 선택에 넣지 않는다.
