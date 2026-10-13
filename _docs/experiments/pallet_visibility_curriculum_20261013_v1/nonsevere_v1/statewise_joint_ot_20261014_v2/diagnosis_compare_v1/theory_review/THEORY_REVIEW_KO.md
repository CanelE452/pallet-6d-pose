# 영역 제약이 있어도 자세 오차가 줄지 않는 이유와 통제 진단

현재 v3는 TEST49에서 기존 PnP보다 평균 위치·회전 오차와 동시 성공 수가 좋지 않았다(3.514cm/6.568°/39 대 3.363cm/6.527°/40). 같은 보정 BI-only보다 좋은 점은 있어도, 이를 기존 방법 전반이나 새 unseen 데이터의 우월성으로 확대할 수 없다. 다음 설명은 원인 가설과 사전에 고정할 통제 실험이며, 새 원인 실험의 결과를 이미 확인했다는 뜻이 아니다.

영역이 정확하면 가능한 자세 F를 줄일 수 있다는 직관은 맞다. 그러나 원하는 결론은 `F가 작다`가 아니라 `선택한 자세가 실제 자세에 더 가깝다`이다. 두 결론 사이에는 상태·영역이 정답을 보존한다는 조건, 관측·모델의 정확성, 최적화 목적과 평가 척도의 일치, 검색·선택의 적절성이 필요하다.

현재 솔버의 목적 L은 **고정 BI 관측의 공분산으로 whiten한 영상 잔차의 radial pseudo-Huber 합**이다. O/T는 각 hard inequality로 들어가며 S/U는 목적에 없다. 최종 선택은 발견한 feasible pose 중 L이 가장 작은 것이다. 반면 평가는 physical translation/rotation/ADDsym를 사용한다. 영역을 추가해 `F₁⊂F₀`이면 전역 최적값은 `min_F₁ L ≥ min_F₀ L`이다. 따라서 영역이 BI fit을 더 좋게 만든다는 주장은 일반적으로 성립하지 않으며, 물리 metric 개선 가능성은 별개의 문제다. 현재 구현의 이 사실은 `scripts/research/pallet_visibility_curriculum_20261013_v1/joint_ot_pnp.py`의 `_objective`, `_constraint_values`, `solutions.sort`에서 확인했다.

정답을 포함하는 올바른 영역도 오차를 키울 수 있다. 정답 `(0,0)`, 관측 `(1,0)`, 목적 `100(x−1)²+y²`에서 제약 전 해는 `(1,0)`이고 Euclidean 정답 오차는1이다. 정답을 포함하는 convex 제약 `x≤y`를 추가하면 최적해는 `(100/101,100/101)`이고 물리 오차는 약1.4002로 커진다. 같은 W norm에서는 정답 오차가 줄지만 다른 평가 norm에서는 커진다. 또 nonconvex 가능집합 `{0,1.9}`에 정답0과 관측1이 있을 때 관측에 가장 가까운 feasible 값1.9가 오히려 정답에서 더 멀다. 재투영을 거친 SE3 가능집합은 각 2D 영역이 convex여도 일반적으로 convex가 아니다. 수치와 유도는 [MATH_COUNTEREXAMPLES.json](MATH_COUNTEREXAMPLES.json)에 남겼다.

더 강한 직관이 성립하는 특별한 경우도 있다. 정답이 들어 있는 closed convex 집합에 대해 **평가와 같은 Euclidean norm의 projection**을 수행하면 projection의 variational inequality로 정답까지의 거리가 증가하지 않는다. 현재 방법은 이 조건 전체를 만족하지 않는다. 이는 ‘영역은 도움이 안 된다’는 결론이 아니라 도움을 보장하는 가정이 무엇인지 구분하는 것이다.

현재 결과에서 먼저 의심할 수 있는 경로는 다음과 같다. 아직 원인으로 확정하지 않았다.

| 가설 | 통제 실험에서 예상되는 관측 |
|---|---|
| BI 보정의 모델 편향 | 같은 Σ에서 raw BI가 corrected BI보다 physical metric/reference 잔차가 좋다. 실제 경계·rounded corner의 pixel 교점이 가상 cuboid vertex와 다를 수 있다. |
| false O/T 또는 틀린 영역 중심 | known-false domain만 제거하거나 reference-center O를 넣을 때 reference가 feasible로 바뀌고 paired metric이 좋아진다. |
| BI hard budget·기하 가정이 정답을 배제 | reference가 영역을 보기 전에 Maha3/nonself/canvas에서 탈락한다. perfect O나 true start만으로 복구할 수 없다. |
| 상태 판정 오류 | 같은 raw q·standard Σ control에 비해 oracle binary/full-human raw state가 좋아진다. initial S/T와 gate 유지 human-known D/E 교정과 나머지 label abstention의 지원점 선택 효과를 구분한다. |
| local search | reference가 feasible이고 reference start 추가가 원 multistart가 놓친 더 작은 목적의 feasible 해를 찾는다. |
| 목적·선택 mismatch | 현재 feasible 후보 중 더 좋은 physical pose가 있어도 작은 visible-loss 후보가 선택된다. 같은 후보에서 nearest-initial selector를 바꾸면 오차가 달라진다. |
| native/reference 대응 문제 | annotation q oracle와 model-projected q oracle가 다른 residual/복구 결과를 보이고, 기존 incompatible native phase에 모순이 집중된다. |
| 정보 부족 | BI2의 residual Jacobian은 4×6이므로 rank≤4이다. broad/inactive O/T는 두 관측만으로 생긴 자유도를 없애지 못하며 가족을 정밀 단일 해로 표현할 수 없다. |

가장 작은 유용한 factorial은 **raw/corrected BI × OT OFF/ON**이다. 같은 현재 state/native/K, 동일 covariance, 동일 model guards와 시작점·선택을 사용해 q 보정과 영역 및 interaction을 분리한다. raw arm에 corrected Σ를 유지하는 것은 ‘q만 변경’하는 인위적 통제 조건이며 raw 측정의 실제 noise가 보정됐다는 뜻은 아니다. BI2의 NO_OT는 원래 미해결·fallback이므로 그 구간은 가족 사례로 따로 기록한다.

그 다음 oracle를 비교할 기준을 맞춘다. ORACLE_BINARY_RAW는 initial S/T를 유지하되 human SELF/UNANNOTATED도 U로 보류하는 controlled support-selection oracle이며, pure D/E classifier-only 인과 효과가 아니다. ORACLE_BINARY_RAW/FULL_HUMAN_RAW는 predeclared B sigma2/I sigma3를 사용하는 **CURRENT_RAW_STANDARD_COV**와 비교해야 state 효과에 noise weighting을 섞지 않는다. full-human raw는 RGB 교점을 다시 측정한 실험이 아니므로 ‘human-state/raw-query structural oracle’라고 부른다. DROP_KNOWN_FALSE_OT는 대응 state를 U로 바꾸고 해당 domain만 삭제한다. BI를 새로 추가하지 않으며 UNANNOTATED를 false로 판정하지 않는다. BUDGET6는 Σ와 L을 고정한 hard-cap 민감도이며 새 calibration이나 현재 방법 변경으로 표현하지 않는다.

검색과 선택도 분리한다. CURRENT_REF_INIT는 같은 F/L에 reference start만 추가한다. reference가 infeasible이면 실패를 local-search 탓으로 해석하지 않는다. reference가 feasible해도 최적화가 다른 더 작은 L의 해로 이동할 수 있으므로 initial/reference/final의 목적·constraint slack·physical metric을 모두 저장한다. nearest-initial selector는 **같은 저장 feasible samples**에서만 비교해 검색 효과와 섞지 않는다. 고정된 dimensionless SO3 거리와 translation/model scale을 쓰되 이것은 다른 진단 estimator이고 새 학습·TEST 튜닝 없이 선언한다. 유한 samples에서 더 좋은 해를 못 찾은 것은 전역 존재·부존재의 증명이 아니다. [SciPy의 SLSQP 설명](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)은 termination과 목적·제약 최적화의 의미를 설명하며 pose accuracy 보장을 제공하지 않는다. [OpenCV PnP 문서](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)는 reprojection geometry와 복수 P3P 해를 설명한다. PnP initializer의 성질을 joint OT 전체의 전역 최적성으로 옮길 수 없다.

reference를 진짜 독립 물리 GT로 부르지 않는다. 기존 pose target은 historical derived reference다. 기존 `_model_compatible` 규칙(gap<1e-6)으로 CAL44/46, TEST42/49만 native pixel 대응이 호환된다. 나머지9건은 annotation/reference-center oracle의 원인 해석에서 eligibility를 별도 표시해야 한다. 전체95 pose metric은 유지한다. physical basis는 `Q=R_cf_originalᵀR_physical_original`, `R_ref_cf=R_ref_physical Qᵀ`이다. 원 native phase·치수·mapping을 고정하고 허용 physical yaw symmetry를 고려해 단일 phase를 인위적으로 infeasible로 판정하지 않는다. GT를 보고 prediction permutation이나 가장 성능 좋은 branch를 선택하면 oracle diagnostic 이외의 정상 비교가 아니다.

정확한 annotation q와 `π(nativeX,reference pose)`도 구분한다. 전자는 rounding·실제 모서리/가상 cuboid 불일치가 남을 수 있다. 후자는 모델이 자기 일관적일 때 동작하는지 보여주는 MODEL_IDEAL이며 실제 RGB 상태 판정이나 실제 O/T 위치 복원 성공을 증명하지 않는다. 모든 GT 입력 arm은 prediction/deployment arm과 별도로 ORACLE_DIAGNOSTIC임을 기록하고 source·input·protocol을 봉인한 후 비교한다. TEST를 이미 여러 차례 읽었으므로 이 진단을 독립 unseen 평가로 표현하지 않는다.

실제 uncertainty도 보수적으로 해석한다. O radius는 eligible CAL7점/2sessions의 normalized empirical max이다. 이상적인 exchangeable independent 연속 residual7개에서도 새 값이 그 max 아래일 확률은 순위 계산상7/8이며95%가 아니다. 현재 데이터의 선택·상관과 여러 O/T의 simultaneous coverage는 그 이상적인 가정도 만족한다고 보장되지 않는다. 공유 raw q·RGB·coarse geometry 때문에 BI/O/state 오류가 독립이라는 가정도 없다.

현재 OTHER-method 비교에서는 동일95영상·평가 metric을 맞추더라도 upstream을 구분한다. literal PoseFix RGB ResNet152와 current N3 learned head는 같지 않으며 PoseFix/Sub는 각 q에서 W/D family를 다시 정하고 current v3는 원 N3 family/Q를 유지한다. physical R와 고정 registry symmetry로 비교해야 basis confound를 피한다. 그 benchmark는 matched full-pipeline 비교이며 region postprocessor 하나의 효과는 위 통제 실험으로 판정한다.

기존 사후 산술 진단도 먼저 검토했다. native 호환86건에서 reference의 BI support 통과14, OT 통과79, 모두 통과12이다. 72건은 BI 지원 조건에 의해 이미 배제된다. 이는 조건의 불일치가 동시 발생함을 보여주며 covariance·state·model·reference 중 원인을 확정하지 않는다. [STAGED_EVIDENCE_REVIEW_V1.json](STAGED_EVIDENCE_REVIEW_V1.json)에 phase 정합과 작은 tolerance 관례 차이의 현재 영향0을 기록했다.

여러 hard gate가 동시에 reference를 배제하면 하나를 고친 arm이 개선되지 않았다는 결과만으로 그 요인이 원인이 아니라고 결론낼 수 없다. BI만 위반/OT만 위반/둘 다/통과의 교차집계를 함께 본다. 영역 제약이 inactive여도 O 판정 때문에 사라진 BI 정보는 OT OFF에도 복원되지 않는다. 또 근접한 objective의 대안만 ambiguity로 표시하는 현재 정책에서 ambiguity=False는 다른 feasible pose가 없다는 뜻이 아니다.

고정된 root plan의 exact arm 이름과 통제 조건은 [FIXED_PLAN_REVIEW_V1.json](FIXED_PLAN_REVIEW_V1.json)에 확인했다. plan의 nearest-coarse selector는 모든8 모델 투영 RMS를 초기 model bboxdiag로 정규화한다. 이는 위의 일반 SO3/translation prior 예시와 다른 고정 진단 규칙이며, S raw나 GT를 사용하지 않는다는 점과 현재 selector와 다른 점을 구분한다. 구현 소스가 준비되면 control·phase·fallback·GT flag를 별도로 확인한다.

선언된 arm·예측·필수 출력과 코드/공식 문서 근거는 [PROTOCOL_REVIEW_V1.json](PROTOCOL_REVIEW_V1.json)에 있다. 리뷰는 frozen core/model/source를 수정하거나 새 inference를 실행하지 않았고, severe/FINAL를 읽지 않았다. 새 결과가 나오면 어떤 가설이 지지되었는지와 아직 분리되지 않은 요인을 따로 갱신해야 한다.

현재 실행 driver 검토와 checkpoint I/O의 의미 불변 보충 기록은 [DRIVER_REVIEW_V1_KO.md](DRIVER_REVIEW_V1_KO.md)에 있다. 입력 감사는 새 solver fitting 없이 95×11arm·5,958 확인 실패0이며, 원 source와 실행 source의 I/O-only 차이는 byte reconstruction으로 확인했다.
