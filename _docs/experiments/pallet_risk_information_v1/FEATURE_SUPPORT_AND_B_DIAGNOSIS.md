# F1 고정 자세 특징과 별도 B 정답 대입 진단

이 문서는 이번 실험의 구현 계약이다. 실제 정확도 향상이나 물리 대응 인증을 뜻하지 않는다. 기존 D1 관측·상태·후보·eligibility와 F0 188차원은 보존한다. 새 PnP·수치 자세 최적화·후보 생성·본체 forward는 수행하지 않는다.

`feature_support.py`는 원 `action_data.actions`의 실제 고유 행동을 같은 순서로 읽는다. N0/E0 alias와 exact numerical duplicate 규약도 그대로다. 행동 ID·물리 자세·alias를 별도 SHA로 묶지만 모델에는 추가 scalar/validity 배열만 전달한다. 원 후보의 허용 여부를 finite 검사로 바꾸지 않는다.

양 family 순서는 원 registry의 `ORIGINAL_WD`, `SWAPPED_WD`다. 같은 물리 자세마다 `R_native = R_physical @ Q.T`, `t_native = centroid - R_native @ mean(native_xyz)`를 원 변환 함수로 적용한다. 한 family만 점수로 골라 버리지 않는다. 원 fixed 비용 선호 family는 별도 bit로만 보존한다.

각 family 72 scalar는 다음과 같다. 총 M=144이고 추가 입력은 값144+validity144=288, 전체476차원이다. 476→64→64→1 모델은 34,753개 파라미터다. 추가 pooling은 없다.

| 추가 정보 | family당 scalar 수 | 의미 |
|---|---:|---|
| 원 fixed/A/B/AB/edge 비용 | 5 | 동일한 G13 고정 자세 재평가; 서로 다른 원 G10/G13 값을 빼지 않음 |
| full/partial/outline 비용 성분 | 3 | 원 fixed audit의 가중 성분; 이 셋의 합이 prior 포함 전체 비용이라는 주장은 하지 않음 |
| 원 제약 minimum slack | 11 | fixed1/A4/B1/AB4/edge1의 원 h별 값 |
| 원 feasible 표시 | 5 | 원 numerical tolerance 그대로 |
| full/partial/outline/O/T/U 관측 개수·가용성 | 7 | 실제 원 컨텍스트에서 계산 |
| full/partial 잔차 및 covariance 요약 | 5 | px 또는 표준편차 단위; 관측 없으면 결측 |
| full 출처 flag별 개수·비율 | 8 | image measurement, pose projection, backbone fallback, unknown source |
| B/I/O/T/S/U 원 상태 개수 | 6 | 초기 예측으로 고정된 상태; 행동별 재분류 없음 |
| 원 fixed 선호 family | 1 | 정보 보존용 bit |
| B bundle/profile/실제 지지/관측 없음 | 7 | 0 profile이면 비율·전체지지 validity=0 |
| finite 거부 원인 | 6 | 전단/후단/평행/투영퇴화/깊이/비유한 |
| numeric B 비용이 있지만 원 profile 지지 부족 | 1 | 임의의 ‘낮음’ 문턱 없이 비용 자체와 함께 제공 |
| 원 B가 소비한 코너 개수 | 1 | 원 bundle 그대로 |
| partial 출처 개수·비율 | 6 | 실제 association provenance만 사용 |

비용과 scalar는 기존 F0의 `finite and abs(value)<1e15` sentinel 규약을 따른다. 결측 수치는 0과 validity=0으로 보관하고 정상화에서 제외한다. partial provenance에 실제 RGB line ID가 있으면 image 관측 근거로 쓸 수 있지만, 화면에 그린 projected coordinate만으로 관측을 만들지 않는다. 출처 flag가 저장되지 않았다면 그 개수/비율도 unknown이다.

추가 normalizer는 원 train768의 실제 행동 중 유효 scalar에만 `sign(x)*log1p(abs(x))`를 적용하고 population mean/std를 계산한다. 분산이 원 규약의 작은 수 기준을 넘지 않으면 scale1을 쓴다. validation·실사·CAL은 fit에 들어가지 않는다. clipping은 없다. train에서 한 번도 유효하지 않은 scalar의 정상화 값은 0으로 두고 원 validity를 보존한다. ZERO 대조는 추가288 전체를 모델에서0으로 만들며 F0는 그대로다.

finite 판정은 이전 `finite_boundary.py`의 고정 원 profile universe와 machine precision 기반 판정을 그대로 호출한다. 코너-이웃 선분과 원 법선 교점의 λ가 실제 선분 안에 있는지 기록한다. 기존 무한 직선 B 비용은 수정하거나0으로 지우지 않는다. 유한 지지는 실제 material owner의 증명이 아니다. profile이0이면 관측 없음이며, 원 helper의 vacuous True를 F1 신뢰값으로 넣지 않는다.

원 비용·제약의 scalar/derivative evaluator만 호출한다. 기존 B/AB/edge의 minimum slack는 원 `candidate_scores`가 사용한 consumed-B replacement package를 가리키고, fixed/A는 원 package를 유지한다. 이를 새 finite 수리 solver의 제약으로 부르지 않는다. 실행 tripwire는 로드된 optimizer/PnP/solve/generator alias와 일반 NN call을 차단한다. 예측 I/O audit은 원 특징·E_RUN·bundle의 명시한 파일과 source/environment 읽기만 허용하고 참조 경로를 거부한다. 이는 Python 계측이며 OS sandbox 인증은 아니다. SciPy Qhull의 진단 tempfile은 새 출력 디렉터리 아래에만 쓴다.

`reference_support.py`는 별도 프로세스·`b_reference_*` private 디렉터리에서만 합성 정답을 읽는다. F1/학습 모듈은 이 모듈이나 정답 객체를 받지 않는다. 원1,003장과768/235 split을 ID로 연결하고 원 행의 C1/C2만 평가한다. 원 phase0과 허용 phase 중 하나라도 성립하는지를 구분하며 직사각형90°는 추가하지 않는다. 대입 자세는 진단 reference일 뿐 production candidate로 저장하거나 bank에 합치지 않는다.

각 phase/family에서 원 full 관측의 soft 잔차와 실제 활성/partial 대체 여부, partial tangent·법선, O/T/U 원 제약과 출처, outline의 원 soft 성분, finite profile의 개별 교점·λ·거부 원인을 보존한다. soft 잔차가 크다는 이유만으로 존재하지 않는 hard residual 예산을 만들지 않는다. 원 base component dictionary에는 prior도 보존하고 outline은 실제 원 가중 성분으로 따로 기록한다. 실제 E0 선택 자세와의 비용 차이는 동일 G13/family/mode/관측 package 안에서만 계산한다. 실패 E0의 비용이 더 낮은지는 원 비용 충돌 진단이지 새로운 선택 정책의 성능이 아니다.

material/mesh/visibility metadata는 실제 원 자료에 존재하는 항목만 진단에 연결한다. 없으면 `MATERIAL_OWNER_UNVERIFIED`다. 원 cuboid edge 투영을 재료 경계 GT로 재명명하지 않는다. 제약·비용 helper 재사용 결과는 독립 수식 검산과 구분해 보고하며, B 진단을 읽고 F1 스키마·관측·R/F/V 설정을 다시 조정하지 않는다.

재현 CLI는 [feature_support.py](../../../scripts/research/pallet_risk_information_v1/feature_support.py), [reference_support.py](../../../scripts/research/pallet_risk_information_v1/reference_support.py)의 `--help`와 최종 캠페인 재현 안내를 따른다. 실제 원 cache·seal·split·config가 필요하며 공개 문서만으로 private 입력이 생기는 것은 아니다.
