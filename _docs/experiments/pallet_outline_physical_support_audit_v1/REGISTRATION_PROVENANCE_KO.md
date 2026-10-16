# 등록·카메라·참조의 확인 범위

고정한 여섯 영상과 기존 G13 후보12개에서 **치수·Q·K·중심 원점·참조 읽기 규약·원 crop 연결은 일치했다.** 이것은 수치 계약의 일치다. 실제 팔레트 경계의 정체나 카메라 교정·참조의 물리적 정확성이 확인됐다는 뜻은 아니다. 이번 감사에서는 자세를 풀거나 원 참조·치수·K·후보를 수정하지 않았다. RGB의 물리 지지 판정은 이 문서의 코드 감사와 분리해 본 보고서의 실제 이미지 검토에서 다룬다.

원결과는 `c56a4e92027a8e56f6704773f24d7c3da78082f1`, 상위 관측·후보 결과는 `407de865445e4987614f50f893d2ccea78d507b6`이다. 각 후보는 항상 `(image_id, candidate_id)`로 연결했고 기존 대표6장·대조 후보를 바꾸지 않았다.

## 확인된 좌표와 등록 규약

여섯 원 RGB는 모두640×480이다. annotation, 원 readiness, 실제 solver packet의 K가 바이트 값 수준으로 같은 수치였다.

```text
K = [[608.2978515625, 0, 326.4007263183594],
     [0, 607.8462524414062, 239.5469512939453],
     [0, 0, 1]]
```

현재 비용과 그림은 이 원영상 좌표를 쓴다. `remainder_native_inputs_v1.predictor_record`는 저장 `raw_hw`와 현재 canvas의 일치, `qFinal == native_points`를 요구하고 `crop_transform=None`으로 전달한다. 기존 갤러리 crop은 원 RGB의 정수 사각형을 그대로 잘라 JPEG quality87/optimize로 저장했다. 여섯 장 모두 같은 인코딩으로 원 crop JPEG 바이트가 재현됐다. 이 분석에 resize·letterbox·K 보정 offset은 없다. 더 앞선 검출기의 letterbox 역변환 전체를 이번에 새 검증한 것은 아니다.

모델은 기하학적 중심을 원점으로 한 등록 직육면체다. canonical +Y는 위에서 아래로 향하는 높이 축이고, X/Z는 등록된 두 footprint 축이다. 플라스틱 등록 치수는 X/Y/Z=`1.10/0.11/1.30m`, 목재는 `0.80/0.14/0.59m`다. 이는 원 prediction registry의 외부 치수 조건이며 이번에 실물 치수를 다시 측정하지 않았다.

native 점을 행벡터로 쓸 때 `X_physical = X_native @ Q`, 회전은 `R_physical = R_native @ Q`다. 각 후보의 자기 Q·XYZ를 사용했을 때 등록된 물리 직육면체의 **순서 없는8개 점 집합**과 일치했다. 일부 native extents의 W/D가 바뀐 것은 명시적인 좌표기저 변경이다. 다른 family의 Q를 섞지 않았다. 등록 일치가 영상점의 semantic owner를 보증하지는 않는다.

## 참조가 실제로 의미하는 것

여섯 사례 모두 기존 추가143장의 `real_pallet_gt_v2` annotation에서 온 **수동 주석에 기반한 PnP 파생 참조**다. `gt_source: manual`은 annotation 전체의 메타데이터다. 여덟 corner가 모두 독립 수동 클릭이라는 뜻이 아니다. 저장 corner에는 `manual_click`, `extrapolated`, `pnp_projected`가 섞여 있고 중심은 `centroid_auto`다. `annotate_io.py`는 이 출처를 구분해 저장한다.

원 scorer의 정책은 다음과 같다. 중심은 `camera_facing_pnp.pose_transform[:3,3]`를 그대로 사용한다. 회전은 원 canonical 후보 두 개가 정확한 t 복사와180° C2 관계를 만족하는지 확인한 뒤 **기존 후보0**으로 고정한다. 예측에 가장 가까운 참조를 고르는 방식이 아니다. 두 후보의 yaw 부호는 여전히 미확정이며,90° W/D 혼동을 C2로 없애지 않는다.

모든 여섯 annotation은 `intrinsics_quality=UNKNOWN`, 원 archive `camera_info.json`의 `PROVIDED_UNVERIFIED` 값을 출처로 기록한다. 세션 메타데이터는 RealSense D435I를 기록하지만 별도 교정 검증을 제공하지 않는다. annotation camera_data에는 distortion 계수가 없다. `UNCONFIRMED_SIGNED_AXIS`와 `MANUAL_REVIEW_REQUIRED` 상태도 여섯 장 모두 그대로다. 목재 registry의 물리 symmetry 상태는 `UNREVIEWED`이고, 이번 C2는 원 평가 규약이지 독립적인 물리 대칭 인증이 아니다.

| 고정 사례 | 원 수동 클릭 corner | 외삽 / PnP 투영 corner 수 | 저장 reprojection error | 원 canonical 후보 | 원 W/D 선택 사유 |
|---|---|---:|---:|---|---|
| plastic_night_01:038630 | 2,4,5 | 1 / 4 | 1.123855px | 90,270° | geometry_rank_clear |
| wood_day_01:005286 | 0,3,4,7 | 1 / 3 | 0.749504px | 0,180° | geometry_rank_clear |
| wood_night_01:030698 | 0,1,4,5 | 0 / 4 | 0.271765px | 0,180° | geometry_rank_clear |
| plastic_day_01:020955 | 0,1,3,5,6 | 0 / 3 | 0.331015px | 90,270° | **geometry_rank_ambiguous** |
| plastic_day_01:014704 | 0,1,2,3,4,6 | 0 / 2 | 0.964464px | 0,180° | geometry_rank_clear |
| wood_night_01:030898 | 0,1,4,5 | 0 / 4 | 0.904792px | 90,270° | geometry_rank_clear |

작은 reprojection error는 선택한 치수·K·주석 대응 아래의 일관성이다. 독립 pose 계측, signed physical 축, 렌즈 교정, 전체 silhouette 대응의 검증으로 바꾸지 않는다. 또한 저장된 반대 W/D annotation 가설에는 요약 점수만 있고 R/t가 없다. 이를 새로 풀거나 임의의 대체 참조 투영으로 만들지 않았다. 이 감사는 원 annotation을 만든 모든 과거 GUI 실행·수동 클릭 행위까지 독립 확인한 것도 아니다.

## A/B/C를 나누어 확인할 최소 질문

아래 좌표는 원 `WITNESS_RECEIPT`의 최대 양의 표본 위치다. 좋은/나쁜 pose 표시는 물리 판정의 정답으로 쓰지 않는다. 각각 해당 부모 선분 전체와 음의 표본 구간을 함께 검토해야 한다.

| 사례와 원 좌표 | A: 저장 경계의 영상 지지 | B: 직육면체 silhouette와 동일시할 조건 | C: 등록·참조 쪽 남은 확인 |
|---|---|---|---|
| 038630, sample28 `(154.78,321.11)` | 실제 RGB에서 팔레트 외곽인지 다른 대비 경계인지 직접 확인 | 가림/내부 구조의 연결 경계와 물체 전체 envelope를 구분 | 수동3점·외삽1점 아래 고정된 축 가정의 지지 여부. 숫자 오차로 mask/참조 오류를 판정하지 않음 |
| 005286, sample53 `(579.71,165.59)` | 오른쪽 화면 끝의 문맥까지 포함해 실제 지지를 확인 | 등록된 전체 cuboid의 어느 외곽을 측정하는지 확인 | 수동4점이 제약한 물리 모서리와 나머지 유도점의 의미를 구분 |
| 030698, sample121 `(516.71,405.95)` | 아래·오른쪽 부모 선분과 반대 부호 선분을 함께 확인 | 목재의 실제 하단 구조/가상 bounding cuboid 외곽이 같은 경계인지 확인 | 수동0,1,4,5와 유도된 나머지4점을 독립 근거처럼 세지 않음 |
| 020955, sample263 `(355.77,91.84)` | **기존 crop y최소97보다 위**이므로 원 전체영상/확장된 문맥에서 확인 | 위쪽 hull 연결이 실제 전체 envelope를 나타내는지 별도 확인 | 원 W/D 결정 자체가 `geometry_rank_ambiguous`; UNKNOWN K와 함께 미확정을 유지 |
| 014704, sample313 `(381.50,147.00)` | 성공 반례에도 같은 원영상 검토 기준 적용 | 실제 상단 envelope가 올바른 s를 보호한 경우인지 확인 | 수동6점의 상대적 지원과 유도2점을 구분하되 독립 정답으로 격상하지 않음 |
| 030898, sample31 `(585.59,237.11)` | 오른쪽 양의 구간과 아래 음의 구간을 같은 기준으로 검토 | 실제 목재 외곽이 두 가까운 pose를 구분하는지 확인 | 90/270° signed 축·수동4점/유도4점·목재 symmetry 미검토를 유지 |

A는 직접 영상 해석이 필요하다. B는 수치 등록을 통과해도 여전히 물리 외곽 대응 질문이다. C에서 확인한 숫자 계약의 일치는 실물의 참자세 인증이 아니다. 따라서 비용이 성공 후보를 벌주는 사실만으로 분할기 오류·치수 오류·참조 오류 중 하나를 확정할 수 없다.

## 원 코드 위치와 재현 범위

- [원 raw-canvas·registry Q 입력 계약](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/remainder_native_inputs_v1.py): `predictor_record`.
- [원 추가143 참조 읽기 정책](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/score_remainder_derived_reference_v4_14.py): `reference_from_document`, `_pair_check`.
- [원407의 고정 참조 reader](../../../scripts/research/pallet_statewise_correspondence_diagnosis_v1/analysis.py): `reference_document`, `fixed_references`.
- [원 좌표/중심/부호 변환](../../../scripts/annotate/pallet_geometry.py): 모듈의 frame contract, `_diagram_points`, `canonical_to_camera_facing_transform`.
- [주석 출처와 canonical 후보 저장](../../../scripts/annotate/annotate_io.py): `_normalise_keypoint_annotations`, `make_annotation`의 v2 payload. `annotate_pnp.py`의 `solve_pose_candidates`는 원 생성 의미를 읽기만 했으며 실행하지 않았다.
- [현재 읽기 전용 감사 코드](../../../scripts/research/pallet_outline_physical_support_audit_v1/registration_audit.py).

현재 실행의 개별 RGB/annotation/source SHA, 고정12후보 R/t/Q 및 raw 좌표 검산은 비공개 `REGISTRATION_AUDIT_V2.json`에 남겼다. 초기 감사6.234초를 보존하고, 추가 CF→canonical 참조 투영의 순서 없는 점 집합 검사를 포함한 최종 감사는6.539초였다. 이 추가 투영 검사 최대 차이는1.137e-13px다. 신경망·PnP·최적화·학습·새 후보0회다. 원 `wood_night_01:033421`의 G10→G13 전달 공백과 후보부재46장은 이번6장 감사로 해결하거나 재평가하지 않았다.

## 저장소의 가까운 관측 지지 처리와 다른 전제

기존 [edge_visibility_v4_28.py](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/edge_visibility_v4_28.py)는 B/I의 제안된 국소 image edge에 대해 이미 조건부 지지를 구분한다. `EdgeVisibilityContext.__init__`는 target과 별도로 출처·RGB·canvas·pixel SHA가 결합된 `NON_TARGET_OCCLUDER` mask를 요구한다. `_occluder_blocked`와 `query_allowed`는 실제 RGB/target remap **이전**에 occluder가 지지한 샘플 위치를 막는다. target mask를 잘라 새 인공 경계를 만드는 방식은 아니다. `classify_expected_edge`는 관측 inlier chain과 target/background 문맥이 있는 지지, 독립 occluder가 있는 gap, 미확인 gap을 구분한다. 이는 모델의 native identity나 상태를 확정하는 함수가 아니다.

반면 기존 whole-outline `observe_global_outline(pallet_mask, image_hw, ...)`는 mask 이외 state/occluder/국소 지지 입력을 받지 않는다. `mask>=.5`의 convex hull에서 화면 안의 모든 유한 선분을 표본화한다. G13도 같은 whole-mask 형상 항을 쓴다. 따라서 국소 경계에 이미 있는 **출처/지지의 조건부 처리 원리**와 전체 convex-envelope 관측의 가정은 다르다. 이를 연결하는 제안을 새로운 발명으로 부르지 않는다. 국소 함수에는 native corner·mate·B/I anchor·실제 chain이 필요하므로 전체 hull에 함수를 그대로 적용할 수 있다는 의미도 아니다.

입력 가용성은 별도 조건이다. `run_joint_geometry_guided_line_cases_v4_28.py`는 그 실험의 실제 cache가 targetSAM/Tiny/raw/clean/alternatives이고 별도 object/occluder mask가 없어서 `independent_occluder_mask=None`이라고 기록한다. `run_initial_only_observed_ray_CAL_v4_32.py`도 target mask를 occluder로 쓰지 않고 독립 mask가 없음을 유지했다. 구현 함수가 존재한다고 이번6장에 인증된 독립 occluder mask가 있었다고 주장하지 않는다. 현재 annotation/state의 O 값이나 target mask의0·보완값을 독립 non-target 지지로 바꿀 수 없다. `boundary_evidence._occluder_evidence`도 target-mask 음성만으로 가림을 인증하지 않는다.

이 선행 비교는 다음 변경안의 필요한 입력과 범위를 명확하게 하는 읽기 전용 소스 감사다. 이번에는 차단 함수를 실행해 mask·관측·비용·후보 순위를 바꾸지 않았다. 새 관측 지원이 필요하면 그것의 가용성과 별도 실행 승인이 먼저 확인돼야 한다.
