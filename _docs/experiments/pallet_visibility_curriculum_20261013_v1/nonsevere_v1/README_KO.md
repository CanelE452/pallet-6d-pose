# Severe를 제외한 가시성·좌표 신뢰도 curriculum

현재 범위는 사용자 요청에 따라 **신규 학습과 주 평가에서 severe를 제외**한다. 완료된 사람의 CLEAN/MODERATE/SEVERE 분류를 이미지 SHA로 결합했다. 기존 세션 분할을 유지하면서 CAL 46장(CLEAN 37/MODERATE 9), TEST 49장(CLEAN 30/MODERATE 19)만 남겼다. severe 78장은 새 모델의 표준화, PCA, 분류, 좌표 회귀와 주 평가에 들어가지 않는다.

RGB 가시성 모델은 실제 CAL 46장에서 새로 추출한 특징으로 학습했다. 좌표 신뢰도·보정 모델도 CAL 46개의 특징 285점/회귀 234점만으로 모든 계수를 새로 계산했다. 파렛트 mask CNN은 외부 가림·cargo·context가 없는 clean-static source TRAIN 426/source CAL 72장으로 처음부터 학습했다. 이전 mixed 모델은 이 설정에 연결하지 않았다. 기존 detector/N3 및 최초 자세 cache는 고정 입력이며 그 과거 학습 범위를 다시 인증하거나 학습한 것은 아니다.

검증은 [NONSEVERE_SCOPE_QA.json](NONSEVERE_SCOPE_QA.json)에 있다. 6,695개 검사에서 위반이 없었으며, 실제 RGB 특징·모델 계수·학습 파일 SHA를 독립적으로 재계산했다. SPLIT의 숫자 요약을 고친 기록과 최초 학습 당시의 원본 protocol도 보존했다. 모델의 CAL 46개 ID와 학습 계수는 이 메타데이터 정정에서 바뀌지 않았다.

## 외곽점부터 추가하는 방법의 구현 결과

`BoundaryCurriculum`에 **RGB 가시성 → 영상 경계선 좌표 보정 → 기하학적 외곽점 seed → 한 점씩 추가 → 마지막 전체 관측 검사**를 연결했다. 가시성은 원래 KP에서 조회하고, 보정은 BOX_ONLY MobileSAM mask의 실제 전이점으로 두 선을 맞춘 교점에서 얻는다. 원래 KP와 교점의 blend는 0.5 이하, 이동은 6px 이하이다. 초기 자세의 투영 좌표는 난이도·geometry·잔차 검사와 숨은 점 출력에 쓰며 관측 좌표로 피팅하지 않는다.

기존 규칙은 현재 쓰지 않은 점이 seed를 거부하거나, 새로운 점의 추가가 앞서 사용한 점의 잔차를 조금 늘리면 전체 관측이 좋아져도 거부했다. 새 선택 옵션 `progressive_active_support=True`는 현재 사용하는 관측 집합의 고정 가중 Huber 합과 동일한 초기 자세 prior가 개선되는지를 검사한다. geometry·depth·초기 자세에서의 이동 제한은 유지한다. 최종 출력은 **처음에 고정한 전체 visible 집합**의 목적함수가 초기 자세보다 개선되는 가장 최근 채택 단계를 사용하며, 없으면 초기 자세를 정확히 보존한다. 최소 GT 오차 단계나 최소 관측 오차 단계를 고르지 않고 마지막 batch refit도 하지 않는다. UNKNOWN의 낮은 가중치 추가에는 기존 visible 보호 규칙을 유지한다.

CAL 46장의 사전 고정 비교에서 FUSED 외곽 seed 채택은 8→43장, 채택된 fit 단계는 48→139개로 늘었다. 위치 평균은 3.7117→3.6883cm, 회전 평균은 7.7353→7.6348°였다. 같은 관측을 함께 쓰는 결과와는 45/46장이 수치상 같았으며 나머지 한 장은 회전과 위치 오차가 서로 반대 방향으로 바뀌었다. **순차 처리의 성능 우위는 확인하지 못했지만, 원래 의도한 중간 추정과 관측 추가는 작동하게 됐다.**

이어 설정·코드·모델을 고정하고 기존 TEST 49장에 대해 7조건 343개 예측을 먼저 봉인했다. 새로 학습하거나 TEST 결과로 정책을 바꾸지 않았다.

| 기존 TEST 49장 전체 | 위치 평균 cm | 회전 평균 ° | 5cm·5° 이내 |
|---|---:|---:|---:|
| 최초 N3/SubPix 자세 | 3.3630 | 6.5271 | 40/49 |
| 원래 KP＋새 순차 보호 규칙 | 3.2995 | 6.5481 | 42/49 |
| 영상 경계선 보정＋새 순차 보호 규칙 | **3.0040** | **6.5243** | **43/49** |
| 같은 보정 KP를 함께 사용 | 3.0040 | 6.5243 | 43/49 |

TEST의 보정 좌표 조건에서 첫 seed는 기존 11장→새 45장, fit 채택은 61→141개로 늘었다. 전체 49장 모두 최종 결과가 together와 `1e-3px` 수준에서 같으므로 순서에서 추가 정확도 이득을 주장하지 않는다. 좌표 보정의 matched 차이는 위치 −0.2955cm, 4세션 bootstrap 95% 구간 `[-0.8667,-0.0329]`; 회전 −0.0238°의 구간은 0을 포함한다. 초기 대비 위치 −0.3590cm의 구간도 0을 포함한다. 재사용 TEST·4세션·annotation 유래 자세 reference라는 한계가 있다.

```python
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_curriculum_api import BoundaryCurriculum

processor = BoundaryCurriculum.from_config(
    "_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/BOUNDARY_PROGRESSIVE_CONFIG.json",
    device="cuda")
result = processor.refine(image_bgr, points_xy, camera_K, initial_pose, registered_dimensions)
boundary_states = result["boundary_visibility"]["states"]
seed_ids = result["seed_corner_ids"]
stages = result["solution"]["stages"]
```

`BOUNDARY_CURRICULUM_CONFIG.json`는 기존 보호 규칙을 재현하는 비교 설정이고, `BOUNDARY_PROGRESSIVE_CONFIG.json`가 위 새 방법을 실행한다. 기존 config의 `progressive_active_support` 기본값은 False다. `BOUNDARY_PROGRESSIVE_FREEZE.json`에 설정과 모델 SHA를 보존했으며 기본 detector/model로 승격하지 않았다. 좌표 분산은 조건부 선 재표집 변동이고 의미적 코너 정확도 확률은 아니다. 기존 CAL의 95% 영역 coverage가 31.8%여서 solver의 새로운 확률 가중치로 쓰지 않았다.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_curriculum_api \
  --image IMAGE.jpg --input input.json \
  --config _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/BOUNDARY_PROGRESSIVE_CONFIG.json \
  --output result.json --device cuda
```

입력 형식은 기존 `EXAMPLE_INPUT.json`이고 실제 실행 예제는 `EXAMPLE_BOUNDARY_PROGRESSIVE_OUTPUT.json`이다. 측정값·순수 가시성·투영·completion은 별도 필드다. 실제 RGB 46장에서는 경계선 교점·blend·가시성·자세가 기존 봉인 관측과 차이 0이었고, 현재 API의 46장×2조건 cached 관측 재실행도 기존/new policy pose 및 final guard와 정확히 같았다. 입력 변조, 화면 밖 점, 보정 bound와 물리 축 검사를 추가한 전체 소프트웨어 검사 118개와 subtest 187개가 통과했다. TEST 49장·343개 예측에 대한 독립 구현 감사도 11,946개 검사에서 위반이 없었다. 전체 suite의 과거 cache/이미지 호환성 검사는 신규 학습·주 성능 평가와 별도다.

근거: [CAL 규칙 실험](progressive_support_guard_v1/RESULTS.json), [TEST 재사용 비교](boundary_progressive_replay_v1/RESULTS.json), [실제 영상 API 일치 검사](boundary_curriculum_integration_v1/QA.json), [현재 API 일치 검사](CURRENT_API_CAL46_PARITY_V2.json), [독립 코드 검토](BOUNDARY_CURRICULUM_REVIEW_V1.json). 실제 단계와 영상 관측은 [진행 그림](boundary_progressive_replay_v1/PROGRESSIVE_TRACE.png)과 export 가능한 PDF에 있다.

## 이전 RGB 신뢰도·보정 결과

모델·설정·입력과 코드를 고정한 뒤 TEST 49장의 245개 예측을 먼저 봉인하고 평가했다. 이 TEST는 과거에 이미 사용한 데이터이므로 독립적인 새 검증이라고 부르지 않는다. 자세 reference는 기존 annotation으로 재구성했으며 별도 물리 실측값이 아니다.

| TEST 49장 전체 | 위치 평균 cm | 회전 평균 ° | 5cm·5° 이내 |
|---|---:|---:|---:|
| 최초 N3/SubPix 자세 | 3.3630 | 6.5271 | 40/49 |
| 새 RGB 가시성＋mask curriculum | 3.3124 | 6.5620 | 42/49 |
| 새 RGB 좌표 신뢰도·보정＋초기 family 유지 | 2.6186 | 6.5923 | 42/49 |
| 위 관측으로 W/D family도 선택 | 3.1923 | 11.8512 | 40/49 |
| 같은 관측·초기 family에서 visible을 함께 사용† | 2.5913 | 6.5468 | 42/49 |

† 마지막 행은 이미 봉인한 비교 후보의 original-family 결과에서 파생한 사후 분석이다. 새로운 학습이나 추론 정책 변경은 없었고, unknown 점의 추가 규칙도 동일하다.

초기 family를 유지한 후보의 초기 자세 대비 위치 차이는 −0.7444cm, 4세션 bootstrap 95% 구간은 `[-1.2768,-0.4891]`이다. 회전 차이는 +0.0652°, 구간 `[-0.0764,+0.2764]`로 개선을 확인하지 못했다. 성공은 3장을 복구하고 1장을 잃었다. W/D 선택은 올바른 방향 3개를 틀리게 바꿨다. 큰 실패나 일치하지 않는 corner phase를 가진 프레임도 전체 49장 자세 결과에서 빼지 않았다.

새 RGB 가시성 후보는 DIRECT 286점 중 269점을 받고, EXTERNAL 12점 중 10점을 제외했다. 이 표본에서는 직접 보이는 점 recall 94.1%, 외부 가림 recall 83.3%였다. 좌표 신뢰도 head의 점수는 `DIRECT_VISIBLE ∧ 원래 좌표 오차≤8px`를 목표로 학습한 사용 가능성이며, 순수한 3D 가시성 확률과 동일하지 않다. 초기 H는 초기 cuboid 형상과 자세에 조건부인 값이며 모든 실제 self-occlusion을 보장하지 않는다.

사용자가 추가로 요청한 **초기 기하에서 배경과 닿을 것으로 예상되는 외곽 KP만** 분리했다. 초기 cuboid 투영의 convex hull vertex를 사용하고 H·원본 좌표/초기 투영의 화면 밖을 제외했다. RGB 점수나 정답 라벨을 subset 선택에 사용하지 않았다. 271개 후보 중 DIRECT 248, EXTERNAL 12였으며 RGB 모델은 직접 보이는 247/248, 외부 가림 10/12를 구분했다. 나머지 6개의 사람 SELF와 5개의 UNANNOTATED는 조건부 초기 H의 한계로 별도 기록했다. 이것은 전체 49장의 기존 예측에 대한 사후 진단이고, 새 threshold 선택이 아니다.

외부 가림을 놓친 두 점을 실제 RGB로 확인했다. 한 점은 잠긴 정답 위치에서 19.55px 떨어진 보이는 파렛트 패치를 읽었으며 정답 코너는 빨간 가림물 경계에 있었다. 다른 사례는 native/reference 대응 자체가 크게 어긋났다. 따라서 단순히 같은 색의 가림물을 잘못 분류한 사례로 설명할 수 없다. 처음 추정한 위치가 목표 모서리와 일치하는지와 그 위치의 재질/경계를 분리해서 확인해야 한다. 원래 대응과 전체 49장 점수는 바꾸지 않았다.

`BoundaryVisibility`는 이 제한된 첫 단계를 pose fitting이나 좌표 회귀 없이 실행한다. 내부점과 제외점은 `NOT_ASSESSED`, 외곽 후보는 `VISIBLE` 또는 `EXTERNAL_OCCLUSION_SUSPECTED`와 확률로 반환한다. `initial_projected_xy`는 기하 판정용이고 RGB를 읽는 위치는 원래 관측 q다. 실제 49장·271개 후보의 점수와 선택 ID는 기존 봉인 결과와 정확히 같았다.

```python
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_visibility import BoundaryVisibility

visibility = BoundaryVisibility.from_config(
    "_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/CAL_VISIBILITY_CONFIG.json")
first_stage = visibility.assess(image_bgr, points_xy, camera_K, initial_pose)
```

근거는 `test_reused_v1/GEOMETRIC_BOUNDARY_VISIBILITY_AUDIT_V1.json`, `BOUNDARY_VISIBILITY_API_QA_V1.json`, `boundary_rgb_failure_diagnostics_v1/BOUNDARY_RGB_FAILURE_DIAGNOSTICS_V1.json`에 있다. Joint API에도 RGB 가시성 원본을 `visibility_assessment`에 보존하고 `keypoint_status`에서 좌표 usability와 별도로 반환한다. 실제 49장에서 RGB로 보인다고 평가하지만 usable 기준을 못 넘은 26점을 분리했으며 기존 pose·완성 좌표는 정확히 유지됐다.

## 아직 해결하지 못한 부분

난이도 순서로 추가하는 **단계 자체의 우위는 확인되지 않았다**. 동일한 관측·좌표·방향에서 staged와 visible-together의 TEST 차이는 위치 +0.0273cm, 회전 +0.0456°, 성공 42/49로 동일했다. CAL 46장 세션 OOF에서도 staged가 조금 더 나빴다. CAL OOF 좌표 오차도 7.036→7.475px로 악화했으므로 TEST의 위치 개선만으로 이 RGB 보정이 일반화한다고 결론 내릴 수 없다.

CAL에 남은 외부 가림 학습점은 7개뿐이다. 새 full-RGB head의 세션 OOF AUC는 .7577이나 선택 범위의 balanced accuracy는 .4962다. 처음 시도한 재사용 cache에는 색 통계가 빠져 있었고 해당 부분 특징 모델은 보존하되 사용하지 않는다. 현재 설정은 실제 CAL RGB에서 다시 추출한 `visibility_logistic_CAL46_v2.json`만 사용한다.

좌표 오차에는 고정된 corner phase와 물리 모델의 대응이 섞일 수 있다. CAL 감사와 영상 윤곽선 교점·불확실성 평가를 완료해 별도 기록했다. 분산만으로 의미적 좌표 정확성을 판단할 수 없었으며, 실제 pose 전체 결과를 좋은 대응 프레임만으로 축소하지 않았다.

## 실행 가능한 선택 후보

`FIXED_FAMILY_CONFIG.json`은 초기 치수 family와 물리 축을 유지하면서 RGB 관측을 순차적으로 추가하는 API다. TEST의 기존 original-family 비교 arm을 실행 형태로 제공한 사후 선택 후보이며, 기본 모델로 승격하지 않았다. W/D를 바꾸는 `JOINT_CONFIG.json`은 실패 비교를 재현하는 연구 설정으로 보존한다.

```python
from scripts.research.pallet_visibility_curriculum_20261013_v1.joint_api import JointSpatialBranch

processor = JointSpatialBranch.from_config(
    "_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/FIXED_FAMILY_CONFIG.json")
result = processor.refine(image_bgr, points_xy, camera_K, initial_pose, registered_dimensions)
pose = result["actual_pose"]
states = result["appearance"]["states"]
```

치수 순서는 `(width,height,depth)`, 단위는 metre다. 8개 corner의 순서는 `common.cuboid()`이며 9번째 center는 보존한다. `observed_points`, `corrected_measurements`, `projected_points`를 구분한다. 숨은 점의 투영 좌표는 최종 출력 보완에만 들어가며 최종 자세 fit에는 들어가지 않는다. alternate 초기 seed의 기존 동작은 원래 예측한 H 좌표도 사용하므로, 비교 연구 API를 모든 단계에서 H 없이 계산한다고 설명하지 않는다. fixed-family API는 alternate seed를 계산하지 않는다.

한 장의 실행 입력은 `EXAMPLE_INPUT.json`과 `EXAMPLE_IMAGE.json`에 있다. CLI는 다음과 같고 output은 새 경로를 써야 한다.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_visibility_curriculum_20261013_v1.joint_api \
  --image IMAGE.jpg --input input.json \
  --config _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/FIXED_FAMILY_CONFIG.json \
  --output result.json
```

API/branch/population 검사 18개가 통과했다. 새 fixed-family API의 실제 49장 RGB 재실행은 봉인된 original-family 결과와 회전행렬·이동·완성 좌표의 차이가 모두 0이었다. `FIXED_FAMILY_API_QA.json`은 이 구현 일치 검증이며 추가적인 성능 검증이 아니다.

원본 결과는 `test_reused_v1/RESULTS.json`, 순서 효과 분리는 `test_reused_v1/MATCHED_FAMILY_STAGE_ANALYSIS.json`, CAL OOF는 `OOF_pose_v1/RESULTS.json`과 `COMPONENT_DIAGNOSTICS.json`에 있다. 요청한 방법론의 분석·실험과 실행 가능한 연구 구현을 위 새 API에 연결해 검증했다. 단계적 순서의 성능 우위나 새로운 독립 검증은 주장하지 않으며 severe를 신규 학습·주 평가에 넣지 않는다.
