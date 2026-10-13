# 초기 자세로 self-occlusion을 제외한 RGB KP curriculum

이 실험은 초기 KP로 자세를 구한 뒤 self-occlusion을 계산하고, 남은 KP를 RGB 증거와 좌표 신뢰도에 따라 쉬운 것부터 추가하는 방법을 구현한다. 초기 자세의 숨은 KP 투영은 출력 보완에만 사용한다. 최적화에 들어가는 좌표는 최초 관측 KP 또는 RGB에서 독립적으로 관측한 경계 교점이다.

방법은 실행 가능한 연구용 후처리로 구현됐다. 첫 고정 검증에서 작은 개선이 있었지만, 외부 가림 판정과 큰 좌표 오류를 해결했다고 할 근거는 아직 부족하다.

현재 사용자 지정 범위는 **학습과 주 평가에서 SEVERE_OCCLUSION을 제외하고 CLEAN/MODERATE만 사용하는 것**이다. 사람의 완료된 분류를 이미지 SHA로 대조해 기존 CAL 76장 중 30장, TEST 97장 중 48장을 제외했다. 새로운 실험은 `nonsevere_v1`의 CAL 46장/TEST 49장으로 진행한다. 아래의 기존 76/97장 결과는 severe를 포함한 과거 연구이며 새 범위의 근거로 사용하지 않는다. 기존 mixed-severity 후보의 W/D 방향 전환 12건은 모두 severe였으므로 해당 개선도 새 범위의 성능을 증명하지 않는다.

현재 범위의 구현과 결과는 [nonsevere_v1/README_KO.md](nonsevere_v1/README_KO.md)에 정리했다. 신규 모델은 실제 CAL 46장과 clean-static source TRAIN 426/source CAL 72장만으로 다시 학습했다. TEST 49장에서 초기 family를 유지한 RGB 후보는 위치 3.363→2.619cm, 성공 40→42/49였으나 회전 개선은 확인하지 못했다. 동일한 관측의 단계 순서만 비교하면 순차 추가가 조금 더 나빠서, 난이도 curriculum의 독립적인 효과는 아직 입증되지 않았다. 전체 목표는 진행 중이다.

## 왜 단순한 색 비교에서 문제가 생기는가

초기 자세와 물체 형상이 주어지면 self-occlusion을 계산할 수 있다. 이 단계와 외부 물체에 의한 occlusion 판정은 별개의 단계다. RGB에서 KP 위치가 파렛트 색인지 확인하면 그 픽셀의 재질 유사성을 얻는다. 그 픽셀이 해당 3D 모서리인지, 같은 색의 앞쪽 물체인지, 잘못 예측한 좌표가 다른 파렛트 면 위에 놓인 것인지는 그 비교만으로 결정되지 않는다.

추가로 다음 두 축을 구분해야 한다.

| 축 | 판정할 내용 | 후처리에서의 역할 |
|---|---|---|
| 가시성 | self-occlusion, 화면 밖, 직접 보임, 외부 가림 의심/불확실 | 사용할 관측 후보를 정함 |
| 좌표 신뢰도 | 관측 좌표가 의미상 3D 모서리를 얼마나 정확히 가리키는가 | 우선순위와 가중치를 정함 |

배경과 접하는 KP는 좌표를 독립적으로 찾기 쉬울 수 있다. 주변이 모두 파렛트인 KP도 직접 보일 수 있다. 따라서 배경과 접하지 않는다는 이유로 가린 점으로 판정하지 않는다. 둥근 모서리, 빈 공간, 격자 구멍에서는 가상 bounding-box 모서리와 실제 재질 경계가 일치하지 않을 수도 있다.

## 구현

1. 초기 camera-facing 자세와 고정된 3D 치수로 self-occlusion mask를 한 번 계산한다.
2. 원본 이미지 좌표 기준 화면 밖 KP를 분리한다.
3. 주변 Lab 색, 파렛트 색 지지율, 경계/모서리 구조를 추출한다. CAL에서 학습한 작은 RGB 가시성 분류기를 적용한다.
4. source 데이터로 학습한 작은 foreground CNN을 추가 증거로 사용한다. mask 픽셀을 KP 좌표나 가시성 정답으로 사용하지 않는다.
5. 보이는 경계 KP를 먼저 선택하고, 기하 조건이 부족하면 다음 관측을 한 점씩 추가한다. 최소 네 점과 충분한 공간 분포를 확보한 뒤 자세를 보정한다.
6. 보이는 내부 KP, 초기 자세와 일치하는 불확실 KP를 낮은 가중치로 순서대로 추가한다. 강건 손실, 초기 자세 prior, 이미 사용한 관측의 오차 검사, 이동량 제한을 적용한다.
7. 수렴한 자세로 self-occlusion KP를 출력에서 보완한다. 이 좌표를 다시 fitting에 넣지 않는다. 관측값과 투영값은 별도 필드로 제공한다.

`VisibilityCurriculum.refine(image_bgr, points, K, initial_pose)`는 데이터셋이나 평가 annotation을 읽지 않는 독립 API다. 원본 이미지, 원본 좌표의 8/9개 KP, 원본 K와 초기 자세를 받는다. 등록 치수/회전은 `initial_pose`에 들어 있으며 단위는 metre다. 모서리 순서는 `common.cuboid()`와 일치해야 한다. 9번째 중심점은 보존한다.

## 데이터와 검증 범위

과거 “DEV319”에는 현재 역할 기준 FINAL 146장이 섞여 있었다. 이번 학습/선택/첫 평가에서는 실제 DEV 173장만 사용했다. material별 session hash로 CAL 76장/5세션과 TEST 97장/4세션을 나눴다. 이 split은 첫 추론과 scoring 전에 고정했다. FINAL은 이번 학습과 평가에 넣지 않았다.

TEST 97장은 이 실험의 CAL 선택에서 제외했지만, 프로젝트의 과거 실험에서 사용된 데이터다. 새로운 독립 물리 검증셋으로 부르면 안 된다. 자세 reference도 기존 annotation에서 기하적으로 재구성한 값이며 별도 실측 자세가 아니다.

source segmentation은 P0 1,024 family 중 train 768/CAL 128로 학습했다. source TEST 128은 체크포인트 선택에 쓰지 않았다. graphite 한 형상과 source 촬영 조건의 범위라는 한계가 있다.

모든 추론은 prediction-only 입력을 사용하고 annotation 접근 canary 하에서 수행했다. 전체 예측을 seal한 뒤 scoring을 열었다. 결과와 코드 snapshot, 모델 해시를 보존했다. 기존 YOLO/N3 좌표 cache를 재사용했으므로 아래 성능은 새 detector 학습 결과가 아니다.

## 첫 고정 TEST 결과: v1

선택 과정과 코드/모델 해시는 `FROZEN_TEST_PROTOCOL.json`에 기록돼 있다. Primary는 `SEGMENTED_RGB_CURRICULUM`이다.

| 지표, 전체 97장 | 초기 N3+SubPix 자세 | RGB+foreground curriculum |
|---|---:|---:|
| 위치 평균 오차 (cm) | 15.0123 | 15.0017 |
| 위치 중앙값 (cm) | 5.0775 | 4.8371 |
| 위치 P90 (cm) | 28.3856 | 29.0346 |
| 회전 평균 오차 (°) | 26.1374 | 26.0934 |
| 회전 중앙값 (°) | 2.3893 | 2.2176 |
| 회전 P90 (°) | 88.1898 | 88.3691 |
| 5cm·5° 이내 성공 | 46/97 | 48/97 |

97장 중 91장에서 새 자세를 채택하고 6장은 초기 자세로 돌아갔다. 성공은 세 장을 복구하고 한 장을 잃어 순증 두 장이다. 실패 프레임을 제외하거나 큰 오류를 clipping하지 않았다.

세션 단위 paired bootstrap의 평균 차이 95% 구간은 위치 −0.01055cm `[-0.49016, +0.68788]`, 회전 −0.04404° `[-0.10856, −0.00771]`였다. 위치 개선은 불확실하다. 회전 개선은 이 작은 4세션 표본에서 구간이 0을 포함하지 않았다. 큰 오류의 P90은 오히려 늘었다.

동일한 segmented 증거와 optimizer를 한 번에 적용한 `SEGMENTED_RGB_ONE_STAGE`와 비교하면 staged 방식의 위치 평균 차이는 −0.08851cm, 구간 `[-0.18556, −0.00452]`였다. 회전 차이는 −0.00132°, 구간 `[-0.01148, +0.01824]`로 불확실하며 성공은 둘 다 48/97이다. 이 matched 비교는 첫 결과 seal 이후 추가 분석이다. 첫 protocol의 secondary reference `RGB_ONE_STAGE`는 raw RGB 방식이므로 순수한 stage 효과 비교와 구분해야 한다.

직접 보이는 KP의 recall은 474/499 = 95.0%였지만, 외부 가림 KP의 recall은 29/95 = 30.5%였다. 외부 가림 66개를 보이는 KP로 잘못 받아들였다. 가림 판정이 해결됐다는 결론은 내릴 수 없다.

## 원인 분리와 추가 실험

CAL oracle factorial에서 초기 body 모델이 reference와 일치한 63장을 분리했다. 원래 예측 좌표에서 가시성만 정답으로 바꾼 경우 성공은 36/63에서 38/63으로 늘었다. 좌표를 정답으로 바꾸고 기존 가시성을 유지한 경우는 48/63이었다. 두 oracle을 함께 넣은 경우는 46/63이었다. 초기 prior와 관측 기하 조건 때문에 두 oracle이 단조롭게 더 좋아지지는 않았다. 이 결과는 좌표 정확도가 큰 병목임을 보여주며, oracle은 배포 방법이 아니다.

나머지 13장은 초기 W/D 모델이 달랐다. 예측 KP의 RGB 가중 재투영 오차로 두 모델을 비교한 후보는 잘못된 13장을 복구하지 못하고, 올바른 3장을 틀리게 바꿨다. 정확한 visible KP를 준 합성 예에서는 복구됐다. 실제 예측 좌표가 잘못된 branch를 지지하므로 해당 후보를 채택하지 않았다.

3점 seed와 더 강한 prior도 합성 예에서는 동작했지만 실제 CAL에서 위치·회전이 함께 좋아지지 않았다. 현재는 최소 4개의 독립 관측을 요구한다. 단순한 `NONE` 후보를 occlusion 정답으로 재사용하지 않았다.

초기 투영 위치의 RGB/CNN 증거를 추가하는 CAL 세션 교차검증도 수행했다. 기존 RGB26 특징의 AUC 0.8775에 비해 CNN을 더한 후보는 0.8673, 투영 위치 증거까지 더한 후보는 0.8733이었다. 개선 근거가 없어 통합하지 않았다. CAL 외부 가림 74개 중 64개가 두 야간 세션에 집중돼 있으며, 주간 outside 세션에서는 기존 분류기가 10개 외부 가림을 모두 놓쳤다.

RGB 경계 좌표 보정은 별도로 36개 조건을 CAL에서 비교했다. 엄격한 후보가 2장/3개 모서리를 보정했고, 모서리 오차는 평균 8.776px에서 0.511px로 줄었다. 세 점 모두 직접 보이는 KP이며, 두 선의 RGB 교점에서 나온 관측이다. 한 선을 두 KP가 공유하는 상관도 보존해 보고했다.

이 좌표를 고정된 segmented curriculum에 넣은 CAL hybrid는 위치 평균 116.2915→116.2010cm, 회전 평균 19.2900→19.2860°, 성공 37/76→39/76이었다. 두 이미지와 한 세션에 집중된 결과이며 36개 CAL 조건을 본 뒤 선택했으므로 추가 검증이 필요하다. 넓게 보정하는 후보는 둥근/가상 모서리를 다른 경계로 끌어 회전을 악화시켰다. CNN 경계는 낮은 해상도의 blur로 좌표를 악화시키는 경우가 있어 선택하지 않았다.

엄격한 후보를 v2로 고정하고 동일 TEST 97장에서 재사용 평가했다. 39장/68개 KP가 보정됐으나 위치 평균은 15.0017→15.2067cm, 회전 평균은 26.0934→26.1189°, 성공은 48/97→46/97로 악화했다. 좌표 중앙값은 3.235→2.933px로 줄었지만 35개 개선/33개 악화였고 자세 개선으로 이어지지 않았다. `LIVE_CONFIG_V2.json`은 이 실패를 재현하는 선택 실험이며 기본 추천은 v1이다. `test_v2_reused`에 최초 seal과 정확한 코드 snapshot을 보존했다. 결과를 본 뒤 v2의 threshold를 다시 맞추지 않았다.

source-only local RGB 모델도 만들었다. RGB patch, 예상 hull, 중심 위치를 입력으로 받고, 확실한 실제 foreground 주변 query에 인공 불투명 가림을 추가해 학습했다. 가상 cuboid 모서리의 가시성 정답이나 원인 불명의 NONE을 그대로 학습 라벨로 쓰지 않았다. 따라서 학습 대상은 local body support proxy이며 3D corner visibility 자체의 인증이 아니다.

이 모델은 실제 CAL 주간 외부 가림 10개 중 8개를 찾았으나 threshold .5에서 나무 파렛트 direct recall은 73.8%였다. source head만 사용한 네 threshold와 RGB 저점수 veto를 고정해 자세도 비교했다. 위치·회전이 함께 개선되는 근거가 없어 기본 API에 넣지 않았다. 같은 RGB를 유지하면서 앞쪽 물체의 소유권만 바꾼 제어군에서는 출력도 완전히 같았다. 엔트로피도 이런 가림을 보장해서 찾아낼 수 없다.

독립 API 수치 검증과 invariant 검사 43개가 통과했다. CAL 76장 전부에서 API v1/v2와 sealed batch 결과가 일치했고 이미지·원본 KP·K·초기 자세·중심점·치수·corner phase를 보존했다. v1 후처리만 76장×3회 측정한 시간은 중앙값 50.6ms/P90 147.1ms였다. 이미지 decode, detector, N3 forward를 포함하는 전체 시스템 latency가 아니다.

선택적 analytic Jacobian을 추가한 뒤 전체 49개 테스트가 통과했다. `LIVE_CONFIG_FAST_V1.json`은 같은 모델·loss·단계 선택을 사용하면서 수치 미분을 정확한 미분식으로 대체한다. DEV173의 847개 단계 승인과 사용 KP가 같았고, 최대 재투영 차이는 0.00000658px였다. 자세 solver 중앙값은 7.42→4.83ms, 전체 후처리 중앙값은 50.6→47.6ms/P90 147.1→138.1ms였다. 기본 v1은 기존 수치 미분을 유지한다.

ImageNet으로 사전학습된 frozen ResNet의 국소 이미지 특징도 CAL 세션 교차검증했다. 두 크기 context를 더한 후보의 visibility AUC는 0.8775→0.9023, external recall은 55/74→60/74였으나 direct recall은 348/367→343/367이었다. 이 확률로 좌표 선택만 바꾼 held-session CAL pose는 위치·회전 평균이 모두 악화했다. 분류 지표 개선을 자세 개선으로 간주하지 않고 기본 모델에 넣지 않았다.

`VISIBILITY_VS_COORDINATE_ERROR.png`와 PDF에는 직접 보이는 CAL KP의 가시성 점수와 좌표 오차를 표시했다. 보인다고 채택한 350개 중 기존 annotation 기준 8px 이내인 점은 187개(53.4%)였다. 이 때문에 현재 추가 실험은 joint observation usability와 독립 RGB 좌표 보정에 집중한다.

joint head는 `DIRECT_VISIBLE ∧ 좌표 오차≤8px`를 usable 라벨로 정의하고, RGB 공간 특징에서 ≤16px 범위의 좌표 보정을 별도로 학습했다. 실제 CAL10-arm 비교에서 작은 좌표 개선은 있었으나 고정 초기 family 안에서는 안정적인 위치·회전 우위를 보이지 않았다. 이 학습/선택은 CAL 세션 교차검증이며 hyperparameter도 같은 교차검증 결과에서 선택한 exploratory 분석이다.

이 독립 좌표와 usable 점수로 두 W/D family를 다시 비교한 조합은 개선 사례를 만들었다. `PCA32_CORRECTED + margin .5px²`가 초기 오류 13개 중 두 개를 복구하고 올바른 family를 틀리게 바꾼 사례는 없었다. 두 회전 오차가 약 87.7→1.21°, 85.2→9.75°로 줄었다. CAL 전체 위치 평균은 116.291→115.545cm, 회전 평균은 19.290→17.269°, 성공은 37→40/76이었다. 세션 bootstrap의 차이 구간은 0을 포함하며, 새 모델의 독립 우위가 확정된 결과는 아니다. `PCA64` 조합은 false flip 세 개가 있어 선택하지 않았다. 이 결과를 바탕으로 PCA32 모델을 고정해 실제 이미지에서 실행하는 후속 후보를 준비한다.

RGB soft-support 특징 두 개가 이름 불일치로 legacy 분류기에 0으로 들어가던 문제는 별도 `soft_support_v2` schema로 수정했다. 기존 legacy 입력/모델 결과는 그대로 재현하며 수정 schema의 CAL AUC는 0.8775→0.8750으로 향상되지 않았다. 따라서 이것만으로 원래 실패 원인을 설명하거나 새 모델을 추천하지 않는다.

사진처럼 세 면이 만나는 내부 KP를 위해 면별 색 전환선 교차점도 시험했다. 합성 primitive는 동작했으나 실제 CAL25개 3-face junction에서 두 개의 지지 선을 얻지 못해 좌표 변경은 0개였다. 구멍, 격자, 면 안의 음영 때문에 면 하나의 대표색이 경계 RGB를 설명하지 못했다. 가시성·색 유사성·좌표 관측 가능성을 구분해야 하는 추가 근거다.

새 정량검증 후보도 메타데이터로 조사했다. 과거 FINAL146과 겹치지 않는 83개 RGB는 있으나 모두 annotation이 없으며, 기존 FINAL과 같은 `plastic_night_01` 세션이고 과거 이미지 review에는 포함돼 있다. 새 독립 reference나 수집 없이 이 83개를 새로운 물리 정확도 검증으로 사용할 수 없다. 조사 단계에서는 이미지 디코딩과 GT 내용 읽기를 수행하지 않았다.

## 실행

API v1은 기존 최초 추론 결과 뒤에 직접 연결할 수 있다.

```python
from scripts.research.pallet_visibility_curriculum_20261013_v1.api import VisibilityCurriculum

processor = VisibilityCurriculum.from_config(
    "_docs/experiments/pallet_visibility_curriculum_20261013_v1/LIVE_CONFIG_V1.json")
result = processor.refine(image_bgr, keypoints_xy, camera_K, initial_pose)
pose = result["actual_pose"]
states = result["appearance"]["states"]
```

하나의 이미지를 처리하려면 `input.json`에 `points`, `K`, `initial_pose`를 넣고 실행한다.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_visibility_curriculum_20261013_v1.api \
  --image IMAGE.jpg --input input.json \
  --config _docs/experiments/pallet_visibility_curriculum_20261013_v1/LIVE_CONFIG_V1.json \
  --output result.json
```

연구 재현용 batch CLI는 `run infer`로 예측을 먼저 seal하고, 동일한 output에 `run score`를 수행한다. output은 새 경로를 사용해야 한다. CAL 이외의 결과를 읽은 뒤 후보를 바꾸는 경우 재사용 평가임을 명시해야 한다.

현재 우선 해결할 문제는 주간 외부 가림 일반화, 초기 branch를 잘못 지지하는 관측 좌표, 가상/둥근 모서리의 위치 정의다. 색 지지율만으로 정확한 모서리를 보증할 수 없으므로, 불확실한 관측을 보류하고 독립적인 이미지 좌표 증거를 얻는 방향으로 실험을 이어간다.
