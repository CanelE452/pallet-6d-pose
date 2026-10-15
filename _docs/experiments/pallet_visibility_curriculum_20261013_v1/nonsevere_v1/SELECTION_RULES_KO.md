# 관측한 경계점으로 보정 범위를 정하는 규칙

현재 보정기는 기존 PoseFix+Sub 또는 N3+Sub가 낸 좌표와 초기 자세에 붙이는 선택 기능이다. 확인된 경계점이 충분할 때만 자세를 보정한다. 학습·주 평가 범위는 severe를 제외한 clean/moderate이며, 기존 기본 추정기는 변경하지 않았다.

## 무엇을 경계점 한 개로 세는가

`geometric_boundary_count`는 초기 자세의 투영 외곽에 해당하고 초기 self-occlusion·truncation 및 실제 입력 좌표의 화면 밖·누락을 제외한 후보 수다. `RGB_material_boundary_count`는 이 후보 중 RGB 가시성 점수 `P_visible >= 0.65`인 수다. 두 수는 실제 두 경계선이 관측된 점 수와 다를 수 있다.

보정 분기에 쓰는 `image_supported_boundary_count`는 후보 조건과 높은 RGB 지지를 만족하면서, 해당 native corner에 연결된 **서로 다른 두 실제 이미지 경계선**과 그 교점이 검증된 점 수다. 선의 실제 샘플, 안쪽 파렛트/바깥 비파렛트 방향, 지지 구간, 잔차와 교점 일치를 검사한다. 교점은 원래 예측에서 12px 이내이며 화면 안에 있어야 한다. 제공된 공분산이 잘못되었거나 독립적인 다른 instance 증거와 충돌하면 이 수에서 제외한다. 내부점과 투영 완성점으로 부족한 수를 채우지 않는다.

이 점 수도 요청한 3D 모서리가 실제 표면에 보인다는 인증은 아니다. 구멍·둥근 모서리·가상 cuboid 모서리, 잘못된 마스크, 같은 색 물체가 영향을 줄 수 있다. 같은 경계선을 공유하는 교점들의 오차도 서로 연관될 수 있다. 수치 rank와 공간 배치를 별도로 검사한다.

| 실제 이미지 지지가 확인된 경계점 수 | 시도하는 보정 | 실패하거나 부족할 때 |
|---|---|---|
| 5개 이상 | 고정된 native 대응과 초기 물리 축을 유지한 전체 6DOF 보정 | 초기 자세·좌표 유지 |
| 4개 | 초기 prior를 포함한 6DOF 보정, 초기값 및 3축 ±8°의 고정 7개 시작점으로 모호성 검사 | 비슷한 비용의 서로 다른 자세가 발견되면 초기 결과 유지 |
| 3개 | 초기 `R_cf`와 `R_physical`을 정확히 유지하고 이동 3DOF만 보정 | 이동 rank·안정성·관측 비용 개선 실패 시 초기 결과 유지 |
| 2개 이하 | 보정하지 않음 | 초기 자세·좌표 그대로 반환 |

모든 보정은 선택한 실제 관측의 가중 Huber 비용과 고정 초기 prior의 개선, 양의 깊이, 점 배치·Jacobian 조건, 초기 자세에서의 이동 한계를 통과해야 한다. 4점 검사는 고정된 시작점 범위의 모호성 검사이며 전역 유일성을 증명하지 않는다. 3점 이동 규칙은 초기 회전 오차를 고칠 수 없다.

기본 `FUSED` 좌표는 실제 두 선의 교점 `x`와 원래 좌표 `q`를 `q + alpha * (x-q)`로 섞는다. `alpha = min(0.5, 4/(4+lambda_max))`이며 `lambda_max`는 검증된 교점 공분산의 최대 고유값이다. 최종 이동은 6px 이내다. 공분산·교점·혼합 계수의 일치를 검사한다. 이 공분산은 선 맞춤의 반복 측정 정밀도이며, 의미적 가시성이나 좌표 정답 확률로 해석하지 않는다. `RAW` 제어군은 원래 좌표를 사용한다.

보정 수락 후 사용하지 않은 **모든** 모서리는 `completed_points`에서 투영으로 완성하며, `projected_completion_ids`에 기록한다. 이 좌표는 출력 완성용이다. `corrected_measurements`와 `used_ids`만 실제 관측 및 적합 정보를 나타내며, 투영을 관측으로 다시 넣지 않는다. 아홉 번째 중심 좌표와 native 순서는 유지한다. 초기 결과로 돌아간 경우 입력 좌표를 그대로 반환한다.

## 다른 물체 가림은 언제 인정하는가

RGB가 파렛트 색과 다르거나 선 지지가 끊겼다는 사실만으로 다른 물체 가림을 확정하지 않는다. 기본 RGB/Tiny/MobileSAM 관측기는 다른 instance·깊이 증거를 생성하지 않으며, 이런 경우 `UNKNOWN_OR_OCCLUSION`으로 남긴다. 초기 self-occlusion과 truncation도 초기 cuboid 자세에 조건부인 상태다.

외부의 검증된 생산자가 `occluding_evidence`를 제공할 수 있다. 원래 native corner와 원래 `query_xy`에 결합된 `independent_occluding_region_v1` 기록이어야 하며, 파렛트 색 판단과 독립적이고 다른 instance가 그 위치를 차지한다는 관측, instance ID와 출처가 필요하다. 유효한 instance 영역 증거만 있으면 `OTHER_OBJECT_SUSPECTED`로 처리하고 보정 지지에서 제외한다. 파렛트와 색이 같아도 이 제외가 적용된다.

`OTHER_OBJECT_OCCLUSION` 확정에는 추가로 `depth_instance`의 동일 카메라 ray 깊이 순서가 필요하다. 요청한 실제 3D 모서리에 독립적으로 결합된 깊이와 그보다 가까운 다른 instance의 깊이를 검증하고, `요청점 깊이 - 전경 깊이 > 깊이 차이 불확실성`이어야 한다. 초기 자세를 투영해 만든 요청점 깊이는 이 확정 근거로 인정하지 않는다. helper는 입력 기록의 계약만 검사한다. 실제 instance·깊이 사실의 검증 책임은 외부 생산자에게 있으며, 현재 기록에는 전체 frame/camera identity 인증을 강제하는 기능이 없다.

## 같은 49장에서의 현재 결과와 사용 선택

아래 성공은 이동 5cm 이내·회전 5° 이내를 모두 만족한 수다. severe를 제외한 동일 49장 전체를 유지했다. 이 TEST49는 앞선 연구에서 이미 본 사진이므로 고정 규칙의 재사용 비교이며 독립 검증이 아니다.

| 방법 | 평균 이동 cm | 평균 회전 ° | 성공 |
|---|---:|---:|---:|
| PoseFix+Sub 초기 | 3.2161 | 6.3495 | 39/49 |
| PoseFix+Sub + RAW 개수 규칙 | 3.2477 | 6.3638 | 40/49 |
| PoseFix+Sub + FUSED 개수 규칙 | 2.7970 | 6.3166 | 42/49 |
| N3+Sub 초기 | 3.3630 | 6.5271 | 40/49 |
| N3+Sub + FUSED 개수 규칙 | 3.0655 | 6.4980 | 40/49 |
| 기존 N3 FUSED 진행 보정 | 3.0040 | 6.5243 | 43/49 |

개수 규칙은 각 입력 추정기에서 19장을 보정했고 나머지 30장은 초기 결과를 정확히 유지했다. 수락한 분기는 3점 7장, 4점 3장, 5개 이상 9장이다. CAL46에서도 보정하지 않은 장면이 32장이어서, 단순 기하학 외곽점 수가 많다고 적극적으로 보정하는 규칙이 아니다.

PoseFix+Sub에 FUSED 규칙을 붙이면 이 비교에서 성공 3장을 회복하고 성공 손실은 0장이었다. 평균 이동 차이 −0.4190cm의 세션 단위 paired-bootstrap 95% 구간은 [−1.5511, +0.0098]cm로 0을 포함한다. 기존 N3 진행 보정은 성공 수가 1장 더 많다. 따라서 PoseFix+Sub를 유지하면서 관측 지지가 확인된 장면에 제한적으로 보정하려면 이 선택 기능을 사용할 수 있지만, 기존 N3 후보를 자동으로 대체하는 승자로 선정하지 않는다. 사진별 정답을 보고 방법을 바꾸는 선택 규칙도 없다.

여기서 PoseFix+Sub는 별도 PoseFix `PRIOR1` 출력을 고정 OpenCV SubPix로 보정한 재계산 조합이다. SubPix는 `winSize=(5,5)`(11×11 탐색 영역), 최대 40회, epsilon 0.001이고 이동 상한은 PoseFix 입력 좌표 기준으로 원래 YOLO 대각선의 1%다. `winSize`는 탐색 영역 변 길이의 절반을 지정한다. [OpenCV 공식 문서](https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html)를 따른다. YOLO 좌표를 이동 제한 기준으로 삼은 대조군도 별도 비교에 남겼다. N3는 별도 학습 head이므로 N3+Sub 초기 결과를 PoseFix+Sub라고 부르지 않는다. 두 전체 파이프라인의 초기 추정과 조건부 가림도 달라질 수 있으므로 후처리만의 인과 비교로 해석하지 않는다.

## 현재 API와 실행 예

`BoundaryRuleCorrector.from_config()`에 고정 설정을 전달하고 `.refine(image_bgr, points, K, initial_pose, registered_dimensions, occluding_evidence=None)`를 호출한다. 입력은 실제 BGR 이미지, native 순서의 8/9 좌표, 카메라 K, 기존 `R_cf`/`R_physical`/`centroid`/`cf_extents`, 등록 치수다. upstream PoseFix 또는 N3를 이 API가 다시 학습하거나 실행하지 않는다. 결과의 `decision`, `keypoint_status`, `corrected_measurements`, `projected_completion_ids`를 함께 확인한다.

저장된 실제 CAL 3점 예제의 입력을 사용하는 CLI는 다음과 같다. 출력 경로는 새 파일이어야 한다.

```bash
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_rule_api \
  --image data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_cad/rgb/1778653030233998336.png \
  --input _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/boundary_rule_replay_v1/EXAMPLE_INPUT.json \
  --config _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/BOUNDARY_RULE_CONFIG.json \
  --output /tmp/pallet_boundary_rule_output.json --device cuda
```

이 예제는 봉인된 CAL 추론에서 첫 3점 수락 사례를 골랐고 정답으로 선택하지 않았다. 실제 RGB 재실행에서 두 초기 회전 행렬이 정확히 유지되고 나머지 5개 모서리만 출력 투영으로 완성됨을 검사했다. 필요한 RGB·모델 파일은 로컬에 있어야 한다. [`CLI_SMOKE.json`](boundary_rule_replay_v1/CLI_SMOKE.json), [`README_KO.md`](boundary_rule_replay_v1/README_KO.md), [`BASELINE_PROVENANCE.json`](posefix_sub_comparison_v1/BASELINE_PROVENANCE.json), [`BOUNDARY_RULE_REVIEW_V1.json`](BOUNDARY_RULE_REVIEW_V1.json)에 근거와 한계를 기록했다.
