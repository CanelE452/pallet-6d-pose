# KP 상태에 따른 관측 보호와 기하학적 자세 후보 평가

KP 상태를 단순한 보이는/안 보이는 필터로 쓰지 않고, 각 상태가 제공하는 좌표·영역·기하학 정보를 구분하는 연구 구현이다. 보정한 이미지 좌표는 보존하며, 가려진 점은 최종 등록 모델의 투영으로 출력한다. 이미지에서 예측한 거친 위치 영역을 자세 후보 평가에 추가했다. 현재 자료에서는 영역 벌점의 추가 자세 개선은 확인되지 않았다.

## 상태별 처리

| 상태 또는 증거 | 구현에서 사용하는 정보 | 처리와 한계 |
|---|---|---|
| 보이는 외곽점 | RGB 지지, 두 개의 실제 이미지 경계선 교차와 오차 정보 | 보정 좌표를 이미지 관측으로 보존한다. 해당 3D 코너의 물리적 소유권을 보증하는 것은 아니다. |
| 모델 내부의 보이는 점 | 보이는 점의 예측 좌표 또는 실제 RGB incident-line junction | 검증을 통과한 관측을 난이도 순으로 추가한다. 이 TEST에서 실제 새 junction의 최종 사용은 N3 0개, PoseFix 1개였다. |
| 초기 모델에서 self-occlusion으로 판단한 점 | 고정된 등록 기하학, 신경망의 원 좌표와 거친 영역 | 정밀 대응점으로 사용하지 않는다. 영역은 후보 벌점용이며 숨은 출력은 한 자세의 투영이다. 초기 self 판정은 조건부이다. |
| 다른 물체 가림 또는 불명확한 점 | 관측이 성립하지 않는 원 예측의 넓은 위치 영역 | 정밀 관측과 분리한다. 현재 실제 depth/다른-instance 생산기가 없어 외부 가림을 독립적으로 확정했다고 주장하지 않는다. |
| 화면 밖 또는 위치 정보가 없는 점 | 원 query의 유효성, 초기 모델의 화면 밖 진단 | 관측에서 제외하고 모델로 완성한다. 원 예측이 화면 밖이라는 사실을 실제 물리적 코너의 truncation 증명으로 사용하지 않는다. |

확실한 외부 가림의 depth 순서, 물리적으로 확인된 truncation의 화면 밖 부등식은 그 증거를 실제로 얻었을 때 적용할 수 있는 추가 제약이다. 이번 실행에 그런 생산기가 들어 있다고 주장하지 않는다. 이 cohort의 사람 주석 TRUNCATED 사례도 0개이다.

## 추정과 출력

보이는 관측의 좌표·가중치와 초기 pose prior를 고정한다. 보정된 외곽 관측의 모델 투영은 한 고정 기준에서 점별 2px 이동, Huber 손실 증가 1px² 이하로 보호한다. 기준 pose가 없으면 초기 pose를 명시적인 고정 기준으로 삼는다. 노이즈가 있는 여러 이미지 좌표를 한 강체 모델에 모두 정확히 고정하지 않는다.

유한 후보는 기존 진행 보정·동결된 RGB joint 보정과, 세 외곽점의 P3P가 생성한 최대 네 해이다. 원 구현의 3점은 회전을 고정한 translation-only 정책이었다. 새 P3P 실험은 실제 세 대응으로 6DoF 해를 생성하는 별도 정책이며, 숨은 영역을 가짜 대응점으로 세지 않는다. 국소 Jacobian rank·조건수, positive depth, 움직임 범위와 관측 보호를 검사한다. 이것은 전역 유일성 증명이 아니다.

거친 영역은 신경망의 **원 이미지 예측**을 중심으로 하고, CAL 오차의 경험적 90% 분위수를 사용한다. pose 투영이나 기존 정보행렬의 타원을 새 이미지 증거로 쓰지 않는다. 초기 PnP와 RGB/예측을 공유하므로 독립 likelihood나 보정된 확률로 해석하지 않는다. 영역 밖 거리의 정규화 벌점에 고정 가중치 0.10을 사용하며, 영역 안에서는 벌점이 0이다. 강제 제거는 사용하지 않는다.

`reported_points` / `completed_points`는 실제 외곽 보정 좌표와 최종 사용한 관측을 보존하고 나머지를 모델로 채운 혼합 2D 추정이다. `rigid_model_corners`는 한 pose에서 나온 별도의 강체 투영이다. 두 배열은 용도가 다르며 다시 all-point PnP에 넣지 않는다.

## 고정 실험 결과

새 학습·주 평가에서 severe를 제외했다. CAL 46장, TEST 49장을 사용하며 N3와 PoseFix는 같은 TEST 사진의 두 입력 변형이다. CAL은 이전 head 학습과 방법 선택에 사용됐고 TEST도 이미 관찰된 자료이므로 독립 일반화 결과라고 주장하지 않는다. primary pose 점수는 모든 사진을 포함한다.

| 입력 / 평가 | 초기 평균 위치 오차 | 상태별 방법의 평균 위치 오차 | 초기 / 최종 5cm·5° 성공 |
|---|---:|---:|---:|
| CAL N3, 46장 | 3.9821cm | 3.6093cm | 34 / 37 |
| TEST N3, 49장 | 3.3630cm | 3.0384cm | 40 / 40 |
| TEST PoseFix, 49장 | 3.2161cm | 2.9474cm | 39 / 42 |

이는 고정된 `FINITE_REGION` 결과이다. 같은 후보 집합에서 영역 가중치만 0으로 둔 `FINITE_VISIBLE`과 자세·오차가 **모두 동일**했다. P3P 선택은 CAL 2장, N3 TEST 0장, PoseFix TEST 1장이었다. 별도의 엄격한 기존 보정기 후보 선택 `PROPOSALS`는 0장이다. 다른 기존 방법보다 항상 우수하다는 결과도 아니다.

영역 벌점이 양수인 새 후보는 CAL 30개, N3 TEST 48개, PoseFix TEST 30개였고, 모두 기존 관측·기하학 검사에서 이미 탈락했다. 영역이 이 후보 집합에 추가 정보를 제공하지 못한 원인이 확인된다. 경험적 영역의 평균 반경은 약 21–26px이다. 반경·가중치를 TEST 결과에 맞춰 다시 조절하지 않았다.

좌표 보존 자체는 이득이 확인된다. 고정된 native 대응과 모델이 비교 가능한 TEST에서 외곽점 100개/101개의 원 오차가 각각 4.236→3.408px, 4.473→3.447px로 줄었다. 같은 코너의 최종 강체 투영 오차는 4.178px/4.355px이다. 숨은 점의 원 좌표에서 투영 좌표로의 개선 대부분은 초기 rigid 투영에도 이미 있었으며, 새 숨은 보정기의 우위로 귀속하지 않는다. reference는 기존 주석에서 재구성된 pose와 코너이다.

## 논문과 재현 자료

- [기본 방법 영문 초안](../anchored_completion_v1/METHOD_DRAFT_EN.md), [선행 연구와 주장 범위](../anchored_completion_v1/PAPER_POSITIONING_V1.md)
- [상태별 기존 결과와 한계](../anchored_completion_v1/PER_STATE_FINDINGS_V1_NOTES.md)
- [완성된 논문 초안](PAPER_DRAFT_EN.md), [상태별 확장 원고](STATEWISE_METHOD_AND_RESULTS_EN.md)
- [사전 고정 프로토콜](PROTOCOL.json), [CAL 영역 모델](REGION_MODEL.json), [독립 추론 감사](INFERENCE_AUDIT.json)
- [전체 검사 기록](FULL_TEST_RECEIPT.json): 255 tests, 199 subtests 통과
- [실제 RGB 추론과 캐시의 일치](live_example/PARITY.json): pose 및 출력 좌표 차이 0
- [그림과 캡션](figures/FIGURE_MANIFEST.json): 네 그림의 PDF / SVG / PNG

연구의 현재 결론은 관측의 상태와 정보의 정밀도를 구분하고 보정된 좌표를 보존하는 구현·검증이며, 숨은 영역의 추가 정확도 향상이나 최고 성능을 입증한 결과가 아니다. [양·파보네의 CVPR 2023 연구](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Object_Pose_Estimation_With_Statistical_Guarantees_Conformal_Keypoint_Detection_and_CVPR_2023_paper.html)도 키포인트 영역을 pose 집합으로 전파하지만, 확률 보장을 위해서는 별도의 보정과 통계 가정이 필요하다.

저장소 루트에서 선택적 실행:

```bash
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.bootstrap_foundation
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.bootstrap_joint
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.protected_completion_api \
  --image IMAGE.jpg --input INPUT.json \
  --config _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/PROTECTED_COMPLETION_CONFIG.json \
  --output OUTPUT.json --device cuda
```

입력에는 원 native KP, K, `R_cf`·`R_physical`·centroid·cf_extents를 담은 초기 pose와 등록 치수가 필요하다. 입력과 이미지의 해상도·좌표계를 유지한다. 새 추론은 GT를 읽지 않는다. 최종 설정은 선택적 연구 API이고 기존 모델을 자동 교체하지 않는다.
