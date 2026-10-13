# 숨은 KP의 2D 가능 범위로 자세 후보를 줄이는 설계

2026-10-13, 향후 확장안. 현재 frozen V1은 변경하지 않았다. 새 추론·GT 접근·영역 추정기 구현·성능 실험은 없다.

**가능하다.** 숨은 점을 정확한 한 좌표로 맞추는 대신, 이미지로부터 예측한 넓은 가능 범위에 그 점의 투영이 들어가는지 확인하면 서로 다른 자세 후보를 구별할 수 있다. 후보마다 숨은 점의 투영 위치가 충분히 달라야 하고, 범위가 정확하면서도 충분히 좁아야 효과가 있다. 크게 줄어드는 정도와 정답 후보 보존은 현재 확인되지 않았다.

## 범위의 원천과 현재 사용 한계

| 범위의 원천 | 의미 | 현재 주장할 수 있는 것 |
|---|---|---|
| 원 RGB의 hidden-KP heatmap, 회귀 좌표와 이미지별 오차 scale | 보이는 몸체의 문맥으로 추론한 이미지 기반 위치 증거 | 직접 보인 점은 아니며, 초기 PnP와 같은 RGB/예측을 공유하므로 통계적으로 독립적이지 않다. |
| 초기/최종 pose의 hidden projection ± 임의 반경 | 기존 pose 주변의 prior | 새로운 이미지 증거가 아니다. 이 범위를 만족했다고 그 pose를 독립 검증한 것이 아니다. |
| 현재 pose 정보행렬에서 전파한 2D ellipse | 등록 모델·관측 noise 가정·prior에 조건부인 출력 불확실성 | hidden image observation으로 넣거나 calibrated 95% 범위로 부르면 안 된다. |

현재 후보 source에는 point prediction과 visibility/accuracy heuristic은 있지만 검증된 hidden-location interval은 없다. `P_visible`이나 `P_usable`은 2D 위치 범위의 coverage가 아니다. 기존 nominal 95% pose-derived ellipse의 실제 포함률은 약 63–68%였으며, accepted·model-compatible corner에 한정된 진단이다. [현재 상태별 분석](PER_STATE_FINDINGS_V1_NOTES.md).

## 선행 연구와 보장의 조건

[Yang·Pavone, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Object_Pose_Estimation_With_Statistical_Guarantees_Conformal_Keypoint_Detection_and_CVPR_2023_paper.html)은 heatmap을 conformal calibration으로 원/타원 prediction set으로 만들고, 기하학적 제약을 pose uncertainty set인 PURSE로 전파한다. §3–5의 핵심은 별도 calibration과 미래 표본의 exchangeability이며, worst-keypoint max score를 통해 여러 점의 joint coverage를 취급한다. 보장은 marginal coverage이고 모든 사진·각 가림 상태의 조건부 보장은 아니다. 단일 영상의 시간 상관은 그 가정을 깨뜨릴 수 있다. [저자 원문](https://arxiv.org/html/2303.12246), [공식 코드](https://github.com/NVlabs/ConformalKeypoint).

따라서 점마다 “95%”라고 말한 뒤 모든 hidden 범위를 동시에 hard intersection해 pose가 95%로 맞는다고 할 수 없다. 별도 보정자료, 고정된 native 정의·카메라·등록 모델, joint error 제어와 촬영 단위의 검증이 필요하다. 이미 head 학습·방법 선택에 사용한 CAL46과 여러 차례 본 TEST49를 새로운 독립 calibration/confirmation으로 재명명하지 않는다. PURSE의 연속 집합 coverage도 우리가 만든 유한 후보 중 하나를 고르는 top-1 accuracy 보장을 자동으로 제공하지 않는다.

## 기존 관측을 보호한 유한 후보 pruning

먼저 같은 visible 측정·native 대응·등록 치수·초기 family에서 **고정된 동일 후보 집합** $\mathcal C$를 생성한다. 경계 관측의 실제 좌표와 fixed-seed 보호·depth·trust 검사도 유지한다. 3개 진짜 대응의 P3P는 여러 유한 full-pose 해를 낼 수 있다. 이를 쓰는 실험은 현재 V1의 3점 translation-only 정책과 다른 사전 선언 arm이어야 한다. 영역을 넣었다고 가짜 대응을 추가하거나 fitted point 수·추정 DOF를 올려 표시하지 않는다.

이미지 기반 hidden 범위를 $\mathcal R_j(I)$라 할 때, hard 방식은

$$\mathcal C_{\rm keep}=\{T\in\mathcal C:\pi_K(TX_j)\in\mathcal R_j(I)\ \text{for all selected }j\}$$

로 후보를 거른다. 충분한 joint calibration이 없는 현재에는 **soft region score**가 더 적절한 연구 출발점이다. 예를 들어 후보 투영과 영역 사이 거리를 confidence에 따라 약하게 벌점화한다. 이를 likelihood라고 부르려면 실제 분포 모델의 검증이 추가로 필요하다. 같은 RGB에서 나온 pose prior와 region score를 독립 likelihood처럼 곱하거나 중복 사용하여 과신하지 않는다. 이는 제안이며 현재 score·반경·추정기가 존재한다는 뜻은 아니다.

모든 후보가 탈락하거나 영역이 무정보이면 기존 guarded 결과를 반환하고 이유를 남긴다. 후보 pool이 정답 근처를 포함하지 않으면 pruning으로 정답을 만들 수 없다. 영역 안의 임의 픽셀이나 기존 pose projection을 새 2D 관측으로 샘플링해 PnP에 다시 넣지 않는다. 숨은 영역은 후보 선택용 증거이며 출력 geometry completion과 분리한다. truncation/모호한 점에 쓸 범위가 없으면 constraint를 생략한다. 화면 내 heatmap만으로 화면 밖 KP의 범위를 보장하지 않는다.

## 다음 검증과 주장 범위

추후 사전 고정 비교는 **동일 후보 pool의 visible-only 선택 대 image-region soft pruning**이어야 한다. 후보 감소율, reference-compatible 후보의 탈락, 전체 pose·hidden pixel error·fallback을 함께 보고한다. hard 영역은 그 이전에 촬영 단위로 분리한 자료에서 coverage·영역 크기·가림 상태별 실패를 확인한다. GT는 봉인 뒤 평가에만 사용하고 hidden 범위나 native slot 선택에 넣지 않는다.

현재 human SELF 2/4점이 initial-H의 오판으로 fit에 들어간 결과, virtual/rounded cuboid corner와 실제 물질 edge의 차이, 고정 native/body phase와 reference 자체의 오차가 남아 있다. 따라서 “숨은 점 범위가 후보를 줄일 수 있다”는 설계는 타당하지만, **현재 이미 calibrated 범위를 갖고 있거나 후보를 크게 줄여 성능을 높였다는 주장은 없다.**
