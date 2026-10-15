# 상태별 실측 관측 보호와 rigid completion: 논문 포지셔닝

독립 방법론 검토, 2026-10-13. 원 논문과 공식 문서만 확인했다. 새로운 추론·학습·파라미터 선택·TEST/FINAL 정답 접근은 수행하지 않았다. 아래는 구현·검증 계약의 제안이며, 새 방식의 성능이나 신규성을 입증한 결과가 아니다.

논문의 핵심 문제는 **보정하기 쉬운 경계 관측을 확보한 다음, 더 불확실한 내부 관측을 추가할 때 처음 확보한 관측을 훼손하지 않고, 관측하지 못한 점은 등록된 rigid 모델로 별도 완성하는 것**으로 정리할 수 있다. 경계점의 이미지 좌표와 자세를 맞춘 뒤 얻은 투영 좌표를 구분하고, 최종 2D 출력과 최종 rigid 자세가 서로 다른 종류의 추정임을 표시해야 한다.

## 가까운 선행 연구와 주장 범위

| 원 논문 | 이미 제시된 내용 | 이번 방법에서 구분할 부분 |
|---|---|---|
| [PVNet, CVPR 2019](https://arxiv.org/html/1812.11788), §3.1–3.2 | 픽셀 방향 투표로 가림·화면 밖 키포인트를 추정하고, 위치 공분산을 사용한 Mahalanobis PnP를 푼다. | 불확실성 가중 PnP나 보이는 부분을 통한 숨은 점 추정을 새 발명으로 주장할 수 없다. 실제 RGB 경계 보정 관측의 고정 provenance와 후속 보정의 보호 계약을 검증해야 한다. |
| [MLPnP, ISPRS Annals 2016](https://isprs-annals.copernicus.org/articles/III-3/131/2016/index.html) | 관측 공분산을 반영한 최대우도 PnP와 자세 정확도 추정을 다룬다. | 가중 최적화 자체가 아니라, 공분산의 조건부 성격과 prior 의존도를 보고하고 실제 관측을 보호하는 절차가 기여 후보이다. |
| [VAPO](https://arxiv.org/html/2403.14559v5), §III-A/C/D, IV-E | 외부 가시성과 self-occlusion을 분해하고, 가시성 기반 중요도로 키포인트를 선택한다. mask와 표면 법선에 기초한 label 생성도 다룬다. | 가시성 taxonomy 또는 쉬운 점 선택 자체의 최초성을 주장할 수 없다. 본 연구의 initial-pose 조건부 상태, RGB 관측, 좌표 정밀도와 실제 선 지지를 분리하는 계약을 비교해야 한다. |
| [HybridPose, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Song_HybridPose_6D_Object_Pose_Estimation_Under_Hybrid_Representations_CVPR_2020_paper.html) | 키포인트·edge vector·대칭 대응을 rigid 모델과 함께 robust하게 정렬한다. 한 표현의 오류를 다른 기하학 신호가 보완한다. | 키포인트와 선 및 등록 모델을 결합한다는 주장만으로 신규성이 성립하지 않는다. 선 교점 관측의 보호와 내부점 추가의 실패·rollback을 별도로 입증해야 한다. |
| [Pix2Pose, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Pix2Pose_Pixel-Wise_Coordinate_Regression_of_Objects_for_6D_Pose_Estimation_ICCV_2019_paper.html) | 픽셀별 3D 좌표와 예상 오류를 예측하고, 가려진 부분을 복원하여 PnP-RANSAC 대응을 구성한다. | geometry completion은 새 개념이 아니다. 본 방법은 숨은 점에 새로운 이미지 관측이 있다고 주장하지 않고, 추정 자세의 출력 투영으로만 완성한다는 점을 명시한다. |
| [DeepIM, ECCV 2018](https://arxiv.org/abs/1804.00175) | 초기 자세에서 렌더링한 이미지와 입력 RGB를 비교해 상대 자세를 반복 보정한다. | 반복 refinement 자체가 신규성은 아니다. 본 방법의 공개된 점별 관측·보호·제외 기록과 matched initializer 비교가 필요하다. |
| [PoseFix, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html) | 기존 추정기에 붙이는 model-agnostic **human** pose refinement를 제시한다. | 저장소의 파렛트용 PoseFix-derived head와 원 논문의 인간 자세 실험은 구분한다. 기존 파렛트 head에 SubPix를 붙인 조합은 별도의 명시된 baseline이다. |

PVNet Fig. 3은 bounding-box corner보다 실제 object-surface keypoint의 localization variance가 작은 예를 제시한다. 파렛트의 둥근 모서리·구멍과 virtual cuboid corner의 차이는 단순 픽셀 잡음과 구분해야 한다. 실제 마스크의 두 선 교점도 요청한 3D 모서리의 이미지라고 자동 인증하지 않는다. VAPO의 법선 기반 internal visibility는 비볼록 물체의 추가 ray intersection을 생략한 근사라는 한계도 원문에 명시되어 있다. [PVNet](https://arxiv.org/html/1812.11788), [VAPO](https://arxiv.org/html/2403.14559v5).

좁은 기여 후보는 **가시성·실측 지지·좌표 보정·rigid completion의 provenance를 유지하면서 경계 관측을 보호하고 내부 관측을 점진적으로 추가하는 검증 가능한 refinement 절차**다. 이 문헌 검토는 최초성을 증명하지 않는다. paper에서는 해당 결합의 구현·통제 실험을 먼저 제시하고, “최초”, “항상 보이는 점을 판별”, “불확실성 보정 완료”, “최고 성능”이라는 포괄적 표현을 피한다.

## 정확한 hard lock의 불가능성과 적절한 보호 계약

등록된 native 3D 점을 $X_i$, 카메라를 $K$, 자세를 $T\in SE(3)$, projection을 $u_i(T)=\pi_K(TX_i)$라 하자. 경계에서 실제 이미지로 얻은 보정 좌표 $a_i$를 변하지 않는 측정으로 보존하는 것은 가능하다. 그러나 모든 anchor에 대해 $u_i(T)=a_i$를 강제하는 것은 별도 문제다.

자세의 자유도는 6이며 $k$개 2D anchor는 일반적으로 $2k$개 등식을 준다. $k\ge4$이면 제약 수가 자유도를 초과한다. 선형화하면 $J_A\,\delta\xi=\delta a$이고 $J_A\in\mathbb{R}^{2k\times6}$이다. 일반적인 측정 잡음에는 $\mathrm{col}(J_A)$ 밖의 성분이 있으므로 이를 만족하는 rigid pose가 보통 존재하지 않는다. 더 적은 점에서도 퇴화·다중해가 가능하다. 이것은 본 검토의 기하학적 도출이며, 단순 점 수만으로 안정적인 자세를 보장하지 않는다는 공식 PnP 문서와도 일치한다. [OpenCV PnP 문서](https://docs.opencv.org/4.8.0/d5/d1f/calib3d_solvePnP.html).

또한 $J_A$의 rank가 6이면 exact lock을 유지한 채 가능한 작은 자세 변화는 일반적으로 0이다. 내부 관측을 추가해도 자세를 움직일 여지가 없다. 불일치를 감추려고 anchor를 seed pose의 투영으로 바꾸면 실제 관측을 모델 결과로 치환하는 순환이 된다.

따라서 측정 $a_i$를 immutable하게 유지하고, boundary seed $T_A$ 기준으로 각 anchor의 허용 변화를 제한한다. 예를 들면 후보 자세에 대해

$$
\|u_i(T)-u_i(T_A)\|_2\le d_A,\qquad
\ell_i(T)\le\ell_i(T_A)+\epsilon_A,\quad i\in A,
$$

$$
\ell_i(T)=\rho_\delta(\|u_i(T)-a_i\|_2).
$$

이는 정확한 pixel lock 대신 명시된 오차 예산으로 관측을 보호하는 계약이다. $d_A,\epsilon_A$ 및 loss의 가중 여부는 예측 전에 고정한다. 매 단계의 직전 자세가 아니라 **같은 seed**를 기준으로 검사해야 누적 drift를 막는다. 후보를 사후 거절하는 guard라면 제약 최적화의 전역해를 구했다고 설명하지 않는다. 현재 solver 제안의 2px·1px² 같은 수치는 사전 고정 가설이며 calibrated noise probability가 아니다.

## 내부점 추가와 숨은 점 완성의 수학적 구분

고정된 eligible visible pool을 $V$, 보호할 boundary anchors를 $A\subseteq V$, 현재 실제 측정 support를 $S\subseteq V$라 하자. 추가 내부점은 원 RGB 기반 좌표 $z_i$를 사용하고, 모델 projection으로 생성한 좌표를 넣지 않는다. 한 후보의 guard용 비용은 예를 들어

$$
F_S(T)=\frac{\sum_{i\in S}w_i\rho_\delta(\|u_i(T)-z_i\|_2)
+\|\operatorname{Log}(T_0^{-1}T)\|_{\Lambda_0}^{2}}
{\sum_{i\in S}w_i}.
$$

단계 전후에는 같은 $S$, weights, fixed initial prior를 사용해 비용을 비교한다. 서로 다른 support의 정규화 비용을 그대로 비교하지 않는다. 가시성 confidence와 coordinate-accuracy cue를 분리하고, “독립 관측”은 pose projection을 재활용하지 않았다는 의미로만 사용한다. 동일 선·동일 segmentation·동일 detector crop을 공유하는 오차는 통계적으로 독립적이지 않다.

3점 seed가 이동만 추정한 뒤 내부점을 추가하여 rank 6을 확보한다면 이후 전체 자세 보정은 가능하다. 이때 “초기 회전 정확히 유지”라는 약속은 3DOF seed 단계에만 해당한다. 최종 단계가 6DOF인지 출력에 기록해야 한다. 관측이 부족하거나 rank·depth·trust·anchor guard를 통과하지 못하면 declared seed/initial fallback을 반환한다.

최종 후보는 immutable $V$의 동일 비용과 초기 reference에 대해 검사해야 한다. seed가 이 최종 guard를 통과하지 못해도 반환하는 정책이라면 **boundary seed의 별도 수락을 보존하는 예외**임을 기록해야 한다. 모든 반환 자세가 all-visible guard를 통과했다고 쓰면 안 된다. “latest passing checkpoint”와 GT 기준 best stage는 서로 다르며 후자는 inference 선택에 금지한다.

실제 CAL 추론 전에 고정하는 `anchor_priority` 정책은 이 예외를 명시적으로 허용한다. 전체 pool 검사와 보호 계약을 통과한 확장 checkpoint가 없으면, 자체 경계 관측·prior 비용·rank·trust 기준을 이미 통과한 boundary seed를 유지한다. seed가 없으면 초기 자세로 돌아간다. `strict_eligible_pool`은 seed까지 전체 pool 검사로 거절하는 대조 정책이다. 이는 합성의 편향된 내부점 사례와 관측 우선순위에 근거한 사전 설계이며 실제 평가 정답에 따른 선택이 아니다. 반환 seed에는 `full_pool_passed=False`, 예외 이유, seed 자체 수락 근거와 seed 기준 보호 결과를 기록한다. 이 모드에서도 약한 관측을 새 projection으로 바꾸거나 관측 좌표를 움직여 예외를 정당화해서는 안 된다.

미관측·self-occluded·truncated 점 및 사용하지 않은 모서리의 완성은

$$\hat q_i^{\mathrm{completion}}=u_i(\hat T)$$

로 정의하고 `PROJECTED_COMPLETION` provenance와 fitted `used_ids`를 분리한다. 이는 주어진 등록 모델과 추정 자세에 조건부인 좌표다. 새로운 RGB 관측, 숨은 점의 실제 가시성 인증, GT 정확도 보장은 아니다. 출력 2D anchor를 측정 그대로 유지한다면 모든 출력이 rigid projection과 정확히 일치한다고 주장할 수도 없다. `corrected_measurements`, `projected_points`, `completed_points`를 별도로 제공한다.

## 관측 불확실성과 prior 의존도

데이터의 국소 정보는 $\mathcal I_{\rm data}=J^T WJ$로 검사한다. 강한 prior가 있으면 $\mathcal I_{\rm data}+\Lambda_0$가 invertible해도 데이터 자체가 자세의 모든 축을 결정한다고 볼 수 없다. calibrated Gaussian 관측과 적절한 국소 근사라는 가정 아래 posterior 형태 $\Sigma_T\approx(\mathcal I_{\rm data}+\Lambda_0)^{-1}$를 쓸 수 있지만, 현재의 line-bootstrap 공분산을 넣었다고 이 가정이 성립하지는 않는다.

따라서 data-only rank·스케일을 명시한 singular values/condition, prior 비용, fallback과 추정 DOF를 함께 기록한다. 회전·이동 단위에 따라 condition이 달라지므로 재현 가능한 파라미터 스케일이 필요하다. 동일 경계선 공유로 생기는 상관을 무시한 block-diagonal weight는 모델링 근사로 설명한다. 관측 공분산과 posterior pose 공분산, calibrated correctness score를 같은 “신뢰도”로 부르지 않는다. uncertainty-aware pose fitting의 선행 사례는 [MLPnP](https://isprs-annals.copernicus.org/articles/III-3/131/2016/index.html)와 [PVNet](https://arxiv.org/html/1812.11788)이다.

외부 물체 가림도 별도 의미 상태다. 낮은 target RGB 지지, mask 밖, 선의 gap은 배경·구멍·잘못된 query·virtual corner와 구별되지 않을 수 있다. 기본값은 `UNKNOWN_OR_OCCLUSION`이다. 독립 다른-instance 영역 증거는 suspected 상태를, 같은 ray의 독립 requested-point depth와 전경 깊이 순서는 confirmed 상태를 뒷받침한다. initial pose의 model depth를 confirmed evidence로 재활용하면 순환이다. 공급된 증거의 실제 검증 책임은 producer에게 있다.

## 사전 고정 통제 실험과 비교 요구

핵심 finite ablation은 같은 초기 추정·H·native 대응·이미지 관측·eligible pool·weights·prior·loss·trust limits를 유지한 다음 다음 차이만 바꾸는 것이다. severe 제외와 entire CAL cohort 유지, prediction seal 후 isolated truth scoring을 동일하게 적용한다.

| 통제 비교 | 확인하는 질문 |
|---|---|
| 기존 boundary-count seed 대 seed + protected visible-interior supplementation | 충분한 실제 내부 관측이 생긴 경우 seed의 제한을 보완하는가? |
| 동일 supplementation에서 보호 guard 켜기/끄기 | anchor 훼손을 실제로 줄이는가, 과도한 fallback이나 전체 pose 악화를 만들지는 않는가? |
| 동일 eligible 관측의 staged 추가 대 together fit | 쉬운 순서 자체의 효과인가, pool·weight·rollback 차이의 효과인가? |
| 동일 solver의 RAW 대 independent RGB FUSED 좌표 | 좌표 보정 효과와 taxonomy/solver 효과를 구분할 수 있는가? |

모든 frame을 유지한 이동·물리 회전·성공·tail error와 paired session 결과를 보고한다. 관측 anchor의 GT pixel error와 seed 대비 drift는 서로 다른 값으로 보고하며, 내부점 보충·rank 회복·거절 이유·fallback·사용한 DOF도 공개한다. 숨은 점의 GT projection error는 completion 평가로 별도 보고하고 “숨은 점 검출 성공”으로 부르지 않는다. 보호 목적함수 개선은 pose accuracy 개선의 증거와 구분한다.

비교기는 현재 로컬에서 실행 가능한 PoseFix-derived head, SubPix, 기존 N3 및 frozen joint/progressive refiner를 먼저 같은 입력/예산에 붙인다. 고정 좌표에서 robust weighted PnP/LM 제어군은 uncertainty fitting과 보호의 차이를 확인하는 데 중요하다. OpenCV SubPix의 `winSize=(5,5)`는 11×11 영역이다. [OpenCV cornerSubPix 문서](https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html). 실제 물리 표면 모서리가 없을 때 물체 내부의 다른 gradient 구조로 이동할 가능성은 이 알고리즘을 virtual corner에 적용할 때의 위험에 대한 본 검토의 추론이다.

DeepIM 같은 learned render-and-compare refiner를 실행하지 않았다면 실험 우위를 주장하지 않는다. 더 넓은 6D refiner 예인 [FoundationPose, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html)는 모델/참조 이미지·관측 modality·큰 외부 학습 데이터·초기화 조건을 별도로 맞춰야 한다. 관련 연구 인용은 직접 실행한 matched baseline의 대체가 아니다.

이미 여러 차례 본 nonsevere TEST49의 추가 고정 replay는 exploratory robustness evidence로만 보고한다. 새 CAL 가설을 고정했다고 이 TEST가 다시 독립적이 되지는 않는다. 아직 보지 않은 촬영 단위의 검증이 없다면 논문은 해당 한계를 명시하되, 재현 가능한 구현·보호 계약·통제 CAL 실험·기존 baseline 비교 결과를 완결된 연구 산출물로 제공할 수 있다. 독립 검증이 없다는 이유로 구현 자체를 미완성으로 남겨둘 필요는 없다.
