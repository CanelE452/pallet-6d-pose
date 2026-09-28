# 선행연구와 현재 정보 계약으로의 전이

검토일: 2026-09-28. [확인] 공개 출판처, 저자 원문 및 공식 저장소를 직접 열어 확인했다. 정확한 접근 상태와 읽은 범위는 [SOURCE_REGISTRY.json](SOURCE_REGISTRY.json)에 있다. `FULLTEXT_METHOD`는 해당 방법 절을 읽었다는 뜻이며 논문의 모든 부록·실험·코드를 재현했다는 뜻이 아니다. `ABSTRACT_AND_OFFICIAL_CODE_README`는 원문 방법을 읽은 것으로 취급하지 않는다. 이 문서의 실행 권고는 [추정]이며 실제 결과는 후속 실행 보고서가 결정한다.

## 현재 조건과 먼저 실행할 작은 대조

[확인] 현재 main은 고정 Replay9장/38점 교사가 보정한 타깃을 RGB 학생의 pose head/flow에 전달한다. Plastic217장 또는 Wood361장과 같은 synthetic replay를 쓰며 기본 320 update다. 실제 inference 정보는 RGB, 기존 K와 등록 치수다. 새로운 depth, 평가 좌표, 평가 pose, 정답 crop, texture mesh를 가져올 수 있다고 가정하지 않는다. 원문에 나타난 성공을 이 조건의 성공 증거로 쓰지 않는다.

[추정] 우선순위는 (1) 같은 frozen pose 후보의 표준 robust residual 점수, (2) TRAIN 타깃 추종·실제 source/target 손실 및 gradient 측정 후 단일 학습 전달 대조, (3) 적격 TRAIN에서만 위치 신뢰성 진단이다. 1은 신규 학습이 없고, 2는 기존 loss를 유지한 RAW/REF 쌍으로 작게 검증할 수 있다. 3은 이미 시행한 stability 필터를 이름만 바꿔 반복하지 않는다. 세부 반론과 필요한 증거는 [CANDIDATE_CARDS.md](CANDIDATE_CARDS.md)를 참조한다.

[확인] 실행팀의 C1은 동일 후보/penalty에서 RMSE9만 Huber12 기반 픽셀 단위 점수로 바꾸었으나 [모든 arm의 선택변경과 gap 회수율이 0](cycles/C1_HUBER_D9/REPORT_KO.md)이었다. 이후 C2는 관측된 affine ON/OFF TRAIN residual 차이를 근거로 real translate/scale만 끄고 RAW/REF 쌍을 비교하도록 선택했다. source gradient 부호가 혼재한 관측만으로 replay 간섭을 확정하거나 PCGrad를 채택하지 않았다. C2는 새 loss가 아니며 결과 해석은 아래 D의 반론을 포함해야 한다.

## 직접 관련 연구 R1–R12

| ID / 출판 | 읽은 범위와 직접 출처 | 원래 입력·감독 | 현재로 옮길 원리 / 현재와 다른 전제 / 결정 |
|---|---|---|---|
| R1 PoseFix, CVPR 2019 | [출판처](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html), [저자 원문 §3–6](https://arxiv.org/html/1812.03595v2), [공식 코드](https://github.com/mks0601/PoseFix_RELEASE) | RGB crop+초기 사람 관절, 실제 사람 오류 통계에 맞춘 합성 입력 오류, GT 관절 | 초기 포즈와 영상의 결합, corruption 분포 일치 점검. 현재는 pallet9·소량9/38·팔레트 교란이며 사람 swap/inversion prior를 재현하지 않는다. 새 teacher부터 학습하지 않고 기존 teacher의 입력/오류·support를 대조. |
| R2 Self6D++, IEEE TPAMI 2024 | [공식 구현/출판 연결](https://github.com/THU-DA-6D-Pose-Group/self6dpp), [저자 원문 §3, RGB/RGB-D 식14–16](https://arxiv.org/pdf/2203.10339) | CAD/renderer, synthetic pretraining, RGB real; depth는 추가 선택지. visible/amodal mask 예측과 deep refiner | RGB-only 버전이 실제 존재한다. 따라서 depth 필수라서 배제하면 잘못이다. 그러나 mask·render comparison·별도 pose/refiner의 전제가 현재 sparse 9점 전달과 다르다. 가림/타깃 품질 분해 원리만 참고; 전체 프레임워크는 보류. |
| R3 ONDA-Pose, CVPR 2025 | [출판 초록](https://openaccess.thecvf.com/content/CVPR2025/html/Tan_ONDA-Pose_Occlusion-Aware_Neural_Domain_Adaptation_for_Self-Supervised_6D_Object_Pose_CVPR_2025_paper.html), [공식 README](https://github.com/tan-tao11/ONDA-Pose). 방법 전체 미독 | CAD-like radiance field, 실제 무주석 영상과 CAD, synthetic pose estimator, global refiner | 합성-실사 외관과 가림을 맞추는 원리. 등록 치수만으로 texture/radiance field를 얻을 수 없다. 실행 예제도 CARF geometry/texture 학습·rendering·refinement·self-training 여러 단계. 이번 저비용 대조로 채택하지 않음; 정확한 loss를 추측하지 않음. |
| R4 Pseudo Flow Consistency, ICCV 2023 | [출판처](https://openaccess.thecvf.com/content/ICCV2023/html/Hai_Pseudo_Flow_Consistency_for_Self-Supervised_6D_Object_Pose_Estimation_ICCV_2023_paper.html), [저자 원문 §3](https://arxiv.org/html/2308.10016v1), [공식 코드](https://github.com/YangHai-1218/PseudoFlow) | calibrated RGB, target 3D mesh, 초기 pose 주위의 여러 렌더, dense optical flow/teacher-student | 같은 3D점의 다중 synthetic-to-real 정합 일관성으로 감독 신뢰성을 측정한다. 여러 렌더를 새 실제 촬영 뷰로 부르지 않는다. 현재 flow는 YOLO RLE flow이며 이 논문의 optical flow가 아니다. 새 mesh/flow/refiner 훈련은 보류. |
| R5 Soft Teacher, ICCV 2021 | [출판처](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html), [원문 §3.1–3.3, §4.2](https://arxiv.org/html/2106.09018v2), [공식 소스](https://github.com/microsoft/SoftTeacher/blob/main/ssod/models/soft_teacher.py) | labeled/unlabeled detection images, EMA teacher, regression head로 jittered proposal 재회귀 | foreground score와 위치 품질을 분리한다. box jitter 후 회귀 결과의 분산으로 regression pseudo-box를 고른다. pallet crop/hint perturbation은 별도 전이 가설이고 같은 알고리즘이 아니다. 기존 stability 대조 재사용; TRAIN reliability 근거 없으면 가중치 학습 보류. |
| R6 Unbiased Teacher v2, CVPR 2022 | [출판처](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.html), [원문 §3.1–3.3](https://arxiv.org/html/2206.09500v1), [공식 구현](https://github.com/facebookresearch/unbiased-teacher-v2) | detector의 boundary별 별도 uncertainty branch; labeled regression으로 uncertainty 학습, teacher/student 비교 | teacher의 상대 위치 불확실성이 더 낮은 boundary만 감독. 현재 detector confidence를 이 uncertainty로 대체할 수 없다. Replay heatmap과 YOLO confidence는 서로 교정된 공통 단위가 아니다. uncertainty 출력을 증명하고 TRAIN calibration이 가능할 때만 후보. |
| R7 SSPCM, CVPR 2023 | [출판 초록](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Semi-Supervised_2D_Human_Pose_Estimation_Driven_by_Position_Inconsistency_Pseudo_CVPR_2023_paper.html), [공식 README](https://github.com/hlz0606/SSPCM). 방법 전체 미독 | labeled/unlabeled 사람 keypoint, 추가 auxiliary teacher, 시기별 두 교사의 disagreement, Cut-Occlude | 상보적 오류가 있는지를 teacher/student 교차표로 먼저 점검. multi-teacher 수만 늘리면 common bias는 남고 비용은 증가한다. 기존 multi-teacher 계약/결과와 비교 전 새 teacher fit 보류; 논문 loss 미구현. |
| R8 PCGrad, NeurIPS 2020 | [출판처/원문 §2–3, Algorithm1](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), [공식 코드](https://github.com/tianheyu927/PCGrad) | 같은 공유 파라미터 위의 여러 task loss와 gradient | 음의 내적이면 상대 gradient의 반대 성분을 투영 제거한다. 원문은 음의 cosine 단독으로 해롭다고 하지 않으며 gradient 크기 차이·curvature도 논한다. replay/real은 이 논문의 다중 task와 같지 않다. 먼저 step0 진단과 단순 가중/노출 대조; PCGrad 자동 도입 금지. |
| R9 PVNet, CVPR 2019 | [출판 초록](https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html), [공식 코드](https://github.com/zju3dv/pvnet). 방법 전체 미독 | RGB→pixel-wise keypoint 방향장→voting, CAD keypoints/학습용 segmentation·pose 준비 | 보이는 픽셀에서 비가시점에 대한 evidence를 모으는 표현. 현재 sparse head나 image edge/Hough와 동일하지 않다. 기존 line/Hough 실패는 PVNet 전체 반증이 아니지만 새 dense target/representation이 필요하여 보류. |
| R10 EPro-PnP, CVPR 2022 | [출판처](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html), [원문 §3.1–3.3](https://arxiv.org/html/2203.13254v3), [공식 코드](https://github.com/tjiiv-cprg/EPro-PnP) | RGB와 2D–3D 대응/학습 weight, target pose distribution, Monte Carlo 적분 | 좌표 surrogate와 pose 목적의 차이, 복수 해 표현. 단순 weighted reprojection만 EPro-PnP라 부를 수 없다. 현재 실사 TRAIN에 정확한 pose를 가정하지 않는다. 먼저 frozen output에서 solver/selector를 대조하고 end-to-end 새 loss는 보류. |
| R11 MegaPose, CoRL 2022 / PMLR 2023 | [출판 초록](https://proceedings.mlr.press/v205/labbe23a.html), [공식 README](https://github.com/megapose6d/megapose6d). 방법 전체 미독 | novel object CAD와 image observation; 대규모 synthetic render 학습, coarse/refiner/score | 기하가 비슷한 후보를 RGB-render evidence로 구분. 등록 cuboid 치수는 대상의 외관 mesh와 같지 않다. coarse silhouette만으로 실제 삽입면을 보장할 수 없다. 신규 model/data 다운로드와 전체 프레임워크 보류. |
| R12 CRISP, CVPR 2025 | [출판처](https://openaccess.thecvf.com/content/CVPR2025/html/Shi_CRISP_Object_Pose_and_Shape_Estimation_with_Test-Time_Adaptation_CVPR_2025_paper.html), [저자 원문 §3, §4.3, §5–6.1](https://arxiv.org/html/2412.01052v1), [저자 프로젝트](https://web.mit.edu/sparklab/research/crisp_object_pose_shape/) | segmented RGB-D, observed depth cloud, learned implicit shape/shape priors; multiview corrector 및 single-view ablation | 보정한 결과를 관측과 비교하고 ambiguity를 별도 점검. certificate는 observed depth와 implicit surface의 consistency이며 sparse reprojection LOO와 같지 않다. HTML 수식 일부 깨짐; 정확한 certificate/loss 재구현 안 함. depth/shape 전제가 달라 방법 자체는 현 계약에서 보류. |

## 인접 원리 A/B/C

### A. 통계적 신뢰성: Guo et al., ICML 2017

[확인] [출판처](https://proceedings.mlr.press/v70/guo17a.html)와 [본문 §2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)를 읽었다. 원래는 분류 정답 사건의 confidence calibration이다. [추정] 현재는 `PCK10 사건`과 `detected/objectness 사건`을 명시적으로 분리하고, TRAIN/calibration의 동일 semantic point로 confidence별 실제 좌표 오차·교사 우세율을 먼저 계산하는 데 전이한다. 분류용 temperature scaling을 좌표 정확도 교정으로 그대로 이식하지 않는다. 38점이 이미 교사에 사용됐으면 그 재평가는 독립 calibration이 아니며 TRAIN 적합성 진단이다.

### B. 최적화·보존: GEM, NIPS 2017

[확인] [출판처](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html), [저자 본문 §3](https://arxiv.org/html/1706.08840v6), [공식 코드](https://github.com/facebookresearch/GradientEpisodicMemory)를 연결했다. GEM은 이전 task memory 손실의 증가를 제한하는 gradient 제약을 사용한다. 이 해석에는 작은 step의 국소 근사와 memory 대표성이 필요하다. [추정] 현재 synthetic replay가 실제 deployment 오류를 대표하는지, source 보존과 real 타깃 추종이 어떤 손익을 가지는지 따로 측정한다. source 손실 증가 자체가 실사 손상 증거는 아니다. GEM solver를 구현하지 않고 source/target norm·내적·실제 손실·학습 노출을 함께 기록하는 것이 첫 대조다.

### C. 기하·관측 가능성: IPPE, IJCV 2014

[확인] [저자 공식 설명과 코드](https://github.com/tobycollins/IPPE), [OpenCV 공식 PnP 문서](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)를 읽었다. planar 또는 거의 affine인 관측에서는 reprojection이 비슷한 두 pose가 생길 수 있다. [추정] 후보가 존재한다는 것과 RGB/기하 cue가 그 후보를 고를 수 있다는 것을 분리한다. IPPE의 planar flip 모호성을 pallet W/D 역할 또는 C2/C4 대칭과 동일시하지 않는다. 모든8점 cuboid에 coplanar solver를 강제로 적용하지 않는다. 추가 후보는 원래 fixed-set oracle와 구분한다.

표준 robust residual의 정확한 정의는 [SciPy 공식 least_squares 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)의 `huber`/`f_scale`를 확인했다. 이는 점수 함수 전이의 출처이며 EPro-PnP, MAGSAC 또는 새 통계적 certificate의 재현이 아니다.

### D. 추가 연결: pseudo-label augmentation consistency

[확인] [FixMatch, NeurIPS 2020 원문 §2.1–2.4](https://proceedings.neurips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf)는 약하게 변형한 영상의 confident 분류 타깃을 강하게 변형한 영상이 따르게 한다. [STAC 원문 §3.1–3.2](https://arxiv.org/html/2005.04757v2)는 검출로 확장하면서 geometric transform을 pseudo-box 좌표에도 적용한다. [공식 STAC 저장소](https://github.com/google-research/ssl_detection)의 인용은 arXiv 2020이며, 이번 확인에서 출판 venue를 확보하지 못했다. 따라서 STAC은 preprint 배경 참고로만 두고 채택 알고리즘/출판논문으로 세지 않는다.

[추정] 현재 sparse keypoint는 class label처럼 변형 불변이 아니라 좌표 변환을 따라야 한다. 이 원리는 target-coordinate/mask consistency와 현재 학생의 변형 대응 부담을 점검할 이유이지 augmentation을 끄면 좋아진다는 논문 근거가 아니다. 오히려 원논문은 적절한 strong augmentation의 이익을 보이므로 C2의 가장 강한 반론이다. 기존 REF가 augmented TRAIN에서 더 큰 residual을 보였다고 label misalignment·loss 오류·gradient 원인을 증명한 것은 아니다. C2는 기존 criterion 그대로 real translate/scale만 OFF하며 HSV/source augmentation을 유지하는 작은 반증 실험이다. 결과가 좋아져도 crop/support·interpolation·노출 변화와 일반화 효과를 완전히 분리하지 못한다.

## 기존 결과와 충돌시키지 않을 부분

[확인] [model-conditioned selector 보고서](../pallet_model_conditioned_selector_v1/REPORT_KO.md)의 H_MANUAL 전용 GEO는 일부 DEV 개선이 있었지만 S1 전용은 old보다 낮았고 synthetic 결과도 동일한 우세를 보이지 않았다. 대상 모델과 수동 hard8 감독이 현재 RAW/REF와 다르다. 이를 현재 학생 selector의 실행 결과로 재명명하지 않는다.

[확인] [기존 augmentation stability](../pallet_posefix_utility_selector_v1/augmentation_stability/RESULTS_KO.md)는 179개 이전 survivor에 7개 RGB 변형을 실행했다. 80% 유지에서 비초록 PCK20은 84.80→87.50%, 초록은 90.64→89.73%였으며 동일 유지율 LOO/confidence를 일관되게 넘지 못했다. 현재217/361 TRAIN 학생학습 검사는 아니지만, 같은 점수를 무근거로 다시 채택할 수 없다는 직접 반론이다.

[확인] 현재 Replay teacher는 [`core.py`](../../../scripts/research/pallet_posefix_replay_v1/core.py)가 연결하는 기존 PoseFix-derived ResNet152와 팔레트 교란을 쓴다. teacher training의 source replay와 학생 training의 source replay를 동일한 loss 경로로 취급하지 않는다. 학생의 [`true_ignore_pose_loss.py`](../../../scripts/self_training_yolo/v3/true_ignore_pose_loss.py)는 `visibility=1`을 ignore, `2`를 감독, `0`을 invisible로 구분한다. 필터링/가중치 전이 시 mask 변환 및 실제 gradient가 먼저 확인되어야 한다.

## 원문과 코드 사이의 작은 차이도 남김

[확인] Soft Teacher arXiv v2 §4.2는 box jitter offset을 uniform ±6%, §3.3은 box 평균 변 길이로 정규화한다고 기술한다. 현재 공식 `soft_teacher.py`의 `aug_box`는 `torch.randn`을 쓰고, `compute_uncertainty_with_aug`는 좌표별 width/height로 나눈다. 따라서 정확한 재현을 하려면 논문/코드 revision을 고정하고 그 차이를 선언해야 한다. 본 후속은 해당 방법을 구현하지 않고 위치 신뢰성과 분류 confidence를 분리하는 원리만 가져온다.

미출판 preprint의 알고리즘을 새로 채택하지 않았다. STAC은 출판 미확인 preprint 배경 참고로 분리했다. arXiv로 읽은 PoseFix/Self6D++/PseudoFlow/Soft Teacher/UTv2/EPro-PnP/CRISP/GEM도 위의 확인된 출판 venue로 표기했다.
