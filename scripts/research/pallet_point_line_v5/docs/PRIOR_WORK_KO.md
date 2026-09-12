# 확인한 선행 근거와 구현 차이

본 패키지는 아래 방법의 전체 재현이 아니다. 코드 신규 작성, 데이터/성능/학술 novelty 미검증.

## 정식 발표 논문

### Deep Hough Transform for Semantic Line Detection(2020/ECCV)

후보 직선을 따라 영상 특징을 누적하고 Hough 공간에서 처리하는 근거다. 이 패키지는 dense-reference 선형 rho voting과 content normalization을 직접 구현했다. 논문의 backbone·multi-scale·CUDA 구현·전체 학습을 재현하지 않는다.

https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/779_ECCV_2020_paper.php

### Deep Hough-Transform Line Priors(2020/ECCV)

Hough 기반 global line prior와 local image representation을 결합하는 선행이다. v5는 HT-IHT feedback을 그대로 복제하지 않고 line observation을 physical pose objective로 전달한다.

https://arxiv.org/abs/2007.09493

### Uncertainty-Aware Camera Pose Estimation From Points and Lines(2021/CVPR)

2D/3D 점·선 대응의 uncertainty를 다루는 camera pose 추정 연구다. v5의 fixed sigma, Huber/mixture LM은 이 논문의 covariance 처리를 충실히 재현한 구현이 아니다. 논문의 향상 수치를 v5의 기대 성능으로 쓰지 않는다.

https://openaccess.thecvf.com/content/CVPR2021/html/Vakhitov_Uncertainty-Aware_Camera_Pose_Estimation_From_Points_and_Lines_CVPR_2021_paper.html

### EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation(2022/CVPR)

자세 계산을 학습 목표에 넣고 다중/확률적 자세를 다루는 근거다. v5는 유한 시작점과 piecewise differentiable unrolled optimizer다. 연속 SE(3) posterior 또는 EPro-PnP의 KL 학습을 구현했다고 주장하지 않는다.

https://openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html

### COPE: End-to-End Trainable Constant Runtime Object Pose Estimation(2023/WACV)

대칭적으로 허용된 geometry correspondence를 학습하는 선행으로 기존 대화 및 저장소 설계가 참조했다. 최종 논문 문구와 식은 실제 전문을 다시 확인한 뒤 인용한다.

https://openaccess.thecvf.com/content/WACV2023/html/Thalhammer_COPE_End-to-End_Trainable_Constant_Runtime_Object_Pose_Estimation_WACV_2023_paper.html

## 공식 구현 문서

OpenCV PnP documentation: object→camera, K, distortion, projection, local refinement의 convention 확인. v5는 cv2.solvePnP를 내부 미분 solver로 호출하지 않는다.

https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html

## 저장소 선행 구현과 구분

- `pallet_dht_joint_v1`: 실제 DHT 특징을 point head에 전달하고 공동학습한 선행 내부 구현.
- `pallet_dht_structured_v2`/v4: endpoint/finite-segment/전체 배치 scorer를 이미 시험.
- `dimension_guided_graph_pose.py`: 치수+point+line을 하나의 pose optimizer에 사용했지만 point-PnP 초기화 및 국소 최적화 한계가 기록됨.
- `pallet_translation_loss_v1`: 물체별 C1/C2/C4 계약과 한 물체의 여러 loss 항이 같은 대칭 가설을 공유하는 원칙을 이미 기록.

따라서 v5의 검증 질문은 “처음으로 점·선을 결합했다”가 아니라, 영상 기반 복수 시작점·학습 그래프 내 공동 pose refinement·치수/대칭 일관 감독이 기존 한계를 실제로 줄이는지다.
