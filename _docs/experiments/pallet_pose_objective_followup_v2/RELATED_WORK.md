# A–F 선행 원리와 현재 코드의 차이

검토일 2026-09-28. 원문 방법절·공식 소스·실제 로컬 코드를 구분해 읽었다. 정확한 범위와 접근 실패는 [SOURCE_REGISTRY.json](SOURCE_REGISTRY.json)에 기록했다. 여기의 설계 권고는 [추정]이며 원논문의 성과를 팔레트 성과로 인정하지 않는다. 최종 주 판정은 사전 고정된 Plastic 자연 Moderate+Severe 99frame의 translation/full rotation median 쌍이다. 현재 집합은 반복 DEV이고 PCK/AUC는 보조다. 이 문헌 조사 자체는 GPU0·fit0·optimizer update0이다.

## 원문에서 가져오는 가장 작은 원리

| 출처 | 실제 읽은 방법·핵심 원리 | 현재와 다른 전제·전이 범위 |
| --- | --- | --- |
| R1, ICCV2021 [원문 §3–4.1](https://arxiv.org/html/2011.12498v2), [공식 설명](https://github.com/xierc/Semi_Human_Pose) | easy 경로의 heatmap은 detach하고 hard 경로만 추종시킨다. 기하 변환에는 타깃 좌표계도 맞춘다. Joint Cutout은 예측 관절 주변을 가린다. | 원문 §3.1은 fixed pseudo teacher를 온라인 상호일관성 collapse와 구분한다. 현 frozen teacher의 실패를 같은 collapse라고 진단하지 않는다. 공식 실험의 GT person box, dense human heatmap, dual network는 현 predicted bbox·sparse pose 학생과 다르다. A는 입력난도 대조이지 전체 재현이 아니다. |
| R2, CVPR2023 [원문 §3](https://arxiv.org/html/2303.04346v1), [공식 저장소](https://github.com/hlz0606/SSPCM) | 두 teacher의 서로 다른 시점 pseudo keypoint 불일치로 heatmap 쌍을 골라 평균한다. Cut-Occlude는 다른 이미지의 pseudo-joint 중심 limb patch를 복사한다. | 현재 noise rectangle과 같은 방법이 아니다. 세 모델·과거 heatmap 저장을 새로 도입하지 않고, 가린 입력과 원래 타깃 분리만 전이한다. 공통오류는 agreement로 검증되지 않는다. |
| R3, ICCV2017 [원문 §3](https://arxiv.org/html/1708.02002v2) | 분류 focal은 `−α_t(1−p_t)^γ log(p_t)`다. 수많은 쉬운 배경의 합을 줄이는 원리다. | `p_t`는 참 클래스 확률이지 좌표 잔차가 아니다. 현재 동결된 detector 분류를 바꾸는 것은 pose 학습 대조가 아니다. 학습되는 kobj BCE도 실제 pos/neg/ignore 및 gradient를 먼저 확인한다. |
| R4, AAAI2019 [원문 방법](https://arxiv.org/pdf/1811.05181), [저자 공식 GHMC/GHMR 소스](https://raw.githubusercontent.com/libuyu/mmdetection/master/mmdet/models/losses/ghm_loss.py) | GHMR은 ASL1 `sqrt(d²+μ²)−μ`와 출력좌표 미분 크기 `abs(d)/sqrt(d²+μ²)`의 histogram 밀도를 쓴다. 조밀한 bin의 총 기여를 줄이며 단순 hard mining과 다르다. | 원래는 GT box offset 회귀다. 네트워크 전체 parameter gradient norm으로 바꾸면 원논문이 아니다. 소량·오류 pseudo의 드문 bin이 과대가중될 수 있어 uniform/단순 capped weighting과 비교해야 한다. |
| R5, CVPR2018 [원문 §4–5](https://arxiv.org/html/1711.06753v5) | Wing은 작은·중간 좌표 잔차의 로그 구간과 큰 잔차의 L1 구간이다. pose-based balancing은 주석 shape를 정렬/PCA한 희소 pose 구간을 더 노출한다. | 큰 오류용 focal의 이름 변경이 아니다. `w=10, ε=2`를 stride/grid/pallet-native 단위 구분 없이 복사하지 않는다. pseudo 잔차를 실제 오답 그룹으로 삼지 않는다. B의 가장 단순한 기준은 정규화 L1/SmoothL1이다. |
| R6, ICCV2019 [원문 §4.1–4.3](https://arxiv.org/html/1904.07399v3), [공식 설명](https://github.com/protossw512/AdaptiveWingLoss) | AWing은 GT heatmap intensity `y`에 따라 비선형 지수 `α−y`를 바꾸고 큰 intensity error에서는 선형이다. 별도 weighted map은 GT heatmap의 dilation으로 foreground 주변을 강조한다. | 위치 거리나 softmax CE용 손실이 아니다. normalized spatial probability는 원문의 peak1 Gaussian intensity와 다르다. AWing·weighted map·Gaussian 타깃을 동시에 바꾸지 않는다. 공식 저장소의 training loss 파일은 이번에 확보하지 못했고 코드 재현으로 쓰지 않는다. |
| R7, ICCV2021 [원문 §3와 알고리즘](https://arxiv.org/html/2107.11291v3) | predicted mean/scale로 표준화한 오차를 normalizing flow와 기준 분포가 설명한다. 추론에는 mean을 사용하고 학습용 flow는 필요 없다. | 이미 학생에 RLE 계열이 있다. 새로운 공간적 불확실성 감독의 첫 도입이 아니다. 원논문 성공이나 confidence 식은 현 scale의 calibration/큰 오차 gradient 정상성을 보증하지 않는다. optical flow와 구분한다. |
| R8, WACV2024 [원문 §3.1–3.3](https://arxiv.org/html/2310.00099v1) | 여러 strong/weak 변형의 heatmap을 pixelwise max로 모으고 threshold 후 argmax 중심 Gaussian으로 정제한다. pixelwise 표준편차의 최대값으로 두 학생의 타깃을 선택한다. | 단순 Gaussian label smoothing 또는 현재 RLE sigma와 같지 않다. 다중 heatmap·두 학생을 요구한다. E/F의 신뢰성 문제를 구분하는 배경이며 새 multi-teacher를 자동 도입하지 않는다. |
| R9, CVPR2019 [PoseFix 원문](https://arxiv.org/html/1812.03595v2), [공식 코드](https://github.com/mks0601/PoseFix_RELEASE) | RGB와 초기 pose를 함께 받아 오류를 수정한다. 초기점 오류분포와 이미지 단서가 중요하다. 이전 방법절 검토를 재사용했다. | 현재 pallet port는 이미 존재한다. 원래 사람 관절 오류 prior와 팔레트 번호/축/가림 오류는 다르다. F는 실제 `I_occ`에서 R0가 낸 초기점·bbox를 넣어야 하며 원본 정확한 hint/GT crop으로 복원 성공을 만들지 않는다. |
| R10, ICCV2021 [Soft Teacher 원문](https://arxiv.org/html/2106.09018v2), [공식 코드](https://github.com/microsoft/SoftTeacher/blob/main/ssod/models/soft_teacher.py) | foreground confidence와 jittered proposal 재회귀의 위치 분산을 분리한다. 이전 방법절/소스 검토를 재사용했다. | box 회귀의 불확실성을 pallet point의 정답확률로 선언하지 않는다. 이전 registry의 paper/code jitter 분포·정규화 차이도 유지한다. 기존9/38 teacher TRAIN을 독립 calibration으로 바꾸지 않는다. |
| R11, CVPR2016 [OHEM 원문 §3.1·4](https://arxiv.org/html/1604.03540v1) | 주석 RoI의 classification+localization loss로 hard RoI를 골라 역전파하고 중복 영역은 NMS로 제한한다. | GT에 대한 hard와 pseudo에 대한 큰 잔차는 다르다. 후자는 잘못된 타깃일 수 있다. 현재 sparse점의 1:1 정답/오답 sampling 근거가 아니다. |
| R12, [공식 v8.4.60 loss.py](https://github.com/ultralytics/ultralytics/blob/v8.4.60/ultralytics/utils/loss.py) | 현재 로컬 버전도8.4.60. KeypointLoss·RLELoss·calculate_rle_loss를 직접 읽고 true-ignore wrapper를 연결했다. | tag 이름만으로 runtime 동일성을 주장하지 않는다. 로컬 SHA와 경로는 registry에 있다. 공식 raw 다운로드의 sandbox DNS 실패로 byte equality는 미확인이다. 실제 graph 진단이 원인 판단의 근거다. |

## A: 기존 S1/S2에서 재사용되는 것과 새로운 공백

[확인] [`augmentation.py`](../../../scripts/research/pallet_clean19_structured_easyhard_v1/augmentation.py)는 bbox 면적의 `{0.1,0.2,0.3}`, 종횡비 `{0.5,1,2}`에서 rectangle 크기를 뽑고 schedule 확률0.5를 사용한다. fill은8×8×3 uniform uint8 noise를 bilinear resize한 것이다. 위치는 변환된 native canvas와640입력의 교집합 안이다. covered supervised corner≥1, 남은 corner≥2, bbox overlap>0을 요구한다. 계획에서 center는 세지 않지만 적용 함수는 좌표·visibility를 바꾸지 않는다.

[확인] 이 함수는 독립 random-only 함수가 아니다. 먼저32개 random 후보를 만든 뒤 supervised cuboid edge를 덮는 구조 후보를 최대32개 만들고, bbox overlap 차이가 `max(1px², bbox area×.01)` 이하인 쌍이 있어야 S1/S2 모두 적용한다. `mask.sum()<4`, edge 부재, pair 실패는 원본 RGB 유지다. 따라서 새 A의 random-only 경로는 구조 pairing/edge 의존을 제거했다는 차이를 명세해야 한다. 최소4점 조건도 그대로 필수라고 일반화할 수 없다. TRAIN feasibility에 따라 선택하되 covered/remaining 계약은 유지한다.

[확인] [과거 감사](../pallet_clean19_structured_easyhard_v1/AUGMENTATION_AUDIT.json): real5,120 occurrence 중 scheduled2,511, applied1,312(25.625%). 실패1,196은 pair 탐색 실패,3은4점 미만이었다. Plastic730/2,560, Wood582/2,560. 같은 적용수/크기/fill에서도 masked target은 S1 1,695, S2 2,653으로 다르다. 구조 효과와 가린점 노출량이 완전히 분리된 결과는 아니다.

- 새로운 것: 현재217/361 filtered pool·고정 RAW/REF·pose/flow-only recipe에서 같은 RGB mask의 짝지은 random occlusion 대조.
- 재사용: 현재 pool/support·R0·기본증강·source replay·solver/evaluator와 기존 rectangle shape/fill 원리.
- 공백: [Clean19 실험](../pallet_clean19_structured_easyhard_v1/REPORT_KO.md)은19장/87근거채널, 다른 교사/학습범위, lr1e−4, 자연평가300장이다. 현재 main의 lr1e−5·217/361·pose/flow-only 인과 대조를 대신하지 못한다. 신규 A는 `FILTERED_INPUT_TO_OCCLUDED`이며 verified clean-to-hard가 아니다.

## B/C/D: 현재 손실과 '어려움'을 혼동하지 않기

[확인] [`TrueIgnorePoseLoss26`](../../../scripts/self_training_yolo/v3/true_ignore_pose_loss.py)은 v2만 위치/RLE 감독하고 v1은 위치·RLE·kobj 전부 제외한다. v0는 존재항 음성이다. RLE scalar aggregate 뒤 `clamp(min=0)`가 있다. 현재 설치 KeypointLoss는 `1−exp(−e)`, `e=d/[8·sigma²·(area+epsilon)]`이고 RLE의 `(pred−target)/sigmoid(scale)`는 finite 검사를 거쳐 각 성분±100으로 잘린다. 위치 gradient가 줄거나 clip된다는 가능성과 실제 원인은 다르다. source/real, assignment, 좌표/scale/head gradient를 같은 실제 graph에서 나눠 확인해야 한다.

[확인] [`PoseOnlyTrainer`](../../../scripts/research/pallet_type_selftrain_v1/recovery_pose_trainer.py)는 detector/box/class/backbone을 고정하고 pose 및 flow만 학습한다. 따라서 dense detector classification Focal은 현재 변경축으로 부적합하다. kobj는 학습되지만 그 불균형·gradient와 위치 회복의 연결을 확인하기 전에는 Focal을 채택하지 않는다. easy background, heatmap pixel, 영상 가림, covered point, noisy pseudo는 서로 다른 그룹이다.

[확인] GHMR 공식 구현은 gradient-length를 `detach()`하고 valid weight>0만 bin에 넣는다. 비어 있지 않은 bin 수로 다시 나누며 optional EMA count를 쓴다. 논문은 μ=.02, EMA=.75를 기술하지만 코드 기본은 bins10/momentum0이다. 이 값들은 box offset 단위의 설정이지 native pixel 기본값이 아니다. 단순 회귀항과 배율 대조 없이 GHMR을 바로 쓰면 loss family·scale·weighting 효과가 섞인다.

## E: 입력 Gaussian과 감독 분포는 이미 다르다

[확인] [`prior_model.py`](../../../scripts/research/pallet_sensors_submission_v1/prior_model.py)는 입력384×288에서 hint Gaussian의 분모162=`2×9²`, peak255를 사용한다. 이는 입력 특징이다. 출력은96×72이고 좌표 expectation에4를 곱한다. 감독은 정답/4 위치의 최대4개 bilinear 이웃 질량이며 finite/in-crop만 유지하고 경계 질량을 재정규화한다. 손실은 spatial softmax CE + output-grid 좌표 L1 + convolution L2다. 이를 처음 Gaussian 감독을 도입하는 모델이라고 부르지 않는다.

[확인, 새 CPU 수학 진단] 실제 `target_distribution`을 합성 fixture6개에만 실행했다(fit0/step0, 약0.85초 process wall). σ1 output-grid=4crop-px의 정규화 Gaussian은 비교용 수학 예시이며 채택 hyperparameter가 아니다. interior 중심의 평균 편향은 수치오차 수준이었다. crop x=1에서 bilinear bias0, Gaussian +1.564833px; x=287에서 bilinear −3px, Gaussian −4.063759px였다. 아래끝 y=383도 같은 현상이다. outside/NaN 두 fixture는 bilinear mass0·CE logits gradient0이었다. 유효 분포의 합은 반올림 오차 내1이다. 실제 데이터 좌표는 사용/공개하지 않았다.

[추정] Gaussian을 finite grid에서 자르면 nominal 중심과 기대값이 달라져 좌표 L1과 요구가 충돌할 수 있다. 이는 기존 bilinear 마지막 cell에도 일부 존재하지만 폭이 커지면 커질 수 있다. sigma의 native 환산은 crop affine scale까지 필요하다. AWing은 peak1 intensity 회귀용이므로 normalized Gaussian CE와 곧바로 교환할 수 없다. E를 실행한다면 먼저 같은 loss에서 분포만 바꾸는 작은 대조가 우선이다.

## F: 교사 적응과 학생 전달은 별도 비용

[확인] 기존 [`Replay protocol`](../pallet_posefix_replay_v1/PROTOCOL.json) 및 [`train.py`](../../../scripts/research/pallet_posefix_replay_v1/train.py)는300step, real8+source8/micro2, TFAdam1e−4, 모든 teacher parameter trainable/BN running stats frozen이다. 원래 시작은 PRIOR1 last6000이며 현재 frozen Replay에서 추가 적응하는 F와 동일한 초기화가 아니다. [`FIT.json`](../pallet_posefix_replay_v1/FIT.json)의300step 종료+최종probe elapsed는184.564471초, peak2,182.879MiB였다. 이것은 새 GPU 비용의 측정치가 아니라 예약 추정의 역사적 참고다.

[추정] 기존300step을 근거로 한 T_KEEP/T_OCC 두 적응과 학생320step 두 전달은 최소4fit/1,240update다. 두 teacher는 동일한 현재 starting teacher에서 시작하고 T_KEEP의 추가학습도 생략하지 않는다. 원본 y_target과 가림 RGB에서 실제 추정한 q_init_occ/box_occ를 구분한다. fixed real membership·common support control을 확보한 뒤 학생 전달을 비교한다. teacher-only screen은 학생 T/R 성과가 아니다. 4fit 재현 reserve와 남은3cycle 상한을 침범하면 F는 `NOT_RUN_BUDGET/LOWER_PRIORITY`이지 방법 실패가 아니다.
