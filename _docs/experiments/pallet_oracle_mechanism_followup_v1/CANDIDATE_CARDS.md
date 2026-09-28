# 실행 후보 카드

검토일: 2026-09-28. 설계 가설은 [추정]이며 실행 상태를 별도로 표시한다. 실제 채택·고정값·실행 arm·예산은 각 cycle의 `SPEC.md` 및 `RESOURCE_LEDGER.json`이 우선한다. 어떤 카드도 평가 oracle의 선택·좌표·가중치를 학습/추론에 전달하지 않는다. 누적 상한은 3 cycles / 12 fits / 7,680 optimizer updates / GPU6h / wall10h이며, 서로 다른 이름으로 같은 상한을 우회하지 않는다.

## C1. 같은 frozen 후보에서 robust reprojection 점수

- 목적: 좋은 pose 후보는 있지만 일부 큰 좌표 잔차 때문에 현재 선택이 손해를 보는지 검증한다. 생략하면 candidate quality와 selection quality를 분리할 수 없다.
- 원문: [OpenCV PnP 문서 전체](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html), [SciPy robust loss 정의](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html), [IPPE 저자 설명](https://github.com/tobycollins/IPPE). 정확한 Huber 정의를 읽었으며 다른 논문 이름의 임의 loss를 만들지 않는다.
- 구조/가정: 같은 point identity·K·치수·W/D 후보, 공통 유효 point set에서 점수만 변경. 일부 큰 residual이 outlier이며 나머지 잔차에 선택 정보가 있다는 가설이다.
- 실제 채택된 가장 작은 구현: 각 후보의 center 포함 9개 radial residual을 보존한다. `rho(z)=z` if `z<=1`, else `2*sqrt(z)-1`; 기존 D9 score에서 RMSE9만 `sqrt(mean(delta^2*rho((r/delta)^2)))`로 교체하고 모든 geometry penalty를 그대로 둔다. `delta=12px`는 기존 [`evaluate_pose_solver_swap.py`](../../../scripts/paper/pose_metric_closure_v1/evaluate_pose_solver_swap.py)의 Huber12에 고정했다. 원래 픽셀 단위를 유지한 이 점수 전이는 SciPy optimizer 실행과 다르다. DEV scale sweep은 하지 않는다.
- 단순 대조: 동일 후보의 현재 D9 선택. baseline pose/실패 penalty를 후보 계약에 포함한다. 입력 점을 바꾸거나 outlier point를 제거하지 않는다. penalty까지 없애는 pure residual 대조는 이번 실행에 포함하지 않았다.
- 필요 코드: 새 namespace의 score-only 모듈, reference 없는 decision dump, hash freeze 후 기존 평가 연결. 순서 불변성과 현재 후보 재현을 확인한다.
- 비용/정보: 신규 fit0/update0. frozen point/K/치수만; CPU scoring. 기존 원시 RGB/카메라 공개 권한은 늘어나지 않는다.
- 강한 반론: 같은 잘못된 identity/기하에 일관된 9점은 robust하게도 틀린다. W/D 후보가 near-ambiguous하면 낮은 residual이 낮은 ADD를 보장하지 않는다.
- 판정을 바꾸는 증거: 같은 set에서 current→robust AUC/coverage와 recording별 개선·악화, fixed-set oracle gap 회수율. 여지는 있지만 회수0이면 이 cue의 제한이지 모든 selector의 불가능이 아니다.
- 상태: `EXECUTED_NO_CHOICE_CHANGE`. [실제 결과](cycles/C1_HUBER_D9/REPORT_KO.md): 두 재료 모든 arm에서 선택변경 0, AUC 변화 0, oracle gap 회수율 0. 이 한 설정은 현재 gap을 회수하지 못했다. 새 subset/RANSAC 후보 생성은 C1의 선택 변경과 섞지 않는다. 실행한다면 별도 pool과 oracle를 보고한다.

## C2. real geometric augmentation을 끈 RAW/REF 전달 대조

- 목적: frozen pseudo target을 변형된 영상에서 추종하도록 하는 부담이, 제한된 pose/flow 학생의 보정 전달을 저해하는지 검사한다. 단순 update 연장이나 새 supervision이 아니다.
- 원문/전이 범위: [FixMatch §2.1–2.4](https://proceedings.neurips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf)의 weak-to-strong consistency 원리. [STAC §3.2](https://arxiv.org/html/2005.04757v2)는 좌표 타깃도 geometric transform을 따라야 함을 명시한다. STAC은 확인된 출판 venue 없이 `arXiv 2020 preprint`로만 기록하며 실행 알고리즘으로 채택하지 않는다. C2는 어느 논문의 재현도 아니고 기존 criterion을 보존하는 ablation이다.
- 현재 관측: frozen REF의 TRAIN normalized-bbox residual은 affine ON/OFF에서 Plastic 0.05873/0.01276, Wood 0.01529/0.01346이었다. Plastic 차이는 tail 영향이 크다. [Plastic 원자료](SIGNAL_DIAGNOSTIC_PLASTIC.json), [Wood 원자료](SIGNAL_DIAGNOSTIC_WOOD.json). augmented input에서 오차가 크다는 사실은 잘못된 label transform이나 loss 구현 오류를 증명하지 않는다.
- 최소 arm: 두 재료 각각 R0에서 RAW/REF 320update 쌍, 총 4fits/1,280updates. 바뀌는 축은 real 데이터의 random translate/scale OFF뿐이다. HSV, source augmentation, source/real 비율, 기존 loss·optimizer·LR·trainable 부위·마지막 checkpoint 평가를 유지한다. 실제 RNG/data-order 일치 여부는 실행 audit가 확인한다.
- 필요 코드/대조: real dataset transform wrapper에서 두 계수만 바꾸고 source 경로는 보존한다. 기존 RAW/REF와 새 RAW/REF를 모두 보며 `(NEW_REF−NEW_RAW)−(REF−RAW)`도 보고한다. TF32/AMP, coordinate transform, visibility/true-ignore 및 source retention을 점검한다.
- 가장 강한 반론: 원논문들은 적절한 strong augmentation의 이익을 보였다. augmentation을 줄이면 TRAIN imitation만 쉬워지고 generalization·robustness가 나빠질 수 있다. crop/support·interpolation·out-of-view 감독도 함께 바뀌므로 개선하더라도 단일 기하 원인이나 잘못된 loss를 확증하지 않는다.
- 판정을 바꾸는 증거: TRAIN target imitation뿐 아니라 고정 DEV의 native2D/pose, recording별 손익, RAW 대비 REF 추가 이득, synthetic 보존. 이 대조를 개발에 이미 사용한 DEV에서 독립 확인으로 부르지 않는다.
- 상태: `SELECTED_FOR_CONTROLLED_FITS`; 최종 실행 상태/결과는 cycle 산출물이 우선한다. 새 consistency loss·confidence gate·새 teacher는 추가하지 않는다.

## C2b. source replay와 real 타깃의 최적화 진단 — fit 미채택

- 목적: REF 타깃의 변화가 학생 파라미터에 충분히 전달되는지 확인한다. 기존640 저LR 연장과 다른 전제는 loss 구성/노출의 관측된 불일치다.
- 원문: [PCGrad §2–3/Algorithm1](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), [GEM §3](https://arxiv.org/html/1706.08840v6). 이 카드의 작은 대조는 두 알고리즘의 재현이 아니다.
- 구조/가정: 같은 pose/flow 파라미터에서 synthetic와 real 좌표 목적이 gradient를 공유한다. source loss 지배나 real 감독 노출 부족이 실제로 관측되어야 한다.
- 먼저 할 진단: 동일 frozen 상태에서 실제 TRAIN batch의 source/target 항별 유효점 수, reduction 분모, loss scale, gradient norm/dot/cosine, 보호 파라미터를 기록한다. AMP unscale, loss gain, detach, unused parameter 처리와 augmentation RNG를 통제한다. optimizer step0이다.
- 최소 arm: 근거가 생긴 경우 기존 recipe와 같은 R0·순서·augmentation·optimizer·320update로 NEW_RAW/NEW_REF 쌍. 바뀌는 한 축은 source/target 목적의 배율 또는 실제 exposure 중 하나다. 고정값은 관측과 기존 설정으로 정하고 sweep하지 않는다.
- 단순 대조/반론: 전체 loss 배율 변경은 effective LR 변경으로 설명될 수 있다. source downweight가 real target imitation만 개선하고 진짜 좌표·pose나 source 보존을 악화시킬 수 있다. 음의 cosine 몇 개는 원인 확증이 아니다.
- 필요 코드: 기존 true-ignore criterion의 의미를 보존하는 loss accounting/gradient probe; 선택된 한 변경의 wrapper. full checkpoint freeze audit, last checkpoint 사용, source retention과 target imitation을 별도 보고.
- 비용: 진단0fit/0update; 채택 시 material당 RAW/REF 2fit×320=640update. 양 재료면4fit/1,280update. 후속 재현 예산을 먼저 남긴다.
- 판정을 바꾸는 증거: REF correction 방향의 학생 변화와 TRAIN imitation, NEW_REF−기존REF 및 NEW_REF−NEW_RAW, source/real 손익과 recording별 DEV 결과. 개선해도 replay 간섭이 유일 원인이라고 단정하지 않는다.
- 상태: `DIAGNOSTIC_ONLY_NOT_SELECTED_FOR_FIT`. 현재 step0 gradient의 부호가 혼재해 PCGrad나 source downweight를 실행할 충분한 원인 증거가 아니다. 위 C2가 선택됐으며 이 카드의 가상 fit은 실제 예산 사용으로 세지 않는다. PCGrad 구현은 단순 대조가 부족하다는 근거가 있을 때만 별도 채택한다.

## C3. 위치 신뢰성과 objectness를 분리한 감독 검사

- 목적: 교사의 confidence/stability가 실제 좌표 정확도를 예측하는지, 학생보다 나은 감독을 골라낼 정보가 있는지 확인한다.
- 원문: [Soft Teacher §3.3](https://arxiv.org/html/2106.09018v2), [UTv2 §3.3](https://arxiv.org/html/2206.09500v1), [Guo §2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf). Soft Teacher는 jitter 후 재회귀 분산, UTv2는 별도로 학습한 regression uncertainty를 쓰며 둘 다 단순 detector score와 구분해야 한다.
- 현재 구조/차이: Replay heatmap, detector objectness, YOLO RLE sigma는 동일 단위가 아니며 공개 inference cache가 모두 sigma를 보존한다고 가정하지 않는다. 실제 saved fields/forward contract를 읽고 비교 가능성부터 기록한다.
- 최소 검사: 적격 TRAIN에서 고정 semantic point의 raw/ref residual·교사우세 사건과 confidence/변형 안정성 관계를 본다. 기존9/38 teacher TRAIN은 독립 보정셋으로 이름을 바꾸지 않는다. SYNTH_CHECK의 exact target도 실사 calibration 증거로 자동 전환하지 않는다.
- 코드/정보: 원본 좌표 변환·mask·support를 보존한 table builder; perturbation은 기존 cache와 계약이 호환되는 것만 재사용. 평가128/45/66의 좌표를 weight 결정에 쓰지 않는다.
- 단순 기준: 무필터, 기존 LOO/confidence, 같은 retention/가중치합/노출의 무작위 또는 단순대조. pointwise mask를 넣는다면 augmentation 뒤에도 동기화되는지를 확인한다.
- 가장 강한 반론: 잘못된 점도 안정적일 수 있고, 좋은 점이 perturbation에 민감할 수 있다. [이전 stability 결과](../pallet_posefix_utility_selector_v1/augmentation_stability/RESULTS_KO.md)는 비초록 일부 이득과 초록 손상을 이미 보였다.
- 달라져야 하는 근거: 현재217/361 TRAIN에 적용 가능하고 감독 신뢰성을 분리해 측정하는 새 자료/관측, 기존 whole-image score와 다른 사전 명세, 같은 retention 대조. 논문 이름만 바뀌면 재시도하지 않는다.
- 비용: cache 진단0fit; 새 teacher/scorer 학습·calibration fit은 모두 전체 예산에 포함. 충분한 별도 calibration이 없으면 새 filter training을 강행하지 않는다.
- 상태: `DIAGNOSTIC_ONLY_UNTIL_TRAIN_RELIABILITY_EVIDENCE`. confidence를 PCK 정답확률로 표기하지 않는다.

## C4. 적격 기존 TRAIN 직접감독으로 학생 capability 확인

- 목적: 현재 허용된 학습부가 정확한 타깃을 따라갈 수 있는지를 확인한다. oracle 정보가 현재 학생에게 전달 가능한지와 일반화는 분리한다.
- 근거/읽은 범위: [PoseFix §4–5](https://arxiv.org/html/1812.03595v2)의 supervised error correction; 실제 학생 학습부/target residual 진단. 이 카드는 새로운 논문 알고리즘이 아니라 통제 실험 설계다.
- 같은/다른 가정: 기존 teacher9/38 또는 exact synthetic TRAIN 감독만 사용한다. 이미 teacher가 쓴9/38 직접 student 감독은 같은 수동 예산의 대안이며217/361 전체 정답학습 upper bound가 아니다.
- 필요한 변경: 기존 trainer의 target source와 support를 정확히 연결하고 R0부터 단일 작은 fit. source replay, optimizer,320 또는 사전 고정640update, 평가 방식 보존. clone fit이면 TRAIN memorization 검사로 표기한다.
- 단순 대조: 동일 TRAIN에서 R0의 잔차, 같은 noisy target과 정확한 target의 비교. 더 복잡한 loss·backbone 해제부터 도입하지 않는다.
- 반론:9/38은 모집단을 대표하지 않고 교사 사용 이력이 있다. synthetic 성공은 real visibility/representation의 충분성을 증명하지 않는다.
- 바꿀 증거: accurate TRAIN도 못 맞추면 구현/최적화/trainable 범위 미분리; TRAIN 성공 DEV 무차이면 학습 가능성과 전이 실패 분리.
- 상태: `CONDITIONAL_ON_ELIGIBLE_TRAIN_AND_UNRESOLVED_FOLLOWING`. 새 manual 데이터·DEV 감독 전환 없이만 실행.

## 대형/추가 조건 후보: 이번 최소 대조로 줄일 수 있는가

| 원문 ID | 가져올 원리·현재와 같은 구조 | 다른 가정 / 가장 강한 반론 | 기존 코드에 필요한 변경 / 더 단순한 대안 | 판정을 바꿀 증거 / 현재 결정 |
|---|---|---|---|---|
| R1 PoseFix | RGB+초기 pose의 오류 수정 | 사람 오류 prior와 팔레트 identity/가림 불일치; 현재 teacher는 이미 존재 | corruption/support audit; 기존 teacher reuse가 기본 | matched real-error coverage가 새로 확인될 때만 teacher 재학습. `REUSE_AND_DIAGNOSE` |
| R2 Self6D++ | corrected pseudo target와 synthetic→real | RGB-only도 가능하지만 CAD/visible-amodal mask/renderer 필요; 네트워크 교체와 정보 효과 혼합 | mask/렌더/refiner 새 경로 대신 현존 타깃/가림 분해 | 기존 자산으로 독립 visual cue가 저비용 구현됨이 확인돼야 함. `DEFER_FRAMEWORK` |
| R3 ONDA-Pose | real 분포에 맞춘 synthetic 및 global correction | CARF/geometry/texture 학습 비용; 치수만으로 외관을 만들 수 없음 | 기존 synthetic-real support 통계 대조; 원문 loss 미구현 | full methods와 현재 재사용 가능한 calibrated views/CAD 확인 전 `DEFER` |
| R4 PseudoFlow | geometry consistency로 pseudo reliability | calibrated mesh renders와 dense optical flow 필요; 현 RLE flow와 무관 | 현재 sparse consistency audit부터, 새 RAFT/render pipeline 보류 | 실제 RGB cue가 오류를 구분하며 준비 자산/예산 존재 시 재검토. `DEFER` |
| R5 Soft Teacher | 위치 신뢰성 분리 | box proposal jitter≠pallet point jitter; paper/code offset·정규화 차이 | C3; 현재 원래 좌표를 유지하고 matched retention 비교 | TRAIN에서 안정성-오류 관계가 단순대조보다 나음. `CONDITIONAL` |
| R6 UTv2 | teacher/student 상대 uncertainty | 별도 학습 uncertainty branch 필요, 현재 출력 교정 불명 | 현재 출력 semantic/scale 확인, C3 | 비교 가능한 uncertainty와 적격 calibration 확보 시만. `DEFER_NEW_BRANCH` |
| R7 SSPCM | 다교사 불일치로 outlier 분리 | 공통오류·동일 pretraining bias; 추가 teacher 비용 | 고정 R0/Replay/RAW/REF 교차표, 기존 multi-teacher 결과 | 관측된 상보성을 GT-free cue로 분리한다는 TRAIN 증거. `DEFER_EXTRA_TEACHER` |
| R8 PCGrad | gradient 공유와 충돌 | 음의 cosine만으로 개선 보장 없음; source는 과거 task와 다름 | C2b step0 norm/dot; 배율 변경 미채택 | 간섭과 실제 손상의 연결이 재현될 때만. `DIAGNOSE_FIRST` |
| R9 PVNet | visible evidence가 hidden keypoint 투표 | dense supervised direction target가 필요; 과거 Hough와 다른 의미 | 기존 point-line/Hough 정상 구현의 작은 capability부터 | 현 sparse oracle 밖 정보와 정상 dense target 파이프라인. `DEFER_NEW_REPRESENTATION` |
| R10 EPro-PnP | point 오차와 pose 목적의 차이 | target pose supervision/Monte Carlo 분포 학습 필요; sparse weight만 바꾸면 원문 아님 | C1 frozen PnP/selection, exact synthetic sanity | 충분한 pose TRAIN과 비용, surrogate mismatch의 실증 필요. `DEFER_END_TO_END_LOSS` |
| R11 MegaPose | RGB-render 비교로 candidate 구별 | CAD/외관/렌더와 범용 synthetic pretraining 전제; cuboid는 외관모델 아님 | 현 RGB/기하 candidate margin을 먼저 측정 | 허용된 기존 mesh/renderer로 cue가 존재할 때만. `DEFER_FRAMEWORK` |
| R12 CRISP | correct→observed consistency→self-train | 실제 segmented depth, implicit shape, certificate의 전제가 다름 | scalar reprojection filter를 certificate라고 부르지 않고 C1의 한계로 기록 | 정보 계약 변경이 필요하면 사용자 결정 사안. `NOT_APPLICABLE_AS_IS` |
| A Guo | confidence와 correctness의 측정 분리 | classification calibration을 위치에 자동 이식 불가; 적은38점 재사용 | C3 descriptive reliability/교차표 | 적격 독립 calibration이 생기면 재검토. `MEASUREMENT_PRINCIPLE` |
| B GEM | source 보존과 target 적합 손익 | memory 대표성/국소 근사, 순차 task와 현재 joint recipe 차이 | C2b source retention+gradient accounting | 실제 source/real 손익 재현. `MEASUREMENT_PRINCIPLE` |
| C IPPE | 후보 생성과 관측으로 구별 가능성 분리 | planar ambiguity와 W/D/C4는 다름;8점 cuboid는 비평면 | C1 candidate margins; 필요 시 새 후보pool 별도등록 | 후보는 좋지만 cue가 무정보이면 다른 observed cue 필요. `MEASUREMENT_PRINCIPLE` |
| D FixMatch / STAC 참고 | pseudo label과 입력 변형 사이의 consistency | 분류 불변성과 좌표 equivariance는 다름; 강한 변형이 일반화에 유리할 수 있음 | C2에서 기존 loss 그대로 real translate/scale 한 축 대조 | target imitation 및 RAW 대비 REF 추가 이득 동시 확인. `CONTROLLED_ABLATION_NOT_REPRODUCTION` |

## 보류의 범위

이 카드의 보류는 현재 정보·기존 자산·작은 인과 대조·누적 예산에서의 우선순위다. 방법 계열의 불가능 판정이 아니다. 유효한 새 관측이 생기면 다음 cycle의 근거로 기록할 수 있지만, DEV를 보고 바꾼 설계는 독립 확인이나 사전등록으로 부르지 않는다.
