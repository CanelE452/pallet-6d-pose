# DHT point–line fusion: failure and selective use

## 1. 제안

[소비처] 팔레트의 점·측면 구조선 결합 방향 결정과 사용자용 실사 시각화.
[문장] 현재 동일 강도 보정의 실패를 코너별로 분해하고, 합성 검증만으로 정한 선택적 결합이 그 피해를 줄이며 실사에서 도움이 되는지 확인한다. 개선을 전제로 하지 않는다.

- 이전 실행의 live 예측692장과 DHT seed3개를 재사용한다. 신경망 재학습·새 추론·추가 실사 평가셋 사용은 없다.
- 진단: 코너 오류≤10 /10–20 />20px별 paired 개선·악화, 코너 이동과 제곱오차 분해, 선의 높이/깊이 역할, 완벽한 GT 선 및 GT 선택 oracle 상한. GT 난이도와 oracle는 사후 진단 전용이다.
- 실제 선택: DOPE 좌표 역변환 대조군 및 YOLO. λ=1 보정 후보를 코너 단위로 적용/유지한다. 점 confidence 단독과 점 confidence+선 peak+이동 상한을 비교한다. 임계값은 합성 검증256장의 예측 분위수25/50/75/100%만으로 산출하고, family마다 baseline 유지 후보를 포함한다.
- 판단 지표: 모든 GT 코너를 분모로 한 PCK@10, 검출 코너 mean/median/P90, 결측 포함 capped normalized error, 보정된 코너·프레임 수, paired seed별 변화. 실사52장을 3seed 반복으로156개 독립 표본처럼 취급하지 않는다.
- 예상 실패: 낮은 confidence가 실제 오차와 다름, 선 peak가 보정되지 않음, 합성→실사 분포차이, 가려진 구조선의 의미 혼동, 점과 선이 동시에 틀림, 선 교점의 퇴화.
- 중단 기준: 고정된 작은 선택 규칙 비교 뒤 보고. 실사 결과를 보고 규칙을 반복 수정하거나 신규 학습을 시작하지 않는다.
- 한계: 이미 결과를 본 real DEV52의 탐색 분석이며 새 독립 test가 아니다. 합성 validation 재사용에 따른 선택 낙관성도 구분한다.

## 2. 선행 근거

- [Deep Hough Transform, ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/779_ECCV_2020_paper.php): 후보 직선 위의 특징을 Hough 공간에 모아 semantic line을 검출한다. 실제 물체 경계의 관측 여부를 자동 보장하지 않는다.
- [PL-SLAM, IEEE T-RO2019](https://arxiv.org/abs/1705.09479): 점이 부족한 저텍스처 환경에서 점·선 결합을 사용한 선행 사례. SLAM의 결과로 우리 팔레트 semantic corner 개선이 증명되는 것은 아니다.
- [Confidence calibration, ICML2017](https://proceedings.mlr.press/v70/guo17a.html): 신경망 confidence와 실제 정답 확률은 구분해야 한다. 본 실험의 heatmap peak를 확률로 해석하지 않는다.

- [Ultralytics PoseLoss26 공식 코드](https://docs.ultralytics.com/reference/utils/loss/#ultralytics.utils.loss.PoseLoss26.calculate_keypoints_loss): 키포인트 confidence감독과좌표손실구분. 수치분석은이기기의고정설치소스를기준으로했다.
- [RLE, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Human_Pose_Regression_With_Residual_Log-Likelihood_Estimation_ICCV_2021_paper.pdf): 회귀출력분포와편차를학습하는선행근거. 현재팔레트의sigma가오차를잘예측하는지는별도검증대상이다.

## 3. 구현에서 확인한 실패 가능 경로

- `scripts/research/dht_pose_integration_v1/fusion.py::fuse_corners`는 각 코너에 연결된 두 선의 거리를 동일 λ로 벌점화한다. 점 confidence, 선 peak, 가시성, 이미지 edge 지지를 사용하지 않는다. `λ=0`만 정확한 baseline identity다.
- 선 하나는 normal 방향만 제약한다. 올바른 선에 수직으로 크게 벗어난 점은 보정할 수 있지만, 선 방향으로 틀린 위치는 그 선 하나로 복원하지 못한다. 두 선이 거의 평행해도 현재 positive point anchor로 해는 안정적이나, 부족한 방향 정보가 생겨나지는 않는다. 순수 교점 추정의 불안정성과 현재 anchor solve를 혼동하지 않는다.
- `deep_hough_side_v1/network.py`는 line loss와 역할별 독립 argmax를 사용한다. 이 8개 선이 하나의 cuboid로 일치하는지, 특정 DOPE/YOLO 점을 개선하는지 직접 최적화하지 않았다.
- line target은 hidden/amodal cuboid supporting line을 포함한다. 관측 가능한 목재/플라스틱 edge의 segmentation 결과가 아니며, 원래 선의 geometry support mask는 평가에만 사용한다.
- DHT head는 frozen VGG feature map50×50에서 normal angle1°/rho0.5cell 격자로 argmax한다. 원본640×480 및100px패딩에서 rho 한 bin은 방향에 따라 약6.8–8.4px에 대응한다. 이를 단순히 모든 오차의 원인이라고 단정하지 않고 GT 격자 대조군으로 구분한다.
- 기존 정확한 resize 역변환 대조군에서도 악화했으므로 해당 DOPE resize 차이 하나로 큰 악화를 설명할 수 없다.

## 4. Confidence가 곧 위치 정확도는 아니다

실제 설치된 Ultralytics8.4.60 `PoseLoss26.calculate_keypoints_loss`에서 키포인트 confidence의 BCE target은 `gt_kpt[...,2] != 0`이다. 따라서 기존 adapter의 `kp_conf`는 특정 픽셀 오차 이내일 확률을 직접 학습한 값이 아니다. DOPE belief peak도 확률로 정규화된 출력이 아니므로1을 넘을 수 있다. DHT의 Hough-bin softmax 최대값 역시 오차 보정 없이 물체 구조선의 정확도 확률로 사용할 수 없다.

실제 사용한 YOLO 체크포인트를 CPU에서 열어 `Pose26`, `kpt_shape=[9,3]`, flow model 및 sigma 파라미터12개를 확인했다. RLE sigma는 일반 eval forward에서는 계산하지 않으며 기존 adapter에도 없다. 별도 추출 후 정답 오차와의 관계를 검증하면 방향별 가중치 후보가 될 수 있지만, 이번70규칙 비교에는 넣지 않았다. 근거와 소스 SHA: `CONFIDENCE_SEMANTICS.json`.

## 5. 결과

### 동일 강도 결합은 일부 이득보다 큰 실패의 손해가 크다

실사52장, 동일 강도 λ=1, seed별 집계 후 평균. '점 난이도'는 GT 오차로 사후 구분했으며 가림/블러 라벨이 아니다.

| 모델 | 기존 점 오차 | 관측 코너 수 | 기존 평균 px | 선 보정 평균 px | 개선한 코너 비율 |
|---|---|---:|---:|---:|---:|
| YOLO | ≤10px |271|4.87|22.50|31.24%|
| YOLO |10–20px |94|13.62|21.32|65.96%|
| YOLO |>20px |35|33.16|48.81|48.57%|
| DOPE 좌표 보정 |≤10px |167|6.62|19.23|43.71%|
| DOPE 좌표 보정 |10–20px |135|14.00|21.00|66.67%|
| DOPE 좌표 보정 |>20px |91|50.61|54.61|59.34%|

일부 구간에서 개선한 코너가 과반이어도 평균은 악화한다. 이는 큰 이동에 따른 손해가 누적되기 때문이다. YOLO의 평균 이동은22.32px로 기존 평균 오차9.40px보다 크다. 제곱오차 변화식 `||e+d||²-||e||²=2e·d+||d||²` 검증 잔차 최대1.75e−10px². 원래 결합 예측과 새 계산은 정확히 일치한다.

### 선이 정확하다면 같은 결합으로 개선 가능하다 — 정답을 사용한 진단

- 실제 DHT 선 대신 GT 구조선을 넣고 같은 λ=1로 계산하면 DOPE 좌표 보정19.34→11.34px, YOLO9.40→5.49px. 예측 가능한 성능이 아니라, 올바른 선이 있다면 같은 수식도 유용함을 보여주는 GT 대조군이다.
- 실제 두 후보(기존 점 / 현재 λ1 보정)에서 정답을 보고 매 코너 더 나은 것을 고르는 best-of-two oracle은 DOPE19.34→14.31px, YOLO9.40→7.80px. 정답 없는 선택기가 이 성능에 도달했다는 뜻은 아니다.
- 실사414개 지원선의 GT를 현재1°/0.5cell 격자에 양자화한 경우 거리 평균2.09/P903.78px. 실제 학습 DHT는 평균28.29/P9068.52px이다. 큰 실패를 격자 해상도만으로 설명할 수 없다. 두 값의 차이를 학습 오류의 인과적 비율로 해석하지 않는다.
- 높이선은 median6.33px이지만 P90106.11px, 깊이선은 median8.65/P9030.15px이다. 중앙값 하나만 보면 큰 높이선 실패를 놓친다.

### 정답 없는 선택적 결합

모델당70개 후보를 합성 validation256장에서만 비교한 뒤 규칙을 고정했다. 아래 joint gate는 점 confidence·두 incident 선의 peak 최솟값·이동 상한을 함께 적용한다. 실사52장 수치는 seed별 통계의 평균이며, 평균 오차는 검출된 코너에 조건부이고 PCK 분모에는 결측도 포함한다.

| 모델 | 기본 평균 px | 무조건 λ1 평균 px | joint gate 평균 px | PCK@10 기본→gate | 수정 코너 수/관측 코너 수 |
|---|---:|---:|---:|---:|---:|
| DOPE 좌표 보정 |19.34|28.03|19.39|40.14→45.59%|183.3/393|
| YOLO |9.40|24.52|10.43|65.14→65.71%|59.3/400|

confidence 단독 family는 DOPE에서Q100, YOLO에서 baseline 유지를 선택했다. DOPE joint도 점 confidenceQ100·선 qualityQ25·이동Q100을 선택했다. 따라서 검출된 검증 코너는 confidence 기준을 모두 통과하고 이동 상한도 넓으며, 주로 선 peak 하한이 보정을 제한한다. 이를 ‘어려운 점을 정확히 골랐다’고 해석할 수 없다. YOLO joint는 점 confidenceQ25·선 qualityQ25·이동Q100을 선택했지만, 합성 test 평균도7.41→7.56px로 악화했다. 실사에서 낮은 YOLO confidence로 점 오류>10px를 구분한 AUROC는0.422였으며, visibility/presence confidence를 위치 정확도 확률로 사용할 근거가 없다.

joint gate−baseline의 대응 변화는 DOPE 평균 오차+0.047px(95%구간 −1.159~+1.234), PCK@10 +5.45%p(+2.56~+8.49)였다. YOLO는 평균 오차+1.032px(+0.212~+2.109), PCK@10 +0.56%p(−0.96~+2.08)였다. 구간은 실사3개 그룹 안에서 프레임을5,000회 재표집하고 같은 프레임의3seed를 함께 평균한 결과다. **이미 본 DEV52에 조건부인 탐색 구간**으로, 새 장면 일반화나 영상 프레임 간 상관·모델 불확실성을 충분히 반영하지 않는다.

현재 확인한 것은 이 DEV셋에서 선택적 결합이 무조건 보정의 손해를 줄이고 DOPE의 PCK를 높였다는 점이다. 평균 오차 개선이나 YOLO의 실사 이득은 확인하지 못했다. 결측·중심·거절된 코너 보존 및 선택 점수 재계산은 모두 통과했다. 근거: [GATE_RESULTS](../../data/pallet/results/dht_pose_integration_v1/live/fusion_diagnosis_v1/GATE_RESULTS.json), [GATE_AUDIT](../../data/pallet/results/dht_pose_integration_v1/live/fusion_diagnosis_v1/GATE_AUDIT.json), [STATISTICAL_AUDIT](../../data/pallet/results/dht_pose_integration_v1/live/fusion_diagnosis_v1/STATISTICAL_AUDIT.json).

다음 검증에서는 역할별 Hough 분포 폭·다중 봉우리와 seed 간 선 위치 불일치를 실제 오차에 대조해 선별 신뢰도를 보정할 수 있다.
선분 주변의 관측 edge 지지와 가시성도 별도 신호로 검증하되, 숨은 구조선 target과 구분해야 한다.
확인된 YOLO RLE sigma head는 점의 방향별 불확실성 후보일 뿐이며, 추출·오차 보정·결합 효과는 아직 검증하지 않았다.
새 신호와 선택 규칙은 별도 합성 검증에서 고정하고, 새로운 독립 실사셋에서 손해 감소와 실제 개선을 함께 확인해야 한다.
