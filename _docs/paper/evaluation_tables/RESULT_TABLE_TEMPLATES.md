# 논문 실험 표

논문 실험에 필요한 표만 유지한다. 모든 정량 표는 아래 열과 순서를 사용한다.
`AP`는 Box AP50:95가 아니라 positive/negative confidence-ranking AP이며,
`—`는 0이 아니라 아직 측정하지 않았다는 뜻이다.

```text
pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑  AP↑  AUROC↑  FPR95↓
```

## Table 1. Main model comparison — frozen FINAL

FINAL population이 동결된 뒤 같은 evaluator로 채운다.

```text
Method / subset                  pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑  AP↑  AUROC↑  FPR95↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SingleShotPose / PLASTIC           —       —        —          —        —        —         —          —         —     —       —        —
DOPE / PLASTIC                     —       —        —          —        —        —         —          —         —     —       —        —
PVNet / PLASTIC                    —       —        —          —        —        —         —          —         —     —       —        —
YOLO26n-Pose G38 / PLASTIC         —       —        —          —        —        —         —          —         —     —       —        —
Proposed S1 / PLASTIC              —       —        —          —        —        —         —          —         —     —       —        —
Real-FT upper bound / PLASTIC      —       —        —          —        —        —         —          —         —     —       —        —
```

## Table 2. Data ablation — PLASTIC DEV140

Population은 `OPEN52 + SEALED88`, negative는 2,689장이다. 아래 값은
`DEVELOPMENT_ONLY`이며 FINAL 값이 아니다. `R med`부터 `AUCall`까지의 pose
열은 axis selector가 해결되기 전까지 upper-bound diagnostic으로만 사용한다.

### Training data composition

모든 arm은 같은 generic synthetic corpus를 공유하고, **평가 대상 팔레트를 담은
target-domain synthetic을 얼마나 더하는가**만 다르다. 학습 조건(60 epoch,
batch 32, seed 42, 동일 pretrained init, 동일 augmentation)은 전부 같다.

```text
Arm  Training data                                        Generic   Target synth   Real
──────────────────────────────────────────────────────────────────────────────────────────
(a)  Generic only                                          38,002             —       —
(b)  Generic + target: 1 geometry, texture-randomised      38,002        17,978       —
(c)  Generic + target: 2 geometries                        38,002        35,914       —
(d)  (c) + real fine-tuning  [upper bound, not controlled] 38,002        35,914     157
```

★ (b)와 (c)의 target subset은 **포함관계가 아니다.** 서로 다른 렌더이고 이미지
해시 교집합은 0이다. (b)는 팔레트 1종을 재질 9가지로 랜덤화한 것이고
(c)는 팔레트 형상 2종이다. 따라서 (b)→(c)를 "target을 더 넣었다"로 읽지 말고
**"어떤 target diversity를 넣었는가"**로 읽어야 한다.

★ (d)는 실제 라벨 157장을 학습하므로 controlled comparison 조건("real 감독 0,
평가 대상 노출 0")을 만족하지 않는다. Table 1의 `Real-FT upper bound` 행에만
쓰고, 평가 프레임 12장이 학습에 포함돼 있어 DAY 지표는 낙관 편향이다.

### Main metrics

```text
Arm / subset      pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑     AP↑  AUROC↑  FPR95↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
(a) / PLASTIC    0.743    12.41    2.76      2.12   0.071   0.634    0.5527    0.1536   0.3018  0.7851  0.9553   0.1625
(b) / PLASTIC    0.850    10.47    2.54      1.51   0.064   0.677    0.5319    0.2947   0.3828  0.8677  0.9837   0.0688
(c) / PLASTIC    0.893     9.84    2.52      1.81   0.059   0.706    0.6059    0.3434   0.4409  0.9247  0.9889   0.0487
(d) / PLASTIC    0.986     6.61    2.30      1.71   0.036   0.722    0.7324    0.4779   0.5724  0.9933  0.9990   0.0000
(d) medium       0.993     6.04    2.21      1.44   0.032   0.749    0.7303    0.4998   0.5854  0.9899  0.9993   0.0015

(b) − (c)       −0.043    +0.63   +0.02     −0.30  +0.005  −0.029   −0.0740   −0.0487  −0.0581 −0.0570 −0.0052  +0.0201
```

★ (b)의 이득은 SEALED에 편중돼 있다. `AUCopen`에서는 (b) 0.5319 < (a) 0.5527로
**(b)가 generic-only보다 낮다.** OPEN/SEALED를 합친 수치만 인용하면 안 된다.

### Provenance — arm과 repo artifact 대응

```text
Arm  Checkpoint
────────────────────────────────────────────────────────────────────────────────────
(a)  runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/best.pt
(b)  spatial_concat_scratch/runs/
     YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt
(c)  runs/stage_a_synth_640_b32_seed42/weights/best.pt
(d)  release/pallet-pose-yolo26n-ft/pallet_yolo26n_pose_ft.pt   (medium: …26m-ft)

Arm  Dataset
────────────────────────────────────────────────────────────────────────────────────
(a)  datasets/g38_generic_only          (G__ 38,002)
(b)  datasets/g38_legacy_v1v2_p0_tex20k (G38__ 38,002 + P0__ 8,989 + TEX__ 8,989)
(c)  datasets/stage_a                   (G__ 38,002 + T__v1 17,964 + T__v2 17,950)
(d)  (c) + real 157 frames + 259 negative frames
```

## Table 3. Architecture ablation — full-model evaluation

동일한 training data, seed, epoch, evaluator로 각 full model을 학습·평가한 뒤
채운다. **현재 값은 전부 미측정이다.** 지금까지의 수치는 YOLO를 동결한 채
후단 probe만 학습한 것이고, 그 probe의 출력은 2-class parity 정확도라
이 표의 pose metric과 같은 양이 아니다.

### Arm 정의

모든 arm은 같은 검출 결과에서 출발하고, **어떤 정보를 어떻게 결합해
물체 축(parity)을 정하는가**만 다르다.

```text
Arm  Model input                                   Fusion                    Params
────────────────────────────────────────────────────────────────────────────────────
(A)  Object dimensions only                        —                          4,834
(B)  Predicted keypoints only                      —                          7,650
(C)  Image feature (single cell) + keypoints + dimensions
                                                   late concatenation        32,034
(D)  Image feature (7x7 neighbourhood) + keypoints late concatenation        32,034
     — no dimensions                               (dimension-free control)
(E)  Image feature (7x7) + keypoints + dimensions  late concatenation        32,034
(F)  Image feature (7x7) + keypoints + dimensions  FiLM modulation           42,802
(G)  Image feature (7x7) + keypoints + dimensions  dimension-query
                                                   cross-attention           42,722
```

`Image feature`는 동결 YOLO의 one2one classification penultimate 활성이며,
검출 원천 위치에서 잘라낸 `7x7x64` 패치다. (C)는 그 중 중심 셀 하나만 쓴다.
`dimensions`는 renderer가 기록한 고정 물리 치수(x,y,z)이고, 프레임마다 달라지는
camera-facing W/D는 **모델 입력으로 금지**된다.

(D)는 치수를 빼고 나머지를 같게 둔 통제군이고, (A)/(B)는 각 입력 단독의 하한이다.

### Main metrics

```text
Arm / subset   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑  AP↑  AUROC↑  FPR95↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
(A) / PLASTIC    —       —        —          —        —        —         —          —         —     —       —        —
(B) / PLASTIC    —       —        —          —        —        —         —          —         —     —       —        —
(C) / PLASTIC    —       —        —          —        —        —         —          —         —     —       —        —
(D) / PLASTIC    —       —        —          —        —        —         —          —         —     —       —        —
(E) / PLASTIC    —       —        —          —        —        —         —          —         —     —       —        —
(F) / PLASTIC    —       —        —          —        —        —         —          —         —     —       —        —
(G) / PLASTIC    —       —        —          —        —        —         —          —         —     —       —        —
```

### Provenance — arm과 repo artifact 대응

```text
Arm  Internal run name (spatial_concat_scratch/architecture_extension/runs/)
──────────────────────────────────────────────────────────────────────────────
(A)  D0_DIMS_ONLY
(B)  K0_KP_ONLY
(C)  C1_CENTER_CONCAT
(D)  S0_SPATIAL_NO_DIMS
(E)  S1_SPATIAL_CONCAT
(F)  S2_SPATIAL_FILM
(G)  S3_SPATIAL_CROSS_ATTENTION
```

### 기존 probe 결과를 이 표에 옮기지 않는 이유

동결 YOLO 위 probe는 synthetic DEV 4,020장에서 2-class balanced accuracy로
비교했고 (E) 0.9042 · (F) 0.9039 · (G) 0.9069 · (D) 0.8544였다. 그 실행은
`irrevocably_exploratory = true`, `paper_final_claim_allowed = false`,
`independent_test_opened = false`, real 평가 0으로 선언돼 있다.
따라서 **이 표의 metric으로 채우려면 각 arm을 full model로 다시 학습해야 한다.**

