# Pallet Pose Estimation Literature Audit

> **문서 성격:** systematic literature review + commensurability audit  
> **용도:** pallet pose 선행연구의 문제정의·아키텍처·데이터·supervision·평가지표·일반화·runtime을 같은 schema로 정리하고, 현재 연구와 어떤 숫자를 직접 비교할 수 있는지 판정하기 위한 근거 문서  
> **작성 기준일:** 2026-09-05  
> **주의:** 이 문서는 meta-analysis가 아니다. 논문별 데이터셋·센서·출력·오차 정의가 달라 effect size를 통계적으로 합산하지 않는다.

---

# 0. Scope, evidence rule, and comparison rule

## 0.1 목적

[확인] 이 문서의 목적은 "가장 숫자가 좋은 논문"을 찾는 것이 아니라 다음 네 질문에 답하는 것이다.

1. 각 연구는 **무엇을 실제 문제로 정의했는가?**
2. 그 문제를 풀기 위해 **어떤 learned component와 geometric/analytic component를 사용했는가?**
3. 논문이 보고한 숫자는 **정확히 어떤 모집단·지표·센서·하드웨어에서 나온 값인가?**
4. 따라서 현재 RGB 기반 pallet pose 연구와 **DIRECT / APPROXIMATE / QUALITATIVE_ONLY / NOT_COMPARABLE** 중 어떤 수준의 비교가 가능한가?

## 0.2 증거 태그

- `[확인]`: 업로드된 원문 PDF 또는 공식/공개 full text에서 직접 확인.
- `[추정]`: 원문에 필요한 정보가 없거나, 여러 관찰을 연결해 내린 해석.
- `NOT_REPORTED`: 논문이 보고하지 않음.
- `NOT_VERIFIED_FROM_FULLTEXT`: 초록/preview까지만 확인했으며 원문 검증이 아직 안 됨.
- `SOURCE-INTERNAL INCONSISTENCY`: 같은 논문 안에서 수치·단위·명칭이 서로 맞지 않음.

## 0.3 비교가능성 등급

| 등급 | 의미 |
|---|---|
| `DIRECT` | 센서/출력/평가모집단/metric 정의가 거의 같아 숫자 비교가 성립함 |
| `APPROXIMATE` | 비슷한 물리량이지만 metric 정의·거리범위·센서조건 일부가 달라 제한적으로 비교 가능 |
| `QUALITATIVE_ONLY` | 어떤 문제를 풀었는지와 경향만 비교할 수 있음 |
| `NOT_COMPARABLE` | 센서·출력·metric·초기조건이 달라 숫자 순위가 의미 없음 |

## 0.4 반드시 분리할 metric layer

[확인] 다음 값들은 서로 같은 성능지표가 아니다.

- Detection: precision, recall, mAP@0.5, mAP@0.5:0.95
- Keypoint: OKS/APKP/KPOKS, pixel/KME
- Translation: axis MAE, 3D Euclidean error, horizontal error, distance error
- Rotation: yaw/tilt/axis error, quaternion geodesic angular error
- Pose-model distance: ADD, ADD-S, ADD-S AUC
- Task success: `5 cm & 5°`, `<5 cm & <3°`, 실제 fork insertion success
- Runtime: detector/model-only FPS, learned pose model FPS, full pipeline latency/FPS

---

# 1. Source-access status

| ID | Paper | Year / Venue | 현재 근거 상태 | 비고 |
|---|---|---|---|---|
| P1 | *Pallet Detection And Localisation From Synthetic Data* | 2024 ACRA / arXiv 2025 | `[확인] FULL LOCAL PDF` | 업로드 원문 확보 |
| P2 | *Estimating the Pose of a Euro Pallet with an RGB Camera based on Synthetic Training Data* | 2022 Logistics Journal: Proceedings | `[확인] FULL LOCAL PDF` | DOI 10.2195/lj_proc_knitt_en_202211_01 |
| P3 | *Pallet Pose Estimation Based on Front Face Shot* | 2025 IEEE Access | `[확인] FULL LOCAL PDF` | DOI 10.1109/ACCESS.2025.3538045 |
| P4 | *Multi-Stage Domain-Adapted 6D Pose Estimation of Warehouse Load Carriers* | 2025 Machines | `[확인] FULL LOCAL PDF` | DOI 10.3390/machines13121126 |
| P5 | *Unmanned Forklift Pallet Positioning Algorithm Based on an Improved Human Pose Estimation Model* | 2025 Annals NYAS | `[확인] FULL LOCAL PDF` | DOI 10.1111/nyas.70001 |
| P6 | *A Hypergraph Computing and Knowledge-Enhanced Framework for Forklift Pallet Pose Estimation* | 2026 Annals NYAS | `[확인] FULL LOCAL PDF` | DOI 10.1111/nyas.70219 |
| P7 | *Real-Time 6DoF Pallet Pose Estimation with Monocular Metric Depth* | 2026 VISAPP | `[확인] OFFICIAL OPEN FULL TEXT` | DOI 10.5220/0014626800004084 |
| P8 | *Occlusion-Robust Pallet Pose Estimation for Warehouse Automation* | 2024 IEEE Access | `[확인] PUBLIC AUTHOR FULL TEXT` | DOI 10.1109/ACCESS.2023.3348781 |
| P9 | *A Point Cloud Data-Driven Pallet Pose Estimation Method Using an Active Binocular Vision Sensor* | 2023 Sensors | `[확인] OFFICIAL OPEN FULL TEXT` | DOI 10.3390/s23031217 |
| P10 | *Recognition and Location Algorithm for Pallets in Warehouses Using RGB-D Sensor* | 2022 Applied Sciences | `[확인] OFFICIAL OPEN FULL TEXT` | DOI 10.3390/app122010331 |
| P11 | *Pallet Localization Algorithm Based on Improved Human Pose Estimation with Transfer Learning* | 2025 Journal of Supercomputing | `[확인] FULL LOCAL PDF` | DOI 10.1007/s11227-025-06973-w; 사용자 제공 원문 33p 확보 |

### 현재 막힌 자료

[확인] **P11 Journal of Supercomputing 원문은 사용자 제공 PDF로 해제되었다.** BDEM, FFDPN, APT-TAL, BA-Wing, LMedS, dataset split, transfer-learning protocol, ablation, runtime을 full text 기준으로 아래에 반영했다.

[확인] 현재 핵심 11편 중 **메인 논문 원문이 막힌 row는 없다.**

[확인] 다만 P5/P6 메인 PDF가 반복적으로 참조하는 별도 **Supporting Information** 파일 자체는 현재 업로드되어 있지 않다. 메인 PDF에 서술된 SI 결과는 사용할 수 있지만, SI 표/그림의 모든 세부 셀과 hyperparameter를 독립 검산하려면 separate supporting PDF가 있으면 더 좋다.

---

# 2. Executive comparison matrix

| Paper | Sensor at inference | Learned core | Geometric / analytic stage | Main pose output | 대표 pose 성능 | Runtime | 가장 강하게 증명한 것 |
|---|---|---|---|---|---|---|---|
| P1 Mueller | `[확인] RGB` | YOLOv8-pose, 4 face corners | PnP | 6DoF | `[확인] front에서 max avg trans. 4.2 cm / rot. 8.2°` | `NOT_REPORTED` | synthetic-only keypoint→PnP feasibility |
| P2 Knitt | `[확인] RGB` | DOPE/VGG19, belief maps + vector fields | PnP | 6DoF | `[확인] NDDS3 filtered 12.9–17.8 cm, robustness 94.9–97.5%` | `[확인] source runtime text internally inconsistent` | synthetic dataset design의 중요성 |
| P3 FFS | `[확인] RGB` | YOLOv4 + WithBNet | KRR + PnP | 6DoF | `[확인] worst avg trans. 7.5 cm; worst success 100%` | `[확인] 49.2 ms total, RTX3090` | unseen **load appearance** robustness |
| P4 CUT+CosyPose | `[확인] RGB image + externally supplied initial pose` | CUT + CosyPose refiners | iterative refinement | refined 6DoF | `[확인] ADD-S AUC >0.81; z MAE <7 cm under tested noise region` | `[확인] 570 ms multi-stage` | synthetic→real refinement under noisy initialization |
| P5 YOLOv11 | `[확인] RGB-D D435i` | Improved YOLOv11s-pose | depth backprojection + weighted/topology EPnP + LM | pose / distance / tilt | `[확인] mean angle <2.9°, mean distance 13–22 mm across tested bins` | `[확인] 44.1 FPS RK3568 **model inference**` | occlusion/stacking + lightweight embedded keypoint detection |
| P6 Hyper-pose | `[확인] RGB-D D455` | Hyper-YOLO/Hyper-pose + HAFB + EGKD | uncertainty + geometric constraints + L-BFGS-B | distance/angle, pose metrics | `[확인] avg angle 1.6°, avg distance 18 mm; ADD-S 95.8%; 5cm5° 93.2% reported` | `[확인] 72.1 FPS Jetson **model inference**` | geometry-aware keypoint modeling + edge deployment |
| P7 Miura | `[확인] monocular RGB` | Depth-Anything-V2 + AsymFormer + dual heads | floor normal; PnP only for augmentation label transform | direct t + quaternion | `[확인] 3.88 cm / 1.65° overall; <5m both-rate 78.1%` | `[확인] full pipeline 27.5 ms / 36.3 FPS RTX4090` | RGB-only metric-depth + load/viewpoint robustness |
| P8 Vu | `[확인] RGB-D` | YOLOv8 + ResNet50 + PointNet + cross-modal reweighting + DenseFusion | optional ICP refinement | 6DoF | `[확인] loaded pallet ADD-AUC 0.66→0.74 with ICP; >70% occlusion 0.42→0.49` | `[확인] 87 ms without ICP, RTX2080Ti` | severe occlusion robustness |
| P9 Shao | `[확인] active binocular point cloud` | 없음/비딥러닝 descriptor | ISS + AGWF + SAC-IA + ICP | rigid transform / deviation / angle | `[확인] angle error ≈0.5°; paper reports translation-related averages 9.8/19.4 mm` | `[확인] descriptor time >30% reduction; full pipeline FPS not headline` | point-cloud registration reference |
| P10 Zhao | `[확인] RGB-D` | 없음, label-template matching | category matrix/template + feet geometry + smoothing | pallet location/angle | `[확인] max reported distance error -101.1 mm, angle 6.07°` | `[확인] 72.44–182.84 ms over 1–4 m` | classical RGB-D warehouse baseline |
| P11 Zhou 2025 JSC | `[확인] RealSense D435i RGB stream; depth-capable camera이나 stated equations는 depth-map sample을 사용하지 않음` | StarNet + FFDPN + BDEM + APT-TAL + BA-Wing | known E-geometry + 20 point-pair hypotheses + LMedS | **distance D + inclination θ** (full 6DoF 아님) | `[확인] D_avg 13–29 mm; θ_avg 2.1–4.0° over tested bins` | `[확인] 104.2 FPS RTX4060Ti **model inference**` | YOLOv8→YOLOv11 계보의 직접 predecessor; edge/keypoint/solver evolution 기준 |

---

# 3. Architecture matrix

## 3.1 Learned vs analytic components

| Paper | Learned components | Analytic / geometric components | 핵심 의도 |
|---|---|---|---|
| P1 | YOLOv8 pose head | known dimensions + PnP | synthetic keypoint prediction을 metric pose로 변환 |
| P2 | VGG19-based DOPE, 9 belief maps, 16 vector fields | PnP + known 3D model | synthetic-only 6DoF |
| P3 | YOLOv4, WithBNet | KRR rear-corner inference + PnP | cargo appearance 영향을 받는 rear-corner CNN 의존도 제거 |
| P4 | CUT, CosyPose coarse/fine refiners | initial pose noise generation | reality gap + wide state-space refinement |
| P5 | StarNet/C3k2_Star, CSP_DEFE, RTA(DFM+TLM), YOLO pose head | D435i depth, weighted EPnP, topology objective, LM | E-shape prior + occlusion/stacking robustness |
| P6 | HGC-SCS hypergraph, HAFB, EGKD(MSFA+EGSA+HKD), uncertainty estimator | RGB-D geometry, Mahalanobis-weighted constraints, L-BFGS-B | high-order geometry + uncertainty + pruning |
| P7 | Depth-Anything-V2, ConvNeXt-Tiny, MiT-B0, AsymFormer-style fusion, dual heads | floor normal PCA; augmentation label EPnP | monocular depth ambiguity를 learned metric depth로 보완 |
| P8 | YOLOv8, ResNet50, PointNet, GAV-FR, VAG-FR, DenseFusion, pose regression | optional ICP | occluded/background feature down-weight |
| P9 | 없음 | pass/voxel/plane segmentation, ISS, AGWF, SAC-IA, ICP | registration robustness와 descriptor 효율 |
| P10 | 없음 | pixel category matrix, label template matching, pallet-foot geometry | 학습 없이 RGB-D structure exploitation |
| P11 | StarNet backbone + FFDPN neck + C2f_BDEM + APT-TAL + BA-Wing + YOLOv8 pose head | 12 E-section keypoints + calibrated pinhole geometry + known H/L + LMedS over point-pair hypotheses | occlusion/stacking keypoint robustness와 distance/inclination 안정화 |

---

# 4. Dataset and supervision matrix

| Paper | Train supervision | Real labels used for training? | Target real image exposure? | Test setting | Known geometry/CAD dependency |
|---|---|---:|---:|---|---|
| P1 | `[확인] synthetic bbox + 4 keypoints` | No | No | real 60 frames / 6 video sequences | `[확인] Euro pallet type 1, known face dimensions` |
| P2 | `[확인] synthetic 6D labels from NDDS` | No | No | real dynamic lab experiments | `[확인] exact EPAL 3D model required by DOPE/PnP` |
| P3 | `[확인] real image bbox/front/rear corner/pose relations` | Yes | Yes | load appearance high/mid/low | `[확인] known 1100×1100×150 mm pallet, fixed camera` |
| P4 | `[확인] synthetic pose labels; CUT uses unpaired real images` | No pose labels for CUT | **Yes** | RPP real 3.2k | `[확인] CosyPose render/refine object model assumptions` |
| P5 | `[확인] real keypoint labels + COCO transfer` | Yes | Yes | indoor/outdoor, occlusion, stack, distance/angle | `[확인] 12 E-section points + depth + known geometric dimensions` |
| P6 | `[확인] real keypoint labels + transfer` | Yes | Yes | 2 pallet standards, many environment conditions | `[확인] standard pallet templates and E topology` |
| P7 | `[확인] real 6DoF labels from AR marker/Visual SLAM` | Yes | Yes | standard + novel viewpoint/load | `[확인] known pallet dimensions; no external CAD model input for inference` |
| P8 | `[확인] supervised RGB-D pose dataset` | Yes | Yes | unloaded/loaded + severe occlusion | `[확인] accurate 3D pallet model assumed` |
| P9 | `N/A` | N/A | N/A | production-shop point cloud | `[확인] source pallet point cloud/template required` |
| P10 | `N/A` | N/A | N/A | warehouse static/dynamic | `[확인] label template and pallet structure required` |
| P11 | `[확인] real 12-keypoint labels + COCO-pretrained YOLOv8s-pose transfer` | Yes | Yes | 4125→12112, 70/20/10 + separate occlusion/overlap + 3300 multi-scene test | `[확인] fixed E-section dimensions/angles + calibrated camera intrinsics` |

## 4.1 중요한 데이터 누출/노출 주의점

### P4 Machines 2025

[확인] RPP는 3.2k real images이며 데이터 절에서는 "only used for testing"이라고 적는다.

[확인] 동시에 CUT은 **WSPP 500장 + RPP real 350장**으로 synthetic→real translation을 학습한다.

[추정] 따라서 "real 6D pose annotation을 학습에 사용하지 않았다"는 말은 맞지만, **target-domain real image 자체는 adaptation에 노출된다.** strict target-free synthetic-only 방법과 동일 조건이 아니다.

[추정] 350장의 CUT target image와 최종 점수 산출 RPP subset의 image-level 중복 여부가 본문에서 명확하게 분리되어 있지 않아, 엄격한 held-out target-domain 평가로 단정하지 않는다.

### P6 Hyper-pose 2026

[확인] 5411 originals를 augmentation으로 16233장으로 만든 뒤 70/20/10 split을 설명한다.

[추정] 원본 단위 group split인지에 대한 명시를 현재 확인한 본문에서 찾지 못했다. 따라서 동일 원본에서 파생된 augmented variants가 서로 다른 split에 들어가지 않았다는 보장은 문서만으로 확정하지 않는다.

---

# 5. Metric comparability audit

| Paper | Detection | Keypoint | Translation | Rotation | ADD/ADD-S | Task success | 직접 비교 수준 |
|---|---|---|---|---|---|---|---|
| P1 | mAP50/mAP50-95 | KME | axis **signed avg**, conclusion aggregate | axis **signed avg** | No | No | `APPROXIMATE` |
| P2 | robustness as successful update ratio | implicit DOPE beliefs | mean 3D Euclidean position error | 정량 주지표 아님 | No | robustness | `APPROXIMATE` for RGB synthetic-to-real position |
| P3 | YOLO threshold | corner-based | 3D translation error | 6DoF result but original headline focuses translation | No | inference success | `DIRECT/APPROXIMATE` for RGB camera-facing load shift |
| P4 | N/A | N/A | axis MAE | axis MAE | ADD-S AUC 0–10 cm | No | `NOT_COMPARABLE` to end-to-end image→pose |
| P5 | P/mAP | APKP (OKS) | distance error | inclination error | No | actual pick/place 95.7% | `NOT_COMPARABLE` to RGB-only because depth used |
| P6 | P/mAP@0.5 | KPOKS | distance / translation | angle / rotation | ADD-S 95.8% | 5cm5° 93.2% | `NOT_COMPARABLE` to RGB-only accuracy; useful deployment reference |
| P7 | detector not main headline | N/A | 3D Euclidean \|\|t_pred-t_gt\|\| | quaternion geodesic angle | No | `<5cm & <3°` | `DIRECT` candidate for monocular RGB 6DoF |
| P8 | detector is first stage | N/A | included in ADD | included in ADD | ADD AUC up to 10 cm | No | `NOT_COMPARABLE` sensor-wise; occlusion reference |
| P9 | N/A | ISS point keypoints | registration displacement | deflection angle | No | total accuracy 97.3% (paper-defined) | `QUALITATIVE_ONLY` |
| P10 | detection rate 92.6% | N/A | distance error | angle error | No | No | `QUALITATIVE_ONLY` |
| P11 | P99.6 / mAP94.4 | AP-KP93.2 | **distance D error**, D_avg 13–29 mm | **inclination θ error**, θ_avg 2.1–4.0° | No | No | `APPROXIMATE` for forward-distance/yaw-like variables; **not full 6DoF** |

## 5.1 특히 금지할 비교

[확인] 다음 식의 순위표는 만들면 안 된다.

```text
Hyper-pose 18 mm
YOLOv11 19 mm
VISAPP 38.8 mm
FFS 75 mm
DOPE 150 mm
→ Hyper-pose가 가장 정확
```

이 순위는 센서와 metric 정의가 다르다.

- P5/P6: active depth가 실제 pose 계산에 들어가는 RGB-D 시스템.
- P7: RGB만 입력하지만 Depth-Anything-V2로 metric depth를 추정하고 direct regression.
- P3: RGB + front/rear 2D geometry + PnP.
- P2: RGB + DOPE cuboid + PnP.
- P4: noisy GT pose에서 시작하는 refiner.

[추정] 따라서 위 숫자는 **"reported operating points"** 로 한 표에 나열할 수는 있지만, `best → worst` 순위를 붙이면 학술적으로 방어하기 어렵다.

---

# 6. Generalization taxonomy

> `[미검증 내부 분류 스키마]` 아래 G0–G8은 표준 학술 taxonomy라고 주장하지 않는다. 이 audit에서 서로 다른 distribution shift를 섞지 않기 위한 내부 분류다.

- `G0` in-distribution
- `G1` unseen load/cargo appearance
- `G2` unseen pallet instance, same basic geometry
- `G3` unseen pallet material
- `G4` unseen pallet geometry/topology/dimensions
- `G5` unseen camera viewpoint/mounting position
- `G6` environment/domain shift: lighting/background/indoor-outdoor/camera
- `G7` distance extrapolation/range shift
- `G8` occlusion/stacking

## 6.1 Matrix

| Paper | G1 load | G2 instance | G3 material | G4 geometry | G5 viewpoint | G6 environment | G7 distance | G8 occlusion |
|---|---|---|---|---|---|---|---|---|
| P1 Mueller | NOT_TESTED | NOT_TESTED | NOT_TESTED | **NOT_TESTED** | limited angles | partial | 1–5 m tested | NOT_TESTED |
| P2 Knitt | NOT_TESTED | NOT_TESTED | NOT_TESTED | **NOT_TESTED** | limited dynamic orbit | **TESTED** lighting/background | 2.5–5.5 m synthetic / ~3.5 m real | incidental only |
| P3 FFS | **TESTED** | NOT_DEMONSTRATED | NOT_TESTED | **NOT_TESTED** | fixed camera | limited | close-range behavior observed | load-caused visibility indirect |
| P4 CUT+CosyPose | NOT_MAIN | NOT_TESTED | NOT_TESTED | NOT_TESTED | NOT_MAIN | **TESTED synthetic→real target domain** | **1–5 m** | not main |
| P5 YOLOv11 | cargo/scene variation present but no held-out protocol | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | front/side scenarios | **TESTED** | close/long test | **TESTED** masked/overlap |
| P6 Hyper-pose | scene variation | C/E both trained/tested | material diversity mentioned, no holdout | **NOT_TESTED AS HELD-OUT** | angle bins tested, not unseen mount | **TESTED** | 1–5 m | **TESTED 0/30/50%** |
| P7 Miura | **TESTED** | not cleanly isolated | not cleanly isolated | NOT_TESTED | **TESTED, but confounded** | same warehouse mostly | 0–9 m | not main |
| P8 Vu | loaded vs unloaded | dataset-specific | not isolated | not isolated | not isolated | warehouse dataset | not primary | **TESTED <50/>50/>70%** |
| P9 Shao | N/A | limited | N/A | template-specific | pose deviation range | production environment | sensor 0.7–6 m | limited |
| P10 Zhao | goods template | multiple pallets detectable | not isolated | template-specific | ±25° | warehouse | **1–4 m** | obstacles tested |
| P11 JSC | NOT_TESTED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | front/45°/60° condition test, held-out mount 아님 | indoor/outdoor/light/background **condition test** | 1/2/3m detection; 1.4–2.4m pose | **TESTED** occluded/overlap sets |

## 6.2 FFS의 "unlearned pallet appearance" 해석

[확인] FFS 원문은 **load appearance**를 "type, size, amount, color of the load on the pallet"로 정의한다.

[확인] 평가 영상은 **same pallets but different load appearances**라고 명시한다.

[확인] High / Medium / Low는 각각 다음이다.

- High: training과 같은 색의 one-tier container box
- Medium: 다른 색의 two-tier container box
- Low: safety cones

[확인] 평가 수는 High 285 / Medium 191 / Low 278.

[추정] 따라서 FFS를 인용할 때 `"unseen pallet generalization"`이라고 쓰면 과장이다. 정확한 표현은 `"unseen load/cargo appearance robustness on the same pallet geometry"`이다.

## 6.3 VISAPP novel viewpoint의 정확한 의미

[확인] novel viewpoint set은 84장이고 manually corrected GT를 사용한다.

[확인] training과 다른 것은 **camera mounting position뿐 아니라 load appearance도 동시에 다르다**. 빈 pallet 또는 cardboard box load이며 training은 container box load다.

[확인] 따라서 이 결과는 `G5 pure viewpoint`가 아니라 **`G5 + G1 compound shift`** 이다.

[확인] 이 compound shift에서 mean translation error = **8.12 cm**, rotation error = **2.75°**, Both Achievement Rate = **34.48%**.

[확인] 단, 이 세 수치의 모집단이 서로 같지 않다. Both Achievement Rate는 정의상 **camera-to-pallet distance 5 m 이내 샘플에 대해서만** 계산된다 (Sec 4.1.4: `"for samples with camera-to-pallet distance within 5 m, we calculate the proportion satisfying Er <= 3.0 deg and Et <= 0.05 m"`). 따라서 34.48%의 분모는 novel viewpoint 84장 전체가 아니라 그중 5 m 이내 부분집합이며, **그 부분집합의 크기는 논문에 보고되지 않았다.**

[추정] 34.48%는 10/29 또는 20/58로 정확히 떨어지므로 해당 부분집합은 29장(또는 58장)으로 보인다. 논문이 명시하지 않았으므로 인용할 때는 장수를 쓰지 말고 `"novel viewpoint의 5 m 이내 부분집합"`이라고만 쓴다.

[확인] 같은 문단이 비교 대상으로 제시한 `(3.88 cm, 78.1%)`도 모집단이 섞여 있다 — 3.88 cm는 0–9 m 기준이고 78.1%는 5 m 이내 기준이다. 따라서 `8.12 cm ↔ 3.88 cm`와 `34.48% ↔ 78.1%`를 한 덩어리로 인용하면 서로 다른 모집단을 같은 조건처럼 읽게 된다. 우리 문서에서 이 논문의 열화 폭을 인용할 때는 두 쌍을 분리해서 쓴다.

---

# 7. Runtime and deployment audit

| Paper | Hardware | Model-only / component | Full pipeline | FPS 해석 주의 |
|---|---|---|---|---|
| P1 | RTX3080 10GB training | inference runtime NOT_REPORTED | NOT_REPORTED | pose accuracy와 speed 비교 불가 |
| P2 | 2× RTX2080Ti workstation | DOPE update 0.17/0.21 s reported | unclear | paper also reports ~0.85 FPS → inconsistency |
| P3 | RTX3090 + i9-10850K | YOLO 19.1ms / WithBNet21.5 / KRR7.4 | **49.2 ms total** | 약 20.3 theoretical FPS, camera 15 FPS |
| P4 | Quadro RTX6000 + Xeon6138 | refiner ~1.8 GFLOPs/iter | **570 ms** multi-stage | CUT 240ms는 offline adaptation preprocessing |
| P5 | RTX4060Ti server / RK3568 | **105.6 / 44.1 FPS pose detector** | NOT_REPORTED | depth+EPnP+LM 포함 end-to-end FPS로 쓰지 말 것 |
| P6 | RTX4090 / Jetson Orin Nano | PC109.6 / **Jetson72.1 FPS Hyper-pose inference** | NOT_REPORTED | TensorRT model inference vs entire RGB-D geometry pipeline 분리 |
| P7 | RTX4090 | YOLOX2.52 / Depth7.73 / floor3.24 / pose5.49 ms | **27.5 ms = 36.3 FPS** | full-pipeline 수치가 명확 |
| P8 | RTX2080Ti | proposed learned path **87 ms**, without ICP | ICP 포함 runtime NOT_REPORTED in table | learned RGB-D network vs refinement 분리 |
| P9 | point-cloud platform | descriptor extraction 비교 | full pipeline headline FPS 없음 | active sensor/reference method |
| P10 | RGB-D | N/A | 72.44/85.45/117.63/182.84 ms at 1/2/3/4m | 거리 증가 시 latency 증가 |
| P11 | RTX4060Ti 16GB + i5-13600KF | **104.2 FPS Improved_YOLOv8s-pose detector/keypoint model** | LMedS 포함 full pipeline `NOT_REPORTED` | 104.2를 end-to-end pallet-positioning FPS로 확대해석 금지 |

## 7.1 72.1 FPS의 정확한 해석

[확인] P6 Table 1에서 Hyper-pose 최종 구성은 PC 109.6 FPS, NVIDIA Jetson Orin Nano 72.1 FPS다.

[확인] 같은 table은 HAFB/EGKD/topological pruning에 따른 **network inference speed**를 비교한다.

[추정] RGB-D acquisition + depth sampling + uncertainty geometric optimization + final distance/angle computation까지 모두 포함한 end-to-end throughput이라고 볼 근거는 현재 본문에서 확인되지 않는다.

따라서 인용 문구는 다음처럼 제한하는 것이 안전하다.

```text
Hyper-pose reports 72.1 FPS for the pruned TensorRT model on Jetson Orin Nano.
The paper does not clearly report an end-to-end RGB-D pose-pipeline latency including all geometric optimization stages.
```

---

# 8. Detailed review — P1 Mueller: synthetic YOLOv8 pose + PnP

## 8.1 Problem

[확인] 수동 real 6D annotation을 피하면서 monocular RGB로 pallet detection과 localization을 수행하는 것이 목표다.

## 8.2 Pipeline

```text
Unity domain randomization
  ↓
synthetic RGB + bbox + 4 face-corner keypoints
  ↓
YOLOv8-pose (n/s/m/l/x 비교)
  ↓
4 projected face corners
  ↓
known face dimensions + calibrated intrinsics
  ↓
PnP
  ↓
translation + rotation
```

## 8.3 Data

[확인] synthetic train sizes는 7.5k / 15k / 30k다.

[확인] Unity에서 primary light intensity, pallet texture, xyz position, orientation을 randomize한다.

[확인] real test는 6 video sequences에서 추출한 manually annotated 60 frames다.

[확인] Euro-pallet type 1 하나만 학습·테스트하며 알려진 face dimensions를 사용한다.

## 8.4 Keypoint / loss

[확인] 4 keypoints를 anchor center에 상대적으로 regress하고, YOLOv8 pose의 Object Keypoint Similarity 계열 loss를 사용한다고 기술한다.

## 8.5 Reported results

[확인] YOLOv8x 30k는 real test에서 mAP50 0.995를 보고한다.

[확인] front configuration의 최대 평균 translation error는 4.2 cm, rotation error는 8.2°라고 discussion에서 요약한다.

## 8.6 Critical caveat

[확인] Table 4의 x/y/z 및 rx/ry/rz는 **signed average errors**다.

[확인] conclusion은 `"total mean position error 3.6 cm"`와 `"total mean rotation error -1.83°"`를 보고한다.

[추정] 음수 rotation "error"와 axis별 signed values 때문에 3.6 cm / -1.83°를 일반적인 MAE 또는 geodesic pose error처럼 사용하는 것은 부적절하다.

[확인] 저자도 `"lack of a reliable ground truth"`와 single pallet만 사용했다는 점을 주요 한계로 명시한다.

### 판정

`QUALITATIVE_ONLY ~ APPROXIMATE`

- synthetic-only RGB keypoint→PnP라는 문제 구조는 현재 연구와 가까움.
- pose accuracy 숫자는 ground-truth 및 signed averaging 문제 때문에 SOTA 순위용으로 약함.

---

# 9. Detailed review — P2 Knitt: NDDS + DOPE

## 9.1 Pipeline

```text
photorealistic EPAL-1 CAD/PBR model
  ↓
NDDS synthetic generation
  ├─ lighting
  ├─ background
  ├─ pallet texture
  ├─ distractors
  └─ camera wiggle
  ↓
DOPE / VGG19
  ├─ 9 belief maps
  │   ├─ 8 cuboid vertices
  │   └─ centroid
  └─ 16 vector fields
  ↓
belief-map peaks + known 3D model
  ↓
PnP
  ↓
6DoF
```

## 9.2 Dataset variants

[확인]

| Dataset | Images | Key changes |
|---|---:|---|
| NDDS1 | 50k | random colors/patterns, 3 random lights |
| NDDS2 | 100k | realistic pallet texture, real/random background |
| NDDS3 | 50k | realistic texture, 8 lights, distractors, camera wiggle, 2.5–5.5m |

[확인] 각 dataset을 90/10 train/test로 나누고 60 epochs 학습한다.

## 9.3 Real evaluation and GT

[확인] Logitech C270 RGB webcam을 사용한다.

[확인] Qualisys optical MoCap로 camera와 pallet ground-truth pose를 측정하며, paper는 mm-range position accuracy와 100 FPS update를 기술한다.

[확인] 실험은 camera가 stationary pallet 주위를 대략 3.5 m 거리로 이동하는 dynamic experiment다.

## 9.4 Results

| Dataset / Exp | Unfiltered cm | Filtered cm | Robustness |
|---|---:|---:|---:|
| NDDS3 Exp1 | 19.2 | 12.9 | 94.9% |
| NDDS3 Exp2 | 16.4 | 17.8 | 96.0% |
| NDDS3 Exp3 | 16.1 | 14.0 | 97.5% |

[확인] NDDS3는 세 lighting experiment 모두 position error <20 cm, robustness >94%를 달성한다.

## 9.5 Generalization boundary

[확인] lighting과 background는 실제 실험 중 변한다.

[확인] 논문은 pallet shape/size가 3D model과 달라지면 failed detection 또는 incorrect pose가 날 수 있다고 명시한다.

[확인] stacked/loaded pallet 및 close-range는 약점이라고 기술한다.

[추정] 따라서 이 논문은 **synthetic-to-real appearance robustness** 근거이지 unseen pallet geometry 근거가 아니다.

## 9.6 Runtime inconsistency

`SOURCE-INTERNAL INCONSISTENCY`

[확인] 논문은 pose update duration이 약 0.17 s 또는 0.21 s에 분포한다고 쓴다.

[확인] 같은 문단에서 average update frequency가 약 0.85 FPS라고 쓴다.

[추정] 0.17–0.21 s의 단순 역수는 약 4.8–5.9 FPS이므로 두 수치는 직접적으로 일치하지 않는다. 원문을 수정해 해석하지 말고 둘 다 기록한다.

### 판정

`APPROXIMATE`

현재 RGB synthetic-only pose track과 문제 설정이 매우 가깝지만, DOPE cuboid representation 및 evaluation 조건이 다르다.

---

# 10. Detailed review — P3 FFS: YOLOv4 + WithBNet + KRR + PnP

## 10.1 Problem

[확인] 핵심 문제는 pallet 자체보다 **load appearance 변화가 whole-image CNN pose estimation을 교란한다**는 것이다.

## 10.2 Pipeline

```text
RGB
  ↓
YOLOv4
  ├─ pallet-body bbox
  ├─ left fork-hole bbox
  └─ right fork-hole bbox
  ↓
WithBNet
  input:
    320×320 grayscale
    + 3 bbox vectors × 4 = 12-D
  output:
    4 front corners = 8-D
  ↓
KRR
  input:
    8-D front corners + 12-D bbox = 20-D
  output:
    4 rear corners = 8-D
  ↓
8 corners
  ↓
PnP
  ↓
6DoF
```

## 10.3 WithBNet

[확인] SSP와 VGG 계열 아이디어를 참고한다.

[확인] grayscale 320×320을 사용하는 이유는 pallet front edge intersection을 잡는 데 edge information이 핵심이라는 설계 논리다.

[확인] convolution branch와 bbox fully connected branch를 결합한 후 8개 coordinate를 출력한다.

[확인] optimizer Adam, loss MSE, 100 epochs이며 final reported loss는 2.6×10^-6이다.

## 10.4 KRR

[확인] explanatory vector `s`는 front corners + body/left/right bboxes = 20-D다.

[확인] Gaussian kernel을 사용하며 실험에서 beta=0.3, lambda=1e-2를 선택한다.

[확인] rear corners를 CNN 대신 KRR로 추정하는 핵심 이유는 fixed camera + known pallet size 조건에서 front/rear geometry 관계를 appearance-independent nonlinear regression으로 모델링하기 위해서다.

## 10.5 Data / GT / assumptions

[확인] known-size 1100×1100×150 mm plastic pallet을 사용한다.

[확인] camera는 forklift에 고정되고 pallet은 horizontal floor에 놓인다고 가정한다.

[확인] 20,000 images를 준비하고 SSP/FFS에 18,000 train, 나머지를 validation으로 사용한다.

[확인] AR marker로 pallet relative 6DoF를 구하고 Visual SLAM으로 camera position을 추정한다.

## 10.6 Load appearance test

[확인]

| Split | 의미 | N |
|---|---|---:|
| High | same-color one-tier container | 285 |
| Medium | different-color two-tier container | 191 |
| Low | safety cones, training과 완전히 다른 load | 278 |

[확인] 같은 pallet을 사용하고 load만 달라진다.

## 10.7 Result

[확인] original FFS paper가 textual headline으로 주는 worst average translation error는 SSP 15.1 cm, SDG 14.9 cm, FFS 7.5 cm다.

[확인] worst inference success rate는 SSP 75%, SDG 28.4%, FFS 100%다.

[확인] SDG는 공개 pretrained weight만 사용했고 training code가 없어 authors' dataset에 fine-tune하지 못했다.

[추정] 따라서 SDG와 SSP/FFS의 비교는 동일 supervision 조건이 아니며, SDG row를 model architecture의 절대 우열로 해석하면 안 된다.

## 10.8 Runtime

[확인]

| Component | ms |
|---|---:|
| YOLO | 19.1 |
| WithBNet | 21.5 |
| KRR | 7.4 |
| **Total FFS** | **49.2** |

[확인] hardware는 Core i9-10850K + RTX3090, camera 1920×1080 15 FPS다.

[추정] 49.2 ms의 단순 역수는 약 20.3 FPS지만 실제 acquisition은 15 FPS로 제한된다.

### 판정

`DIRECT`에 가장 가까운 RGB-only pallet-specific comparison 중 하나.  
단, generalization claim은 반드시 **load appearance**로 한정한다.

---

# 11. Detailed review — P4 Machines: CUT + multi-stage CosyPose

## 11.1 Problem

[확인] synthetic real gap과 1–5 m wide state-space를 해결하는 것이 목표다.

## 11.2 Pipeline

```text
WSPP synthetic RGB
  ↓
CUT (unpaired synthetic→real appearance)
  ↓
adapted-WSPP
  ↓
CosyPose coarse refiner
  ↓
CosyPose finer refiner
  ↓
refined 6DoF
```

## 11.3 CUT

[확인] one generator + one discriminator를 사용하는 CUT을 선택한다.

[확인] CUT train = WSPP 500 images + RPP real 350 images.

[확인] adapted-WSPP는 55k이며 50k train / 5k validation.

[확인] CUT은 pose estimator inference path에 들어가지 않고 **training-data preprocessing**으로 사용할 수 있다고 명시한다.

## 11.4 Refiners

[확인] single-stage basic refiner training noise:

- xyz STD 30 cm
- roll/pitch/yaw STD 15°

[확인] finer refiner training noise:

- x/y STD 1 cm
- z STD 5 cm
- roll/pitch/yaw STD 5°

[확인] multi-stage는 first refiner 2 iterations + finer refiner 4 iterations로 구성된다.

## 11.5 Evaluation contract — 가장 중요한 caveat

[확인] real RPP test에서 **ground-truth pose에 random noise를 더해 initial pose estimate를 만든다.**

```text
RPP GT pose
  + artificial Gaussian noise
      ↓
initial pose
      ↓
refiner(s)
      ↓
refined pose
```

[추정] 따라서 이 논문의 ADD-S AUC는 **detector부터 시작한 end-to-end image→pose** 성능이 아니라, noisy initialization을 복구하는 **pose refinement** 성능이다.

## 11.6 Metrics/results

[확인] ADD-S AUC threshold는 0–10 cm다.

[확인] proposed multi-stage domain-adapted pipeline은 tested region에서 ADD-S AUC >0.81을 보고한다.

[확인] depth-noise experiment에서 final iteration의 z MAE는 약 7 cm 아래, y 약 2.5 cm, x 약 0.5 cm 수준으로 기술된다.

[확인] yaw input noise STD=10°일 때 roll/pitch/yaw MAE가 모두 3° 미만이다.

## 11.7 Runtime

[확인] Quadro RTX6000 + Xeon Gold 6138.

[확인] original CosyPose 440 ms, proposed multi-stage 570 ms.

[확인] CUT 240 ms이나 offline computation이므로 pose online latency에 포함하지 않는다고 설명한다.

### 판정

`NOT_COMPARABLE` to end-to-end pallet pose accuracy.  
`QUALITATIVE_ONLY` for synthetic→real adaptation and refinement design.

---

# 12. Detailed review — P5 Improved YOLOv11s-pose

## 12.1 Pipeline

```text
RGB-D D435i
  │
  ├─ RGB
  │   ↓
  │ Improved YOLOv11s-pose
  │   ├─ StarNet / C3k2_Star
  │   ├─ CSP_DEFE
  │   │   ├─ Scharr / spatial-time features
  │   │   └─ FFT frequency-domain features
  │   └─ RTA
  │       ├─ DFM
  │       └─ TLM
  │   ↓
  │ 12 E-section keypoints
  │
  └─ D435i depth
      ↓
keypoint 3D backprojection
      ↓
visibility-weighted EPnP
      ↓
topological constraints
  ├─ horizontal
  ├─ right-angle
  ├─ horizontal edge length
  └─ vertical edge length
      ↓
Levenberg–Marquardt optimization
      ↓
pose / distance / inclination
```

## 12.2 StarNet

[확인] YOLOv11s-pose backbone을 Star operation 기반으로 재구성해 feature interaction과 lightweight deployment를 노린다.

## 12.3 CSP_DEFE

[확인] Scharr-based edge extraction과 FFT/IFFT frequency-domain processing을 결합한다.

[확인] 의도는 occlusion/stacking 상황에서 local edge detail과 global repetitive pallet structure를 동시에 유지하는 것이다.

## 12.4 RTA = DFM + TLM

[확인] DFM은 spatial focus + frequency attention으로 pallet edge/corner 특징을 강조한다.

[확인] TLM은 keypoint feature correlation matrix에 predefined structure prior와 learnable position embedding을 더해 E-shaped spatial topology를 명시적으로 모델링한다.

## 12.5 Pose stage

[확인] depth value를 사용해 each keypoint를 camera 3D 좌표로 backproject한다.

[확인] EPnP reprojection objective에 visibility weight를 넣는다.

[확인] topology objective를 reprojection objective와 결합하고 LM으로 refine한다.

[확인] 따라서 **실험된 시스템은 RGB-only가 아니다.**

## 12.6 Dataset

[확인] 4125 originals → 12112 augmented.

[확인] 80/10/10 split + 별도 occlusion/stacking validation sets.

[확인] COCO-pretrained YOLOv11s-pose에서 transfer learning.

[확인] training: RTX4060Ti 16GB, 180 epochs, batch8, lr0.001, SGD.

## 12.7 Detection/keypoint result

[확인] final ablation row:

- P 99.6%
- mAP 95.1%
- APKP 94.2%
- 19.21 M parameters
- 15.1G FLOPs
- 105.6 FPS server
- 44.1 FPS RK3568

### Naming caveat

`SOURCE-INTERNAL TERMINOLOGY ISSUE`

[확인] abstract는 `"object detection accuracy 95.1%"`라고 부르지만 Table 1에서 95.1은 **mAP**, precision은 99.6이다.

[추정] 외부 표에서는 95.1을 단순 `"detection accuracy"`로 쓰지 말고 `"mAP = 95.1%"`로 명명하는 것이 안전하다.

## 12.8 Robustness

[확인] masked dataset final APKP 92.2%, overlap dataset 93.3%.

[확인] long-distance test APKP 92.7%, complex-context 93.6%.

[확인] 이 multiscene 수치는 condition-specific robustness지만 **held-out pallet geometry/material** 실험은 아니다.

## 12.9 Pose errors

[확인] angle −60°…60°에서 combined visibility+topology method의 average angle error는 2.1–2.9° 범위이며 max ≤3.7°.

[확인] distance 1300–2500 mm에서 combined method average distance error는 13–22 mm, max ≤37 mm.

[확인] 실제 automated forklift pick-and-place를 100회 이상 수행해 95.7% success rate를 보고한다.

[확인] 다만 final insertion 단계에서는 progressive approach와 visual servoing 전략도 사용한다고 명시한다.

## 12.10 Runtime interpretation

[확인] 105.6 / 44.1 FPS는 model table의 YOLO pose inference speed다.

[추정] D435i depth processing + EPnP + LM을 모두 포함한 full pipeline FPS는 명확히 보고되지 않는다.

### 판정

- Accuracy rank vs RGB-only: `NOT_COMPARABLE`
- Edge deployment reference: `HIGH VALUE`
- Occlusion/stacking evaluation design: `HIGH VALUE`

---

# 13. Detailed review — P6 Hyper-pose 2026

## 13.1 Problem

[확인] E-shaped pallet의 rigid high-order keypoint relationships, occlusion/stacking uncertainty, edge deployment를 동시에 다룬다.

## 13.2 Hypergraph formulation

[확인] 12 semantic keypoint nodes.

[확인] 12 hyperedges:

- 3 horizontal
- 3 vertical
- 6 symmetry

[확인] predicted keypoint 주변 feature map에서 7×7 RoI를 뽑고 RoI pooling + FC로 256-D node feature를 만든다.

[확인] hyperedge weight는 geometric prior와 node feature cosine consistency를 결합한다.

[확인] 3-layer hypergraph convolution으로 structure-wide information을 전달한다.

## 13.3 HAFB

[확인] Hierarchical Attention Fusion Block은 2×2와 4×4 patch Local-Global Attention을 병렬 사용한다.

[확인] base branch + 2 attention branches를 concat한 뒤 1×1 conv와 RepConv로 fuse한다.

## 13.4 EGKD

[확인] Enhanced Geometry-Aware Keypoint Detection은:

- MSFA: multiscale adaptive fusion
- EGSA: E-shaped geometry spatial attention
- HKD: hierarchical keypoint detection

으로 구성된다.

[확인] HKD는 heatmap probability, subpixel offset, visibility/confidence 정보를 사용하며 coarse→constraint→fine localization 전략을 취한다.

## 13.5 Topology-preserving pruning

[확인] backbone / neck / head pruning rate를 각각 약 35.2% / 22.6% / 15.8%로 차등 적용한 구성이 보고된다.

[확인] overall 44% compression point에서 성능/속도 trade-off가 가장 좋다고 주장한다.

## 13.6 Uncertainty optimization

[확인] keypoint confidence conversion + response-map analysis + local-gradient evaluation을 uncertainty covariance로 결합한다.

[확인] dynamic threshold로 unreliable keypoint를 down-weight한다.

[확인] back-end에서는 Mahalanobis distance + geometric constraints를 구성하고 L-BFGS-B로 optimize한다.

## 13.7 Actual sensor and pose stage

[확인] Intel RealSense D455 RGB-D를 사용한다.

[확인] RGB는 12 keypoint localization, depth는 해당 keypoint의 spatial information을 제공한다.

[확인] 본문의 actual geometric pose stage는 단순 PnP 한 줄이 아니라 pallet dimensions, keypoint pair geometry, distance/angle formulas 및 uncertainty optimization을 사용한다.

[추정] 따라서 P6를 `"RGB keypoints + PnP"`라고 요약하면 실제 시스템 구조를 과도하게 단순화한다.

## 13.8 Dataset

[확인]

- C-pallet 1200×1000 mm
- E-pallet 1200×800 mm
- originals 5411 = C 3247 + E 2164
- augmented 16233
- train/val/test = 70/20/10

[확인] indoor/outdoor, occlusion, overlap, strong/weak light, angle, close/remote categories를 포함한다.

[확인] C/E ratio 6:4를 모든 split에 유지한다.

[추정] 두 pallet type이 모두 train/test에 들어가므로 이 실험은 **held-out pallet type generalization**이 아니다.

## 13.9 Model result / Jetson

[확인] final model table:

- P 99.5%
- mAP@0.5 97.6%
- KPOKS 97.3%
- Size/M 20.2
- FLOPs 23.2G
- PC 109.6 FPS
- Jetson Orin Nano 72.1 FPS

### Internal inconsistency 1: 20.2 MB vs 20.2 M

`SOURCE-INTERNAL INCONSISTENCY`

[확인] abstract/conclusion은 storage/model size를 `"20.2 MB"`라고 표현한다.

[확인] Table 1과 body는 `Size/M = 20.2`를 **20.2 M parameters**라고 설명한다.

[추정] 20.2M parameters와 20.2MB는 같은 단위가 아니므로 외부 문서에서는 둘을 하나의 확정값으로 합치지 않는다.

### Internal inconsistency 2: "detection accuracy"

[확인] abstract는 97.5% detection accuracy, table은 P99.5 / mAP97.6을 제시한다.

[추정] abstract의 97.5가 정확히 어느 table metric을 뜻하는지 명시적 mapping이 불완전하므로, 표에서는 **P99.5 / mAP97.6**을 우선 사용한다.

## 13.10 Pose results

[확인] final reported average angle error는 약 1.6°, average distance/translation error는 약 18 mm.

[확인] paper body가 summarise하는 S18 comparison에서는 ADD-S 95.8%, 5cm5° success 93.2%를 보고한다.

[확인] 50% occlusion에서 88.7% accuracy를 보고하고, full uncertainty optimization에서는 angle 1.9° / distance 22 mm를 보고한다.

## 13.11 Generalization limitation

[확인] conclusion은 method가 E-shaped pallet에 특화되며 other pallet geometries에는 수정이 필요하다고 인정한다.

[확인] four-corner, nine-leg, double-sided 등 다른 geometries는 **future work**로 남아 있다.

[추정] 따라서 P6의 강한 수치를 `"unseen pallet geometry generalization"` 근거로 사용하면 안 된다.

### 판정

- Edge deployment: `HIGH VALUE`
- RGB-only pose accuracy: `NOT_COMPARABLE`
- geometry/uncertainty architecture reference: `HIGH VALUE`

---

# 14. Detailed review — P7 VISAPP 2026: monocular metric depth direct regression

## 14.1 Pipeline

```text
Monocular RGB
  ├─────────────┬─────────────────────┐
  ↓             ↓                     ↓
YOLOX       Depth-Anything-V2       RGB image
pallet/     Metric-Indoor-Small
holes            ↓
bboxes       metric depth
                  ↓
             floor normal (PCA)
                  │
                  ↓
          AsymFormer-inspired
       ┌───────────────────────┐
       │ ConvNeXt-Tiny RGB     │
       │ MiT-B0 depth          │
       └───────────────────────┘
                  ↓
          SCC + cross-attention
                  ↓
          image feature 256-D
                  +
       bbox embeddings + floor normal
                  ↓
              shared MLP
             ┌─────┴─────┐
             ↓           ↓
       translation    quaternion
          head          head
             ↓           ↓
              direct 6DoF
```

## 14.2 Important modality distinction

[확인] 물리 sensor input은 **monocular RGB only**다.

[확인] 그러나 network 내부에서는 RGB로부터 Depth-Anything-V2가 **metric depth map을 생성**하고 RGB-D-style fusion을 수행한다.

[추정] 따라서 active RGB-D sensor methods와는 구별되지만, "depth cue를 전혀 사용하지 않는 RGB-only" 방법과도 architecture assumption이 다르다.

## 14.3 Depth model

[확인] `Depth-Anything-V2-Metric-Indoor-Small`, ViT-Small 24.8M parameters를 사용한다.

## 14.4 Output and loss

[확인] direct output은 `t ∈ R^3`와 normalized quaternion `q ∈ R^4`.

[확인] translation은 axis-weighted MSE를 쓰며 z-axis를 더 크게 가중한다.

[확인] rotation은 quaternion cosine similarity 기반 loss를 사용한다.

## 14.5 Virtual camera augmentation

[확인] virtual camera motion으로 RGB/depth를 homography warp한다.

[확인] warped image와 pose label의 consistency를 맞추기 위해 3D vertices → 2D projection → homography → transformed 2D → EPnP 순서로 새 pose label을 계산한다.

[추정] 여기의 EPnP는 main inference output을 만드는 stage가 아니라 **augmentation label generation**을 위한 geometric stage다.

## 14.6 Standard result

[확인] Table 1:

| Method | High | Middle | Low | Overall |
|---|---|---|---|---|
| SSP | 1.37cm / 0.12° | 7.07 / 0.97 | 15.1 / 1.83 | 8.12 / 1.00 |
| SDG | 9.02 / 0.85 | 14.9 / 3.06 | 11.9 / 5.89 | 11.7 / 3.36 |
| FFS | 5.72 / 0.55 | 7.49 / 1.14 | 7.46 / 1.20 | 6.86 / 0.96 |
| Ours <5m | 3.45 / 2.48 | 3.34 / 1.34 | 3.66 / 1.10 | 3.50 / 1.65 |
| Ours <9m | 3.51 / 2.41 | 4.42 / 1.39 | 3.91 / 1.03 | **3.88 / 1.65** |

[확인] translation error는 3D Euclidean distance `||t_pred-t_gt||2`.

[확인] rotation error는 quaternion angular difference.

[확인] within 5m Both Achievement Rate (`Et <=5cm AND Er<=3°`) = **78.1%**.

## 14.7 Novel viewpoint

[확인] N=84.

[확인] camera mounting + load appearance가 동시에 바뀐다.

[확인] result = **8.12 cm / 2.75° / Both 34.48%**.

[추정] 이 결과는 camera generalization에 중요한 evidence지만 pure-camera-viewpoint causal effect만 분리한 실험은 아니다.

## 14.8 Ablation

| Condition | Translation cm | Rotation ° | Both % |
|---|---:|---:|---:|
| Full | 3.88 | 1.65 | 78.1 |
| w/o Depth | 4.52 | 1.01 | 71.2 |
| w/o Floor Normal | 7.57 | 1.01 | 25.1 |
| w/o BBox | 6.90 | 1.27 | 41.9 |
| w/o Dual Head | 11.62 | 2.92 | 12.6 |
| w/o VCam Aug | 6.48 | 1.53 | 34.9 |
| w/o Depth Correction | 5.16 | 1.08 | 47.9 |

[확인] floor normal과 dual-head, virtual-camera augmentation의 contribution이 task success에 매우 크다.

## 14.9 Runtime

| Component | ms | FPS |
|---|---:|---:|
| YOLOX | 2.52 | 397 |
| Depth-Anything-V2 ViT-S | 7.73 | 129 |
| Floor normal CPU | 3.24 | 309 |
| Pose model | 5.49 | 182 |
| **Full pipeline** | **27.5** | **36.3** |

[확인] RTX4090.

[확인] 논문은 cloud-based computation을 지향하며 edge-forklift onboard result는 아니다.

### 판정

현재 문헌 중 **monocular RGB full-pipeline pose accuracy + runtime + load appearance + novel viewpoint**를 가장 명확히 동시에 보고한 축에 속한다.

---

# 15. Detailed review — P8 Vu 2024: occlusion-robust RGB-D

## 15.1 Pipeline

```text
RGB-D
  ↓
YOLOv8 bbox
  ├─ RGB crop → ResNet-50 → visual features
  └─ depth crop → 3D point cloud → PointNet → geometric features
                         ↓
             cross-modal feature re-weighting
               ├─ GAV-FR
               └─ VAG-FR
                         ↓
                    DenseFusion
                         ↓
                   pose regression
                         ↓
                    optional ICP
```

## 15.2 Purpose

[확인] bounding crop 안에 cargo/background가 섞여 pose model이 잘못된 feature를 쓰는 문제를 줄이기 위해 RGB와 point-cloud feature를 서로 이용해 re-weight한다.

## 15.3 Dataset

[확인] newly collected pallet dataset은 **80,000 annotated RGB-D images**를 포함한다고 공개 full text에 기술된다.

[확인] RealSense 및 ASUS Xtion 계열 sensor가 언급된다.

## 15.4 Metrics/results

[확인] paper는 ADD metric threshold AUC를 사용한다.

[확인] unloaded pallet에서 proposed method:

- without ICP: 0.77
- with ICP: 0.83

[확인] loaded pallet:

- without ICP: 0.66
- with ICP: 0.74

[확인] severe occlusion analysis에서 proposed method는 다음 AUC를 보고한다.

- occlusion <50%: 0.88 / 0.96 (w/o / w ICP)
- occlusion >50%: 0.78 / 0.85
- occlusion >70%: 0.42 / 0.49

[확인] 즉 >70%에서는 성능이 크게 감소하지만 다른 비교방법보다 높은 값이라고 보고한다.

## 15.5 Runtime

[확인] proposed learned model without ICP runtime = **87 ms**.

[확인] runtime comparison은 RTX2080Ti, 640×480에서 수행됐다고 기술한다.

[추정] ICP refinement 포함 full latency를 동일 table에서 명확히 주지 않으므로 0.83/0.74 refined accuracy와 87ms를 같은 operating point의 speed/accuracy pair로 묶지 않는다.

### 판정

`NOT_COMPARABLE` sensor-wise to monocular RGB, but **G8 occlusion benchmark design reference로 매우 유용**.

---

# 16. Detailed review — P9 Shao 2023: active binocular point cloud

## 16.1 Pipeline

```text
active binocular point cloud
  ↓
pass-through filtering
  ↓
voxel downsampling
  ↓
plane segmentation
  ↓
ISS keypoints
  ↓
AGWF descriptor
  ↓
SAC-IA coarse registration
  ↓
ICP fine registration
  ↓
rigid R,t
  ↓
horizontal deviation + deflection angle
```

## 16.2 Core contribution

[확인] 기존 FPFH의 fixed neighborhood radius와 Euclidean-distance-only weight를 개선한다.

[확인] local neighborhood entropy 기준으로 radius를 adaptive하게 고른다.

[확인] Gaussian weighting으로 neighborhood influence를 재정의한다.

## 16.3 Result

[확인] web-accessible official full text는 tested normal operating range에서 deflection angle average error 약 **0.5°**를 보고한다.

[확인] translation 관련 평균으로 약 **0.0098 m**와 **0.0194 m**를 기술한다. 현재 추출 문맥만으로 두 값의 정확한 축 명칭이 충분히 명료하지 않아 임의로 x/z로 이름 붙이지 않는다.

[확인] horizontal deviation >0.15m 또는 angle >15° 같은 out-of-limit cases에서는 FOV/initial-pose 문제로 registration failure가 발생한다.

## 16.4 Feature descriptor efficiency

[확인] AGWF는 traditional descriptor와 비교해 feature extraction time을 30% 이상 줄이고 error를 35% 이상 줄였다고 결론낸다.

### 판정

`QUALITATIVE_ONLY` for RGB project.  
센서 modality가 완전히 다르지만 **정밀 registration의 상한/engineering reference**로 유용.

---

# 17. Detailed review — P10 Zhao 2022: RGB-D label-template matching

## 17.1 Pipeline

```text
RGB-D
  ↓
color-based pixel classification
  ↓
category matrix
  ↓
labeled template
  ├─ goods
  ├─ pallet
  └─ ground
  ↓
compression + template matching
  ↓
pallet region
  ↓
pallet-foot geometry + depth
  ↓
pose/location
  ↓
sliding-average smoothing
```

## 17.2 Result

[확인] detection rate = **92.6%**.

[확인] detection times:

- 1 m: 72.44 ms
- 2 m: 85.45 ms
- 3 m: 117.63 ms
- 4 m: 182.84 ms

[확인] 1–4m, within ±25° 조건에서 paper가 보고한 maximum distance and angle estimation error는 **-101.1 mm**와 **6.07°**다.

[추정] distance max가 음수로 표기되어 있으므로 signed bias/error convention이 개입되어 있다. 일반적인 absolute max error처럼 그대로 순위 비교하지 않는다.

### 판정

`QUALITATIVE_ONLY` — classical RGB-D baseline.

---

# 18. Detailed review — P11 Journal of Supercomputing 2025: Improved YOLOv8s-pose + LMedS

## 18.1 Problem definition

[확인] 논문의 목표는 warehouse에서 pallet occlusion, dense stacking, lighting/background interference 때문에 생기는 **12개 E-section keypoint detection error**를 줄이고, 그 keypoint로부터 forklift가 사용할 **pallet inclination angle과 distance**를 안정적으로 계산하는 것이다.

[확인] 제목과 본문은 "pose estimation/localization"이라고 부르지만, 최종 정량 pose stage에서 직접 계산·평가하는 물리량은 주로 다음 두 개다.

```text
D      : pallet front center와 camera 사이 horizontal distance
theta  : pallet front orientation / inclination angle
```

[추정] 따라서 이 논문의 최종 output은 일반적인 `R ∈ SO(3), t ∈ R^3` 전체 6DoF와 같지 않다. 현재 연구의 `forward distance z`와 `yaw`에는 대응시킬 여지가 있지만, lateral x까지 포함한 full 3DoF/6DoF와는 별도 취급해야 한다.

## 18.2 End-to-end architecture

```text
RealSense D435i RGB image
        ↓
Improved_YOLOv8s-pose
  ├─ StarNet backbone
  ├─ FFDPN neck
  │    └─ FocusFeature multi-scale focus/diffusion
  ├─ C2f_BDEM
  │    ├─ Sobel-x / Sobel-y edge branch
  │    └─ spatial convolution branch
  ├─ APT-TAL label assignment
  └─ BA-Wing keypoint loss
        ↓
12 E-section keypoints
        ↓
calibrated pinhole geometry
+ known pallet E-section dimensions
        ↓
many admissible point-pair estimates
        ↓
LMedS robust selection
        ↓
distance D + inclination theta
```

### Sensor interpretation — 중요한 점

[확인] 실험 장치는 Intel RealSense **D435i depth camera**이고 논문은 이를 여러 번 "depth camera"라고 부른다.

[확인] 그러나 Section 3의 실제 distance/inclination 계산식은 **2D keypoint pixel coordinates + camera intrinsics + known pallet dimensions H/L**에서 `Z`, `theta`, `D`를 해석적으로 계산한다.

[확인] P5 YOLOv11 후속논문과 달리, P11 본문에는 각 keypoint의 **depth-map 값을 읽어서 3D로 back-project하는 절차가 기술되어 있지 않다.**

[추정] 따라서 P11을 `"RGB-D depth를 사용한 pose solver"`라고 단정하는 것도, 반대로 `"일반 RGB webcam만 사용"`이라고 단정하는 것도 부정확하다. 가장 안전한 표기는:

```text
RealSense D435i hardware, RGB/keypoint + calibrated projective geometry in the stated pose equations;
direct use of the depth map is not described.
```

이다.

## 18.3 StarNet backbone

[확인] 기본 YOLOv8s-pose backbone을 StarNet으로 재구성한다.

[확인] 목적은 pallet/cargo의 세부 feature를 유지하면서 parameter와 compute를 줄여 resource-constrained deployment에 맞추는 것이다.

[확인] ablation에서 StarNet 단독은 baseline 대비:

- mAP 90.9 → 91.8
- AP-KP 87.6 → 88.1
- Size 35.41M → 25.43M
- FLOPs 29.8G → 17.4G
- FPS 84.8 → 100.4

를 보고한다.

## 18.4 FFDPN — Feature Focused Diffusion Pyramid Network

[확인] YOLOv8s-pose의 PAFPN neck을 FFDPN으로 재구성한다.

[확인] 설계 목적은 세 가지다.

1. **Feature focusing:** 각 scale에서 유용한 feature를 더 강조.
2. **Diffusion connection:** 서로 다른 scale/layer 사이에 context를 넓게 전달.
3. **Lightweight design:** 계산량과 storage를 억제.

[확인] P3/P4/P5 세 scale을 FocusFeature가 받아 channel concat 후, 서로 다른 kernel의 depthwise convolution과 1×1 pointwise convolution을 거쳐 multi-scale feature를 섞고 residual connection을 사용한다.

[확인] FFDPN 단독 ablation:

- mAP 92.4%
- AP-KP 89.4%
- Size 32.27M
- FLOPs 31.1G
- FPS 86.2

## 18.5 BDEM — Boundary Detail Extraction Module

[확인] pallet keypoint가 대부분 E-section edge/corner에 있다는 domain prior를 이용한다.

[확인] BDEM은 Sobel horizontal/vertical edge operator를 feature branch로 사용하고 convolutional spatial branch와 channel-wise concat한 뒤 residual하게 합친다.

[확인] Sobel kernel은 channel-independent group convolution 형태로 적용되어 cross-channel mixing 없이 edge feature를 추출한다.

[확인] BDEM 단독은 baseline 대비 mAP을 90.9→93.1, AP-KP를 87.6→90.7로 올린다.

[추정] P11의 BDEM은 후속 P5의 CSP_DEFE와 역할상 직접적인 조상이다. P5에서는 단순 Sobel/spatial edge 결합을 넘어 **Scharr + FFT/IFFT dual-domain**으로 확장된다.

## 18.6 APT-TAL — Adaptive Power Transformation Task-Aligned Labelling

[확인] 원래 TAL은 classification score와 IoU를 결합해 positive sample을 고른다.

[확인] P11은 dense stacking에서 high-overlap prediction들을 더 강하게 구분하기 위해 IoU를 power-transform한다.

```text
IoU < 0.5   → IoU^2       (low-quality match를 더 약화)
IoU >= 0.5  → sqrt(IoU)   (high-quality match를 더 강화)
```

[확인] transformed IoU를 prediction score와 결합해 task-alignment metric을 계산한다.

[확인] threshold sweep 결과 0.5를 선택했다고 설명한다.

[확인] APT-TAL 단독 ablation은 mAP 93.5%를 기록한다.

## 18.7 BA-Wing — Boundary-Aware Wing Loss

[확인] 기본 YOLOv8s-pose의 OKS keypoint loss 대신 boundary-aware Wing loss를 사용한다.

[확인] keypoint가 pallet boundary에 가까울수록 더 큰 weight를 주는 구조이며, boundary distance `d_i` 기반 weight와 Wing loss를 곱한다.

개념적으로:

```text
w_i = 1 + alpha * exp(-d_i^2 / (2 sigma^2))

BA-Wing = sum_i w_i * Wing(x_i)
```

[확인] BDEM이 추출한 boundary feature도 keypoint prediction에 결합한다고 설명한다.

[확인] BA-Wing 단독은 AP-KP를 baseline 87.6 → 91.7로 높인다.

## 18.8 Geometric distance / inclination estimation

[확인] 12 keypoints 중 pallet outline의 geometric constraint를 만족하는 point pairs를 선택한다.

[확인] camera intrinsics `(fx, fy, cx, cy)`와 known pallet dimensions `H, L`을 사용해 projected vertical edge length와 horizontal relationships로 depth-like quantity `Z`, inclination `theta`, horizontal distance `D`를 계산한다.

[확인] 이 stage는 generic PnP/EPnP가 아니라 논문이 직접 유도한 pallet-specific projective equations다.

## 18.9 LMedS refinement

[확인] raw point-pair 계산은 소수 keypoint error에 민감하기 때문에 Least Median of Squares (LMedS)를 사용한다.

[확인] 알고리즘은 **20 sets of point pairs**를 random selection하고, 각 set에서 distance와 inclination을 계산한다.

[확인] 각 result와 나머지 result 사이 residual median을 계산한 뒤, median residual이 가장 작은 parameter estimate를 선택한다.

[확인] threshold는 다음처럼 제시된다.

```text
angle difference d  <= 2°
distance difference dD <= 0.15 m
```

[추정] 후속 P5에서는 이 pallet-specific multiple-pair + LMedS solver가 **depth backprojection + weighted EPnP + topology constraints + LM**으로 대체된다.

## 18.10 Dataset / training

[확인]

- original real images: **4,125**
- augmented: **12,112**
- split: **70% train / 20% validation / 10% test**
- 별도 occluded pallet / stacked pallet validation sets
- image data described as 480×480 for dataset
- camera: Intel RealSense D435i
- COCO-pretrained YOLOv8s-pose transfer
- optimizer: SGD
- nominal training: 200 epochs
- batch: 8
- lr: 0.001
- confidence threshold: 0.2
- server: Windows 10, i5-13600KF, RTX4060Ti 16GB, PyTorch 2.2, CUDA 11.8

[확인] occluded keypoint annotation은 visible points와 fixed E-geometry로 invisible point location을 추론하여 붙인다.

### Transfer-learning comparison

[확인]

| Strategy | mAP | AP-KP | convergence | training time |
|---|---:|---:|---:|---:|
| Scratch | 85.8 | 85.4 | 300 epochs | 48 h |
| Direct migration | **94.4** | **93.2** | **130** | **20 h** |
| Hierarchical fine-tuning | 92.8 | 91.5 | 180 | 30 h |
| Progressive fine-tuning | 93.1 | 92.0 | 160 | 26 h |

[확인] final protocol은 COCO weight를 직접 fine-tune하는 direct migration이 가장 좋았다고 보고한다.

## 18.11 Final detector/keypoint result

[확인] final Model 11:

- Precision: **99.6%**
- mAP: **94.4%**
- AP-KP: **93.2%**
- Size: **25.71M parameters**
- FLOPs: **17.6G**
- FPS: **104.2**

### Runtime interpretation

[확인] 104.2 FPS는 Table 1/3에서 network architecture들끼리 비교되는 **Improved_YOLOv8s-pose detector/keypoint model FPS**다.

[확인] LMedS geometric position calculation을 포함한 end-to-end latency는 별도로 보고하지 않는다.

[추정] 따라서 `"full pallet positioning at 104.2 FPS"`라고 쓰지 않는다.

## 18.12 Occlusion / overlap

[확인] final model:

| Condition | Precision | mAP | AP-KP |
|---|---:|---:|---:|
| Occluded | 97.2 | 92.7 | 91.2 |
| Overlapping | 97.8 | 92.9 | 92.1 |

[확인] baseline YOLOv8s-pose는 occluded AP-KP 85.2, overlap 85.4였다.

## 18.13 Multi-scene evaluation

[확인] 별도 **3300 images / 11 scenes** test를 구성한다.

조건:

- indoor bright / low / normal light
- outdoor natural light
- front
- side 45°
- side 60°
- distance 1 / 2 / 3 m
- complex context

[확인] average:

- Precision 99.0%
- mAP 95.6%
- AP-KP 94.0%
- FPS 103.9

[추정] 논문은 이를 "generalisation ability"라고 부르지만, 이 3300장의 pallet identity/material/scene가 training에서 strict hold-out되었다는 membership contract는 명시하지 않는다. 따라서 **condition robustness**로 기록하고 unseen pallet generalization으로 승격하지 않는다.

## 18.14 Pose/distance result

[확인] pose experiment:

- angle: −50° to +50°, 10° intervals
- distance: 1400–2400 mm, 100 mm intervals

[확인] stage definitions:

```text
Original:
  YOLOv8s-pose keypoints + direct geometry

Improvement 1:
  Improved_YOLOv8s-pose keypoints + direct geometry

Improvement 2:
  Improved keypoints + LMedS
```

[확인] Improvement 2 angle results:

- mean error: **2.1–4.0°**
- maximum error: **3.1–4.6°**

[확인] Improvement 2 distance results:

- mean distance error: **13–29 mm**
- maximum distance error: **27–48 mm**

### Headline wording caveat

[확인] abstract는 angle error `"within 4.0°"`와 distance error `"within 29 mm"`라고 쓴다.

[확인] Table 7에는 mean angle error가 정확히 **4.0°**인 bin이 있고 max는 4.6°까지 있다.

[추정] 따라서 안전한 표기는:

```text
mean inclination error <= 4.0°
mean distance error <= 29 mm
```

이며 `"all angle errors <4°"`라고 쓰면 안 된다.

## 18.15 What it does NOT demonstrate

[확인] full 6DoF `(R,t)`를 정량적으로 보고하지 않는다.

[확인] unseen pallet material/geometry hold-out을 하지 않는다.

[확인] 104.2 FPS에 LMedS full pipeline이 포함됐다는 근거가 없다.

[확인] code/dataset은 laboratory confidentiality로 공개하지 않는다고 Data Availability에서 밝힌다.

### 판정

- `APPROXIMATE` for camera-based forward-distance / yaw-like pose variables.
- `HIGH VALUE` as the direct predecessor of P5 YOLOv11.
- `NOT_DIRECT` for full 6DoF / x-z-yaw simultaneous comparison.

---

# 19. Architecture evolution across the literature

## 19.1 Synthetic RGB keypoint lineage

```text
Knitt 2022
NDDS synthetic
→ DOPE cuboid belief maps
→ PnP
→ 6DoF
    │
    └── limitation: 10–20 cm scale error, fixed pallet CAD dependence

Mueller 2024
Unity DR
→ YOLOv8 pose
→ 4 face corners
→ PnP
→ pose
    │
    └── simpler modern detector, but single pallet / weak GT / signed error issue
```

## 19.2 Appearance-robust geometric decomposition

```text
whole-image / all-corner CNN
       ↓
load appearance changes disturb rear-corner prediction

FFS 2025
YOLOv4 detections
→ CNN only for visible front corners
→ KRR for rear geometry
→ PnP
       ↓
same pallet + unseen cargo appearance에서 accuracy 유지
```

[추정] FFS의 핵심은 "더 큰 CNN"이 아니라 **appearance-sensitive visual inference와 geometry-predictable inference를 분리**한 것이다.

## 19.3 Human-pose transfer → pallet-specific structure

```text
P11 — Improved YOLOv8s-pose
  StarNet backbone
  + FFDPN multi-scale diffusion
  + BDEM Sobel boundary features
  + APT-TAL label assignment
  + BA-Wing keypoint loss
  → 12 E-keypoints
  → pallet-specific projective geometry
  → LMedS
  → distance D + inclination theta
       ↓
P5 — Improved YOLOv11s-pose
  StarNet retained
  + CSP_DEFE (spatial edge + FFT frequency domain)
  + RTA
      ├─ DFM
      └─ TLM explicit topology
  → 12 E-keypoints
  + D435i depth backprojection
  → visibility-weighted EPnP
  → topology objective + LM
  → richer 3D pose / distance / angle
       ↓
P6 — Hyper-pose
  Hyper-YOLO / spatial-semantic hypergraph
  + explicit 12-node / 12-hyperedge topology
  + HAFB
  + EGKD
  + topology-preserving pruning
  + keypoint uncertainty covariance
  → RGB-D geometry
  → Mahalanobis + dynamic constraints + L-BFGS-B
```

[확인] P11과 P5는 저자 3명이 동일하고, **4125 original / 12112 augmented images, D435i, 동일 camera intrinsic matrix, 동일 server hardware**를 보고한다.

[확인] 다만 split은 P11이 **70/20/10**, P5가 **80/10/10**이고 training epoch도 P11 200, P5 180으로 다르다.

[추정] 이 일치성은 두 논문이 **같거나 매우 밀접하게 재사용된 base corpus와 camera calibration** 위에 있을 가능성을 강하게 시사한다. 그러나 P5가 "P11과 동일한 exact image population을 재사용했다"고 명시적으로 선언한 문장은 현재 확인하지 못했으므로 `SAME_DATASET`으로 확정하지 않는다.

[추정] 기술적으로 가장 큰 변화는 세 단계다.

1. **Feature extraction:** FFDPN+BDEM → CSP_DEFE의 spatial/frequency dual-domain feature.
2. **Geometry prior:** APT-TAL/BA-Wing의 training-time boundary emphasis → TLM/RTA의 feature-level explicit topology.
3. **Pose solver:** image geometry + LMedS → depth-assisted weighted EPnP + topological nonlinear refinement.

[추정] P6에서는 이 topology prior가 다시 **explicit hypergraph node/hyperedge representation**으로 승격되고 uncertainty까지 solver에 연결된다.

## 19.4 Synthetic→real adaptation lineage

```text
Knitt:
domain randomization only
      ↓
Machines 2025:
unpaired target-real images
→ CUT
→ real-like synthetic
→ CosyPose multi-stage refinement
```

[확인] 단, Machines는 end-to-end detection 평가가 아니라 noisy-GT-initialized refinement 평가다.

## 19.5 Monocular metric-depth lineage

```text
RGB-only keypoint/PnP
      ↓ depth ambiguity
VISAPP 2026
RGB
→ learned metric depth
→ floor geometry
→ asymmetric RGB-depth fusion
→ direct t/q regression
```

[추정] active depth sensor를 쓰지 않으면서 metric translation precision을 높이려는 가장 직접적인 최근 방향이다.

---

# 20. What the literature actually establishes

## 20.1 "Jetson에서 real-time"을 실제로 보여준 논문

[확인] **P6 Hyper-pose**: Jetson Orin Nano TensorRT network inference 72.1 FPS.

[확인] **P5 YOLOv11**: RK3568 44.1 FPS.

[추정] 두 논문 모두 **full sensor→geometry→final pose pipeline FPS**라고 확정할 근거는 부족하다. 따라서 edge model throughput 근거로는 강하지만 end-to-end system throughput 근거로는 제한적이다.

## 20.2 Full-pipeline runtime이 가장 명확한 최근 논문

[확인] **P7 VISAPP**: component breakdown + full pipeline 27.5 ms / 36.3 FPS on RTX4090.

[확인] **P3 FFS**: YOLO + WithBNet + KRR total 49.2 ms on RTX3090. PnP가 별도 runtime item으로 분리되진 않는다.

## 20.3 Unseen load appearance

[확인] **P3 FFS**가 가장 명시적으로 설계했다.

[확인] **P7 VISAPP**도 같은 high/mid/low load-similarity framework에서 FFS와 비교하며 own translation error가 3.34–3.66 cm 수준으로 안정적임을 보였다.

## 20.4 Unseen pallet material / unseen pallet geometry

[확인] 이 audit에 포함한 주요 최근 논문 가운데 **training에서 특정 pallet material/geometry를 통째로 hold out하고, unseen pallet material/geometry에서 정량 6DoF를 보고한 clean protocol은 확인하지 못했다.**

[확인] P6는 C/E 두 standard sizes를 쓰지만 둘 다 dataset splits에 들어가며, other geometries는 future work다.

[확인] P3는 same pallet + load change다.

[확인] P1/P2는 특정 Euro-pallet/CAD에 묶여 있다.

[추정] 따라서 held-out wood/plastic instance 또는 held-out geometry를 별도 test로 구성한다면, 그것은 기존 literature에서 비어 있는 **더 강한 category/instance generalization evidence**가 될 수 있다.

## 20.5 Viewpoint generalization

[확인] P7이 novel camera mounting test를 명시한다.

[확인] 그러나 load appearance도 동시에 변경되어 pure G5가 아닌 compound shift다.

[추정] camera mount만 바꾸고 pallet/load identity는 고정하는 clean G5 test가 있으면 원인 해석이 더 명확하다.

## 20.6 Occlusion

[확인] P5는 masked/overlap keypoint performance를 별도 보고한다.

[확인] P6는 0/30/50% occlusion condition과 50% pose error를 보고한다.

[확인] P8은 <50%, >50%, >70% occlusion ADD-AUC를 명시해 severity curve가 가장 분명한 편이다.

## 20.7 Synthetic-only real generalization

[확인] P2 Knitt는 real pose labels 없이 synthetic NDDS로 학습하고 real dynamic MoCap test에서 <20 cm를 보인다.

[확인] P1 Mueller도 synthetic-only train → 60 real frames test를 수행한다.

[확인] P4 Machines는 pose labels는 synthetic-only지만 CUT adaptation에 target real images를 사용한다.

[추정] 그러므로 `"synthetic-only labels"`와 `"target-free"`는 반드시 분리해야 한다.

---

# 21. Commensurability with a monocular RGB pallet-pose study

## 21.1 가장 직접적인 비교 후보

### Tier A — 가장 가까움

1. **P7 VISAPP 2026**
   - monocular RGB sensor
   - metric 3D translation + rotation
   - full pipeline runtime
   - load shift + camera-mount shift
   - direct regression

2. **P3 FFS 2025**
   - monocular RGB
   - known pallet geometry
   - pallet-specific
   - PnP final pose
   - load appearance robustness

3. **P2 Knitt 2022**
   - monocular RGB
   - synthetic-only training
   - known CAD/PnP
   - real GT pose evaluation

4. **P1 Mueller 2024**
   - monocular RGB
   - synthetic-only
   - YOLO pose + PnP
   - 단 pose error reporting quality가 약함

### Tier B — evaluation/design reference

- P5/P6: geometry-aware keypoints, occlusion, edge throughput, but active depth used.
- P8: severe occlusion evaluation, but RGB-D.
- P4: synthetic→real/refinement design, but evaluation contract differs.

### Tier C — sensor-based engineering lower/upper references

- P9 point-cloud registration
- P10 classical RGB-D template

---

# 22. Implications for our evaluation design

> 이 절은 **선행연구가 요구하는 evidence를 역으로 정리한 설계 제안**이다. 현재 repository의 pose metric gate가 열리지 않았다면 아래 pose 수치를 paper claim으로 승격하면 안 된다.

## 22.1 Minimum pose metrics

[추정] pallet alignment 목적이라면 최소:

- `MAE_x` 또는 lateral error
- `MAE_z` 또는 forward/depth error
- `MAE_yaw`
- 3D translation Euclidean error `||t_pred - t_gt||2`
- rotation geodesic error 또는 yaw-specific error

를 분리하는 것이 좋다.

이유:

- P4는 depth z가 가장 어려움을 명시.
- P7도 z-axis weighted loss를 사용.
- P5/P6도 실사용에서 distance + tilt/angle을 별도 보고.
- 단일 "translation error"만으로는 forklift lateral/forward failure mode가 가려진다.

## 22.2 Task-success metric

[추정] VISAPP의 `Et<=5cm AND Er<=3°`처럼 **동시 성공률**을 넣는 것이 평균 MAE보다 downstream 의미가 강하다.

후보 예:

```text
Success_5cm3deg
Success_xzYaw = (|x| <= τx) AND (|z| <= τz) AND (|yaw| <= τyaw)
```

threshold는 실험 결과를 보고 정하면 안 되고 forklift tolerance / protocol에서 사전 고정해야 한다.

## 22.3 Generalization split

우선순위:

1. `G2`: held-out pallet instance, same topology
2. `G3`: held-out material
3. `G1`: held-out cargo/load
4. `G5`: held-out camera mount
5. `G6`: session/day/night/indoor-outdoor
6. `G7`: distance bins
7. `G8`: occlusion bins

[추정] 특히 G2/G3는 현재 조사한 literature에서 clean evidence가 부족해 차별화 가치가 높다.

## 22.4 Runtime reporting

반드시 다음을 따로 측정한다.

```text
detector / pose-network only
geometry/PnP only
pre/post-processing
full pipeline
```

각 숫자에:

- hardware
- input resolution
- batch size
- FP32 / FP16 / INT8
- TensorRT 여부
- warmup
- N frames
- mean / median / p95 latency

를 붙인다.

[추정] Jetson Orin Nano deployment가 최종 목표라면 desktop GPU FPS보다 **Jetson full-pipeline latency**가 더 강한 claim이다.

## 22.5 Fair baseline principle

- 같은 sensor modality
- 같은 real evaluation population
- 같은 pose coordinate convention
- 같은 metric implementation
- 같은 failure handling

을 맞추지 못하면 reported-results comparison과 empirical benchmarking을 구분한다.

---

# 23. Source-internal issues and audit warnings

| Paper | Issue | 처리 |
|---|---|---|
| P1 Mueller | signed axis errors + conclusion rotation `-1.83°`; GT reliability limitation | absolute SOTA error로 사용 금지 |
| P2 Knitt | 0.17/0.21s update와 ~0.85 FPS가 서로 안 맞음 | source values 병기 |
| P3 FFS | SDG만 public pretrained, SSP/FFS는 authors' data train | architecture absolute ranking 주의 |
| P4 Machines | RPP "test only"라고 하면서 CUT target으로 RPP 350 real images 사용 | target-domain exposure 명시 |
| P4 Machines | evaluation starts from noisy GT pose | end-to-end image→pose 성능으로 인용 금지 |
| P5 YOLOv11 | abstract "detection accuracy 95.1%" vs table P99.6/mAP95.1 | mAP95.1로 명명 |
| P5 YOLOv11 | 44.1 FPS가 full geometric pipeline인지 불명확 | model inference로만 기록 |
| P6 Hyper-pose | abstract 20.2 MB vs table/body 20.2M params | 단위 불일치 명시 |
| P6 Hyper-pose | abstract detection 97.5 vs table P99.5/mAP97.6 | table metric 이름 우선 |
| P6 Hyper-pose | 72.1 FPS가 final geometry optimizer까지 포함했는지 불명확 | network inference로 기록 |
| P6 Hyper-pose | C/E both in splits; other geometries future work | unseen geometry claim 금지 |
| P7 VISAPP | novel viewpoint와 unseen load가 동시에 변함 | G5+G1 compound shift |
| P8 Vu | reported 87ms는 without ICP, refined scores는 with ICP 별도 | refined accuracy + 87ms pair 금지 |
| P10 Zhao | max distance error가 -101.1mm로 signed | magnitude ranking 주의 |
| P11 JSC | abstract의 `detection accuracy 94.4%`는 body/table에서 mAP94.4, precision99.6 | metric name을 mAP로 고정 |
| P11 JSC | D435i를 사용하지만 stated pose equations는 depth-map samples가 아니라 RGB keypoints+intrinsics+known geometry를 사용 | 단순 RGB-D pose method로 분류하지 않음 |
| P11 JSC | `pose`라고 부르지만 정량 최종 output은 distance D + inclination θ | full 6DoF 비교 금지 |
| P11 JSC | 104.2 FPS는 detector/keypoint network table의 값 | LMedS 포함 full-pipeline FPS로 쓰지 않음 |

---

# 24. Papers/materials still needed from the user

## 현재 main-paper blocker

[확인] **없음.** P11 full PDF가 추가되어 핵심 11편의 main-paper level audit은 모두 열렸다.

## 있으면 좋은 optional material

### P5/P6 separate Supporting Information

필요 이유:

- main text가 Figure/Table S-series를 광범위하게 참조.
- 현재 main text가 직접 요약한 값은 이미 반영했지만,
  SI의 모든 ablation row, hyperparameter, pruning schedule, pose-comparison table을 독립 검산할 수 있음.

### 기타

- VISAPP 2026: official SCITEPRESS full text 확보 완료.
- Vu 2024: public author full text 확보 완료.
- Shao 2023 / Zhao 2022: official open access 확보 완료.
- P11 Journal of Supercomputing: **사용자 제공 full PDF 반영 완료.**

---

# 25. One-page takeaways

[확인] **가장 직접적인 monocular RGB 최근 비교:** VISAPP 2026, FFS 2025.

[확인] **P11 YOLOv8 predecessor는 이제 full text 검증 완료:** 104.2 FPS는 model-only이며, 최종 정량 output은 full 6DoF가 아니라 distance D + inclination θ다.


[확인] **synthetic-only RGB 계보:** Knitt 2022 DOPE → Mueller 2024 YOLOv8-pose.

[확인] **강한 occlusion reference:** Vu 2024 RGB-D, YOLOv11 2025, Hyper-pose 2026.

[확인] **edge inference reference:** Hyper-pose 72.1 FPS on Jetson Orin Nano; 단 full pipeline으로 확대해석 금지.

[확인] **full pipeline runtime이 가장 투명:** VISAPP 36.3 FPS RTX4090.

[확인] **unseen load appearance를 clean하게 정의:** FFS. 하지만 unseen pallet geometry/material이 아님.

[확인] **camera mounting shift를 정량 평가:** VISAPP. 단 load shift와 confounded.

[확인] **synthetic→real adaptation:** Machines 2025. 단 real target images를 CUT이 보고, evaluation은 noisy-GT pose refinement.

[추정] **현재 literature gap 후보:** held-out pallet instance/material/geometry를 명시적으로 분리한 monocular-RGB quantitative pose generalization.

---

# 26. Bibliographic checklist

1. Mueller, H.; Kim, Y.; Gee, T.; Nejati, M.  
   *Pallet Detection And Localisation From Synthetic Data.*  
   ACRA 2024; arXiv version 2025, arXiv:2503.22965.

2. Knitt, M.; Schyga, J.; Adamanov, A.; Hinckeldeyn, J.; Kreutzfeldt, J.  
   *Estimating the Pose of a Euro Pallet with an RGB Camera based on Synthetic Training Data.*  
   Logistics Journal: Proceedings, 2022. DOI 10.2195/lj_proc_knitt_en_202211_01.

3. Kai, N.; Yoshida, H.; Shibata, T.  
   *Pallet Pose Estimation Based on Front Face Shot.*  
   IEEE Access, 2025. DOI 10.1109/ACCESS.2025.3538045.

4. ElMoaqet, H.; Rashed, M.; Bakr, M.  
   *Multi-Stage Domain-Adapted 6D Pose Estimation of Warehouse Load Carriers: A Deep Convolutional Neural Network Approach.*  
   Machines, 2025. DOI 10.3390/machines13121126.

5. Zhou, Z.; Lu, Y.; Lv, L.  
   *Unmanned Forklift Pallet Positioning Algorithm Based on an Improved Human Pose Estimation Model.*  
   Annals of the New York Academy of Sciences, 2025. DOI 10.1111/nyas.70001.

6. Ye, T.; Wang, Z.; Qin, Y.; Gao, Z.; Wang, Y.  
   *A Hypergraph Computing and Knowledge-Enhanced Framework for Forklift Pallet Pose Estimation.*  
   Annals of the New York Academy of Sciences, 2026. DOI 10.1111/nyas.70219.

7. Miura, A.; Uchiyama, H.; Yamaguchi, M.; Kai, N.; Shiroshima, T.; Saito, H.  
   *Real-Time 6DoF Pallet Pose Estimation with Monocular Metric Depth.*  
   VISAPP 2026. DOI 10.5220/0014626800004084.

8. Vu, V.-D. et al.  
   *Occlusion-Robust Pallet Pose Estimation for Warehouse Automation.*  
   IEEE Access, 2024. DOI 10.1109/ACCESS.2023.3348781.

9. Shao, Y.; Fan, Z.; Zhu, B.; Lu, J.; Lang, Y.  
   *A Point Cloud Data-Driven Pallet Pose Estimation Method Using an Active Binocular Vision Sensor.*  
   Sensors, 2023. DOI 10.3390/s23031217.

10. Zhao, J.; Li, B.; Wei, X.; Lu, H.; Lü, E.; Zhou, X.  
    *Recognition and Location Algorithm for Pallets in Warehouses Using RGB-D Sensor.*  
    Applied Sciences, 2022. DOI 10.3390/app122010331.

11. Zhou, Z.; Lu, Y.; Lv, L.  
    *Pallet Localization Algorithm Based on Improved Human Pose Estimation with Transfer Learning.*  
    The Journal of Supercomputing, 2025. DOI 10.1007/s11227-025-06973-w.  
    **Status: full local PDF verified.**

---

# 27. Recommended repository placement

```text
_docs/audits/paper/PALLET_POSE_LITERATURE_AUDIT.md
```

[추정] 기존 `EXTERNAL_BASELINE_AUDIT.md`가 같은 evaluator에서 직접 재평가한 baseline의 공정성을 다룬다면, 이 문서는 published reported-results와 literature-level comparability를 담당하도록 역할을 분리하는 것이 가장 깔끔하다.

논문 본문/최종 표로 옮길 때는 반드시 현재 `_docs/paper/final/` claim/metric lock과 대조하고, 이 literature audit의 외부 수치를 현재 연구의 자체 pose claim으로 오해시키지 않는다.
