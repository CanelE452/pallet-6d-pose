# Result Table Templates

아래 표는 **값을 비워둔 논문용 템플릿**이다. final test가 동결되고 평가가 완료되기 전에는 숫자를 채우지 않는다.

## Table 1. Main comparison on the frozen in-house final test

| Method | Training data | Target-specific training? | CAD at inference? | Real supervision? | Box AP50:95 ↑ | ADD(-S) AUC ↑ | Rotation med. (°) ↓ | Translation med. (cm) ↓ | Yaw med. (°) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SingleShotPose |  |  |  |  |  |  |  |  |  |
| DOPE |  |  |  |  |  |  |  |  |  |
| PVNet |  |  |  |  |  |  |  |  |  |
| YOLO26n-Pose (G38) | G38 generic synthetic | No | No | No |  |  |  |  |  |
| Proposed method |  |  |  |  |  |  |  |  |  |
| Real-FT upper bound |  | Yes | No | Yes |  |  |  |  |  |

> MegaPose-RGB를 넣을 경우 CAD-at-inference / bbox-input 계약이 다르므로 별도 reference block으로 분리한다.

---

## Table 2. Synthetic data ablation

| Arm | Generic unique data | Distribution change | Extra exposure/repeat | Additional support data | Box AP50:95 ↑ | ADD(-S) AUC ↑ | Rotation med. ↓ | Translation med. ↓ | Yaw med. ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A42 legacy 10K |  |  |  |  |  |  |  |  |  |
| C43 broad 10K |  |  |  |  |  |  |  |  |  |
| G38 generic 38K |  |  |  |  |  |  |  |  |  |
| G38EXP exposure control |  |  |  |  |  |  |  |  |  |
| Support-data arm (if retained) |  |  |  |  |  |  |  |  |  |

---

## Table 3. Self-training / adaptation ablation

| Method | Pseudo-label source | Confidence filtering | Geometry/reprojection filter | Consistency filter | Real labels used? | Box AP50:95 ↑ | ADD(-S) AUC ↑ | Rotation med. ↓ | Translation med. ↓ | Yaw med. ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic-only | None | No | No | No | No |  |  |  |  |  |
| Naive self-training | Model prediction | Yes | No | No | No |  |  |  |  |  |
| Confidence-only | Model prediction | Yes | No | No | No |  |  |  |  |  |
| Geometry-aware | Model prediction | Yes | Yes | No | No |  |  |  |  |  |
| Full proposed | Model prediction | Yes | Yes | Yes | No |  |  |  |  |  |
| Real-FT upper bound | Ground truth | — | — | — | Yes |  |  |  |  |  |

---

## Table 4. Robustness by domain

| Method | Domain | N | Box AP50:95 ↑ | ADD(-S) AUC ↑ | Rotation med. ↓ | Translation med. ↓ | Yaw med. ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-Pose (G38) | DAY |  |  |  |  |  |  |
| YOLO26n-Pose (G38) | NIGHT |  |  |  |  |  |  |
| Proposed method | DAY |  |  |  |  |  |  |
| Proposed method | NIGHT |  |  |  |  |  |  |

---

## Table 5. In-house final-test dataset composition

| Split / condition | Frames | Independent capture sessions | DAY | NIGHT | Occlusion annotated? | Truncation annotated? | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Final test total |  |  |  |  |  |  |  |
| DAY |  |  |  |  |  |  |  |
| NIGHT |  |  |  |  |  |  |  |
| Low elevation / edge-on |  |  |  |  |  |  |  |
| Mid/high elevation |  |  |  |  |  |  |  |
| Small projected pallet |  |  |  |  |  |  |  |
| Medium projected pallet |  |  |  |  |  |  |  |
| Large / close pallet |  |  |  |  |  |  |  |
| Occluded |  |  |  |  |  |  |  |
| Truncated |  |  |  |  |  |  |  |

---

## Table 6. Annotation reliability

| Reliability check | N | Median | P90 | Unit / interpretation |
|---|---:|---:|---:|---|
| Keypoint disagreement (NME) |  |  |  | normalized image/object scale |
| Rotation disagreement |  |  |  | degrees |
| Translation disagreement |  |  |  | cm |
| Yaw disagreement |  |  |  | degrees |

> 이 표는 모델 성능이 아니라 GT annotation noise floor를 나타낸다.

---

## Table 7. Failed negative-data ablation (appendix / failure analysis only)

| Arm | Negative selection | Negative count | Loss | Positive recall / AP | Negative rejection metric | Verdict |
|---|---|---:|---|---:|---:|---|
| Y0E exposure control | None (positive repeat) | 0 | Stock |  |  |  |
| YN random negative | Random synthetic negative | 9,000 | Stock |  |  | Failed / operating-point shift |
| HM hard-mined negative | Model-mined hard negative | 1,900 | Stock |  |  |  |
| HF hard-mined + focal | Model-mined hard negative | 1,900 | Focal-negative |  |  |  |

이 표는 main result가 아니라 negative-data 축이 왜 최종 방법에서 제외됐는지를 설명하는 appendix용이다.
