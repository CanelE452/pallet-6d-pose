# YOLO·DOPE·ResNet-18 N3 방법 일치표

세 경로는 하나의 가중치를 공유하지 않는다. 각 frozen base에 같은 **N3 절차와 감독**을 적용해 별도 헤드를 학습하며, 이 표는 그 방법 계약이 같은지를 확인한다.

| 항목 | YOLO | DOPE | ResNet-18 CONSTANT-fold |
|---|---|---|---|
| base neural input | RGB only | RGB only | RGB only |
| N3 feature taps `(C,stride)` | `(64,8)`, `(128,16)` | `(256,4)`, `(128,8)` | `(128,8)`, `(256,16)` |
| base selection/decoder | fixed detected instance | corner heatmap, affinity unused | fixed DSNT 9-point decoder |
| N3 input | features + initial points + box + valid + `[W,D,H]` | same | same |
| corrected points | corners 0–7 | corners 0–7 | corners 0–7 |
| preserved | center8, box, score, instance/order, missing mask | same | same |
| candidates | 13 directions × 17 radii + null = 222 | same | same |
| maximum candidate radius | `0.08 × predicted-box diagonal` in network coordinates | same | same |
| visual stencil | 4×8 samples; `stencil_fraction=0.1310373991727829` | same | same |
| dimension feature | five normalized log features from canonical metres | same | same |
| metadata path | `5→16→16`, then role8 + displacement2 + null1, `27→32→1` | same | same |
| initialization | metadata final layer zero; additive logits | same | same |
| symmetry supervision | frozen initial prediction selects one approved whole-object permutation, identity-first tie | same | same |
| target/loss | Gaussian, sigma=`0.08×diag/17`, corner CE then frame mean | same | same |
| inference decode | softmax expectation, lambda=1 | same | same |
| original-coordinate cap | circular `1% × original image diagonal` | same, full anisotropic inverse | same, full anisotropic inverse |
| PnP | base와 N3 모두 동일 K·치수 | same | same |
| N3 training | existing 3×6,000 | new 3×6,000 | new 3×6,000 |
| temperature | synthetic calibration only | synthetic calibration only | synthetic calibration only |

DOPE와 ResNet-18의 기존 P/D 실험은 이 표의 N3 증거로 사용하지 않는다. 새 fit receipt에는 `dimension_input_to_N3=true`, `symmetry_supervision=true`, `base_receives_dimensions=false`가 모두 있어야 한다.

좌표 역변환은 모든 기반에서 `delta_original = inverse(A_b) delta_net`을 적용하며 패딩 이동을 변위에 더하지 않는다. YOLO의 등방 변환과 달리 DOPE·ResNet-18은 두 축 gain을 각각 사용한다. cap은 역변환 뒤 원본 영상에서 적용한다.

이 실험이 지지할 수 있는 표현은 “동일 N3 절차를 세 frozen estimator에 적용해 각 기반의 보정 전후를 비교했다”이다. 서로 다른 base와 별도 N3 head를 사용하므로 무학습 plug-and-play 전이나 모든 backbone에 대한 보편적 robust를 뜻하지 않는다.

