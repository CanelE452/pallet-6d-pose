# Development results — not independent confirmation

T2a. Original-pixel point errors. Pooled9-point median is computed per seed, then seed means; no ensemble. All-GT PCK uses2818 supervised points, including failures. Each raw model has its own row.

| Model | Median px | P90 px | Frame mean px | Gross20 % | PCK5 % | PCK10 % | PCK20 % | Matched / GT points |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 | 6.6157 | 38.6700 | 21.6734 | 17.199 | 37.189 | 63.733 | 80.979 | 311/319; 2756/2818 |
| P1 | 5.8509 | 38.0254 | 20.9305 | 15.566 | 42.264 | 67.637 | 82.576 | 311/319; 2756/2818 |
| P2 | 5.9427 | 37.7279 | 21.0096 | 15.893 | 42.158 | 67.069 | 82.257 | 311/319; 2756/2818 |
| P3 | 5.9211 | 37.5460 | 20.9486 | 15.421 | 41.661 | 67.317 | 82.718 | 311/319; 2756/2818 |
| L1 | 6.0518 | 37.2041 | 21.1446 | 16.074 | 41.235 | 66.572 | 82.079 | 311/319; 2756/2818 |
| L2 | 6.0373 | 37.3017 | 21.0853 | 15.929 | 41.057 | 66.927 | 82.221 | 311/319; 2756/2818 |
| L3 | 6.1207 | 37.4530 | 21.1644 | 16.001 | 41.022 | 66.856 | 82.150 | 311/319; 2756/2818 |
| D1 | 6.3665 | 37.4788 | 21.4105 | 16.110 | 39.141 | 64.656 | 82.044 | 311/319; 2756/2818 |
| D2 | 6.4571 | 37.8098 | 21.4561 | 16.437 | 38.751 | 64.620 | 81.725 | 311/319; 2756/2818 |
| D3 | 6.4672 | 37.4101 | 21.4353 | 16.219 | 38.964 | 64.443 | 81.938 | 311/319; 2756/2818 |
| P seed mean | 5.9049 | 37.7664 | 20.9629 | 15.627 | 42.027 | 67.341 | 82.517 | same fixed population |
| L seed mean | 6.0699 | 37.3196 | 21.1315 | 16.001 | 41.105 | 66.785 | 82.150 | same fixed population |
| D seed mean | 6.4303 | 37.5662 | 21.4340 | 16.255 | 38.952 | 64.573 | 81.902 | same fixed population |

T2b. Canonical MAIN pose against geometry-reconstructed reference; no external6D metrology claim.

| Model | Rotation deg | Yaw deg | Translation cm | IoU3D | ADDsym AUC | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 2.2625 | 1.2306 | 7.8969 | 0.60318 | 0.42847 | 319/319 |
| P1 | 2.0892 | 1.0738 | 6.8647 | 0.64216 | 0.45350 | 319/319 |
| P2 | 2.1059 | 1.1361 | 6.9167 | 0.63735 | 0.45606 | 319/319 |
| P3 | 2.0731 | 1.1557 | 7.6775 | 0.63715 | 0.45405 | 319/319 |
| L1 | 2.1100 | 1.1147 | 7.6092 | 0.63743 | 0.44997 | 319/319 |
| L2 | 2.1031 | 1.1221 | 7.3644 | 0.63607 | 0.45188 | 319/319 |
| L3 | 2.1358 | 1.1316 | 7.6719 | 0.62499 | 0.44504 | 319/319 |
| D1 | 2.2005 | 1.1915 | 7.5697 | 0.62107 | 0.44069 | 319/319 |
| D2 | 2.1808 | 1.1821 | 7.5400 | 0.61891 | 0.44167 | 319/319 |
| D3 | 2.1882 | 1.1920 | 7.6817 | 0.61603 | 0.43976 | 319/319 |

T3. Shared paired draws across model seeds,13 observed sessions,10,000 resamples,seed20260914. Negative favors P. Primary P−R0 is the only designated primary; P−D and secondary outputs are exploratory. Intervals do not establish independent generalization.

| Comparison | Delta median px | Session95% interval px | Frame95% interval px |
|---|---:|---|---|
| P−R0 | -0.71077 | [-1.17374, -0.41108] | [-0.98144, -0.52944] |
| P−D | -0.52536 | [-0.98002, -0.21012] | [-0.70441, -0.36542] |

P−R0 translation median reduction averaged over seeds is7.439mm. This is a difference of geometry-reference error statistics, not directly measured insertion improvement and not a pixel-to-mm conversion. Per-session and leave-one-session-out results are in the experiment CSV/JSON. Tail and pose intervals are in EXPLORATORY_PAIRED.json.

P has a lower development median than this fixed direct control. P−D P90 difference is 0.200px (session95% [-1.855,1.247]); positive means P has the larger P90. The observed D runtime is lower; neither all-metric dominance nor equivalence is claimed. D's fixed train/cal probe curves and complete update traces must accompany that claim. No saturation or global convergence guarantee follows from6,000 updates. Shared evidence and+2.57% parameters do not match FLOPs, output support or supervision. D regresses unbounded residuals with L1; P uses candidate-distribution supervision and expectation.

T4. RTX3080 desktop.26 prespecified images,20 warmups/model,5 balanced order blocks, batch1 and4 CPU threads. Decoding/model loading excluded. All samples and allocated/incremental memory are retained. Display/RustDesk stayed active.

| Model | Image→2D median/mean/P90 ms | +PnP median/mean/P90 ms | Paired added2D median ms | Refiner parameters | Single-model peak allocated MiB |
|---|---|---|---:|---:|---:|
| R0 | 9.406/9.593/11.033 | 10.800/11.035/12.559 | 0.000 | 0 | 190.09 |
| P1 | 13.529/13.729/15.955 | 15.047/15.216/17.314 | 4.029 | 18962 | 190.17 |
| P2 | 13.716/13.889/16.036 | 15.506/15.432/17.459 | 4.286 | 18962 | 190.17 |
| P3 | 13.540/13.792/15.807 | 15.251/15.291/17.213 | 4.291 | 18962 | 190.17 |
| L1 | 17.872/17.736/19.444 | 19.333/19.193/20.854 | 8.076 | 19810 | 190.17 |
| L2 | 17.767/17.619/19.311 | 19.145/19.079/20.853 | 8.020 | 19810 | 190.17 |
| L3 | 17.826/17.652/19.570 | 19.294/19.101/20.880 | 8.118 | 19810 | 190.17 |
| D1 | 12.777/12.945/14.736 | 14.485/14.386/16.255 | 3.430 | 19450 | 190.17 |
| D2 | 12.554/13.038/14.950 | 14.252/14.479/16.247 | 3.214 | 19450 | 190.17 |
| D3 | 12.686/12.967/14.865 | 14.224/14.422/16.225 | 3.546 | 19450 | 190.17 |

Feature sampling is included, not a cached-feature deployment benchmark. End-to-end adds the canonical prediction-only selector and selected-pose solver. Embedded export/Jetson performance was not measured. No fastest repeat was selected.

T5. Independent confirmation: NOT_YET_MEASURED. Formal prior-method performance: NOT_YET_MEASURED. Neither missing value is represented by zero. See the separate collection and prior protocols.
