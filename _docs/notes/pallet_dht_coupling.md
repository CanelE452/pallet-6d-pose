# Pallet DHT coupling v2

## Actual results — 2026-09-09

**12개 추가 학습·실제 평가·통계·독립 검증·시각화 전달까지 완료했다. 안정적인 종합 성능 개선은 확인되지 않았다.** [실제 HTML](../../data/pallet/results/pallet_dht_coupling_v2/index.html), [판정](../../data/pallet/results/pallet_dht_coupling_v2/VERDICT.json), [전체 통계](../../data/pallet/results/pallet_dht_coupling_v2/SUMMARY.json).


브라우저 실제 창 표시를 확인하고 Discord 완료 알림을 HTTP204로 전달했다(2026-09-09 00:06KST). [전체 완료 기록](../../data/pallet/results/pallet_dht_coupling_v2/COMPLETION.json), [실제 창 확인](../../data/pallet/results/pallet_dht_coupling_v2/GALLERY_OPEN.json). 완료는 정확도 향상이나 모든 수치 검사 통과를 뜻하지 않으며 runtime parity 실패는 유지했다.

새 네 방법×3seeds 모두55,980장×2epochs,6,998실제 updates를 완료했다. 기존 같은 seed의 초기 tensor와 전체 augmented input trace가 일치했다. 새 모델은 각각319positive+2,689negative를 실제 추론했으며 총3,828positive+32,268negative model-forwards이다. 기존9개 대조군 결과는 검증 후 재사용했다. 동일한 최종epoch EMA 평가이며 실사 기반 checkpoint/방법 선택은 없다.

아래는 각 seed의 전체 평가 지표를 구한 뒤 세 seed 평균이다. 점 지표는 IoU 매칭된 프레임의 감독점이며, 모델별 매칭 수가 달라질 수 있다.

| 방법 | 점 median px↓ | 점 P90 px↓ | 회전 °↓ | 이동 cm↓ | IoU3D↑ | ADDsym AUC↑ |
|---|---:|---:|---:|---:|---:|---:|
| 점 전용 | 6.865 | 40.395 | 2.376 | 7.731 | 0.595 | 0.424 |
| Hough 특징 | 6.773 | 36.320 | 2.434 | 7.975 | 0.597 | 0.428 |
| 기존 Hough joint | 6.689 | 38.690 | 2.319 | 7.800 | 0.595 | 0.424 |
| 손실 비중 조정 | 6.896 | 47.801 | 2.385 | 7.783 | 0.583 | 0.419 |
| PCGrad | 6.761 | 40.178 | 2.343 | 7.743 | 0.598 | 0.426 |
| 비중 조정 + PCGrad | 6.720 | 42.920 | 2.378 | 7.529 | 0.593 | 0.426 |
| 점·선 incidence | 6.883 | 42.420 | 2.386 | 7.366 | 0.588 | 0.416 |

사전 지정48개 비교(4방법×2기준×6지표)의 Bonferroni session bootstrap 구간은 모두0을 포함했다. 보정 전95%에서도 유의한 개선 구간은 없었다. 이는 이번3seed/재사용 DEV13세션/2epoch 조건의 결과이며, 모든 DHT 결합 구조의 불가능성을 뜻하지 않는다. 공통 관측 프레임의 paired bootstrap 차이와 위 전체 매칭 모집단 평균은 서로 다른 통계다. 모든6D평가 coverage는319/319이나, PCGrad의2D매칭은[306,308,307], incidence는[308,306,302]로 point전용[307,308,310] 대비 모든seed에서 보존되지 않았다.

실제로 확인한 경사 문제는 후반의 비중 변화와 전역·국소 방향 차이다. 고정선계수 PCGrad의 line/stock 전역 norm비 중앙값은 epoch2에서6.40–7.20배 커졌다. 균형을 맞춘 PCGrad에서는0.827–0.882배였지만 이것이 정확도 향상으로 이어지지는 않았다. 실제 PCGrad 적용은6,653/20,994회(31.69%), 균형+PCGrad는6,276/20,994회(29.89%)였다. Hough 내부와 전역의 부호가 다른 사례가 있으며, 국소 진단은 군별24고정지점(유효Hough cosine21)뿐이다. Main incidence 자체의 별도 gradient norm은 미측정이다. [세 seed 경사 해석](../../data/pallet/results/pallet_dht_coupling_v2/provenance/three_seed_gradient_summary/INTERPRETATION.md).

선은 같은 GT 점에서 만들어져도 긴 경계를 모으는 표현과 기하 제약을 제공한다. 다만 무한선 하나는 끝점의 접선방향 위치와 순서를 고정하지 못하고, 함께 잘못된 점·선은 서로 일치할 수 있다. PCGrad는 이런 의미·가시성·끝점 정보를 추가하지 않는다. 이것은 구조의 수학적 한계이며 관측된 모든 오차의 원인을 증명한 것은 아니다. [분석과 해석 예제](../../data/pallet/results/pallet_dht_coupling_v2/provenance/architecture_limits/ARCHITECTURE_LIMITS.md).

구도별로도 혼합 효과다. 낮은 앙각(<5°,55장)에서 PCGrad 점P90은55.94px(점전용67.70px)이지만 이동오차는19.78cm(18.40cm)였다. 정면에 가까운 면적비 그룹(≤0.1,144장)은 incidence 이동7.21cm(점전용7.67cm)이나 점P90은26.52px(25.61px), 매칭수[137,136,133]([138,138,140])였다. 이는 사전지정 proxy 구도의 기술 통계이며 보정된 실제yaw/가시성 또는 소그룹 개선 검정이 아니다.

시간은21모델×26고정프레임×3반복=1,638회 수집했다. Seed별 median의 평균은 점전용8.758ms, Hough joint10.670ms, 새 방법10.475–10.788ms이다(이미지 decoding·PnP 제외). 엄격한atol1e-4/rtol0 parity는7/1,638회 실패했다. 최대 점차0.0061035px/box차0.0013428px이며 기준은 변경하지 않았다. RUNTIME.PASS=false를 유지하고 시간 수집 완료와 분리했다. [원기록](../../data/pallet/results/pallet_dht_coupling_v2/RUNTIME.json).

독립 감사는21모델 저장 결과·원좌표 overlay6,699행·48비교 판정·대표100k bootstrap을 검증했다. 실제 시각 QA는21모델×319프레임,252개 Hough 증거,깨진 이미지0을 확인했다. 후속 작업에는 GT 역할/끝점·가시성의 식별 가능성, 국소 Hough에서 점 위치 task와 선 task의 결합을 별도 검증할 필요가 있다. 이번 결과만으로 계수나 구조를 실사에서 선택하지 않는다.

### 원본 실패 사례의 추가 검토

`eval_pallet07:1778652166837872128`의21개 저장 예측은 모두 원래 번호 기준8점(7코너+중심) median232.93–273.87px로 큰 오류가 남았다. 동일한 GT 진단 순열로만 재대응하면11.40–59.85px여서 번호 대응 오류의 비중이 컸다. 이 순열은 실제 운영 교정 규칙이 아니고 GT를 바꾸지 않는다.

더 구체적으로 역할7(GT4–7)에서 joint·balanced·PCGrad·balanced-PCGrad의12개 출력 모두 잘못된 원래 P4–P7보다 GT pair를 더 지지했다. Joint seed1의 peak 선은 GT 두 endpoint까지 평균1.68px인데 잘못된 예측 endpoint까지277.64px였다. 따라서 이 사례를 모두 ‘점과 선이 함께 같은 곳으로 틀렸다’고 설명할 수 없다. 일부 정확한 선 증거를 해당 semantic ID의 점 배치로 연결하지 못한 문제가 남았다.

Incidence의 세 seed는 이 역할 peak의 방향오차가79.30–81.30°이고, GT pair cost1.50–1.68과 잘못된 예측 pair cost1.57–1.64가 비슷했다. 이는 단일역할의 실제 관측으로, 무한선 mixture 일치가 정답 semantic edge를 보장하지 못하는 예시다. 최종 검출점으로 사후 계산했으므로 실제 TAL-positive 학습 loss 재현이나 모든 역할의 원인 증명으로 해석하지 않는다. [독립 사례 검토](../../data/pallet/results/pallet_dht_coupling_v2/provenance/problem_case_final_review/REVIEW.md), [전체 수치](../../data/pallet/results/pallet_dht_coupling_v2/provenance/problem_case_final_review/REVIEW.json).

## Proposal — 2026-09-08, before new training

The user authorizes actual gradient diagnosis and matched experiments for loss balance, PCGrad and explicit point–line relations, while keeping synthetic-only supervision and the single full-network DHT + point architecture. No pseudo labels, filter/self-training, test-time snapping or real checkpoint selection. This follows completed `pallet_dht_joint_v1`; that study and all original sources/results remain unchanged.

The new output family is `data/pallet/results/pallet_dht_coupling_v2/`, with code under `scripts/research/pallet_dht_coupling_v2/`. A separate family is needed because loss/optimizer treatment and the experimental questions change. The primary consumer is the user's architecture decision and an interactive actual-result report.

Same R0, synthetic train55,980/val4,020, identical per-seed augmented batches, FP32 batch16, AdamW1e-4 and two full continuation epochs (6,998 updates) as v1. Reuse v1's nine completed point-only / Hough-features / Hough-joint controls only after exact data, protocol, initial-state, batch-trace, checkpoint and saved-evaluation binding checks. Newly trained cells must infer all319positive+2689negative images, with the unchanged original-coordinate canonical2D/MAIN6D pipeline. All real data are reused13-session DEV; FINAL remains untouched.

New arms, each seeds1/2/3: (1) balanced auxiliary coefficient0.1×one2many_coefficient/0.8; (2) fixed0.1 auxiliary with symmetric two-task PCGrad; (3) both balance and PCGrad; (4) fixed0.1 auxiliary plus an explicit point–line incidence loss. The fourth arm's exact distribution/assignment normalization and coefficient will be fixed after geometry tests and synthetic-only plumbing, before main training or new real predictions. No latent inference architecture, input geometry, point numbering or postprocessing change is planned.

PCGrad tasks are the original combined stock box/class/pose/RLE objective and the weighted line objective. Project only parameters structurally shared by both tasks; preserve task-private gradients. Sum the projected task gradients so non-conflicting updates equal the original summed objective. Also diagnose isolated point-location+RLE versus line gradients, and keep this distinction explicit. Fixed actual training steps2,16,256,1750,3499,3500,5249,6998 will record cosine, norms and per-group statistics without extra BN updates or altered augmentation/RNG. A separately declared GPU batch16 synthetic diagnostic will replace neither the unexecuted v1 CPU diagnostic nor its resource guard.

Compare all four new arms against the matched-budget point-only and original joint references, without selecting a real-data winner. Six metrics: pooled point median/P90 and MAIN rotation/translation/IoU3D/ADDsymAUC. Report seed means/sampleSD, full-population matching/pose coverage, actual negative FP/AP, original user failure case and predeclared view cohorts. Use100,000 paired-session bootstrap draws and Bonferroni intervals across the48 registered arm/reference/metric comparisons; ordinary95% intervals are descriptive. Overall superiority requires all six directions and family-adjusted intervals plus per-seed coverage preservation. Thirteen reused sessions and only three training seeds limit generalization regardless of significance.

Failure hypotheses remain separate: sparse forward/backprojection holes were already fixed; gradient conflict is unmeasured; amodal infinite-line targets lack physical visibility and endpoint extent; global pooling can retain background/distractor evidence; symmetry/occlusion can make corner identity ambiguous. Nonzero gradients do not prove conflict, and a point–line consistency loss can be satisfied by jointly wrong outputs. Geometry/mathematics, shared-mask/private-gradient invariance, zero-conflict optimizer equivalence, real batch16 resource/finite-update checks and exact augmentation tracing must pass before main. Failed numerical/protocol checks are archived and repaired before continuing, never silently relabeled.

Completion means actual diagnosis, all registered training/evaluations, independent metric/visual checks, an HTML window visibly opened and the already-authorized Discord notification. Completion is not a guarantee of improved accuracy. No commit or changes to unrelated workspace work.

## Synthetic preflight observations

The separate GPU diagnosis completed three disposable raw-final-checkpoint forwards on the same fixed16 synthetic images; it performed zero optimizer steps and changed no parameters. Full-stock versus weighted-line cosine over the shared intersection was +0.04576, −0.00381, +0.02894. The corresponding Hough-only seed2 cosine was −0.27908 (isolated point-location+RLE versus line: −0.42834), so weak global conflict does not mean every layer agrees. Weighted-line/full-stock gradient norm ratios were0.0497/0.0533/0.0389 globally but1.93/1.11/1.76 inside Hough. These are one-batch observations, not population prevalence or a causal accuracy explanation. Evidence is new `GRADIENT_DIAGNOSIS.json` and its separately frozen GPU protocol; the earlier unexecuted CPU guard remains unchanged.

Incidence uses the actual stock positive assignment and decoded predicted endpoints, with the same predicted line hypothesis supporting both endpoints. It marginalizes over the full image-predicted Hough score distribution using a stable log mixture of pseudo-Huber residual costs. GT determines existing instance assignment and supported roles, without restricting the predicted mixture to a GT line. Endpoint collapse cannot evade the loss. All60,000 source samples are singleton objects; physical per-edge occlusion is not labeled. The coefficient remains0.1 pending synthetic scale/resource checks and main protocol freeze. Seventeen geometry tests and13 trainer/surgery CPU tests passed, including nonzero-residual neural-forward equality with v1.

PCGrad task-private gradient preservation is a statement about the projection stage, before original global clipping. Changed global norm clipping can scale private gradients differently afterward; final private parameter updates are not claimed bit-identical. Separate same-graph CUDA backward calls are also not assumed bit-identical: synthetic B16 diagnostics record reconstruction error, task cosine and projection-decision agreement explicitly.


## Read-only checks during main training

The first balanced seed1 completed all6,998 updates with the exact old seed1 full augmented trace and eight planned probes. PCGrad seed1 epoch1 applied projection in1,140/3,499 actual updates; this is an observed early-epoch frequency for one run, not a population claim or evidence of real accuracy improvement.

A separate CPU same-forward-graph check compared the original JointCriterion against the new balanced epoch1 criterion at nonzero residual weights. Seven original loss components and all518 parameter gradients/None masks were bit-exact. The final scalar sum differed by one FP32 ULP (7.629e−6), reproduced solely by appending the zero incidence component before sum; double-precision sums agreed. This validates the checked objective/gradient implementation, not numerical identity of complete CUDA trajectories. Evidence: `provenance/criterion_epoch1_equivalence/RESULTS.json` and `SUM_REDUCTION.json`. No GPU, optimizer, main source or protocol change was made.

## Seed 1 실제 경사 기록 요약

완료된 네 seed1 모델의 저장 경사 기록을 비교했다. 전체 step 기록이 있는 PCGrad의 선/stock 전역 norm 비 중앙값은 epoch1 0.00621에서 epoch2 0.04470으로 약7.2배 증가했다. balanced-PCGrad는0.00642→0.00566이었다. 실제 투영은각각2316/6998,2230/6998회였다. 이는 경사 규모에 대한 관측이며 실사 정확도 개선이나 인과 증거가 아니다. Local Hough 진단은8개 고정 probe뿐이고 전체 충돌률로 해석하지 않는다. 근거: [해석](../../data/pallet/results/pallet_dht_coupling_v2/provenance/seed1_gradient_summary/INTERPRETATION.md), [SHA가 포함된 JSON](../../data/pallet/results/pallet_dht_coupling_v2/provenance/seed1_gradient_summary/SUMMARY.json). Main에는 incidence 자체의 분리 경사가 없으므로 line/stock 비를 incidence/stock 비로 읽지 않는다.
