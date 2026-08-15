# 모델 카탈로그

학습된 DOPE 모델 목록. 각 모델의 학습 설정, 데이터, 평가 결과 기록.

> ⚠️ **트랙/convention 안내 (2026-06-04)**: 프로젝트는 **논문용(`paper_*`)** / **과제용(`challenge*`, `dope_cropaug_ft*`)** 2 트랙. 현행 convention = **camera-facing 0123**.
> 아래 `mixed_v*` / `v8_ablation` / `selftrain_r1` 및 "Active 모델 요약"·"v8 Ablation"·평가표는 전부 **폐기된 v8(object-frame) 자산** — 새 작업에 사용 금지, history 참고용으로만 보존.
> 현행 논문용 모델 = **`paper_base`** (아래 `paper_base.md`). 자세히는 CLAUDE.md "핵심 방향" + memory 3종.

## 문서 구조

```
파일                     내용
──────────────────────────────────────────────────────────────────────
paper_base.md            ★ 논문용 base (camera-facing, 합성+squash+truncation, v1/v2 제외) — 명세+로드맵
dope_architecture.md     DOPE 모델 구조, 학습 설정, annotation 형식, 평가 메트릭
training_loss.md         Loss 함수 상세 (기본 MSE + Geometric Loss + Symmetric Loss)
── 이하 폐기 v8(object-frame) 자산 (history 참고용) ──
mixed_v1.md              baseline (Isaac 4K + Blender 4K)
mixed_v2.md              mixed_v1 + blender_manydir 2K (개선 없음)
mixed_v3.md              geo loss 최초 적용 (cuboid 형태 개선, 감지율 하락)
mixed_v6_full.md         dark + view 데이터 추가, augmentation 조정 (PCK 최고)
mixed_v8.md              test_blender 데이터, Real PnP 최고, 8장 self-training
v8_ablation.md           Structural/Reliability loss ablation (A/B/C/D/E) — coord(A) B∧C 최고, rel(E) PnP/B 최고
selftrain_r1.md          Self-training Round 1 (pseudo-label 751장)
archive.md               초기 실험 모델 (pallet_category, pallet_v11, blender_v1, combined_v1 등)
```

## Active 모델 요약

```
모델               Weight 경로                                      Epochs   학습 데이터                              이미지 수    초기 weight       특수 loss
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1           weights/mixed_v1/final_net_epoch_0060.pth         60      Isaac 4K + Blender 4K                     8,000      scratch           MSE only
mixed_v2           weights/mixed_v2/final_net_epoch_0091.pth         91      mixed_v1 8K + manydir 2K                 10,000      mixed_v1 ep60     MSE only
mixed_v3           weights/mixed_v3/final_net_epoch_0091.pth         91      mixed_v2_train 10K                       10,000      mixed_v1 ep60     geo loss
mixed_v4_aug       weights/mixed_v4_aug/final_net_epoch_0060.pth     60      mixed_v1_train 8K (강한 aug)              8,000      scratch           MSE only
mixed_v6_full      weights/mixed_v6_full/final_net_epoch_0060.pth    60      mixed_v1 8K + dark 100 + view 1K          9,100      scratch           MSE only
mixed_v7_sym       (학습 중)                                         91      mixed_v6_full_train 9.1K                  9,100      mixed_v6 ep60     symmetric
selftrain_r1       weights/selftrain/selftrain_r1/final_net_epoch_0070.pth     70      mixed_v1 8K + pseudo-label 751            8,751      mixed_v1 ep60     MSE only
```

## v8 Ablation 모델 (Structural / Reliability Loss)

모두 mixed_v8 (ep60) 위에 mixed_v8_train (9000장)으로 5 epoch finetune. LR=5e-5, batch=4.

```
Ablation   Loss 설정                                         Weight 경로
─────────────────────────────────────────────────────────────────────────────────────
A (coord)  struct_coord=0.003                                weights/v9_ablation_A_coord/
B (edge)   struct_edge=0.003                                 weights/v8_ablation_B_edge/
C (co+ed)  struct_coord=0.003, struct_edge=0.002             weights/v8_ablation_C_coord_edge/
D (flip)   struct_flip=0.02                                  weights/v8_ablation_D_flip/
E (rel)    rel_loss, rel_lambda=0.005, rel_lambda_log=0.5    weights/v8_ablation_E_rel/
```

### noapril 추론 결과 (capture0403noapril, 188장)

```
             PnP Rate    A pass    B pass    C pass    B∧C
v8 (base)    49.5%       43장      1장       0장       0장
A (coord)    62.2%       19장      16장      6장       6장  << B∧C 최고
B (edge)     54.3%       43장      20장      4장       4장
C (co+ed)    55.3%       31장      13장      2장       0장
D (flip)     29.3%       19장      7장       2장       1장
E (rel)      62.8%       24장      25장      4장       2장  << PnP/B 최고
```

## 평가 결과 비교 (Synthetic Val, mixed_v1_val 200장)

```
모델               PCK@3px ↑   PCK@10px ↑   PnP Rate ↑   Reproj mean ↓   Vol Ratio   Vol<20% ↑
──────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1           0.469       0.731        72.5%        88.1 px         1.159       55.3%
mixed_v2           0.466       0.693        77.5%        112.4 px        1.048       54.6%
mixed_v3           0.470       0.719        70.5%        88.6 px         0.764       33.0%
mixed_v4_aug       0.439       0.612        63.0%        215.2 px        0.901       51.0%
mixed_v6_full      0.495       0.632        66.0%        143.8 px        -           52.1%
```

## Real Data 추론 결과 (capture0403noapril, 188장 어두운 팔레트)

```
모델               Avg KP ↑   PnP Rate ↑   비고
───────────────────────────────────────────────────────────────────────
mixed_v1           3.2/9      30.9%        밝은 팔레트 OK, 어두운 팔레트 0/9
mixed_v2           2.9/9      27.1%        데이터 추가했으나 악화
mixed_v3           2.3/9      27.1%        cuboid 3D 형태 개선, 감지율 하락
mixed_v4_aug       2.6/9      22.3%        aug 과도, 어두운 팔레트 centroid 1개 감지
mixed_v6_full      2.9/9      26.6%        어두운 팔레트 3/9 감지, PCK 최고
mixed_v7_sym       (학습 중)               symmetric loss로 앞뒤 혼란 해소 기대
```
