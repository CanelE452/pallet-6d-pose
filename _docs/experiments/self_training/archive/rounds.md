# F1. Self-Training — Single-Round Real-Only Fine-tune

상태: **★ 완료** (2026-04-14 재작성, 원본 multi-round 계획은 폐기)

## 설계 변경 공지

원래 계획: multi-round RANSAC self-training (R0→R1→R2→R3) 수렴 관찰.
실제 진행: **single-round real-only ft**가 mixed training 보다 압도적으로 잘 됨.
R2/R3 수렴 figure 는 포기 (mixed training 실패로 의미 소실), single-round ablation
으로 재설계.

## 핵심 발견

```
1. Mixed training (synthetic + PL, α=1.0) — 전부 실패
   T1 none (64 PL):   <20px 17.3% (baseline 18.9% 보다 낮음)
   T1 ransac (6 PL):   <20px 16.4% (oversampling 5625회/장 → memorize)

2. Real-only ft (PL 만 사용, synthetic 제거) — 성공
   ST_8only (8 PL B∧C):     37.3%  (baseline 대비 +18pp)
   F5 (2 PL RANSAC+LOO):    60.5%  (baseline 대비 +39pp) ★
```

**해석**: 소량의 clean PL 로 real-only ft 하면 domain gap 이 극단적으로 빠르게
좁혀진다. Synthetic 을 섞으면 오히려 synthetic domain 쪽으로 끌려감.

## Single-Round Ablation (middle 440, NN matching <20px)

```
model                              base    filter          PL #   ep    <20px   median
──────────────────────────────────────────────────────────────────────────────────────────
baseline ep60                       —       —              —      60    18.9%   30.4px
v8_A coord ft ep65                  ep60    —              —      +5    21.6%   66.2px
ST_8only (real-only)                ep60    canonical B∧C   8     +91   37.3%   33.5px
F4 RANSAC 9PL (real-only)           ep65    RANSAC c≥6      9     +90   21.8%   77.9px
F5 RANSAC+LOO 2PL (real-only) ★     ep65    RANSAC+LOO      2     +96   60.5%   12.2px
F5 재현 (다른 seed)                  ep65    RANSAC+LOO      2     +96   53.9%   —
```

Metric: `gt_final_isaac` projected_cuboid vs DOPE raw keypoint, Hungarian NN
matching, per-frame mean error < 20px. 검증: [`../eval/metric_validation.md`](../eval/metric_validation.md).

## 실행 명령 (F5)

```bash
python scripts/self_training/self_train.py \
    --config config/stage3_selftrain.yaml \
    --filter_type ransac_loo \
    --real_only \
    --num_rounds 1 \
    --epochs_per_round 96 \
    --lr 5e-5 \
    --pretrained weights/v8_ablation_A_coord/final_net_epoch_0065.pth \
    --real_dir data/pallet/raw_data/capture0403noapril/rgb \
    --output_dir weights/misc/f5_noapril_ransac_loo_realonly \
    --seed 4165
```

PL pool: `output/pl_noapril_ransac_loo_only/` (2 장).
ST_8only PL pool: `data/pallet/eval_results/st8only_pl8_frames/` (8 장, canonical B∧C).

## 평가 명령

```bash
python scripts/data_prep/eval/eval_nn_matching.py \
    --weights weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth \
    --test_dir data/pallet/raw_data/capture0403middle \
    --gt_dir data/pallet/raw_data/capture0403middle/gt_final_isaac
```

## 재현성 검증

F5 는 PL 이 2 장뿐이라 seed 에 민감. 2 seed 실험:

```
           원본(seed=4165)   재현(seed=random)
ep90       58.2%             52.0%
ep96       60.5%             53.9%
```

발표 / 논문 시 단일 수치 대신 **54~60% 범위**로 보고. F5 > F4(21.8%) 순서는
seed 에 관계없이 유지되므로 방향성은 robust.

## 한계 (committee 대응)

1. **PL 수 2장은 memorization 위험** — 그러나 noapril(pool)과 middle(test)는
   독립 frame set 이고 AprilTag 유무도 달라 완전 overlap 아님
2. **Seed 민감도 6~7pp** — seed 2 개로는 분포 특성화 불충분, 최소 5 개 필요.
   발표에서는 범위로 보고, 논문에서는 추가 seed 확보 예정
3. **Single dataset evaluation** — Seen/Unseen AprilTag GT 평가 미완성.
   일반화 주장은 보류, middle 440 내에서의 효과로 제한

## 관련

- 필터 선정: [`../filter/selection.md`](../filter/selection.md)
- 필터 ablation (T1 포함): [`../filter/ablation.md`](../filter/ablation.md)
- Metric 검증: [`../eval/metric_validation.md`](../eval/metric_validation.md)
- 모델 카탈로그: [`../model_catalog.md`](../model_catalog.md)
- 구현: `scripts/self_training/self_train.py`, `scripts/self_training/geometric_filter.py`
