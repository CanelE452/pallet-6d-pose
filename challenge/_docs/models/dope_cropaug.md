# dope_cropaug (crop+padding 증강 DOPE, truncation 강건)

YOLO 트랙의 truncation crop 증강을 DOPE에 이식한 라인. pretrain은 synthetic-only crop, ft는 real 2단계(원본 long → +crop short).

## 모델 체인

| 단계 | weight | base | epoch | 데이터 |
|------|--------|------|-------|--------|
| pretrain | `weights/dope/dope_cropaug_pretrain/final_net_epoch_0060.pth` | scratch | 60 | mixed_v8 9000 + synth crop 8831 |
| ft s1 | `weights/dope/dope_cropaug_ft_s1/final_net_epoch_0150.pth` | pretrain | +90 (누적150) | real GT 251 |
| **ft s2 ★** | `weights/dope/dope_cropaug_ft_s2/final_net_epoch_0180.pth` | ft s1 | +30 (누적180) | real 251 + crop 485 |

- 공통: input 448, sigma 4.0, batch 4, lr 5e-5(ft)/1e-4(pretrain), loss=belief+affinity only.
- 재현: 데이터 `challenge/scripts/gen_truncation_crops.py` → `pad_truncation_crops.py`, 학습 `scripts/ft_dope_cropaug.sh`.
- **train.py는 finetune 시 net_path epoch에서 이어받고 EPOCHS=누적 목표**로 해석 (s1=150, s2=180).

## crop+padding 방식

- DOPE belief map(50×50)은 입력 범위 내 keypoint만 Gaussian → crop 후 화면 밖 corner는 supervision 없음.
- 해결: crop별 dynamic 대칭 pad + reflect border + 640×480 resize-back, **MARGIN_FRAC=0.20** (output 50/σ4 → 변에서 16% 안쪽 필요). belief 커버리지 100% 달성, 화면 밖이던 corner도 belief 8/8 supervised.

## 성능 (real truncation, 485 ft_real crop, order-free PnP)

| model | det≥6 | PnP% | reproj med |
|------|-------|------|-----------|
| baseline_v8_A | 12.8% | 22.7% | 134.5px |
| ft_s1 | 51.8% | 80.0% | 48.7px |
| **ft_s2** | **94.2%** | **98.8%** | **19.3px** |

- truncation에서 baseline(synthetic-only)은 거의 붕괴(0/9 kp), ft_s2는 6/9 복원 + 안정적 cuboid.
- clean에선 s1≈s2 — crop 가치는 truncation에서 가장 큰 격차.

## 성능 (clean synthetic val, order-free 평가)

| 모델 | PCK@3/5/10 | corner med | reproj med |
|------|-----------|-----------|-----------|
| baseline_v8_A | 79/85/88 | 28px | 39px |
| ft_s1 | 81/89/96 | 16px | 21px |
| ft_s2 | 78/86/94 | 18px | 20px |

→ **crop-aug는 clean에서도 baseline보다 우세** (corner localization, reproj 모두). truncation 강건성을 clean 정밀도 손해 없이 얻음.

## 주의

- `evaluate_on_val.py`는 2026-06-02 order-free PnP로 수정됨(백업 `.bak`). flat object 평가는 **EPnP 금지·ITERATIVE 필수**, 매칭은 48 automorphism order-free, median 기준. 수정 전 same-index 평가는 crop-aug를 과소평가했음. 상세: `_docs/history/2026-06-02.md`, 메모리 [[evaluate-on-val-convention-bug]].
- real holdout 없음(251장 전부 학습) → real 정량은 상대비교만.
