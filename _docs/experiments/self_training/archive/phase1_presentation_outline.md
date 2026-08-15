# 3차 발표 슬라이드 Outline

상태: Phase 1 결과 기반 (2026-05-28)

## 슬라이드 구성 (15분 발표 기준)

```
#    슬라이드                                        분량
─────────────────────────────────────────────────────────
1    표지                                            1
2    1·2차 보완사항                                  1-2
3    시스템 전체 구조 + 데이터 흐름                  1
4    개발 환경 / 데이터셋                            1
5    합성 데이터 생성 (mixed_v8, 9000장)              1
6    DOPE + coord loss (Step 1 anchor, ep65)         1
7    Geometric Filter (RANSAC subset consensus + LOO) 1-2
8    Phase 1: Self-Training 라운드별 곡선            1   ← figure 1
9    Phase 1: 정량 매트릭스 (3 도메인 × 3 라운드)    1-2
10   Phase 1: 정성 비교 panel                        1   ← figure 2
11   핵심 발견 (R1 optimal, R2 over-iteration, filter quality) 2
12   한계 + 4차에 남은 작업                          1
13   Q&A 예비                                        1
```

## 슬라이드 별 핵심 내용

### #2 — 1·2차 보완사항

2차 발표 한계 (본인 인정):
- 학습/평가가 동일 실내 환경 → 도메인 갭 미검증
- Self-training 반복 (R1→R2→R3) 미실험 (F5 는 R1 단발)

3차에서 직접 보완:
- **outside (data/outside/, 11 시퀀스 9894 frame)** + **night (data/night/, 10 시퀀스 9134 frame)** 추가 — 학습 도메인(실내)과 다른 환경
- 각 도메인 manual_gt 평가셋 통합: indoor 440 / outside 129 / night 90
- Self-training R0 → R1 → R2 모든 라운드 학습/평가

### #3 — 시스템 전체 구조

```
                        ┌─────────────────┐
                        │  Isaac+Blender  │
   합성 9000장 ─────────→│  DOPE pretrain  │── anchor (ep65, coord loss ft)
                        │  (Step 1)       │
                        └────────┬────────┘
                                 ▼
                ┌──────────────────────────────────────┐
                │  R0 anchor → real unlabeled 추론     │
                │  → RANSAC subset consensus + LOO     │ (Step 2)
                │  → pseudo-label 통과                  │
                └────────┬─────────────────────────────┘
                         ▼
                ┌──────────────────────────────────────┐
                │  R1: real-only ft on PL (Step 3)     │
                │  R2: R1 → PL 재추출 → ft (R1 iter)   │
                └──────────────────────────────────────┘
```

### #4 — 개발 환경

```
OS:         Windows 10
GPU:        RTX 3080 10GB
Python:     3.10 (conda env pallet-pose)
Framework:  PyTorch 2.10 + CUDA 12.6
Synthetic:  Isaac Sim 4.5.0 + Blender (Replicator 기반)
DOPE base:  NVIDIA Deep_Object_Pose (VGG-19 backbone, 9 belief maps + 16 affinity)
Camera:     RealSense D435i (640x480, fx=605.9, fy=605.9)
Pallet:     KS T-11형 1100×1100×150mm
```

### #5 — 합성 데이터 (mixed_v8)

```
mixed_v8_train 9000장 = Isaac 2000 + Blender 2000 + test_blender 4000 + variant 1000
도메인 랜덤화: 조명 (5000K/dome 2000-3500K), 텍스처, 색상, 디스트랙터
60 epoch scratch → mixed_v8 baseline NN<20px (val) 46.0%
```

### #6 — DOPE Step 1 (anchor)

```
v8_ablation_A_coord (★ Phase 1 anchor):
  base: mixed_v8 ep60
  +5 epoch ft with coord Huber loss (λ=0.003)
  output: weights/v8_ablation_A_coord/final_net_epoch_0065.pth
  ─────────────────────────
  (Loss ablation 결과 — coord 단독 최적, edge/flip/VP loss 실패)
```

### #7 — Geometric Filter

```
RANSAC subset consensus (n_iter=50, k=5, τ=5px, c≥6) 
+ LOO PnP stability check (loo_tau=0.05)
+ Physical size sanity (0.5~2.5m)

선정 근거 (2026-04-11): GT 기반 P/R 분석 후 23 후보 중 최종 선정
F1 점수 0.833 (canonical B∧C 의 0.235 대비 압도)
```

### #8 — Phase 1 라운드별 곡선 (figure 1)

```
figure 경로: _docs/figures/phase1_round_curve.png
(a) 라운드별 NN<20px 곡선 (3 도메인)
(b) PL pool 양 변화 (R0/R1/R2)
```

### #9 — Phase 1 정량 매트릭스

```
=== indoor (capture0403middle, 440 frame, NN<20px per-frame) ===
                    R0       R1            R2
indoor anchor       21.6%    60.5% (F5)    30.5%   ↓
outside anchor      -        58.4%         15.9%   ↓
night anchor        -        43.9%         33.4%   ↓

=== outside (manual_gt, 129 frame) ===
                    R0       R1            R2
indoor anchor       -        31.8% (F5)    11.6%
outside anchor      27.9%    39.5%         24.8%   ↓
night anchor        -        22.5%         20.2%

=== night (manual_gt, 90 frame) ===
                    R0       R1            R2
indoor anchor       -        32.2% (F5)    21.1%
outside anchor      -        33.3%         11.1%   ↓
night anchor        21.1%    26.7%         26.7%   =
```

Best per-domain:
- indoor: R1_indoor (F5) **60.5%** (+38.9 pp vs R0)
- outside: R1_outside_loo **39.5%** (+11.6 pp)
- night: R1_outside_loo **33.3%** (+12.2 pp)

### #10 — 정성 비교 (figure 2)

```
figure 경로: _docs/figures/phase1_qualitative.png
3 도메인 × 2 frame
좌: R0 (ep65) 빨강 keypoint
우: R1_outside_loo 파랑 keypoint
녹색: manual GT cuboid
```

### #11 — 핵심 발견 (발표 main message)

```
1. Self-training R1 한 라운드가 모든 도메인에서 best
   → R0 → R1 큰 향상 (+11~39 pp)
   → R1 → R2 모든 트랙에서 성능 정체/하락 (over-iteration)

2. PL 양이 증가해도 quality 향상 X (confirmation bias)
   ─────────────────────────────────────────
   indoor: 2 → 31 (15.5x), outside: 167 → 514 (3.1x), night: 105 → 973 (9.3x)

3. Filter quality 가 PL 양보다 중요
   ─────────────────────────────────────────
   ransac (c≥6) 1289 PL  → 학습 실패 (R0 보다 떨어짐)
   ransac_loo    167 PL  → 학습 성공 (R1_outside best)

4. outside PL 학습이 가장 generic 향상
   ─────────────────────────────────────────
   outside-trained R1 이 indoor(58.4), night(33.3) 에서도 best transfer

5. 도메인 갭 검증
   ─────────────────────────────────────────
   R0 frame-with-prediction: indoor 71% → outside 53% → night 29%
   R1 학습 후 모든 도메인 향상 (학습 도메인 = outside 만 학습했는데도)
```

### #12 — 한계 + 4차

한계 발견:
- R2 over-iteration: F5 이후의 self-training 추가 라운드는 confirmation bias 로 향상 한계 도달
- night PL pool 105 (R0) / 973 (R1) — 작은 PL 수가 학습 안정성에 영향
- ransac vs ransac_loo 의 큰 차이 — filter strict 가 critical

4차 발표 (최종) 에 남은 작업:
- Cross-domain transfer 의 정량 분석 (figure 보강)
- 추가 seed (현재 seed 1) — F5 처럼 6-7pp 변동성 측정
- 실패 케이스 분석 (어떤 frame 이 R1 에도 fail 하는지)
- 프로토타입 데모 영상 (DOPE inference cycle)
- (선택) over-iteration 회피 strategy (예: PL pool ensemble, curriculum 등)

## Phase 1 결과물 위치

- 결과 JSON: `_docs/experiments/self_training/phase1_results.json`
- 설계 문서: `_docs/experiments/self_training/phase1_3rd_presentation.md`
- 5/27 history: `_docs/history/2026-05-27.md`
- 5/28 history: `_docs/history/2026-05-28.md`
- Figure 1 (round 곡선): `_docs/figures/phase1_round_curve.png`
- Figure 2 (정성 panel): `_docs/figures/phase1_qualitative.png`
- 평가 결과 텍스트: `data/pallet/eval_results/phase1_{R0,R1_*,R2_*}/*.txt`
- 학습된 weights: `weights/selftrain/r1_outside_loo/`, `weights/selftrain/r1_night_loo/`, `weights/selftrain/r2_outside_loo/`, `weights/selftrain/r2_night_loo/`, `weights/selftrain/r2_indoor_loo/`
- 실패 라인 (참조용): `weights/selftrain/r1_outside_ransac/` (ransac c≥6, 1289 PL → 학습 실패)
