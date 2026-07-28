# C1. Self-training R0→R1→R2 × 도메인  (★논문 핵심 Figure F1 + Table)

> 상태: **R0/R1 완료 (R1 회귀 — 아래 결론)** | 의존: paper_base 학습, 필터 선정(diag/diag∧ratio)
> 구분: **다시** (v8 발표 셋업을 camera-facing + paper_base로 재현)

## 목적 (한 줄)
2D 기하 필터로 선별한 신뢰 PL로 self-training 반복 시, **도메인별(indoor/outside/night) 성능이 R0→R1→R2로 향상**되는가.

## 판단 지표
도메인별 **per-frame 검출 정확도(NN<20px) + reproj(9kp)**, R0/R1/R2 곡선.
(발표 교훈: PL 수보다 품질. indoor 소량 PL로 R1↑, outdoor/night 다량인데 R2↓ → 좋은 필터로 재현 검증)

## 설정
- anchor R0 = `paper_base`
- 필터: outside=`diag`, night=`diag∧ratio` (indoor=PL 신뢰 낮음 → 1라운드 후 재필터)
- unlabeled pool: outside 9894 / night 9134 / indoor(noapril) 188 (TBD: camera-facing 재확인)
- GT 평가셋: outside_combined(129)·night_combined(90)·capture0403middle(440) [exclude.txt 반영]
- 학습: train.py finetune, 누적 epoch (memory `dope-finetune-cumulative-epoch`)

## 방법
1. R0(paper_base) → unlabeled 추론 → 필터 → PL 추출
2. PL로 R1 finetune → 도메인별 평가
3. R1 → PL 재추출 → R2 → 평가
4. R0/R1/R2 매트릭스 + 곡선

## 결과 — R0 vs R1  (2026-06-06, order-free 9kp 평가)

> 스크립트: `scripts/data_prep/eval/c1_r0_vs_r1.py`
> 산출: `scripts/data_prep/eval/c1_r0_vs_r1_results.json`
> 지표: 검출률(≥6 corner) / reproj order-free(Hungarian 8corner + centroid, GT projected_cuboid 대비) median / good(<10px)%
> R0=paper_base/0060, R1_outside=diag PL 697, R1_night=diag∧ratio PL 107

```
model       domain    n    det%   rep8_med  rep9_med  rep9_mean  good%
──────────────────────────────────────────────────────────────────────
R0_base     outside   128  45.3    18.65     18.91     22.09     12.1
R0_base     night      90  30.0    23.46     21.88     33.86      0.0
R0_base     indoor    440  26.6    13.09     13.02     13.94      3.4
R1_outside  outside   128  51.6    27.27     26.22     28.72      0.0
R1_outside  night      90  43.3    33.43     33.24     33.22      0.0
R1_night    outside   128  46.1    29.92     29.90     34.01      0.0
R1_night    night      90  47.8    25.91     26.17     28.18      0.0
```

### R0→R1 향상폭 (in-domain)
```
도메인    지표        R0      R1      Δ
─────────────────────────────────────────────────
outside   det%        45.3    51.6    +6.2   (검출↑)
outside   rep9_med    18.91   26.22   +7.31px (정확도↓ 악화)
outside   good%       12.1    0.0     붕괴
night     det%        30.0    47.8    +17.8  (검출↑ 큼)
night     rep9_med    21.88   26.17   +4.29px (정확도↓ 악화)
night     good%        0.0     0.0    변화없음
```
cross-domain: R1_outside@night rep9_med 33.24 / R1_night@outside rep9_med 29.90 — 둘 다 R0보다 악화.

## 결론 (2026-06-06)

**self-training R1은 검출률은 올렸으나 keypoint 정확도(reproj)는 모든 도메인에서 악화시켰다.**
in-domain·cross-domain 모두 reproj median +4~9px 증가, good(<10px)%는 outside 12.1→0%로 붕괴.

- 해석: diag/diag∧ratio 필터가 "centroid 부근 대각선 일관성"은 만족하나 **전체 8 corner의 절대 위치가 부정확한 PL을 다량 통과**시킨 것으로 보인다 (memory `filter-goal-reliable-pl-full-keypoints`의 경고가 그대로 재현 — 단일/2D 기하 조건은 무게중심만 보정, corner 정확도는 보장 못 함). 모델은 "더 자주, 더 거칠게" 찍도록 학습됨 (det↑ / px정확도↓).
- night det +17.8%는 다량(107) PL의 효과지만 정확도가 따라오지 않음 → 발표 교훈("다량 PL인데 성능↓")이 품질 필터로도 그대로 재현.
- 교차검증: R0의 기존 synthetic val(`weights/paper_base/eval_results/eval_summary.json`)은 PCK@5px=98.8%지만 corner2d_median=11.7px / reproj_median=14px로 절대 px는 원래 두 자리 → 본 real 평가(R0 18px)와 일관, 평가 코드 신뢰 가능.
- 주의(provenance): paper_base 학습 데이터에 `mixed_v8_train` 포함(header.txt). CLAUDE.md "v8 폐기" 방침과 충돌 가능 — base 자체의 convention/라벨 점검 필요(별도 이슈).

**다음**: (1) 필터를 전체 9kp 정확도 기준으로 강화(diag 단독 부적합), (2) PL을 corner reproj 임계로 hard-gate, (3) R2 진행 전 R1 회귀 원인을 PL 시각화로 확인.

## 산출물 (예정)
- round figure(F1), PL pool 증가표, 도메인 cross 평가
