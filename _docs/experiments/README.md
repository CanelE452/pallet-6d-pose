# Experiments — camera-facing 논문 실험 인덱스

각 실험은 **하나의 Table/Figure 단위** 파일. 양식(빈 틀)을 미리 만들어두고, 실험하면 그 파일을 채운다.

> 2026-06-04 재편: v8(object-frame) 실험은 `archive/` 로 격리. 아래는 **camera-facing 0123 / 논문용(v1/v2 제외, 일반화)** 새 실험들.
> 방향: CLAUDE.md "핵심 방향" + memory 4종. 검증할 주장: `_docs/method/{overview,step1~3,evaluation}.md`.

## 실험 인덱스 (구분 / 논문 / 상태 / 의존)

```
#    파일                                    구분   논문    상태      의존
──────────────────────────────────────────────────────────────────────────────────────
A1   data/A1_paper_base_perf.md              다시   T       미시작    paper_base 학습
A2   data/A2_squash_ratio_ablation.md        새로   T       미시작    paper_base ±squash
A3   data/A3_truncation_padding_ablation.md  새로   T       미시작    paper_base ±padding
B1   filter/pr_screening.md                  다시   T1      ★일부완료 ft_s2/pretrain (paper_base 재확인)
B2   filter/B2_filter_selftraining.md        다시   T       미시작    paper_base, C1
C1   self_training/C1_rounds.md              다시   ★F1     미시작    paper_base, 필터선정
C2   self_training/C2_pl_quality_vs_quantity 다시   T       미시작    C1
D1   eval/D1_generalization_seen_unseen.md   새로   T       미시작    paper_base, real GT
D2   eval/D2_real_test.md                    다시   T       미시작    paper_base/r1, real GT
D3   eval/D3_pnp_solver.md                   새로   T       부분      real GT (challenge 검증됨)
F2   eval/F2_qualitative.md                  다시   F       미시작    위 실험들
T10  related_work.md                         유지   T10     예정      논문 draft
```

★ = 핵심 / 부분·일부완료 = 채울 데이터 일부 있음

## 의존 순서 (실행 경로)

```
[다른 머신] paper_base 학습
      │
      ├─→ A1 base 성능, A2 squash, A3 padding  (데이터/학습 검증)
      ├─→ B1 필터 P/R 재확인 (paper_base)
      │      │
      │      └─→ 필터 선정 (outside diag / night diag∧ratio)
      │             │
      │             └─→ C1 self-training R0→R1→R2 ──→ B2 필터별 downstream
      │                        │                       C2 PL 품질vs수량
      │                        └─→ D1 일반화, D2 real test, D3 PnP, F2 정성
      └─→ (논문 draft) T10 related work
```

## Metric 정책 (camera-facing)

```
필터 P/R 스크리닝   통과 PL의 전체 9kp order-free(Hungarian) 평균오차 — 필터 목적=믿을만한 PL
self-training       도메인별 per-frame 검출(NN<20px) + reproj(9kp)
real 6D (dims known) ADD, 5cm5°, reproj — SQPnP, order-free
주의                evaluate_on_val convention 버그 → order-free PnP 필수 (memory)
                    monocular라 5cm5° 약함 → reproj median이 keypoint 품질 신호
```

## 데이터셋 정책

```
GT 평가셋     outside_combined(129) + night_combined(90) + forklift(32) + capture0403middle(440)
제외          data/_eval_sets/_exclude.txt (1778652125245035520 = bad manual GT)
unseen 정의   논문용은 v1/v2(내 파렛트) 제외 학습 → real 파렛트가 곧 unseen
누수 주의     평가모델이 GT를 학습했는지 확인 (ft_s2=누수 / pretrain·paper_base=held-out)
```

## 관련 폴더
- `_docs/method/` — 검증할 주장 (overview / step1~3 / evaluation)
- `_docs/filter/` — 필터 전용 (pr_screening, survey)
- `_docs/models/paper_base.md` — 논문 base 모델 명세
- `data/pallet/eval_results/` — 평가 결과 원본
- `archive/` — 폐기 v8 실험 (참고용)
- [pallet_boundary_action_repair_v1/README.md](pallet_boundary_action_repair_v1/README.md) — 유한 B 지지·L 행동 목표 실험: 전체192장·별도 CAL46 결과, 독립 검산과 원영상 갤러리
