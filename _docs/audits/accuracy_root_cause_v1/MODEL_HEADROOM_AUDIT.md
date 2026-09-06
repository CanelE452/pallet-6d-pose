# 헤드룸 감사 — 이미 가진 것 안에 답이 얼마나 있는가

작성 2026-09-06 · HEAD `2e5ec0e` · 새 추론 0 회 · 모집단 role = **DEV**
오라클은 배포 결과가 아니다. 아래 오라클 수치를 성능 주장으로 쓰지 않는다.

---

## 요약

```
CANDIDATE_ORACLE_HEADROOM   거의 없음   후보 0인 프레임 0장, IoU50 매칭 311/319 (97.5%)
RANKING_HEADROOM            작음        AUROC 0.9921, misrank 9/161 (다른 모델·모집단)
PERMUTATION_HEADROOM        작음        8! 자유 배정이 98장 중 13장만 구제, p50 4% 개선
PNP_SOLVER_HEADROOM         없음        PNP_NOT_PRIMARY_LEVER (기존 판정 2건, 재확인)
SELECTIVE_REJECTION         약함        kp_conf_min AUROC 0.744, 30% 버려 실패율 1/3
TEACHER_ORACLE_HEADROOM     큼          그러나 배포 가능한 selector 로 회수 실패 (4회)
CAPACITY_HEADROOM           미확정      medium 이 기각된 적 없음. PAPER_EVAL 319 미채점
REAL_SUPERVISION_HEADROOM   큼 · 실증됨  unseen 세션에서 코너 중앙값 -33.5%
```

---

## 1. 후보 오라클 (§7.1) — 헤드룸 없음

[확인] `R0_per_frame.csv` 319 positive: `candidate_count == 0` 인 프레임 **0장**,
`top_iou50_match` 311/319 (97.5%), `top_target_iou` p50 0.919.
정답 후보를 만들지 못하는 문제가 아니다.

[확인] 다른 모델(`yolo26n_paper_generic_v1_seed42`)·다른 모집단(161)의 `_rr_detail.json`
4분류도 같은 방향이다 — `A_GOOD_CANDIDATE_MISRANKED` 9장(5.6%) 대
`B_CORRECT_BOX_BAD_KP` 59장(36.6%). ★ 두 표를 같은 줄에 놓지 말 것.

## 2. 순열 오라클 (§6.2) — 헤드룸 작음

[확인] `AXIS_FAILURES.json`. `identity` 최대 코너 오차 > 25 px 인 98장 중:
계약 대칭 {0,180} 구제 **0장**, C4 구제 15장, 8순열 15장, Hungarian(8! 자유) 13장.
최대 코너 오차 p50 은 16.37 → 15.75 px (4%) 밖에 안 내려간다.
→ role/축 semantics 가 아니라 실제 위치추정 문제. 상세 `FAILURE_DECOMPOSITION.md`.

## 3. PnP / solver 오라클 (§7.3) — 헤드룸 없음

[확인] 이미 두 방향에서 닫혔다. 평가측 교체(`solver_swap_v1`, 2026-09-06):
7 arm × 7 키 전부 baseline 재현(1e-9), D1_GN_LS 는 rot 2.262 → 2.296°(+1.48% 악화),
D3(SQPnP init + GN)은 최대 상대변화 3.67e-08 = 변화 없음.
학습측(`diffpnp_yolo_v1`): 합성 val 은 소수점 넷째 자리까지 같은데 real 은 7~23% 악화.
`KEYPOINT_SUBSET_PNP.json` 에 top7/6/5/4 · LOO 8개 · near/far/top/bottom-only PnP 가
이미 계산돼 있다(rows 116).
→ **PNP_NOT_PRIMARY_LEVER.** solver 탐색을 닫는다.

## 4. selective rejection (§8) — 약하지만 실재

[확인] 직접 계산(`selective_signals.py`, 319 프레임). 배포 가능 신호만:
`kp_conf_min` AUROC 0.744 > `box_conf` 0.697 > 공간대각 교점 0.649 > 연결선 CV 0.606
> 면대각 0.585 > centroid 자기일관성 0.444.
coverage 0.70 에서 gross 30.7% → 21.1%. 상세·곡선 `SELECTIVE_RISK_AUDIT.md`.

★ **기하 자기일관성이 정보를 주지 않는다**는 것이 이 감사의 부수 소득이다.
R0 가 틀릴 때 틀린 코너끼리는 여전히 정합한 육면체를 이룬다 — 육면체가 통째로
잘못된 자리에 놓인다. `corner-residual-systematic-not-noise` 와 같은 방향.

## 5. 교사 오라클 (§7.2) — 크지만 회수 불가 (4회 확인)

[확인] `MULTITEACHER_FINAL_REPORT.md` 및 memory
`multiteacher-consensus-gives-confidence-not-labels`: oracle p90 43.9 → 13.3 은 실재하나
GT 없이 못 고른다. 합의 필터는 **R0 가 이미 맞히는 자리**를 고른다.
좌표 median/medoid 융합·고전 CV 코너 선택기·도메인 정렬 전부 닫힘.
회수 시도 feature 의 AUC: pnp_reproj 0.597, domain 0.482, dims 회수율 2.56%.
→ "어디가 틀렸는지는 아는데 무엇이 맞는지는 모른다" 가 네 트랙에서 반복됐다.

## 6. capacity (§17) — 기각된 적이 없다

[확인] `CAPACITY_AND_REAL_SUPERVISION_AUDIT.md`.
`LARGER_MODEL_REJECTED_UNDER_MATCHED_CONDITIONS = NO`.
Wilcoxon p = 0.1433 은 NOT_ESTABLISHED 이지 기각이 아니다.
`runs_arch_baseline`(v8n/11n/26n)은 전부 nano — architecture family 비교이지 capacity 가 아니다.
s 계열은 사전학습 가중치만 있고 학습 run 0개, l 계열은 문자열도 없다.
**medium 은 PAPER_EVAL 319 에서 한 번도 채점된 적이 없다.**

[확인] 그럼에도 SEALED 105 에서 medium 이 nano 를 위치추정·pose 전 축에서 이긴다:
corner 6.80 vs 7.63, IoU3D 0.66 vs 0.57, R med 2.78 vs 3.08, ADD-S 0.10 vs 0.12
(검출만 0.952 vs 0.971 로 뒤진다).
★ matched 는 아니다 — medium base 는 서버 학습이라 로그·`results.csv` 가 이 머신에 없고
epoch 수 UNKNOWN, DDP 2GPU 라 유효 batch·BN 통계가 다르다. 단일 seed.

[확인] R0 에 `epoch0/5/…/55.pt` 13개가 남아 있고 `paper_real_eval.py` 가 `--weights` 를
받으므로, **재학습 없이 real 수렴곡선**을 뽑을 수 있다.

## 7. real labeled supervision (§22) — 크고, 실증됐고, 실패 레짐에 정확히 꽂힌다

[확인] `_docs/history/2026-08-20.md` 원문. SEALED 105 = pallet07/09 · night08/09 로
FT 학습 세션(night01~07 · pallet02/03/04/05/08 · forklift_20260528)에 없고,
같은 세션의 인접 non-eval 53장까지 제외해 프레임 겹침 0 을 문서화했다.

```
SEALED 105        det    pnp  corner  R med  ADD-S   IoU   5cm5
yolo26n_synth   0.838  0.809   10.51   2.90   0.12  0.57  0.219
yolo26n_ft      0.971  0.952    7.63   3.08   0.12  0.57  0.314
yolo26m_ft      0.952  0.933    6.80   2.78   0.10  0.66  0.324
```

[확인] 메인 세션 직접 계산(`ft_by_elevation.py`) — **이득이 저앙각에 몰린다.**
정본 140장 중 세 모델 공통 검출, SEALED 세션만(73장, 전부 앙각 <15°):

```
앙각      N   synth   n_ft   m_ft   n_ft개선   m_ft개선   synth>25   n_ft>25
<8       49   9.22   5.86   5.49    36.4%     40.5%     53.1%     28.6%
8-15     24  11.49   8.49   7.87    26.1%     31.5%     50.0%     37.5%
전체      73   9.62   6.40   6.53    33.5%     32.1%
```

[확인] 그리고 **학습 세션과 겹칠 수 있는 나머지 51장에서는 이득이 오히려 작다(25.8%)**.
암기라면 반대여야 한다. 정본 140 전체로 보면 고앙각에서는 이득이 사라지거나 뒤집힌다
(15-30° −6.0%, ≥30° −7.4%이고 ≥30° 의 gross 는 11.8% → 29.4% 로 악화).
즉 real supervision 이 채운 것은 **저앙각 레짐 그 자체**다.

[확인] 다만 이건 **new-session** 일반화다. **new-shape(다른 물체) 일반화는 측정된 적이 없다** —
`paper_real_ft_v1` 은 착수 전 중단됐고 이유가 라벨 결함이다(402장 중 106장 좌우 코너
순서 위반, 187장 90도 stale).

## 8. memory 정정 2건

[확인] `live-gt-ft-split-and-aug-decide-the-verdict` 의 "pose 0.35 → 0.98" 은
**same-session** 이다 — `datasets/live_gt_v2|v4|v5_nocrop/_prepare_live_gt.json` 이
전부 `split_mode: "interleave", val_every: 6` 이다. 새 세션 일반화 근거가 아니다.
그리고 split(촬영단위→interleave)과 aug(ultralytics 기본→base 계약)가 **동시에** 바뀌어
교락돼 있다. 촬영단위 split + base-contract aug 조합은 한 번도 돌린 적이 없다.

[확인] `hf-published-pallet-pose-models` 의 "medium 이 명확히 낫지 않다" 는 OPEN 56 기준이다.
일반화를 시험하는 SEALED 105 에서는 medium 이 위치추정·pose 전 축에서 낫다.
