# Table B — detection, ranking and fine 2D localisation

population   PAPER_EVAL 319 positive + 2,689 negative, role DEV
source       data/pallet/results/paper_eval_v1/arms/<ARM>.json
             ranking CIs: paper_framing_closure_v1/PAPER_STATIC_STAT_AUDIT.json

```text
arm                               AP50-95 ↑   AP50 ↑   AUROC ↑   FPR95 ↓  kp med px ↓  kp p90 px ↓
---------------------------------------------------------------------------------------------------
R0  합성만 학습                      0.7688   0.9363    0.9921    0.0417       6.6157        38.67
R0-CONT  추가학습만 (ST 없음)        0.7609   0.9367    0.9872    0.0573       6.9113        45.19
R1  ST 필터없음                      0.7622   0.9292    0.9913    0.0558       7.1201        35.12
R2  ST 신뢰도                        0.7635   0.9467    0.9923    0.0469       7.0372        43.61
R3  ST +재투영                       0.7643   0.9417    0.9920    0.0487       7.0436        41.29
R4  ST +코너제거                     0.7578   0.9366    0.9911    0.0502       6.9987        39.34
R5  ST 전체일관성 (제안)             0.7585   0.9580    0.9953    0.0283       7.2099        41.38
```

How to read it in the meeting

- Ranking is the one axis where the proposed filter is best of all arms.
- paired R5 - R0 AUROC +0.00318 frame CI [+0.000092, +0.006898] — excludes zero.
- paired R5 - R0 FPR95 -0.01339 frame CI [-0.02566, +0.00558] — contains zero.
- The session-clustered ranking interval is NOT computable: negative rows
  carry no session identifier. The frame-level interval was computed after
  the point estimate was seen, so it is Tier B.
- Fine 2D localisation: no arm is below R0's 6.6157 px. That is the
  paper's central negative result.

Paper role: **main table** (detection/ranking/2D).
