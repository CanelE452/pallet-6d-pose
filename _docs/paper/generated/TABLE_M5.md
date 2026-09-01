# Table M5 — Robustness and pallet morphology generalization

Synthetic-only(R0) 대 Proposed(R5).  새 학습 없이 subgroup 평가만 한다.

```text
Condition          N  R0 corner↓  R5 corner↓        Δ  R0 det↑  R5 det↑
──────────────────────────────────────────────────────────────────────────
Plastic          194       4.323       3.985   -0.338    0.959    0.979
Wood             125       4.481       4.362   -0.119    1.000    0.992
Daytime           70           —           —        —    1.000    0.986
Nighttime         50       5.478       5.271   -0.207    0.840    0.960
Clean            184       4.169       4.031   -0.138    1.000    0.989
Occlusion        135       4.976       4.626   -0.350    0.941    0.978
Truncation        51       6.581       6.122   -0.458    0.922    0.922
Far               59       3.590       3.285   -0.305    1.000    1.000
```

조건은 서로 중복될 수 있다. N 은 PAPER_EVAL manifest 와 workspace tag 에서 온다.
Low/Mid/High 와 넓은 lighting 분할은 Appendix 로 뺀다.
