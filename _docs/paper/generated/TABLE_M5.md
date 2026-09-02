# Table M5 — Robustness and pallet morphology generalization

Synthetic-only(R0) 대 Proposed(R5).  새 학습 없이 subgroup 평가만 한다.

`src` 열이 그 행의 keypoint 통계 출처다.

```text
strict   evaluator 의 supervision mask (reviewed visibility)
diag     all-annotated — visibility 가 unknown 인 legacy 점까지 포함.
         visible/occluded 주장이 아니다.  strict 가 0 개인 조건에서만 쓴다.
```

```text
Condition          N    src   n_kp  R0 corner↓  R5 corner↓        Δ  R0 det↑  R5 det↑
────────────────────────────────────────────────────────────────────────────────────────────
Plastic          194 strict    594       4.323       3.985   -0.338    0.959    0.979
Wood             125 strict    720       4.481       4.362   -0.119    1.000    0.992
Daytime           70   diag    630      10.928      11.592   +0.664    1.000    0.986
Nighttime         50 strict    198       5.478       5.271   -0.207    0.840    0.960
Clean            184 strict    936       4.169       4.031   -0.138    1.000    0.989
Occlusion        135 strict    378       4.976       4.626   -0.350    0.941    0.978
Truncation        51 strict    207       6.581       6.122   -0.458    0.922    0.922
Far               59 strict    468       3.590       3.285   -0.305    1.000    1.000
```

조건은 서로 중복될 수 있다. N 은 PAPER_EVAL manifest 와 workspace tag 에서 온다.

`diag` 행과 `strict` 행의 절대값을 서로 비교하지 않는다 — 다른 모집단이다.
행 안에서의 R0 대 R5 비교는 같은 모집단이므로 유효하다.

Low/Mid/High 와 넓은 lighting 분할은 Appendix A7 로 뺀다.
