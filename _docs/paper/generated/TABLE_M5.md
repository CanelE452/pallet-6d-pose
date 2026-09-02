# Table M5 — Robustness and pallet morphology generalization

Synthetic-only(R0) 대 Proposed(R5).  새 학습 없이 subgroup 평가만 한다.

`src` 열이 그 행의 keypoint 통계 출처다.

```text
strict   evaluator 의 supervision mask (reviewed visibility)
diag     all-annotated — visibility 를 무시하고 좌표가 있는 점을 전부 센다.
         visible/occluded 주장이 아니다.  strict 가 0 개인 조건에서만 쓴다.
```

```text
Condition          N    src   n_kp  R0 corner↓  R5 corner↓        Δ  R0 det↑  R5 det↑
────────────────────────────────────────────────────────────────────────────────────────────
Plastic          194 strict   1603       7.067       7.783   +0.716    0.959    0.979
Wood             125 strict   1111       5.965       6.224   +0.259    1.000    0.992
Daytime           70 strict    609      10.556      11.576   +1.020    1.000    0.986
Nighttime         50 strict    339       7.045       8.897   +1.852    0.840    0.960
Clean            184 strict   1631       5.582       5.659   +0.077    1.000    0.989
Occlusion        135 strict   1083       8.746       9.351   +0.605    0.941    0.978
Truncation        51 strict    379       9.600      11.132   +1.532    0.922    0.922
Far               59 strict    531       3.823       3.551   -0.271    1.000    1.000
```

조건은 서로 중복될 수 있다. N 은 PAPER_EVAL manifest 와 workspace tag 에서 온다.

`diag` 행과 `strict` 행의 절대값을 서로 비교하지 않는다 — 다른 모집단이다.
행 안에서의 R0 대 R5 비교는 같은 모집단이므로 유효하다.

Low/Mid/High 와 넓은 lighting 분할은 Appendix A7 로 뺀다.
