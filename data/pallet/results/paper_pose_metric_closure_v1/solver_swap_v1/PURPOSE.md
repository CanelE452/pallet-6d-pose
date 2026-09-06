# PURPOSE — solver_swap_v1

[소비처] 논문 §evaluator — "pose read-out solver 를 미분가능 PnP 로 바꾸면 수치가 어떻게
         변하는가" 한 표. `metric_split_lock` §3.1 의 SQPnP LOCK 을 유지할지 푸는지의 근거.

[문장]   같은 keypoint 에서 solver 만 미분가능 Gauss-Newton 으로 바꾸면 pose 정확도가
         (개선된다 / 변하지 않는다 / 나빠진다) 중 무엇인지를 짝지은 프레임에서 보인다.

설계·게이트: `_docs/notes/pnp-solver-swap.md` 와 `SOLVER_SWAP_METHOD_LOCK.json`.
학습 0 step · 새 추론 0 회.
