[소비처] `_docs/notes/diffpnp-yolo.md` §2 결과 — 논문 §method 의 "미분가능 PnP 를 학습에
         넣으면 pose 가 좋아지는가" 한 절. 평가 solver 교체(solver_swap_v1)가 REJECT 로
         끝났으므로, 남은 갈래인 학습측 DiffPnP 를 같은 기준으로 판정한다.

[문장]   평가가 pose 를 읽는 연산(예측 2D → PnP)을 학습 loss 안에 넣으면, 그 외 모든 것을
         고정한 채로 실제 pose 정확도(rotation·translation·IoU3D·ADD AUC)가 개선되는가.

설계·게이트: `_docs/notes/diffpnp-yolo.md`.  선행 결과: solver_swap_v1(평가측 교체)=REJECT,
2026-08 predseed DiffPnP=REJECT, 2026-07 DOPE Q1/StageB=PARTIAL.
