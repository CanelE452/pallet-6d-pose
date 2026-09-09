# PURPOSE — dimension_conditioning_probe

[소비처] 논문 method 결정 — "known object dimensions 를 PnP/네트워크에 넣을지" 를
Table(baselines) 확정 전에 판정한다. full YOLO 재학습 승인 여부의 전제 근거.

[문장] "현재 prediction-only W/D parity selector 실패는 real DEV pose 오차의
지배적 병목이며(또는 아니며), image feature + known dimensions 로 그 headroom 을
회수할 수 있다(또는 없다)."

training-0 (Phase A) → frozen-feature probe (Phase C) 순서로만 진행한다.
30ep full training, backbone 변경, keypoint head FiLM 은 이 작업에서 금지.
