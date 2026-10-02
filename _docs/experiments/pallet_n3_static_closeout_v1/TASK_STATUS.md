# 실행 상태

| Task | Status | Evidence / remaining input |
| --- | --- | --- |
| N0/N1 6D | RECOMPUTE + VERIFIED + INSERTED | 1914 frame-method rows,6 seeds; same PnP |
| R0/P/N2/N3 | REUSE + RAW REGRESSION | 26 original score arrays identical |
| ResNet model | CORRECTED + INSERTED | Actual checkpoint10epoch/34990steps; fold parity passes |
| Pose uncertainty | RECOMPUTE + INSERTED | 13 session paired10000 draws; all3seeds and ablations |
| Occlusion/material | REUSE + REAGGREGATE + INSERTED | same reviewed128 andunknown191 across backbones |
| Visibility | PARTIAL_LABELS + INSERTED | visible66/external5/unknown2428; old coordinates retained |
| Square | VERIFIED + INSERTED | both600/602 modes;6D x |
| D/L/PoseFix | REUSE + RAW REGRESSION + INSERTED | 319/8corners |
| Student alternatives | REUSE + RAW REGRESSION + INSERTED | 128 separate cohort |
| Runtime | REUSE2 + MEASURE1 + INSERTED | DOPE/ResNet reused;YOLO new;environments separated |
| LaTeX | READY_TO_INSERT | Exact-label fragments; matching source absent |
| Lifter | OUT_OF_SCOPE_USER | No execution; chapter unchanged |
