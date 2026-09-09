# PURPOSE — 무패딩(Jetson Nano 배포용) 재학습

[소비처] Jetson Nano 실시간 추론 배포 결정 — 100px reflect padding 을 뺄지 말지.
판정과 근거는 `_docs/notes/c4-rotation-symmetry.md`.

[문장] 학습·추론 모두 padding 을 빼면 입력 텐서가 640x544 에서 640x480 으로 줄어
추론이 실측 12.8% 빨라지는데, 그 대가로 잃는 잘림(truncation) 강건성이
Jetson 배포에서 감수할 만한 크기인지 수치로 정한다.

padding 은 학습과 추론이 같아야 한다(train/infer parity). 추론에서만 끄면 성능이
떨어지므로 학습부터 다시 한다. 같은 데이터·같은 분할·같은 레시피, padding 만 0.
