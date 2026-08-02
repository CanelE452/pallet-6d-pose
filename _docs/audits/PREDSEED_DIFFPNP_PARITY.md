# Parity

```
baseline 재현        strict 87 / GT-2D 87 / predicted 70 / yaw 6.025216 / reproj 23.161629
F1 24  F2 35
valid membership     70 frames, sha256 10a8b40b508698dfabe80f196c61b9b4563cde0a2171d5fe72ba5555b1dd699b
D0 / D1              동일 70 frame, 동일 9-point correspondence
correspondence       8 corner + centroid (centroid 제거 금지 조항 준수)
seed                 canonical solve 결과 재사용 (새 solvePnP 호출 없음)
training steps       0     optimizer steps 0
checkpoint           c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
```

[확인] 이번 harness 의 D0 reproj 23.1616px 는 canonical baseline 23.161629px 와 **일치**한다.
직전 screen 에서 centroid 를 뺐을 때 24.91px 였던 것과 달리, 이번에는 지시문대로
centroid 를 포함해 canonical 경로와 동일하다.
