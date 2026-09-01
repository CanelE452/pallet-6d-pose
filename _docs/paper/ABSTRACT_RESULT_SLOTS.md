# Abstract result slots

초록에 넣을 수 있는 값과, 아직 넣을 수 없는 값을 구분한다.

```text
Strongest baseline        Synthetic-only YOLO26n-Pose (R0)
Primary metric            supervised keypoint location median px (PAPER_EVAL)
Baseline value            4.420 px
Proposed value            4.180 px
Improvement X             5.4 %
Worst-condition before    — px
Worst-condition after     — px
YAW_RESULT_SLOT           BLOCKED
```

`YAW_RESULT_SLOT = BLOCKED` 이므로 초록의
"reduces median yaw error by [Y]" 문장은 **사용할 수 없다**.
다른 2D metric 을 yaw 라고 바꿔 쓰지 않는다.

pose evaluator 가 READY 가 되면 R0~R5 를 같은 evaluator 로 다시 배치 평가하고
이 파일과 M1/M2/M3/M5 의 pose 열을 함께 갱신한다.
