# Table G — backbone 비교와 속도 (★ 다른 모집단)

★★ **이 표는 Table A/B/F 와 모집단이 다르다.** real n=128, 30 epoch 이고
   PAPER_EVAL 319 가 아니다.  같은 줄에 놓고 비교하면 안 된다.

source   challenge/yolo_pose_one_model/runs_arch_baseline/ARCHITECTURE_BASELINE_TABLE.md
         (동일 recipe·동일 evaluator 로 세 backbone 을 G38 합성만으로 30ep 학습)

## real n=128 (실제 성능)

```
model            scope    cbox  9kp med  9kp p90  gross20
------------------------------------------------------------
YOLOv8n-Pose     ALL     0.773    12.13    77.00    0.306
YOLO11n-Pose     ALL     0.836    14.46   104.59    0.387
YOLO26n-Pose     ALL     0.828    12.52    69.71    0.334
```

## 속도 (RTX 3080, batch1, imgsz640, warmup 30 / run 200)

```
model             lat med ms  lat p90 ms     FPS     params  GFLOPs
--------------------------------------------------------------------
YOLOv8n-Pose            5.18        6.08   193.2  3,142,182     8.7
YOLO11n-Pose            6.88        7.72   145.3  2,721,174     6.9
YOLO26n-Pose            8.68       10.28   115.2  3,043,704     7.6
```

★ 읽을 때 붙일 단서

- 합성 val 에서는 세 backbone 이 거의 같다(poseMAP 0.9006~0.9059).
  차이는 real 에서만 나고, 축마다 승자가 다르다 —
  cbox 는 11n(0.836), kp median 은 v8n(12.13), kp p90 은 26n(69.71) 이 최고다.
  **단일 승자가 없다.**
- 속도는 **RTX 3080 workstation** 값이다.  Jetson 값이 아니다.
- n=128 은 FT 누수 12 장을 뺀 셋이며 PAPER_EVAL 319 와 다르다.

Paper role: **appendix**.  상담에서는 '속도 이야기의 출발점' 으로만 쓴다.
