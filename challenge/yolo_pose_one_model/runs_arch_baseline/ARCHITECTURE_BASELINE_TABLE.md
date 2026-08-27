# ARCHITECTURE BASELINE — G38 synthetic-only, 30ep, 동일 recipe/evaluator

```
model                params  GFLOPs  synPose   cbox  9kp med  9kp p90  NIGHTtop1  FP/img@.4    FPS
------------------------------------------------------------------------------------------------
YOLOv8n-Pose      3,142,182     8.7   0.9058  0.773    12.13    77.00      0.357     0.0041  193.2
YOLO11n-Pose      2,721,174     6.9   0.9006  0.836    14.46   104.59      0.536     0.1629  145.3
YOLO26n-Pose      3,043,704     7.6   0.9059  0.828    12.52    69.71      0.500     0.0941  115.2
```

## synthetic (G38 val 1,998)
```
model             boxmAP50   boxmAP  poseMAP50  poseMAP  9kp med  9kp p90   cbox
------------------------------------------------------------------------------
YOLOv8n-Pose        0.9947   0.9347     0.9440   0.9058     2.88    10.00  0.997
YOLO11n-Pose        0.9947   0.9301     0.9398   0.9006     2.90    10.90  0.996
YOLO26n-Pose        0.9938   0.9279     0.9400   0.9059     2.28     8.88  0.994
```

## real n=128
```
model            scope    cbox  9kp med  9kp p90  gross20
------------------------------------------------------------
YOLOv8n-Pose     ALL     0.773    12.13    77.00    0.306
YOLOv8n-Pose     DAY     0.890    11.41    76.55    0.279
YOLOv8n-Pose     NIGHT   0.357    25.46    89.58    0.537
YOLO11n-Pose     ALL     0.836    14.46   104.59    0.387
YOLO11n-Pose     DAY     0.920    13.14    88.36    0.346
YOLO11n-Pose     NIGHT   0.536    29.73   137.51    0.633
YOLO26n-Pose     ALL     0.828    12.52    69.71    0.334
YOLO26n-Pose     DAY     0.920    11.86    59.70    0.300
YOLO26n-Pose     NIGHT   0.500    23.67   151.21    0.554
```

## NIGHT candidate  (★ cand/frame·wrong% 는 모델 간 직접 비교 금지)
```
model                any    top1   cand/f   wrong%   margin
------------------------------------------------------------
YOLOv8n-Pose       0.571   0.357     4.00    0.964   +0.035
YOLO11n-Pose       0.964   0.536    16.75    0.929   +0.048
YOLO26n-Pose       0.714   0.500     7.89    0.929   +0.315
```

## real negative n=2,689
```
model                  AP    AUROC   FPR@95  FP/img@.4  detrate@.4  neg p90
------------------------------------------------------------------------------
YOLOv8n-Pose       0.8673   0.9619   0.1164     0.0041      0.0041   0.0446
YOLO11n-Pose       0.4326   0.9377   0.1848     0.1629      0.1592   0.5584
YOLO26n-Pose       0.6460   0.9374   0.3020     0.0941      0.0915   0.3634
```

## efficiency (RTX 3080, batch1, imgsz640, warmup 30 / run 200)
```
model             lat med ms  lat p90 ms     FPS     params  GFLOPs
--------------------------------------------------------------------
YOLOv8n-Pose            5.18        6.08   193.2  3,142,182     8.7
YOLO11n-Pose            6.88        7.72   145.3  2,721,174     6.9
YOLO26n-Pose            8.68       10.28   115.2  3,043,704     7.6
```

## checkpoint SHA256
```
YOLOv8n-Pose     be9be2fb6c452f7ff9f4fef27ff96246c753333a8e33d00bf7c217a2b764b9a3
  pretrained     c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212
YOLO11n-Pose     1faa376f58b9567598b9397b7ffaf6aee9f40d519d26fc866c9a898793fae72e
  pretrained     869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0
YOLO26n-Pose     37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1
  pretrained     eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9
```

**VERDICT = ARCHITECTURE_BASELINE_TABLE_COMPLETE**

★ 승자 선정 아님. 60ep·s/m 확대 없음.
★ cand/frame·wrong% 는 v8/11(NMS) 과 26(end2end one2one) 의 후보 정의가 달라 직접 비교 금지.
★ negative FP/image 는 박스 개수 기반이라 NMS 영향을 받는다 — 프레임 단위 detrate 를 같이 볼 것.
★ 6D/PnP 는 Y0 정본 evaluator 가 2D 전용이라 제외(새 solver 금지).
★ NIGHT n=28, seed 1.