# SEED42_VS_SEED43_CONFIG_DIFF

seed43 은 seed42 의 **실제 args.yaml 을 복제**해 만들었다. 기억으로 재구성하지 않았다.

## 의도한 차이

```
seed        42 -> 43
name/save_dir  run 이름만
```

## 그 외 전 항목 (seed42 == seed43)

```
model            /home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo26n-pose.pt
data             /home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/datasets/broad40k_fixed/data.yaml
epochs           60
batch            32
imgsz            640
optimizer        SGD
lr0              0.01
lrf              0.01
momentum         0.937
weight_decay     0.0005
warmup_epochs    3.0
cos_lr           True
mosaic           0.3
close_mosaic     10
fliplr           0.0
flipud           0.0
scale            0.25
translate        0.1
hsv_h            0.015
hsv_s            0.5
hsv_v            0.35
deterministic    True
patience         0
workers          4
amp              True
save_period      10
single_cls       True
multi_scale      False
cache            False
device           0
rect             False
dropout          0.0
val              True
plots            True
```

```
resume_from_seed42 = False   (pretrained yolo26n-pose.pt 에서 clean start)
data manifest      동일 (broad40k_fixed)
label manifest     동일 (sha 94804a10803c3156)
```
