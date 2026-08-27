# FIXED_60EP_CONFIG_DIFF

recipe source = `runs_paper/yolo26n_paper_generic_v1_seed42/args.yaml` (실제 artifact를 읽음).
**기억으로 재구성하지 않았다** — 그랬으면 아래 증강 차이를 놓쳤다.

## ★ paper 60ep 와 5ep baseline 은 recipe 가 다르다

```
항목                  paper 60ep    5ep baseline
 --------------------------------------------
model           challenge/weights/pretrained_yolo/yolo26n-pose.pt/home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo26n-pose.pt
epochs                      60               5
cos_lr                    True           False
mosaic                     0.3             1.0
scale                     0.25             0.5
hsv_s                      0.5             0.7
hsv_v                     0.35             0.4
save_period                 10              -1
single_cls                True           False
```

따라서 FIXED 60ep 은 **paper 60ep recipe** 를 따른다 (5ep 것이 아니다).
epochs/name/project 은 당연히 다르고, 의미 있는 유일한 차이는 **data** 다.

## FIXED 60ep 에 잠근 값

```
model            /home/minjae/Documents/github/pallet-pose/challenge/weights/pretrained_yolo/yolo26n-pose.pt
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
seed             42
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
resume_from_5ep        = False
5ep_weight_as_init     = 금지 (pretrained yolo26n-pose.pt 에서 처음부터)
ORIGINAL_5EP_GATE      = FAIL  (보존)
ADAPTIVE_60EP_CONFIRM  = USER_AUTHORIZED
```
