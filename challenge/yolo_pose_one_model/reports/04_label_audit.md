# 04 — Label audit

대상: `datasets/stage_a`  
생성: `python challenge/yolo_pose_one_model/scripts/audit_yolo_pose_labels.py --dataset datasets/stage_a`

```
train: images=73916 labels=73916 errors=0
val: images=4009 labels=4009 errors=0
train/val sha256 중복: 0
flip_idx involution: OK
```

판정: **PASS**