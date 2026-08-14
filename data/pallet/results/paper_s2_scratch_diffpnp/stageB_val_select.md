# stageB synthetic-val composite selection

val=full (n=1500, V8=1363)  weights_dir=/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB
guard tol: front+1.5px det-5.0 good-8.0 gross+5.0
primary=rank(rear)+rank(honest8) among guard-pass, tie=corner

```
ep    rear  honest8 front corner  det   good  gross  guard  best
-----------------------------------------------------------------
45      8.8     8.9   8.5    8.3  86.5  63.0    8.5     Y
48      8.2     8.6   8.3    8.0  88.2  65.5    8.6     Y
51      8.5     8.8   8.2    8.2  87.9  63.7    7.7     Y
54      8.4     8.5   8.1    8.0  88.8  65.7    8.1     Y
57      8.2     8.4   8.0    8.0  88.9  67.0    7.5     Y  <==
```

**BEST = epoch 57** -> `/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth`
rear=8.2 honest8=8.4 front=8.0 corner=8.0 det=88.9 good=67.0 gross=7.5
