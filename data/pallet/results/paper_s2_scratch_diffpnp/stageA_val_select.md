# stageA synthetic-val composite selection

val=full (n=1500, V8=1363)  weights_dir=/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageA
guard tol: front+1.5px det-5.0 good-8.0 gross+5.0
primary=rank(rear)+rank(honest8) among guard-pass, tie=corner

```
ep    rear  honest8 front corner  det   good  gross  guard  best
-----------------------------------------------------------------
3      11.3    11.7  11.2   10.8  60.5  48.2   17.3     .
6      10.1    10.1   9.8    9.7  72.5  53.9   13.1     .
9       8.6     8.7   8.6    8.5  77.1  62.4    8.7     .
12      8.6     8.7   8.5    8.5  82.8  61.9    9.8     .
15      8.9     9.2   8.8    8.7  83.6  59.8   10.4     Y
18      9.2     9.5   9.2    9.0  86.3  57.9   11.4     .
21      8.6     8.7   8.9    8.4  83.4  62.0    9.0     .
24      8.9     9.0   8.8    8.6  87.1  60.8    9.5     Y
27      8.6     9.0   8.5    8.3  86.1  62.4    9.0     Y
30      8.5     8.7   8.4    8.2  86.3  64.1    8.3     Y
33      8.7     8.8   8.7    8.3  86.9  63.4    9.5     Y
36      8.7     8.6   8.4    8.4  82.1  62.9    9.0     .
39      8.6     8.9   8.6    8.4  87.1  62.9    9.0     Y
42      8.1     8.3   8.0    7.8  87.7  66.6    8.7     Y  <==
45      8.8     9.1   8.6    8.5  88.1  61.4    9.3     Y
48      8.6     8.7   8.4    8.3  87.9  63.6   10.3     Y
51      8.4     9.1   8.6    8.3  86.9  63.4   10.1     Y
54      9.0     9.1   8.8    8.6  87.1  61.4   10.2     Y
57      8.9     9.1   8.6    8.6  88.5  62.0    9.9     Y
60      9.0     9.3   8.7    8.7  87.9  61.3   11.2     Y
```

**BEST = epoch 42** -> `/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageA/net_epoch_0042.pth`
rear=8.1 honest8=8.3 front=8.0 corner=7.8 det=87.7 good=66.6 gross=8.7
