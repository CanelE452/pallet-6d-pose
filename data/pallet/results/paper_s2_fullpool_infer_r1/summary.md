# s2 diffpnp full-pool inference (6 domains, manual 제외)

- weights: paper_s2_stageB net_epoch_0057 (diffpnp λ0.005, squash-parity)
- n_det = 검출 코너 수(>=6 검출). scores f1~f7 는 GT-free 필터 입력.
- wood dims=(0.8,0.59,0.14), 그 외 (1.1,1.3,0.12).

```
domain          frames   det>=6    det%
----------------------------------------
outside           2976     1454   48.9%
cad               1179        7    0.6%
noapril            188       73   38.8%
night             1624      936   57.6%
----------------------------------------
ALL               5967     2470   41.4%
```
