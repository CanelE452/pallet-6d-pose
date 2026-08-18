# s2 diffpnp full-pool inference (6 domains, manual 제외)

- weights: paper_s2_stageB net_epoch_0057 (diffpnp λ0.005, squash-parity)
- n_det = 검출 코너 수(>=6 검출). scores f1~f7 는 GT-free 필터 입력.
- wood dims=(0.8,0.59,0.14), 그 외 (1.1,1.3,0.12).

```
domain          frames   det>=6    det%
----------------------------------------
outside           2976     1242   41.7%
cad               1179       15    1.3%
noapril            188       75   39.9%
night             1624      805   49.6%
wood_indoor        136       93   68.4%
wood_outdoor        99       78   78.8%
----------------------------------------
ALL               6202     2308   37.2%
```
