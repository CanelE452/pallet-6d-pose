# s2 diffpnp full-pool + FULL7(f3 포함) 필터 결과

- weights: paper_s2_stageB net_epoch_0057. FULL7 = f1&f2&f3&f4&f5&f6&f7.
- TAU: {k: TAU.get(k) for k in FULL7[:6]} = {'f1':0.5,'f2':1.5,'f3':10.0,'f4':5.0,'f5':0.5,'f6':0.06}, f7=posdepth(bool).
- pass 오버레이: {domain}/pass/*.jpg (pred cuboid). 원본은 raw_data.

```
domain          frames   det>=6  FULL7 pass pass%(det)
-------------------------------------------------------
cad               1179       15           2      13.3%
night             1624      805         112      13.9%
noapril            188       75           7       9.3%
outside           2976     1242         392      31.6%
wood_indoor        136       93           0       0.0%
wood_outdoor        99       78           0       0.0%
-------------------------------------------------------
ALL               6202     2308         513      22.2%
```
