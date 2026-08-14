# Paper-S1 all-frames overlays — 세션별 검출/신뢰도 요약

weights: `/home/minjae/Documents/github/pallet-pose/weights/paper_s1_maskaux/net_epoch_0065.pth`  
전처리: reflect-pad100 (near-field, official 아님)  
★ PnP 없음 — belief-peak 키포인트만. GT 없음(raw)  
★ 정확도 표시 불가 → det(검출 코너수) / peak(belief peak) 프록시  
★ challenge/palletobj 계열은 S1(paper-track)엔 unseen → 도메인갭  
★ cad near-field 미검출 다수 예상 [추정]

```
session                  N  det>=6   det%  det_mean  peak_med
--------------------------------------------------------------
capturepalletcad      1179     163   13.8      3.13     0.807
capture0403noapril     188     106   56.4      5.87     0.917
capturepallet02        298     168   56.4      5.72     0.827
capturepallet03        308     174   56.5      5.84     0.743
capturepallet04        192     132   68.8      6.06      0.61
capturepallet05        248     101   40.7      4.58       0.6
capturepallet08       1930     842   43.6      4.47      0.77
--------------------------------------------------------------
TOTAL                 4343    1686   38.8
```

new=4343  skip(existing)=0  bad=0  elapsed=3.0m
컬럼: det>=6=검출성공(코너 6+) 프레임수, det%=검출률, det_mean=평균 검출 코너수(/8), peak_med=mean belief peak 중앙값.
