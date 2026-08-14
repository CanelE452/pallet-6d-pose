# STAGE22 PART C — coord vs control paired ablation (pilot, B2 init +8ep, epoch_size6000)

best ckpt: control=net_epoch_0085.pth coord=net_epoch_0085.pth
real inference=reflect-pad(pad=100). GT=projected_cuboid[:8]. real same-idx per-corner.

## control real summary (N=145, det=100 rate=0.69)
```
front_med=11.65(n100)  rear_med=19.79(n100)  full8_med=16.34(n87)
rear good(<10)=0.04  rear gross(>20)=0.49
V=8(n91): front=11.51 rear=19.71
elev bins (front/rear med):
     <3 n= 36 front=11.43 rear=21.34
    3-8 n= 58 front=11.69 rear=19.94
     8+ n=  6 front=12.58 rear=14.48
```
## coord real summary (N=145, det=100 rate=0.69)
```
front_med=13.67(n100)  rear_med=17.63(n100)  full8_med=15.76(n85)
rear good(<10)=0.12  rear gross(>20)=0.45
V=8(n91): front=13.46 rear=17.64
elev bins (front/rear med):
     <3 n= 35 front=13.31 rear=20.18
    3-8 n= 59 front=13.88 rear=17.3
     8+ n=  6 front=14.44 rear=11.43
```

## PAIRED (same-frame control vs coord; delta<0 = coord 개선)
```
  front: n=97 control=11.71 coord=13.63 delta=+0.78 improve=20 worse=42
   rear: n=97 control=19.86 coord=17.64 delta=-0.81 improve=45 worse=23
  full8: n=84 control=16.34 coord=15.68 delta=-0.07 improve=25 worse=18

rear paired by elev bin:
     <3: n=35 control=22.24 coord=20.18 delta=-0.75 imp=14 wrs=5
    3-8: n=57 control=19.86 coord=17.62 delta=-1.28 imp=29 wrs=17
     8+: n=5 control=12.42 coord=11.65 delta=-0.77 imp=2 wrs=1
```

## 병리진단 — coord rear 채널 hard-argmax↔soft-argmax μ 거리 (belief px)
```
rear  med=0.95 (n100)
front med=0.73 (n100)
hard-argmax peak↔soft-argmax μ 거리(belief px). rear>>front 면 soft-argmax가 rear에서 원거리 false-peak 견인.
```

## 판정 (task 3-way)
- (ii) rear 무변화 delta=-0.8px → 슬라이드 '표준기법 채택(Integral Pose Regression 계열)'으로 낮출 근거 + ablation 숫자 확보.

★ 페어 N 명시. real 소표본=예비. loss-ratio proxy(≠gradient) 로 λ 캘리브(0.24, ~7.5%).