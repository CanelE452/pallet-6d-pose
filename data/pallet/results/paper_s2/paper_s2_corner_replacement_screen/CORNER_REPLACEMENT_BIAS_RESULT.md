# Signed bias / tail / PnP

C1-refined 는 gate 붕괴(g~1e-9)로 C1-base 와 수치적으로 동일하므로
(최대 차이: corner error 3.3e-08 px), 아래에서 의미 있는 대비는 C0 vs C1-base 다.

```
arm            F2 far   signed   F2 near  near     >20px  >50px  PnP    reproj    yaw
               median   bias     median   median   tail   tail   /87    median    median
────────────────────────────────────────────────────────────────────────────────────────
C0 (ep57)       44.59    20.58     7.86    6.88     203    120    70    23.16     6.03
C1-base         44.93    19.91     6.82    7.67     215    137    69    25.57     5.97
C1-proposal    160.34    71.57    16.49   18.97     462    387    87   108.30    42.59
C1-refined      44.93    19.91     6.82    7.67     215    137    69    25.57     5.97
```

- [확인] signed bias 만 20.58 → 19.91 (-3.3%) 로 줄었다.  기준은 -20% 였다.
- [확인] >50px tail 은 120 → 137 로 **늘었다**(기준 -20%).
- [확인] reproj 23.16 → 25.57 (+10.4%), PnP 70 → 69.
- [확인] F2 paired far error: improved 6 / worsened 18 (기준 improved > worsened).
- [확인] weak corner(base peak<0.1) 115 → 99, confident-wrong(peak>=0.5 & err>20)
  181 → 170 으로 소폭 줄었으나 tail 과 PnP 는 악화했다.
- [확인] C1-proposal 의 PnP 87/87 은 정확도가 아니라 **항상 8 점을 내놓기 때문**이다
  (far median 160px, yaw 42.6°).

[확인] synthetic L_DOPE 는 0.001088 → 0.001018 로 계속 내려갔다.
**synthetic 개선이 real N87 로 전이되지 않았다.**
