# 5개 도메인 필터 통과 수 — GT-free 전체 AND (f1~f7, f3 flip 포함) — Stage B

- weights: paper_s2_stageB net_epoch_0057, squash-parity, flip=(W-1)-x
- FULL7 = f1&f2&f3&f4&f5&f6&f7 (GT-free 전체). DEPLOY6 = f3 제외(배포).
- f8/f9 = GT 필요라 배포 불가로 제외. purity = 통과 중 GT-good(corner_med<10px).
- ★f3(flip) = 모델 좌우 비대칭으로 정상도 대량 탈락(broken) → FULL7 급감 예상.

```
domain      N  det>=6  GTgood  FULL7 pass  purity    DEPLOY6 pass  purity
--------------------------------------------------------------------------------
outside    44      30      12          10     50%              28     39%
cad        11       0       0           0       -               0       -
manual     36      35      11           3     33%              32     31%
noapril     6       4       4           1    100%               4    100%
night      43      29       3           5      0%              26      8%
--------------------------------------------------------------------------------
ALL       140      98      30          19     37%              90     30%
```

해석: FULL7(f3 포함)은 f3 하나 때문에 통과가 급감(정상 프레임도 탈락) = f3 broken 확인. purity(통과 중 실제 정확 비율)는 도메인 base 품질 종속: outside/manual↑ night↓.
