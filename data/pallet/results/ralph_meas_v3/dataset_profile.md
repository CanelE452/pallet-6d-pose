# dataset profile — 평가셋들이 무엇이 다른가 (GT 기반)

```
set                   n  dist m   elev°  width px  trunc%  reproj  rearClick%
------------------------------------------------------------------------------
outside_ft_p07       27    2.04   10.12       503      37    2.30          31
outside_ft_p09       36    2.85    2.36       313       6    1.05          31
outside_fv_CANON     22    3.66    3.38       269       0    0.68          25
outside_NONcanon     32    3.72    6.16       258       9    1.47          40
noapril_CANON        12    3.48   19.10       273       0    0.52          27
cad_CANON            22    1.68   31.71       547      23    1.80          50
night_ft_n08         17    2.59    5.89       419      41    1.48          49
night_ft_n09         25    2.70    5.95       333      28    2.20          47
night_fv_n05_07      43    2.60    6.04       384      23    2.42          49
------------------------------------------------------------------------------
```

앙각 분포 (프레임 수):
```
set                   0-5   5-10  10-20  20-90
----------------------------------------------
outside_ft_p07          0     12     14      0
outside_ft_p09         35      0      0      0
outside_fv_CANON       14      8      0      0
outside_NONcanon       12     20      0      0
noapril_CANON           0      0      9      3
cad_CANON               0      3      0     19
night_ft_n08            7      7      3      0
night_ft_n09            5     16      4      0
night_fv_n05_07         6     35      2      0
----------------------------------------------
```

세션 구성 / dims / split:
```
outside_ft_p07:  sessions={'capturepallet07': 27}
                 dims={'1.1x1.3': 17, '1.3x1.1': 10}  split={'eval': 27}
outside_ft_p09:  sessions={'capturepallet09': 36}
                 dims={'1.1x1.3': 28, '1.3x1.1': 8}  split={'eval': 36}
outside_fv_CANON:  sessions={'capturepallet08': 13, 'capturepallet02': 3, 'capturepallet05': 3, 'capturepallet03': 2, 'capturepallet04': 1}
                   dims={'1.1x1.3': 13, '1.3x1.1': 9}  split={'eval': 22}
outside_NONcanon:  sessions={'capturepallet08': 12, 'capturepallet03': 7, 'capturepallet04': 5, 'capturepallet02': 4, 'capturepallet05': 4}
                   dims={'1.3x1.1': 19, '1.1x1.3': 13}  split={'(none)': 29, 'train': 3}
noapril_CANON:  sessions={'capture0403noapril': 12}
                dims={'1.3x1.1': 12}  split={'eval': 12}
cad_CANON:  sessions={'capturepalletcad': 22}
            dims={'1.1x1.3': 18, '1.3x1.1': 4}  split={'eval': 22}
night_ft_n08:  sessions={'capturenight08': 17}
               dims={'1.3x1.1': 16, '1.1x1.3': 1}  split={'(none)': 17}
night_ft_n09:  sessions={'capturenight09': 25}
               dims={'1.3x1.1': 15, '1.1x1.3': 10}  split={'(none)': 25}
night_fv_n05_07:  sessions={'capturenight07': 16, 'capturenight06': 15, 'capturenight05': 12}
                  dims={'1.1x1.3': 16, '1.3x1.1': 27}  split={'(none)': 33, 'train': 10}
```
