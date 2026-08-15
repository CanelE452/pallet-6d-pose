# Paper-S1 on cad(22): 9 PL accept filters (each INDIVIDUAL)

- weights: `weights/paper_s1/paper_s1_maskaux/net_epoch_0065.pth`
- data: cad manual GT, N=22 | detected(>=6 corner)=4 | GT-good(corner_med<10px)=0/22
- inference: reflect-pad100 (near-field 검출 확보; NOT official eval)
- precision = 통과 프레임 중 corner_med<10px 비율 (order-free Hungarian median)
- 모든 필터 공통 전제: 코너>=6 검출 (PL 후보). 검출 안 된 프레임은 통과 불가.
- ★ N=22 극소 + cad=paper-track S1 엔 unseen(도메인갭). 과결론 금지.

```
filter             tau  n_pass  pass%  good  precision   desc
--------------------------------------------------------------------------------------------
f1_peak            0.5       3    14%     0      0.000   heatmap peak >=tau (min over 8 corners)
f2_peak_ratio      1.5       4    18%     0      0.000   1st/2nd local peak >=tau (min over 8 corners)
f3_flip           10.0       1     4%     0      0.000   L-R flip TTA mean dist <=tau px
f4_tta_stab        5.0       4    18%     0      0.000   scale/brightness TTA pos std <=tau px
f5_rear_conf       0.5       4    18%     0      0.000   rear(4-7) peak >=tau (min)
f6_frsep          0.06       4    18%     0      0.000   pred depth-sep/cuboid-diag >=tau (not collapsed)
f7_posdepth          -       4    18%     0      0.000   solve_pose all 9 cam-z>0 & t_z>0
f8_size_env          -       1     4%     0      0.000   pred bbox size&aspect inside GT envelope
f9_bbox_iou        0.5       2     9%     0      0.000   pred bbox vs GT bbox IoU >=tau
```

## Sweep (score-based filters)
```
f1_peak:
    tau=0.3    n_pass=4   good=0   precision=0.000
    tau=0.4    n_pass=4   good=0   precision=0.000
    tau=0.5    n_pass=3   good=0   precision=0.000
    tau=0.6    n_pass=1   good=0   precision=0.000
f2_peak_ratio:
    tau=1.2    n_pass=4   good=0   precision=0.000
    tau=1.5    n_pass=4   good=0   precision=0.000
    tau=2.0    n_pass=4   good=0   precision=0.000
f3_flip:
    tau=5.0    n_pass=0   good=0   precision=n/a
    tau=8.0    n_pass=0   good=0   precision=n/a
    tau=10.0   n_pass=1   good=0   precision=0.000
    tau=15.0   n_pass=3   good=0   precision=0.000
f4_tta_stab:
    tau=3.0    n_pass=4   good=0   precision=0.000
    tau=5.0    n_pass=4   good=0   precision=0.000
    tau=8.0    n_pass=4   good=0   precision=0.000
    tau=10.0   n_pass=4   good=0   precision=0.000
f5_rear_conf:
    tau=0.3    n_pass=4   good=0   precision=0.000
    tau=0.4    n_pass=4   good=0   precision=0.000
    tau=0.5    n_pass=4   good=0   precision=0.000
    tau=0.6    n_pass=2   good=0   precision=0.000
f6_frsep:
    tau=0.03   n_pass=4   good=0   precision=0.000
    tau=0.05   n_pass=4   good=0   precision=0.000
    tau=0.08   n_pass=4   good=0   precision=0.000
f9_bbox_iou:
    tau=0.3    n_pass=4   good=0   precision=0.000
    tau=0.5    n_pass=2   good=0   precision=0.000
    tau=0.7    n_pass=1   good=0   precision=0.000
```

## 판정
- precision>=0.7 로 통과시키는 단일 필터 없음.
- ★ 근본 원인: base(S1)가 cad 22장 중 corner_med<10px 프레임 0개 (검출>=6 도 4/22뿐, best corner_med≈11px). good PL 자체가 존재하지 않으므로 어떤 필터도 precision>0 불가 — 필터 실패가 아니라 base 천장 문제. (memory: 필터 천장=base 코너 정확도)
