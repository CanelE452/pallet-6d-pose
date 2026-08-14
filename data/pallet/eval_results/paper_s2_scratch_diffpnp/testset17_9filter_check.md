# Stage B (paper_s2 net_epoch_0057) — 9-filter check on testset17

- weights: `weights/paper_s2_stageB/net_epoch_0057.pth`  (squash-parity decode, THRESH=0.3)
- set: cad11 + noapril6 = 17 (hand-anno GT, HIGH-ANGLE; memory: NOT real-representative)
- dims (W,D,H) = (1.1, 1.3, 0.12) m  (f7 posdepth / honest8)
- filters/helpers reused verbatim from `scripts/stage0/s1_cad_9filters.py` (M.apply_filter, M.TAU, ...)
- corner_med = order-free Hungarian median (dims-indep); honest8 = full-8 PnP reproj (dims (1.1, 1.3, 0.12))
- tau: f1_peak=0.5, f2_peak_ratio=1.5, f3_flip=10.0, f4_tta_stab=5.0, f5_rear_conf=0.5, f6_frsep=0.06, f9_bbox_iou=0.5; f7 posdepth(bool); f8 size-env(GT envelope)
- f8 size envelope (GT p2.5-p97.5): size=0.367-1.262, asp=1.27-2.77

**Detection: 4/17 have n_det>=6** (all noapril). 13/17 under-det (11 cad near-field + 2 noapril) = pre-filter stage, no f1..f9.

## Detected frames — f1..f9 pass(P)/fail(.) + GT contrast
```
frame                det     cm   hon8  f1 f2 f3 f4 f5 f6 f7 f8 f9  #pass strong weak
-------------------------------------------------------------------------------------
1775201411535812864    8    7.2    7.8   P  P  .  P  P  P  P  .  P     7     4    3
1775201432466607872    6    5.2   39.2   P  P  .  P  P  P  P  .  .     6     3    3
1775201442780546816    7    8.6    7.3   P  P  P  P  P  P  P  P  P     9     4    4
1775201447585014272    6    4.4    9.7   P  P  .  P  P  P  P  P  P     8     4    4

filter legend: f1=peak f2=peak_ratio f3=flip f4=tta_stab f5=rear_conf
               f6=frsep f7=posdepth f8=size_env f9=bbox_iou
strong(appearance/conf)=f1,f4,f5,f9 | weak(2D-geom)=f2,f6,f7,f8
```

## Detected frames — raw filter values
```
frame                 f1pk  f2rt  f3fl  f4tta  f5rr  f6frs  f7pd   f8sr  f8as  f9iou
------------------------------------------------------------------------------------
1775201411535812864   0.89 66.02  12.6    0.3  0.89  0.207    OK  0.295  2.68   0.86
1775201432466607872   0.81 36.84  27.2    0.5  0.81  0.484    OK  0.272  1.31   0.47
1775201442780546816   0.84 18.57   8.2    0.2  0.90  0.370    OK  0.607  2.29   0.84
1775201447585014272   0.86126.30  16.2    0.5  0.91  0.369    OK  0.468  2.28   0.83
```

## Under-det frames (n_det<6, pre-filter stage — no f1..f9)
```
frame                     dom  det   V
1778653004700882176       cad    2   5
1778653007153689088       cad    4   7
1778653010109701120       cad    2   3
1778653016761774848       cad    1   4
1778653018743942912       cad    2   5
1778653022742179328       cad    0   2
1778653030872511232       cad    5   8
1778653042732726272       cad    0   4
1778653056641040128       cad    4   7
1778653095042360064       cad    2   5
1778653097293288448       cad    2   4
1775201436599073280   noapril    4   7
1775201439118590208   noapril    5   6
```

## Per-filter pass count (over detected n)
```
filter             tau  n_pass   desc
----------------------------------------------------------------------
f1_peak            0.5       4   heatmap peak >=tau (min over 8 corners)
f2_peak_ratio      1.5       4   1st/2nd local peak >=tau (min over 8 corners)
f3_flip           10.0       1   L-R flip TTA mean dist <=tau px
f4_tta_stab        5.0       4   scale/brightness TTA pos std <=tau px
f5_rear_conf       0.5       4   rear(4-7) peak >=tau (min)
f6_frsep          0.06       4   pred depth-sep/cuboid-diag >=tau (not collapsed)
f7_posdepth    bool/env       4   solve_pose all 9 cam-z>0 & t_z>0
f8_size_env    bool/env       2   pred bbox size&aspect inside GT envelope
f9_bbox_iou        0.5       3   pred bbox vs GT bbox IoU >=tau
```

## 판정 (selectivity — N=4 detected, tiny sample, high-angle set)

핵심 대비: **corner_med(2D order-free)** vs **honest8(3D pose reproj, dims 1.1x1.3x0.12)**.
불량 프레임 ...432466 = corner_med **5.2px(우수)** 인데 honest8 **39.2px(W/D swap)**.
= 8코너 2D 위치는 정확, 3D 포즈 해석만 W/D 뒤바뀜. 2D 신호로는 원리적으로 안 보임.

강신호(appearance/conf) 거동 — f1/f4/f5:
- 3 정상 + 불량 ...432466 **전부 pass**. 불량 프레임도 peak0.81 / rear0.81 / tta0.5 =
  "confidently-wrong" (memory: rear=confidently-wrong 정합). → W/D swap 못 잡음.

약신호(2D geometry) 거동 — f2/f6/f7:
- 4장 전부 pass (불량 포함). f2 peak_ratio=36.8(확신 단봉), f6 frsep=0.484(붕괴 아님, 오히려 최대),
  f7 posdepth OK. = W/D swap 은 기하적으로 valid cuboid 라 2D 자기일관 신호에 **blind**.
  (memory: dims-free 2D 기하 PL 필터 원리 불가 실측 재확인.)

불량 프레임(...432466)을 실제로 reject 한 필터 = **f9 bbox_iou(0.47<0.5) / f8 size_env(sr0.272<envelope) / f3 flip(27.2>10)** 뿐.
- f9: 불량만 reject, 정상 3장 pass = 이 셋에서 유일한 clean 분리. ★단 f9/f8 은 **GT 참조**
  (bbox/size envelope) = unlabeled PL 에선 배포 불가. f9 저값은 n_det=6 truncated bbox 영향도 있어
  N=4 에서 우연 가능성 배제 못 함.
- f3 flip: 불량 잡지만 정상 3장 중 2장(...411535 12.6, ...447585 16.2)도 reject = 과다 reject(비선택적).
- f8: 불량 + 정상 ...411535(sr0.295, 작게 찍힘) 둘 다 reject = "화면서 작음"에 반응(swap 특이신호 아님).

정상 3장 통과: strong-combo(f1∧f4∧f5∧f9)=3장 전부 pass / 불량은 f9 에서 탈락 → 이 셋에선 완전 분리.
단 all-9 통과는 ...442780 1장뿐(나머지는 f3/f8/f9 중 하나 실패). f3 가 정상까지 깎아 all-9 는 과엄격.

## S1(2026-07-07) 대비 — 같은 경향 + 정제된 통찰
- 동일: 강=appearance/conf(f1/f4/f5) 가 detection-quality 판별, 약=2D geometry(f2/f6/f7)
  거의 무력. Stage B 도 그대로.
- 차이/정제: 이번 유일 confidently-wrong 실패는 **2D corner 우수(5.2px)한 W/D swap(3D pose)** →
  appearance-conf(f1/f4/f5)도 못 잡음(2D 는 완벽하니 신뢰도 높음). 즉 강신호는 "2D 코너 품질"에는
  강하나 "3D 포즈 모호성(W/D swap)"에는 직교 → 배포 가능 self-supervised 필터(f1~f7) 중 이 실패를
  잡는 건 **없음**. 잡는 건 GT 참조(f9/f8) 또는 잡음 큰 flip(f3)뿐.
- 결론(memory 정합): W/D swap 류 pose 모호성은 dims-known reproj-gate(치수 알려진 데이터) 또는
  GT 참조로만 걸러짐 — 2D 기하/2D 신뢰도 self-supervision 으론 원리적 불가.

## Caveat
- ★N=4 검출 극소표본, honest8=39.2 불량 = 단 1장. f9 의 clean 분리는 우연 배제 불가.
- 이 17장 = 고앙각 손어노셋(memory: real 대표 아님, 저앙각 rear-collapse 레짐 회피). 여기 수치로
  real PL 필터 선택성 일반화 금지.
- squash-parity 추론(Stage B 학습 정합). GT10px good 임계 취약 — corner_med 는 order-free 라
  dims-indep, honest8 만 dims(1.1x1.3x0.12) 의존.

