# Corner 0,1 Spike Diagnosis (B2, reflect-pad, per-channel)

records=519 (real=98 syn=421), spike threshold=15.0px per-corner.

convention: 0,1=top-front (진단대상), 2,3=bottom-front (대조군), 4,5=top-rear, 6,7=bottom-rear.

## 진단1 — per-corner err (channel-i, px)
```
corner    real_med  real_p90  real_n   syn_med   syn_p90   syn_n
----------------------------------------------------------------
0 TF         11.05     20.84      96     24.98    215.38     409
1 TF          9.89     48.35      90     17.73    238.14     403
2 BF*        14.02     51.54      90     19.38    240.32     402
3 BF*        11.15     21.01      95     24.63    217.58     403
4 TR         20.91      47.9      96     28.76    240.63     415
5 TR         14.33     56.12      98     45.06    212.71     419
6 BR         17.23     58.88      98     44.74    215.22     419
7 BR         20.22     39.62      96     29.29    242.16     415
```
(TF=top-front 진단대상, BF*=bottom-front 대조군)

## 진단1b — 0,1 보이는데 튀나 (vis01 & per-corner>spike)
```
set        n_vis01   n_spike01  n_spike23_ctrl
real            88          41              51
syn            390         240             229
```

## 진단2 — 합성 vs real (e01 mean per frame, px)
```
metric               n    median       p90
real e01            98     11.75     42.52
syn e01            421     30.54    217.88
real e23 ctrl       98     13.26     38.74
syn e23 ctrl       421     24.63    221.07
```

## 진단4 — 180° yaw symmetry swap (B 검증) — real/syn 분리
```
REAL: n_spike01=41 swap_improves_2px=1 frac=0.024
SYN : n_spike01=240 swap_improves_2px=147 frac=0.613 (gross_broken e01>100: 127 → 신뢰X)
REAL examples (dom fid e01_orig e01_swap):
  outside     1778651569891693056      34.3    215.3
  outside     1778651571906584064      21.6    226.1
  outside     1778651579432408064      55.0    212.2
  outside     1778651583329453568      46.2    197.6
  outside     1778651585345475584      30.5    214.4
  outside     1778651587361112320      31.9    223.7
  outside     1778651607668742912      40.9    210.9
  outside     1778651609885579008      49.1    196.7
  outside     1778651611935350016      52.4    199.8
  outside     1778651641769099008      29.7    218.2
```

## 진단C — GT front-face 사다리꼴 비율 (1.0=직사각형)
```
spike01    : median=1.01 p90=1.07 n=281
non-spike  : median=1.01 p90=1.03 n=197
```

## 진단3 — floating/두께
```
{"n": 98, "height_m_set": [0.11], "note": "팔레트 납작(height≈0.11m)→top/bot 코너 이미지상 근접→0,1↔2,3 구분 단서 약함(depth 붕괴 취약). top 코너는 deck 표면 위라 '허공'은 아님."}
```

## 판정 (A/B/C) — 자동 가이드

- real 0,1 med≈10.5 vs 2,3 med≈12.6; syn 0,1 med≈21.4
- 합성에서도 0,1 튐 → 구조/convention 문제(A·B 쪽). sim2real 전이 단독 원인 아님.
- (B) REAL swap 개선 2% 뿐 (swap 후 오히려 악화) → 180° 대칭 꼬임 아님. (syn frac=0.613 은 gross-broken 오염 artifact, 무시.)
- (A) 위 B·C 가 약하고 0,1 이 보이는데도 per-corner 만 크면 → box corner 시각단서 약함. surface-point 키포인트 검토.

★ 표본 작으면 예비. overlay 로 GT 정상/예측만 튐 여부 눈검증 필수.
---

## ★ 정정된 핵심 결론 (auto-verdict 보완 — 손수 분석)

**전제 자체가 틀렸다: 0,1 은 특별히 튀지 않는다. 진짜 문제는 REAR(4-7).**

REAL per-corner spike-rate (err>15px, detected corner 기준):
```
0TF 24%  1TF 28% | 2BF 43%  3BF 25% | 4TR 61%  5TR 45%  6TR 61%  7BR 69%
FRONT(0-3) 평균 30%  vs  REAR(4-7) 평균 59%
```
- 0,1(top-front)은 8 코너 중 **가장 정확한 축**에 든다. 2(bottom-front)가 오히려 43%.
- domain 별 per-corner median(px)도 동일: 0,1 은 7~15px, 4~7 은 14~36px.

진단2(합성) 정정: pooled syn 0,1=21px 는 **gross-broken 프레임(e01>100px) 127/240**
(주로 v3 held-out=challenge palletobj, 어려운 occlusion/truncation)에 오염된 값.
healthy syn(전체코너 med<20px) 만 보면 0,1=10~11px = 나머지 코너와 동일, 특이성 없음.

진단4(B) 정정: real swap 개선 2%(1/41), swap 후 30→215px 로 악화 → 180° 대칭 꼬임 아님.
pooled 53%/syn 61% 는 broken 프레임의 무작위 artifact.

진단C: spike vs non-spike GT 사다리꼴 비율 차이 없음(1.01 vs 1.01) → C(원근 오해) 아님.

진단3: 팔레트 height=0.11m 로 납작 → top/bot 코너가 이미지상 근접 → flat/edge-on
뷰에서 depth 붕괴 취약. top 코너(0,1,4,5)는 deck 표면 위라 '허공'은 아님.

overlay 눈검증(zoom_spike_* vs zoom_ok_*): 같은 팔레트라도 **저앙각/edge-on** 뷰에서
예측 cuboid 의 **깊이(front↔rear)가 붕괴**(red 큐보이드 납작) — 0,1 만이 아니라 큐보이드
전체, 특히 rear 가 무너짐. 앙각이 조금만 높아 top face 가 보이면 전 코너 정확.

### A/B/C 판정
- **B (180° 대칭 꼬임): 기각.** real swap 2% 만 개선, swap 시 악화.
- **C (원근 오해): 기각.** spike/non-spike GT 사각형 비율 동일.
- **A 변형 채택, 단 "0,1" 가 아니라 "REAR/flat-view depth":** box-corner 의 깊이
  단서가 flat/edge-on 에서 약해 **rear 코너**가 무너진다. 0,1(front-top)은 영상에
  직접 보여 안정적. → surface-point 키포인트가 도움될 수 있는 건 **rear/depth**
  쪽이지 0,1 이 아니다.

### 처방
1. "0,1 surface-point 재생성" 의 전제(0,1 이 튄다)는 데이터로 부정됨 → **0,1 만을
   겨냥한 재생성은 헛수고.** 비싼 재생성 전 이 결론으로 방향 재설정.
2. 진짜 레버는 (a) **저앙각/edge-on + rear 코너** 커버리지·표현, (b) flat 물체
   depth 붕괴 완화. rear 를 surface-point(예: deck 뒷모서리 위 실표면점)로 바꾸는 건
   시험가치 있음 — 단 front 가 아니라 rear/깊이축 대상.
3. ⚠ 표본: real spike01 n=41(소표본), syn pool 오염 큼 → "예비". 다만 front<rear
   패턴은 4개 real 도메인 전부 일치해 방향성은 견고.
