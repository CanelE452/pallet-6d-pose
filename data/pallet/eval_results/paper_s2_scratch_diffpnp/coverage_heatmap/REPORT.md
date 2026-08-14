# Stage B coverage/belief/mask 진단 — 5 frames

weights: /home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth
out: /home/minjae/Documents/github/pallet-pose/data/pallet/eval_results/paper_s2_scratch_diffpnp/coverage_heatmap

```
domain   fid                  tag        nd  pk8min pk8avg rearpk frontpk f2max f2avg segmax maskA/kpbbox
---------------------------------------------------------------------------------------------------------
noapril  1775201432466607872  FAIL(f3; W 6/8 0.01   0.67   0.88   0.45    0.97  0.35  1.00   0.90
night    1779449194023912448  pass       8/8 0.77   0.89   0.90   0.87    0.26  0.22  0.97   0.36
night    1779449196392532480  pass       8/8 0.44   0.66   0.69   0.64    0.40  0.28  0.37   0.00
night    1779449266426633216  pass       8/8 0.61   0.84   0.76   0.92    0.60  0.29  0.75   0.01
outside  1778651651444080384  pass       8/8 0.75   0.85   0.83   0.87    0.23  0.20  0.84   0.04
```

## 육안 관찰 (관찰 전용, 과결론 금지)

### FAIL frame (noapril 1775201432466607872)
- 실제 팔레트는 크고 납작하며 우측으로 길게 뻗음. keypoint cuboid(빨강)는 좌측 ~40%
  에만 붕괴(collapse) — c1(front-top-우), c2(front-bottom-우) 가 **미검출**(pk 0.02/0.01,
  weak). 나머지 6코너가 좌측에 뭉침 = "일부만 잡음"의 전형.
- belief: 검출된 6코너는 **날카로운 단봉**(f2 0.15~0.22) = confidently-wrong. f2max=0.97 은
  미검출 채널 c2 의 noise/noise 비일 뿐(진짜 2차peak 아님) → 강한 검출에 2차봉 없음.
- mask(raw sigmoid): keypoint 가 놓친 **우측까지 포함해 팔레트 전체**에 fire(=keypoint 보다
  넓음). 단 0.5 이진화하면 우측(cooler, <0.5)이 잘려 좌측만 남아 maskA/kpbbox=0.90 로 보임.

### PASS frames (night x3, outside x1)
- keypoint cuboid 가 팔레트 **전체**를 덮음(정상). belief 8/8 검출, f2 낮음(0.2~0.6).
- mask(raw): 전부 팔레트 좌측 **일부에만 작은 blob** 으로 약하게 fire (max 0.37~0.97,
  대부분 부분). 팔레트 전체를 덮지 않음. 0.5 이진화 시 maskA/kpbbox 0.00~0.36.

## 핵심: 두 신호가 "일부만 잡음"을 구분하는가?
- **belief 2차peak(f2): NO.** fail 의 붕괴 코너는 오히려 단봉(sharp)이고, 높은 f2 는
  미검출 채널의 노이즈 아티팩트. fail↔pass 를 못 가름. 진짜 fail 신호는 "특정 코너
  (front-우 c1/c2) 미검출 + 나머지 collapse" 이지 2차peak 가 아님.
- **mask vs keypoint 커버리지: 애매(경향은 있음).** raw mask 범위로 보면 fail 프레임만
  mask 가 keypoint cuboid 밖(팔레트 전체)까지 뻗고, pass 4장은 mask 가 keypoint 안쪽
  부분 blob. 즉 "mask 범위 > keypoint 범위" 가 under-cover 후보 신호가 될 여지. 그러나
  (1) 0.5 이진 metric 은 반대로 나옴(low-thresh raw 범위 필요), (2) n=1 fail, (3) fail 은
  밝은 실내 대형 팔레트라 mask 가 세게 fire → **밝기/크기 confound**, (4) mask 헤드는
  학습전용·real 도메인갭으로 pass 에서도 절반만 fire(신뢰낮음). 확언 불가.

## caveat
- 5장 극소표본(fail 1 / pass 4). mask 헤드는 원설계상 추론 미사용(학습전용)을 진단용 추출.
- squash-parity 전처리, belief/seg 50x50 격자 -> 원본 (W/50,H/50) 매핑.
