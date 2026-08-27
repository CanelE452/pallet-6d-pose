# perturb_summary — PHASE B

```
CORNER_SPECIALIZATION            = True
LINE_SPECIALIZATION              = False
RANDOM_CONTROL                   = False
COMPLEMENTARY_EVIDENCE_SUPPORTED = False
```

**사전등록 gate 에 따라 여기서 멈춘다. attention-diversity loss 를 만들지 않는다.**

training 0 · 새 architecture 0. checkpoint 는 locked candidate
`screen_A1_CORNER_LINE_e3confirm25k_seed{1,2}/step_25000.pth`.
forward 경로는 `mh_screen` 의 것을 그대로 썼다 — 내부 정합 증거로, 무섭동
조건 `I0` 의 corner RMS 가 PHASE A 의 SPLIT_LATE 값과 **정확히 일치**한다
(seed1 0.7129, seed2 0.7243).

## 기하 (PURPOSE.md 에 사전 고정, 길이 단위 하나에서 유도)

```
r = 2 x CORNER_SIGMA = 4.0 cells = 32.0 px      (IMAGE 400 / GRID 50 -> 8 px/cell)
IC  8 개 GT corner 중심 반지름 r 인 disk 의 union
IE  12 개 projected edge 에서 거리 r 이내 band  MINUS  IC
IR  같은 프레임에서 IC 또는 IE 와 면적을 맞춘 random disk, 4 draw
연산자  마스크 내부만 Gaussian blur (sigma 8 px, kernel 33), 4 px feather
```

실측 마스크: IC 14,302 px · IE 10,488 px · IR/IC 0.997 · IR/IE 0.998~0.999.
IC∩IE soft 겹침은 IC 면적의 **1.96%** — feather ring 의 필연적 잔재이며,
IE 를 약간 corner-파괴적으로 만들어 `S_corner` 를 **보수적으로 낮춘다**.

## 주효과 (delta = 조건 − I0, median)

### seed 1
```
metric             I0        IC        IE       IR0       IR1       IR2       IR3
---------------------------------------------------------------------------------
corner_rms     0.7129   +6.2647   +0.0042   +0.0003   +0.0000   +0.0004   +0.0000
line_angle     2.2093   +3.1595   +0.0000   +0.0000   +0.0000   +0.0000   +0.0000
line_offset    0.9885   +2.1349   +0.0117   +0.0000   +0.0000   +0.0000   +0.0000
```
### seed 2
```
metric             I0        IC        IE       IR0       IR1       IR2       IR3
---------------------------------------------------------------------------------
corner_rms     0.7243   +6.1558   +0.0103   +0.0006   +0.0000   +0.0004   +0.0000
line_angle     2.3186   +3.7136   +0.0000   +0.0000   +0.0000   +0.0000   +0.0000
line_offset    1.0299   +2.0146   +0.0068   +0.0000   +0.0000   +0.0000   +0.0000
```

## 사전등록 특이성 점수

### seed 1
```
score              observed  relative     CI low    CI high   CI>0
------------------------------------------------------------------
S_corner            +4.9895    +6.999    +3.7808    +6.7857   True
S_line_angle        -2.6667    -1.207    -3.5451    -2.0335  False
S_line_offset       -1.8348    -1.856    -2.2461    -1.5175  False
```
### seed 2
```
score              observed  relative     CI low    CI high   CI>0
------------------------------------------------------------------
S_corner            +4.2088    +5.811    +3.3209    +5.2435   True
S_line_angle        -3.4776    -1.500    -4.0733    -2.7233  False
S_line_offset       -1.6013    -1.555    -1.9541    -1.3106  False
```

## 판정 1 — corner branch 는 corner 근방에 특이적으로 의존한다 `[확인]`

`S_corner` = +4.99 / +4.21, 기저 대비 +700% / +581%, 두 seed 모두 CI 가 0 배제.
IC 는 corner RMS 를 **+6.26 / +6.16** (기저 0.71 의 약 9 배) 올리는데 IE 는
**+0.004 / +0.010** 만 올린다. 면적통제 대조(IC vs 면적정합 IR)도 두 seed 모두
CI 가 0 을 배제한다(+5.92 / +5.31).

## 판정 2 — line branch 도 corner 근방에 지배된다 `[확인]`

`S_line_angle` = **−2.67 / −3.48** (음수), `S_line_offset` = −1.83 / −1.60.
즉 **edge interior 보다 corner 근방을 가릴 때 line 이 더 망가진다.**
IC 는 line angle 을 +3.16 / +3.71 (기저 2.2~2.3 의 약 1.5 배) 올린다.

따라서 **두 branch 는 상보적 증거를 쓰지 않는다.** 둘 다 corner 근방에
닻을 내리고 있다. 이는 기존 메모리 `attention-is-not-the-line-bottleneck`,
`five-class-line-cannot-identify-corner-12edge-can` 과 같은 방향이다.

## ★ 반드시 함께 읽을 검산 — median 은 이 출력에 대해 퇴화한다

line 출력은 **격자 argmax** 다. 작은 증거 변화는 argmax 를 못 움직여 delta 가
정확히 0 이 되고, 그런 프레임이 29% 라 median 이 0 으로 눌린다.
사전등록 gate 는 median 을 썼으므로 판정은 그대로 두지만,
**median 만 보고 "line 은 edge 를 안 본다" 고 읽으면 틀린다.**

민감한 통계로 다시 보면:

### seed 1
```
condition      angle 변화율  angle mean|d|     corner 변화율  corner mean|d|
----------------------------------------------------------------------
IC                 0.996          9.925          1.000          11.770
IE                 0.707          2.066          1.000           1.932
IR0                0.646          0.780          1.000           0.771
IR1                0.436          0.684          1.000           0.662
IR2                0.680          1.671          1.000           1.025
IR3                0.438          0.440          1.000           0.645
```
### seed 2
```
condition      angle 변화율  angle mean|d|     corner 변화율  corner mean|d|
----------------------------------------------------------------------
IC                 0.998          9.639          1.000          10.697
IE                 0.711          1.788          1.000           2.558
IR0                0.666          1.387          1.000           0.986
IR1                0.430          0.576          1.000           0.813
IR2                0.650          1.110          1.000           0.822
IR3                0.432          0.545          1.000           0.743
```

IE 는 line angle 을 **70.7% / 71.1%** 의 프레임에서 바꾸고 mean |Δ| 가
2.07 / 1.79 다. 면적을 맞춘 random(IR1/IR3)은 43.6% / 43.0%, mean |Δ|
0.68 / 0.58 이다. **line branch 는 edge interior 를 실제로 읽는다** — 다만
corner 근방보다 훨씬 약하게 읽는다. `[확인]`

## 설계상의 비대칭 (해석 시 감안할 것)

`IE = band − corner disk` 이므로 IE 는 **edge 의 끝점을 일부러 남긴다**.
반대로 IC 는 12 개 edge 전부의 **끝점을 파괴**한다. 따라서 IC 가 line 에
주는 타격에는 "corner 증거 제거" 와 "edge 끝점 제거" 가 섞여 있다.
`S_line` 이 음수인 것은 이 비대칭으로도 일부 설명된다 `[추정]`.
이 비대칭은 브리프가 지정한 마스크 정의에서 나온 것이고, 결과를 보고
정의를 바꾸지 않았다.

## 다음에 하지 말아야 할 것

```
attention-diversity loss    금지 (gate 미통과)
orthogonality loss          금지
새 router / 새 backbone     금지
```

`COMPLEMENTARY_EVIDENCE_SUPPORTED = False` 이므로 브리프대로 **원인 진단이
먼저**다. 지금 자료가 가리키는 원인은 "line head 가 edge 를 안 본다" 가 아니라
**"line head 가 edge 를 보되 corner 근방에 훨씬 크게 의존한다"** 이다.
