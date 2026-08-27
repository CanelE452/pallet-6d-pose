# BROAD_APPEARANCE_V2 — 후보

## 왜 appearance 인가 (근거)

```
luma x failure_type (DEV 161)
  dark<60      n=35   NO_BOX 0.43  KP_BAD 0.34  GOOD 0.23
  dim60-100    n=69   NO_BOX 0.28  KP_BAD 0.16  GOOD 0.55
  mid100-140   n=57   NO_BOX 0.19  KP_BAD 0.11  GOOD 0.67
```
어두울수록 단조롭게 나빠진다. night 세션 luma median 48~52 로 DEV 최저.

## ★ 그러나 geometry 와 얽혀 있다 — 분리했다고 주장하지 않는다

```
elevation x luma (CHALLENGE 105)
  elev<8  & bright>=80   n=41   NO_BOX 0.46  GOOD 0.32
  elev<8  & dark<80      n=30   NO_BOX 0.63  GOOD 0.20
  elev>=8 & bright>=80   n=23   NO_BOX 0.00  GOOD 0.70
  elev>=8 & dark<80      n=11   NO_BOX 0.09  GOOD 0.18
```
**저앙각이면 밝아도 NO_BOX 가 0.46 이다.** 어둠만의 문제가 아니다.
반대로 `elev>=8 & dark` 에서는 NO_BOX 0.09 인데 KP_BAD 가 0.73 으로 뛴다 —
어둠은 **검출보다 키포인트 정밀도**를 깎는다.

그리고 target 세션과 night 세션은 **같은 물체**다. 세션 수준에서 geometry 와
appearance 가 얽혀 있어 이 데이터로 인과를 가를 수 없다.

## 축

```
조명      bright indoor / dim indoor / outdoor day / outdoor night
색온도    혼합, local strong highlight, deep shadow
센서      noise / motion blur / defocus / exposure / white balance / low contrast
```
real DEV 이미지를 배경이나 텍스처로 **복사하지 않는다.** real 통계는 구간을
정하는 development reference 로만 쓴다 (real luma p50 ~123, 야간 세션 ~48).

## 후보

```
후보              photometric strata   frame     real DEV support gain        cost   위험
A_CONSERVATIVE     +3 (야간·저조도)    [보류]   night 셋(n=42)만 겨냥        낮음   generic 보호는 쉬움
A_BALANCED         +6 (조명+센서)      [보류]   dark/dim 전반 + blur/noise   중간   가장 무난
A_BROAD            +10 (전 축)         [보류]   전 구간                      높음   generic 분포 희석 위험
```
