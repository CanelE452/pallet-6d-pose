# Oracle O0 실패 3/86 원인 분해

```
frame                              domain  trunc  valid_corners  gt_upright  margin
night:1779449185084437248          night   False  8              True        0.0156
night:1779449198527179520          night   False  8              True        0.0122
night:1779449320806001408          night   False  8              True        0.1569

정상 프레임 margin: median 0.3275   p10 0.0673
```

## 분류: **E — candidate energies 가 사실상 tie**

- [확인] 3/3 모두 **night 도메인**이고 non-truncated 이며 corner 8개가 모두 보인다.
- [확인] 올바른 upright 후보가 **모두 존재**했다 (`gt_upright_available` True).
- [확인] 두 건의 margin(0.0156, 0.0122)은 정상 p10(0.0673)보다도 작다 = 명백한 tie.
  세 번째(0.1569)도 정상 median 의 48% 수준이다.
- [확인] **F(evidence 생성 오류) 0건, G(projection/class mapping 오류) 0건**.
  따라서 target/evidence 코드를 수정할 필요가 없고, oracle gate 를 다시 돌리지 않는다.
- [추정] 야간이라 top/base edge 의 image gradient 가 약해 두 polarity 의 support 가
  비슷해지고 에너지가 붙은 것으로 보인다.

## 판정

A~E 범주이므로 **oracle ceiling 한계**로 기록한다.  실패 3건을 제외하거나 숨기지 않았다.
