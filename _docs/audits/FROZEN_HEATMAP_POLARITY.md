# Frozen heatmap polarity re-ranking — FAIL

학습 없음.  기존 ep57 belief cache(stage 4/5/6, 동일 가중 1/3)의 **spatial softmax**
분포로 후보를 재순위했다.  raw peak 를 confidence 로 쓰지 않으므로 additive offset 에
불변이고, 무응답 map 은 거의 균등분포가 되어 강한 투표를 하지 못한다.

## 결과 (n=86)

```
지표                    값        기준            판정
inversion              26/86     <=10            FAIL
signed_rot>90°         0.314     <=0.10          FAIL
point-fail correct     8/17      >=12            FAIL
indexed reproj median  16.48px   <=46.4          PASS
  (155.6px 대비 89% 감소, baseline 23.16px 보다도 낮음)
yaw median             1.81°
corner_sym median      0.158 m
undecidable            0
```

## 해석

- [확인] heatmap 은 **reproj 를 극적으로 낮췄다**(155.6 -> 16.5px).  즉 후보 재순위 자체는
  작동하고, 잘못된 후보를 상당수 걸러낸다.
- [확인] 그러나 **polarity 정확도는 70%** 에 그친다.  특히 point-fail 프레임에서 8/17 로,
  point 가 약한 곳에서는 heatmap 도 약하다 — 같은 feature 에서 나오므로 당연하다 [추정].
- [판정] **Frozen heatmap PPD: REJECT**.  가장 싼 해결책이지만 충분하지 않다.
