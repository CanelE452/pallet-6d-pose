# PURPOSE — E4, the capacity-matched control for E3

[소비처] 논문 §method 의 architecture 절. E3(SPLIT_LATE_2HEAD)를 최종 아키텍처로
  쓰면서 "왜 좋은가" 를 한 문장으로 쓸 수 있는지가 걸려 있다. 이 실험이 끝나면
  architecture search 를 종료하고 view-dependent scale 축으로 넘어간다.

[문장] "line 과 corner 는 서로 다른 late representation 을 필요로 한다" — 이 문장이
  참인지, 아니면 E3 의 이득이 그냥 corner 쪽 +5.0M 파라미터 때문인지 결정한다.
  후자면 그 문장을 쓰지 않는다.

[판단 지표] 사전등록. 두 seed 모두, paired frame bootstrap.
  CASE A  E3 > E4 : line 둘 다 A0 대비 ≤0.5% 열화 AND corner E3 가 ≥5% 우세
                    AND PATH-C R 또는 t 중 하나 ≥5% 우세·다른 축 ≤2% 열화
                    AND front_rear_shift 또는 affine_scale_isotropic 가 E3 쪽으로
          → LATE_FEATURE_SPECIALIZATION_SUPPORTED
  CASE B  E4 ≈ E3 : corner 차이 <3%, PATH-C R/t 각각 <3%, geometry 동률
          → CAPACITY_EXPLAINS_E3_GAIN, LATE_SPLIT_NECESSITY = NOT_ESTABLISHED
  CASE C  E4 > E3 → FULL_SPLIT_LATE_NOT_NECESSARY
  CASE D  E4 line 깨짐 → 먼저 control 설계 실패를 의심, E4_INVALID_CONTROL 로 폐기

## 이 control 이 분리하는 것과 못 하는 것

E3 가 E2 를 이기는 변화는 두 가지가 묶여 있다.

```
(a) corner branch 에 +5,014,912 trainable
(b) corner feature 를 frozen early(256ch)에서 새로 계산 (line 의 late 출력 128ch 이 아니라)
```

E4 는 (a)만 주고 (b)를 뺀다 — E3 의 파라미터 예산과 E2 의 feature 출처.

**분리 못 하는 것**: (b) 자체가 다시 두 가지다 — task-specific representation 이라는 것과,
256 early 채널에 접근한다는 것(E4 는 128ch 병목을 거친다). E4 는 이 둘을 못 가른다.
결과 해석에서 이 한계를 유지한다.

## 설계 근거 (wiring 으로 검증 완료)

```
파라미터   E3 extra 5,014,912  vs  E4 extra 5,015,168   +0.0051%  (기준 ≤2%)
step0      corner_capacity(F50) − F50 = 0.0  → E2 와 동일 출발 (zero-init residual)
line       E0 대비 max|diff| = 0.0  (E2·E3 와 동일)
gradient   L_line → corner 쪽 정확히 0.000e+00 / L_corner → line 쪽 정확히 0.000e+00
replay     20 step ×2, line logits diff 0.0
```

한 가지 편차: E4 의 block 은 128ch 에서 시작해 conv 가 5개(E3 는 256ch 에서 4개)라
receptive field 가 11 vs 9 다. 파라미터를 0.005% 로 맞추면서 표준 채널폭을 쓰려면
이 편차가 불가피했다. 기록하고 넘어간다.

## 범위 밖

- E3 재학습 (25k × 2 seed 이미 완료, historical fact 로 고정)
- 새 head·새 fusion·새 CIGM — architecture search 는 이 판정으로 종료
- real 전이, sealed set
