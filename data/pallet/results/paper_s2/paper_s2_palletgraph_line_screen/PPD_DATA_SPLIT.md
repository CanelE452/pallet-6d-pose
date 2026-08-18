# PPD data split lock — paper_4pallet_mask_v1 단독

## Audit (실측)

```
frames                10,000
mask_rle              10,000/10,000 (100%)
K / dimensions / R,t  100%
V=8                   100%   (truncation 0%)
mesh asset            4종 (Pallet_0~3)
dimension triples     621+ (연속 랜덤화, W 0.87~1.33 / D 0.87~1.33 / H 0.12~0.22 m)
hdri                  7종      floor 11종      background 1종(industrial)
camera_mode           6종 (top_down 2552, mid 1871, top_down_crop 1508,
                          close_crop 1459, far 1411, close_full 1199)
```

[확인] 지시문의 "asset 4종" 과 이전 보고의 "dimensions 621종" 은 **둘 다 사실**이다.
4 mesh 를 치수 랜덤화한 구조다.

## Group split

group key = `hdri | background | floor` (pose·GT 미사용) -> 77 groups.
hdri 별로 group 을 정렬해 round-robin 배정하여 hdri 편중을 줄였다.

```
split       frames   groups
train        3,039       23
val          1,045        8
untouched    5,916       46
train/val group overlap = 0
```

stratification (max |train - val| 비율차):

```
asset  0.019     mode  0.018     hdri  0.134     floor  0.240
```

[확인] floor 0.240 은 group key 자체가 floor 를 포함하기 때문에 생기는 구조적 한계다
(val 8 group 으로 floor 11 종을 고르게 만들 수 없다).  hdri 는 초기 0.349 에서
0.134 로 개선했다.

overfit32 = 32 frames, train 안에서 asset x camera_mode 를 커버하도록 결정적 추출.

## 무결성

- N87 membership 0 (구성상 다른 root)
- final-test membership 0
- 허용 root 는 `paper_4pallet_mask_v1` 하나뿐, hash 기록됨
- mask source 는 JSON `mask_rle` 뿐 (PNG 미사용)

## 필수 caveat

이 데이터는 **V=8 full-view 100%, truncation 0%** 다.  clean full-view 에서의
polarity extraction capability 만 평가할 수 있고, close-range / truncation
일반화를 이 데이터만으로 주장할 수 없다.
