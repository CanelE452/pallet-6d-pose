# MASK DATASET USAGE — paper_4pallet_mask_v1

이번 실행에서 이 root 는 **search prior 유도**에만 사용됐다.  학습은 하지 않았다 (G2 미완).

## Audit (실측)
```
frames(json)              10,000
mask_rle                  100% (sample 1500/1500)
dimensions_m / pose / K   100%
projected_cuboid(+centroid) 100%
V=8 full-view             100%   (truncation 0%)
distinct dimensions       621+ (연속 랜덤화)
```

## ⚠ 지시문과의 불일치 1건

지시문 H1 은 'asset 4종' 을 조건으로 두었으나, 실제 `dimensions_m` 은 **621종 이상**으로
연속 랜덤화되어 있다 (W 0.87~1.33 / D 0.87~1.33 / H 0.12~0.22 m).
4개 메시를 치수 랜덤화한 것으로 보이나 [추정], JSON 만으로는 asset 정체를 셀 수 없다.

## 필수 caveat

이 dataset 은 **V=8 full-view 100%, truncation 0%** 다.  따라서 clean full-view 에서의
line extraction 만 검증할 수 있고, **close-range/truncation 일반화를 이 데이터만으로 주장할 수 없다**.

## 이번 사용 내역
- search prior (tz/tx/ty/tilt p1..p99) 유도: 3,000 frame 샘플
- tz search range 1.481..4.677 m
- 학습: 없음
- 데이터 파일 수정: 0건

