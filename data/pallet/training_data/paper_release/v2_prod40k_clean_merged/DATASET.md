# v2_prod40k_clean_merged

`data/pallet/training_data/paper_release/v2_prod40k_clean_merged`

이미지는 저장소에 올리지 않는다. 이 카드는 폴더를 실측한 요약이며,
`scripts/data_prep/validate/gen_dataset_cards.py` 로 다시 만들 수 있다.

## 규모

```
파일      160,001
이미지    120,000
JSON       40,000
용량        28.2G
```

## 해상도

이미지 400장 표본.

```
640x480         183
960x540         120
720x480          50
560x560          47
```

## 라벨

JSON 400개 표본 (전체 40,000개).

```
[split]
  (none)                          400
[gt_source]
  (none)                          400
[class]
  pallet                          400
```

## 파일명 예

```
f17335.png
f31669.png
records.jsonl
```
