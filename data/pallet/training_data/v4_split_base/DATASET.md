# v4_split_base

`data/pallet/training_data/v4_split_base`

이미지는 저장소에 올리지 않는다. 이 카드는 폴더를 실측한 요약이며,
`scripts/data_prep/validate/gen_dataset_cards.py` 로 다시 만들 수 있다.

## 규모

```
파일        8,000
이미지      4,000
JSON        4,000
용량         1.8G
```

## 해상도

이미지 400장 표본.

```
640x480         400
```

## 라벨

JSON 400개 표본 (전체 4,000개).

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
b000_000011.png
b004_000093.json
b007_000010.json
```
