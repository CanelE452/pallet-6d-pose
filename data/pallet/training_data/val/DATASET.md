# val

`data/pallet/training_data/val`

이미지는 저장소에 올리지 않는다. 이 카드는 폴더를 실측한 요약이며,
`scripts/data_prep/validate/gen_dataset_cards.py` 로 다시 만들 수 있다.

## 규모

```
파일        3,000
이미지      1,500
JSON        1,500
용량       491.2M
```

## 해상도

이미지 400장 표본.

```
640x480         400
```

## 라벨

JSON 400개 표본 (전체 1,500개).

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
000344.png
000868.png
000873.json
```
