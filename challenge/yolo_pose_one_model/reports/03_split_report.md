# 03 — Split report

생성: `python challenge/yolo_pose_one_model/scripts/build_splits.py`  
seed=42, 결정적(해시 순위) — 재실행하면 같은 분할이 나온다.

범위: 합성 G/T 만. real 은 이번 라운드에서 만들지 않았다(사용자 지시).

## 결과
```
domain             train     val   val%
generic_synth      38002    1998   5.0%
target_synth       17957    2011  10.1%
```

## 누수 검사 [확인]
```
G train∩val sample_id      0
T train∩val sample_id      0
T train∩val scenario       0   (train 3691 / val 392 scenario)
```

## G 그룹 근거

`records.jsonl` 의 seed 가 40000/40000 전부 고유하고 `diagnostic_mode` 가
`clean-static` 이다 = 프레임마다 독립 렌더, 카메라 궤적 없음. 시퀀스 누수가
원천적으로 없다. 그래도 image-level 무작위를 피하려고
(pallet_type, scene_preset, background_asset) 층화 + seed 해시 순위로 결정적 분할했다.

```
stratum                                           n   val
Pallet_0|indoor|industrial                     1185    59
Pallet_0|indoor|parking_lot                    1287    64
Pallet_0|outdoor-day|industrial                1440    72
Pallet_0|outdoor-day|parking_lot               1443    72
Pallet_0|outdoor-night|industrial              1159    58
Pallet_0|outdoor-night|parking_lot             1226    61
Pallet_0|random-mix|industrial                  931    47
Pallet_0|random-mix|parking_lot                 953    48
Pallet_1|indoor|industrial                     1250    62
Pallet_1|indoor|parking_lot                    1290    64
Pallet_1|outdoor-day|industrial                1533    77
Pallet_1|outdoor-day|parking_lot               1488    74
Pallet_1|outdoor-night|industrial              1271    64
Pallet_1|outdoor-night|parking_lot             1264    63
Pallet_1|random-mix|industrial                  983    49
Pallet_1|random-mix|parking_lot                1016    51
Pallet_2|indoor|industrial                     1211    61
Pallet_2|indoor|parking_lot                    1359    68
Pallet_2|outdoor-day|industrial                1414    71
Pallet_2|outdoor-day|parking_lot               1541    77
Pallet_2|outdoor-night|industrial              1259    63
Pallet_2|outdoor-night|parking_lot             1301    65
Pallet_2|random-mix|industrial                  993    50
Pallet_2|random-mix|parking_lot                1021    51
Pallet_3|indoor|industrial                     1267    63
Pallet_3|indoor|parking_lot                    1266    63
Pallet_3|outdoor-day|industrial                1528    76
Pallet_3|outdoor-day|parking_lot               1544    77
Pallet_3|outdoor-night|industrial              1206    60
Pallet_3|outdoor-night|parking_lot             1325    66
Pallet_3|random-mix|industrial                  984    49
Pallet_3|random-mix|parking_lot                1062    53
```

## T 그룹 근거

`frame_meta.scenario` 가 같은 장면의 여러 카메라 프레임을 묶는다
(v1 2782 scenario/9997 frame, v2 1305/9994). scenario 를 통째로 한쪽에만 넣었다.

```
stratum(set|background_3d)          frames   scen  val_fr  val_sc
v1|industrial                         4172   1216     419     101
v1|parking_lot                        5814   1563     585     160
v2|industrial                         3877    511     392      50
v2|parking_lot                        6105    793     615      81
```

## 산출
```
manifests/generic_train.txt  generic_val.txt
manifests/target_train.txt   target_val.txt
manifests/split_manifest.json
```