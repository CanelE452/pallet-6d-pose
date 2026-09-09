# Stage 0A — R0 가 실제로 학습한 치수 전수

입력은 동결 manifest `challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl`
(sha256 `20b863b7...`, 선언값과 일치 확인). 60,000 행 전수. 치수 결측 0.
산출: `data/pallet/results/pallet_translation_loss_v1/SYNTHETIC_DIMENSION_AUDIT.json`.

모집단은 예상과 일치한다 — train 55,980 / val 4,020 / total 60,000,
G38 38,002 + P0 8,989 + TEX 8,989.

## ★ 핵심 결과 — exact-square 는 4,220 장이 아니라 **1 장**이다 [확인]

```
train 55,980            N        %
EXACT_SQUARE (<=1e-6)        1    0.0018
NEAR_SQUARE_1 (1.00,1.02] 1,613   2.8814
NEAR_SQUARE_2 (1.02,1.05] 2,606   4.6552
RECT_MODERATE (1.05,1.20]20,052  35.8199
RECT_STRONG   ( >1.20  )31,708  56.6417
```

기존 감사의 "footprint ratio <= 1.05 가 약 7.5%, 약 4,220 장" 은
`1,613 + 2,606 = 4,219` 로 정확히 재현된다. 그러나 그것은 **near-square** 이고,
저장된 두 float 가 실제로 같은 프레임은 60,000 중 **1 장**뿐이다.

즉 지시문 §48 이 금지한 "현재 <=1.05 인 4,220 장을 전부 C4 로 친다" 는 판단은
금지 이전에 **사실과 다르다**. R0 의 학습 모집단에 정사각 기하는 사실상 없다.

## asset 별

```
asset                                    n(60k)   ratio min/p50/max      exact
eur_pallet_bk_cc0.glb                    10,182   1.1251/1.4964/1.9824       0
woodpallet_block_jtoastie_ccby.glb       10,099   1.0000/1.1754/1.5640       1
scene_1.usd                              10,095   1.0001/1.2024/1.6037       0
scene.usd  (G38 9,624 + P0/TEX 20,000)   29,624   1.0000/1.1982/1.6091       0
```

`eur_pallet_bk_cc0.glb` 는 최소 비율이 1.1251 이라 near-square 구간에 아예
들어오지 않는다. 유일한 exact-square 1 장은
`woodpallet_block_jtoastie_ccby.glb` 것이고, 그 asset 은 대칭 미확인이라
계약상 C1 이다 (`LOSS_SYMMETRY_CONTRACT.json`).

## 유효 mesh 는 4 개가 아니라 사실상 3 개

legacy P0/TEX 17,978 장(train)의 `source_asset` 은 10,000/10,000 이 `scene.usd`
다 (`SOURCE_REAL_GAP_AUDIT.md`). 따라서 asset 별 train 분포는

```
scene.usd                          27,121   48.4%
eur_pallet_bk_cc0.glb               9,675   17.3%
woodpallet_block_jtoastie_ccby.glb  9,593   17.1%
scene_1.usd                         9,591   17.1%
```

## §9 target leakage 판정 — PARTIAL_VERIFICATION

```
TARGET_ASSET_USED        아니라고 말할 수 없다
TARGET_ASSET_NOT_USED    주장 불가
PARTIAL_VERIFICATION     ★ 이것으로 기록한다
```

근거:

- R0 학습셋의 **65.5%(train 36,712 장)가 `scene*.usd`**, 즉 이 저장소의
  `data/pallet/raw_data/models_usd/` 에 있는 자체 USD 팔레트 계열이다.
  legacy 아카이브 이름 자체가 `legacy_v1v2_p0_*` 이고, CLAUDE.md 는 v1/v2 를
  **과제(challenge) 전용, 논문 트랙 제외**로 선언한다.
- 두 generic `.glb` mesh 는 이 머신에 파일이 없어(2026-08-22 확인) 평가 물체와
  mesh 수준 비교가 불가능하다.
- 평가 물체(`plastic 110x130x11`, `wood 80x59x14`)의 mesh 는 저장소에 없다.
  따라서 mesh hash 비교로 "같은 물체가 아니다" 를 **증명할 수 없다**.

결론: Stage A 는 **R0-matched mechanism experiment** 로 부른다.
target-free 일반화 주장으로 쓰지 않는다. 이것은 이번 track 이 만든 오염이 아니라
R0 에 이미 존재하던 성질이고, 숨기지 않고 기록한다.
