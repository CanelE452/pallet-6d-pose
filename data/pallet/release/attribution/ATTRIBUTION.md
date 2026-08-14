# CC-BY Asset Attribution (paper appendix)

논문 데이터셋 공개 시 필수 표기(attribution) 목록. **CC-BY 라이선스 에셋은 저작자 표기 의무**가 있으므로 논문 부록/데이터셋 릴리스에 아래를 포함한다. (CC0 에셋 — Poly Haven HDRI 30, wood 텍스처 9, EUR-Pallet, pallet_full.obj[본인 photogrammetry] — 은 표기 의무 없음. occluder는 대부분 CC0이나 Sketchfab CC-BY 8종 추정 포함 / floor 텍스처 14는 전부 CC0 확정 — §3 참조.)

Canonical per-asset 출처는 각 폴더의 `SOURCES.txt` / `license.txt` 에도 있음. 이 파일은 그 통합본.

---

## 1. Distractor — Sketchfab (16종, 전부 CC-BY 4.0)

gap-fill로 추가(2026-07-24). 저장: `data/pallet/assets/distractors/library/{tier}/sf__<name>/`.

```
name                     sketchfab title                     author            license      url
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
forklift_01              Forklift                            fdgasd7           CC-BY 4.0    https://sketchfab.com/3d-models/forklift-bdb03db7036e436286f4e2fd34c02a89
forklift_02              Forklift                            mansta9           CC-BY 4.0    https://sketchfab.com/3d-models/forklift-d40cae50e04145dd997cdca415cd72ad
cargo_truck_01           Isuzu Cargo Base Truck              VuckyZ123         CC-BY 4.0    https://sketchfab.com/3d-models/isuzu-cargo-base-truck-6f5765ef13294287b5d14df4ba64d5bf
delivery_truck_01        DELIVERY TRUCK                      jasmin.daniel     CC-BY 4.0    https://sketchfab.com/3d-models/delivery-truck-1d53f7fa474849db812102dfa5d070d0
delivery_van_01          European Delivery Van               evan.hiltz        CC-BY 4.0    https://sketchfab.com/3d-models/european-delivery-van-0b2f1ad95a79419f9a092420024d329c
hand_truck_01            Industrial hand truck               ittoKubashi7      CC-BY 4.0    https://sketchfab.com/3d-models/industrial-hand-truck-a7b424b174ba456f9d84624c1835a2f5
hand_truck_scan_02       Quixel Megascans Metal Hand Truck   Guay0             CC-BY 4.0    https://sketchfab.com/3d-models/3d-scan-quixel-megascans-metal-hand-truck-df28fe2186a3417090b912d63daca2b4
storage_rack_01          Storage Rack                        andersta          CC-BY 4.0    https://sketchfab.com/3d-models/storage-rack-f990c9d601bd480798d12fd5a60dcb5a
no_parking_sign_01       No Parking Sign                     polygroun         CC-BY 4.0    https://sketchfab.com/3d-models/no-parking-sign-3e6e0c4e68794d0d852a28be2f30c766
construction_sign_01     construction sign                   SweetLemons       CC-BY 4.0    https://sketchfab.com/3d-models/construction-sign-7f84fa84c2064de496a68e2cab2acf51
water_dispenser_01       Brio Water Dispenser                dana.digital      CC-BY 4.0    https://sketchfab.com/3d-models/brio-water-dispenser-818b5e12dd3c47c4939e5b4c9c45b6a5
hard_hat_01              Safety Helmet                       muradyanhovo1117  CC-BY 4.0    https://sketchfab.com/3d-models/safety-helmet-f9c17905f17a45d885442ebace25a66f
hard_hat_02              Hard Hat3                           kristiyan         CC-BY 4.0    https://sketchfab.com/3d-models/hard-hat3-cc19391032eb4ff7872b274df375801e
bollard_01               Bollard                             MaX3Dd            CC-BY 4.0    https://sketchfab.com/3d-models/bollard-aa382530c7624927a782547def4c85cb
construction_barrier_01  construction site barrier           lwse              CC-BY 4.0    https://sketchfab.com/3d-models/construction-site-barrier-3917d39740eb4c008924a08c273412d1
traffic_barricade_01     Traffic Barrier                     chamindu918       CC-BY 4.0    https://sketchfab.com/3d-models/traffic-barrier-53fb77ce2c9248319d8913f3526cd047
```

권장 표기 형식: `"<title>" by <author>, licensed under CC BY 4.0, via Sketchfab (<url>).`

---

## 2. 그 외 표기 필요 CC-BY 에셋 (통합 릴리스 시 함께 포함)

- **Distractor — Google Scanned Objects (128종, CC-BY 4.0)** — *2026-07-24 D 확장: 32 → 128 (+96)*. 저작자 = Google Research, via Gazebo Fuel(GoogleResearch). 확장분 96종 구성: box 39 · container 31 · office 14 · warehouse 10 · other 2. 개별 모델 URL·이름은 `data/pallet/assets/distractors/library/distractors_manifest.csv`(`source=gso`, `url` 열 = `https://app.gazebosim.org/GoogleResearch/fuel/models/<name>`) 및 각 폴더 `SOURCES.txt` 참조. 표기: `Google Scanned Objects, Google Research, CC BY 4.0.`
  - **우리가 가한 변경(고지 의무 — CC-BY §3(b) "indicate if changes were made")**: ① 접지 정규화(mesh min-z → 0), ② XY 중심 정규화, ③ (일부) glb/obj 재익스포트, ④ MTL `map_Kd` 텍스처 경로 재bind. 원본 지오메트리/텍스처 콘텐츠는 불변, 좌표/포맷/경로만 조정.
- **배경 glTF (2종, CC-BY 4.0)**: `modular_buildings_industrial_area` (author BazukaliKartal), `parking_lot` (author Veterock) — 정확 URL은 `data/pallet/assets/scenes/backgrounds/background/<name>/license.txt` 참조.

## 3. 표기 불필요/제외 (참고)
- CC0: Poly Haven HDRI 30, wood 텍스처 9, EUR-Pallet(BlenderKit), pallet_full.obj(본인 photogrammetry) → 표기 의무 없음(권장만).
- **occluder(blend 내장 ×19)**: B2 오탐 정정(2026-07-24) 후 = 대부분 **Poly Haven CC0**(표기 불요) + **Sketchfab CC-BY 8종(추정, 표기 필요)** + 불명 3. ⚠️ CC-BY 8종 확정 시 §1/§2에 추가할 것(현재 미확정이라 미기재).
- floor 텍스처 14: **전부 Poly Haven CC0 확정(2026-07-24, B4 해소)** — 미검증 5종을 동등 Poly Haven CC0로 교체(`textures_floor/SOURCES.txt` = 14/14 CC0). → CC0라 표기 불요.
- 제외됨: `modern_city_block`(Sketchfab Standard 비-CC, 미사용) → 2026-07-24 데이터셋에서 격리 제거.
