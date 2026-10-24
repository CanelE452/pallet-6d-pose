# V204 TRAIN intervention 픽셀과 원 query의 정렬

핵심은 기존 E 약라벨이 항상 원 query 중앙 픽셀 변경을 뜻하지 않는다는 것이다. TRAIN의 known D/E/D109쌍을 모두 검사했다. COVER64/109는 중앙2×2에 byte 변화가 없고, SHAM6/109는 중앙2×2에 변화가 있다. 이는 라벨 오류를 새로 확정한 것이 아니라 finite-end 감독과 query 중심 입력의 질문이 다름을 보여 주는 입력 증거다.

| source | pair | TRAIN triplets | h=.5 (2×2) | h=1.5 (4×4) | h=3 (6×6) | h=6 (12×12) | h=12 (24×24) |
|---|---|---:|---:|---:|---:|---:|---:|
|SYNTH_FINITE_END|COVER|82|30|36|45|65|76|
|SYNTH_FINITE_END|SHAM|82|2|2|2|8|34|
|REAL_SEMANTIC_END_WEAK|COVER|27|15|17|20|22|25|
|REAL_SEMANTIC_END_WEAK|SHAM|27|4|5|7|11|17|

표는 max-channel absRGB≥1인 유효 변화 픽셀이 창 안에 하나 이상 있는 triplet 수다. 창 중심은 RAW96의(47.5,47.5), integer pixel center에 닫힌 square를 적용했다. h=.5는 정확히4개 pixel center다. threshold8·32도 사전 고정했고 모두 CSV에 보존했다. nearest distance는 변화한 픽셀 중심까지의 Euclidean 거리이며, 변화가 없으면0이 아닌 null이다.

합성 COVER의 nearest 변화 거리: min0.707 / median2.550 / max29.707 RAW96 px. 공개실사 weak COVER: min0.707 / median0.707 / max19.092 px. whole96에서는 COVER와 SHAM 모두109쌍 전부 변화했다. COVER 변화 픽셀 수는 합성36–100, 공개실사283–2633이었다. 중앙 변화가 없다는 것은 전체 crop이 동일하다는 뜻이 아니다.

분모는 원 TRAIN2784행이며 known direct327행=109×3이다. 합성82쌍은41base·41component, 공개실사27쌍은12frame이나1component다. 109개의 독립 frame 실험으로 세지 않는다. 모든 native0–7의 UNKNOWN/U 행을 별도 분모표에 남겼고 native6/7 known0을 물리적 part 부재로 해석하지 않는다. 공유 NPZ3144행을 물리적으로 풀어도 TRAIN slice만 비교했으며 HOLDOUT 픽셀/통계는 분석하지 않았다.

## 실제 본 그림과 권위

저자는 이미 픽셀 전 고정된 stablehash 순서의 첫 합성 V189_BASE036_NATIVE3, 첫 공개실사 Q0052_D1 카드 두 개만 실제 toolview했다. 전체12/109개를 판독했다고 주장하지 않는다. 두 카드 모두 원 clean96와 확대 표시 사본, 변화 heatmap, q cross와 고정창이 구분된다. 합성에서는 작은 patch가 query 근처와 더 먼 SHAM 위치로, 공개실사에서는 큰 붉은 donor가 다른 위치로 보인다. 이 두 예시가 전체 source의 원인·빈도를 대표하지 않는다.

- [고정12 전체 gallery](actual_train_pixel_alignment_v1/PORTABLE_GALLERY_KO.md)
- [수치요약 JSON](actual_train_pixel_alignment_v1/SUMMARY.json)
- [개별 window3270행 CSV](actual_train_pixel_alignment_v1/QUERY_CENTER_FIXED_WINDOWS.csv)
- [window90행 집계](actual_train_pixel_alignment_v1/QUERY_CENTER_WINDOW_SUMMARY.csv)
- [whole96/threshold654행](actual_train_pixel_alignment_v1/PAIR_WHOLE96_THRESHOLDS.csv)
- [allnative TRAIN UNKNOWN/U 분모](actual_train_pixel_alignment_v1/TRAIN_ALL_NATIVE_TARGET_STATUS.csv)
- [실제2개 도구 관찰 영수증](ACTUAL_FIXED2_GENERATED_CARD_VISUAL_QA.json)
- [prospective protocol](PROTOCOL_BEFORE_TRAIN_PIXEL_ARRAY_DIAGNOSTIC.json)

## 해석 한계

원 q/L/native·RGB/valid bytes와 약라벨을 변경하지 않았다. 원 source packet/endmask/새 GT는 열지 않아 실제 edit가 어느 물리적 끝부위인지 새로 인증하지 않는다. E를 query 중앙에 foreign이 생긴 gold로 바꾸거나, SHAM의 중앙 변경을 오라벨로 고치지 않는다. 학습된 모델·실사 평가·threshold·architecture·새 데이터 선택에 대한 결론도 내리지 않는다. 가장 직접적인 후속 source-only 질문은 기존 finite-end witness와 원 query 창의 의미를 구분해 사전에 정의하는 것이며, 이 진단 자체는 새 감독·학습·입력 이동을 실행하지 않았다.
