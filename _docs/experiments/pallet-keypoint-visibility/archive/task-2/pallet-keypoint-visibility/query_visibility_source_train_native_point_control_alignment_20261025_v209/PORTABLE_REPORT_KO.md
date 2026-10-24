# V209: SOURCE point와 기존 control의 대응

**82개 요청 point 모두 BEFORE CLEAR → COVER COVERED → SHAM CLEAR였다.** 원 query 중심과 선언 point가 다르다는 이유만으로 기존 point 타깃이 잘못됐다고 볼 수 없다. 이 결과는 SOURCE raster 대응 진단이며 새 native D/E 라벨·실사 개선·학습 admission이 아니다.

고정 TRAIN 82시도·41개 원 base·328개 원 point의 변형/native8을 보존한 1968행을 독립 확인했다. 원 q/L/native/initializer/sampler와 input U=0이 유지됐다. 요청 이외 native 관계 1722개는 모두 CLEAR였다.

| 변형 | 전체 행 | CLEAR | COVERED | 요청 point CLEAR/COVERED |
|---|---:|---:|---:|---:|
| BEFORE | 656 | 656 | 0 | 82/0 |
| COVER | 656 | 574 | 82 | 0/82 |
| SHAM | 656 | 656 | 0 | 82/0 |

원 BEFORE point 상태와 현재 opacity는 별도 축이다. 원 point D×CLEAR 1640행, D×COVERED 82행, S×CLEAR 246행이며, 이를 새 D1640/E82/S246 학습라벨로 변환하지 않았다. CLEAR는 저장 raster의 양수 tap이 모두 비불투명이라는 뜻이며 깊이 tie/비전면도 포함할 수 있어 가시성 인증이 아니다.

## Coverage와 실제 본 범위

기존 SOURCE TRAIN finite-end COVER 약타깃 82건의 input-native0..7 건수는 **[26, 19, 13, 22, 2, 0, 0, 0]**다. canonical0..7 건수는 **[12, 12, 11, 11, 11, 12, 9, 4]**로 모두 등장한다. 두 순서를 혼동하거나 input-native 빈도를 곧바로 물리 부위 D/E 결손으로 해석하지 않는다.

시각화는 사전 hash·quota로 고정한 8 base, input-native0/1 각4개에 한정된다. root가 카드8개와 full-context triplet2개(고유 PNG10개)를 실제 보았다. renderer의 원 crop parity192개·PNG136개 생성은 실제 view 횟수와 다르다.

[고정8 원 clean/case gallery](fixed8_source_native_point_visual_v1/PORTABLE_GALLERY_KO.md) · [root 실제 view 영수증](ROOT_ACTUAL_FIXED8_SOURCE_POINT_VISUAL_QA.json)

일부 고정 SOURCE 예에서는 작거나 target과 유사색인 평면 개입이 관찰됐다. 이는 SOURCE appearance/scale 가설이며 전체82의 대표성, 학습 정보 증가, 실사 모델 오류의 원인을 입증하지 않는다. 원 source NPZ를 재개봉하지 않고 고정24 clean PNG만으로 COVER/SHAM 대 BEFORE 변화 픽셀수·최소 변화 bbox·절대 채널 변화를 기술 집계했다. RGB 변화 bbox는 opaque alpha footprint·정밀 부위 mask가 아니며 비슷한 색이면 opacity가 있어도 변화가 작을 수 있다.

| 고정 case | 변형 | 변화 pixel/9216 | 변화 bbox (inclusive) | 절대 채널차 합 |
|---|---|---:|---|---:|
| V189_BASE063_NATIVE1 | COVER | 56 | [37, 44, 44, 50] | 13144 |
| V189_BASE063_NATIVE1 | SHAM | 49 | [52, 44, 58, 50] | 1713 |
| V189_BASE110_NATIVE1 | COVER | 56 | [42, 48, 49, 54] | 1313 |
| V189_BASE110_NATIVE1 | SHAM | 54 | [58, 48, 65, 54] | 5938 |
| V189_BASE088_NATIVE0 | COVER | 49 | [47, 51, 53, 57] | 5495 |
| V189_BASE088_NATIVE0 | SHAM | 49 | [61, 51, 67, 57] | 4356 |
| V189_BASE082_NATIVE0 | COVER | 48 | [44, 46, 51, 51] | 2934 |
| V189_BASE082_NATIVE0 | SHAM | 42 | [59, 46, 66, 51] | 620 |
| V189_BASE069_NATIVE1 | COVER | 57 | [39, 44, 46, 51] | 10117 |
| V189_BASE069_NATIVE1 | SHAM | 56 | [55, 44, 61, 51] | 13842 |
| V189_BASE000_NATIVE0 | COVER | 43 | [45, 50, 51, 56] | 1410 |
| V189_BASE000_NATIVE0 | SHAM | 40 | [60, 50, 66, 56] | 256 |
| V189_BASE098_NATIVE1 | COVER | 64 | [50, 48, 57, 55] | 9249 |
| V189_BASE098_NATIVE1 | SHAM | 56 | [65, 48, 71, 55] | 8401 |
| V189_BASE001_NATIVE0 | COVER | 44 | [44, 46, 50, 52] | 1044 |
| V189_BASE001_NATIVE0 | SHAM | 48 | [58, 46, 64, 52] | 244 |

## 표·방법·검산

- [전체 1968행 alpha 관계 CSV](report_preparation_b/ALL1968_ALPHA_RELATIONS.csv)
- [원 point/alpha 별도축 CSV](report_preparation_b/SOURCE_POINT_AND_ALPHA_SEPARATE_AXES.csv)
- [input-native coverage CSV](report_preparation_b/ALL82_INPUT_NATIVE_COVERAGE.csv)
- [canonical coverage CSV](report_preparation_b/ALL82_CANONICAL_COVERAGE.csv)
- [고정8 RGB 변화 CSV](report_preparation_b/FIXED8_RAW96_RGB_CHANGE_STATS.csv)
- [PNG 통계 사전규칙](report_preparation_b/RGB_STATISTICS_PRECOMMIT_PROTOCOL.json)
- [source82 실제 COMPLETE](source_train_native_point_control_alignment_v1/COMPLETE.json)
- [독립 실제 검산 PASS](independent_actual_alignment_review_b/REVIEW.json)
- [고정8 렌더 COMPLETE](fixed8_source_native_point_visual_v1/COMPLETE.json)
- [이 보고서 builder](report_preparation_b/build_source_alignment_report.py)

저장 reference의 모든 엄격 양수 bilinear tap이 alpha True이면 COVERED, 모두 False면 CLEAR, 혼합이면 EDGE_UNKNOWN, 누락·프레임 밖이면 UNKNOWN으로 분리했다. RGB 차분·region state·crop-center owner로 native 상태를 대체하지 않았다. 새 labels/fit/admission/forward·q 교정·PnP는 0이며 목표는 미해결이다.
