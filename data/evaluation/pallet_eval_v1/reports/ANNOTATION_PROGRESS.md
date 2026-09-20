# 통합 평가 데이터 어노테이션 진행률

평가 데이터 전체에서 같은 image를 SHA256으로 한 번만 세어 아래
통합 집계만 표시한다.

```text
Positive total        319 / 300

Object
Plastic               194 / 180
Wood                  125 / 120

Lighting
DAY                   168
NIGHT                 106

Condition coverage
Clean                 155 / 80
Occlusion             135 / 80
Truncation             51 / 50
Far                    59 / 50

Elevation
Low                   122 / 60
Mid                   138 / 60
High                   57 / 40

Negative
Negative unique      2688 / 1500   SATISFIED

UNKNOWN_METADATA      319
```

`UNKNOWN_METADATA`는 통합 positive 중 object/lighting/condition metadata가 하나라도
`unknown`인 frame 수다.

## Dataset readiness

```text
DATASET_READY        TRUE

DATASET_READY 는 네 조건을 동시에 만족해야 참이다
  total >= minimum                  true
  MAIN domain coverage              true
  morphology coverage               true
  robustness minimum coverage       true
```

## Metadata unknown — 축별

한 덩어리로 세면 `view` 하나 때문에 전 행이 unknown 이 되어 domain readiness 를
읽을 수 없다. 축을 나눈다.

```text
CORE_DOMAIN_METADATA_UNKNOWN        113   object_type · acquisition_domain
ROBUSTNESS_METADATA_UNKNOWN           2   occlusion · truncation · distance · elevation
AUX_METADATA_UNKNOWN                319   view
```

domain experiment(M2 / M5) readiness 는 AUX 때문에 FAIL 시키지 않는다.

## Main domain evaluation readiness

```text
Condition     Object      Frames  Minimum  Preferred  Sessions  MinSess   Status
----------------------------------------------------------------------------------------
Daytime       Plastic         70       50         60         3        2   PREFERRED_READY
Nighttime     Plastic         50       50         60         3        2   READY
```

내부 provenance 대응은 `reports/PAPER_DOMAIN_COVERAGE.md` 를 본다.
내부 capture id 별 집계는 `reports/DOMAIN_COVERAGE.md`(engineering audit).
`Positive total`만 보고 domain experiment 진척으로 읽지 말 것 —
DATASET_READY 는 위 네 조건을 모두 만족해야 참이다.

<!-- GREEN_REVIEW_PROGRESS_BEGIN -->
## 초록 정사각형 팔레트 — 추가 검토 진행률 (자동)

**아래는 별도 검토 폴더 집계이며 위 논문 평가셋 합계에는 포함하지 않습니다.**

| 항목 | 수 |
|---|---:|
| 전체 촬영 프레임 | 30400 |
| 최초 복사한 기존 라벨 | 851 |
| 현재 유효 라벨 | 862 |
| 새로 라벨링한 프레임 | 11 |
| EVAL로 지정·저장한 검토 후보 | **862** |
| TRAIN으로 남아 있는 라벨 | 0 |
| 미어노테이션 프레임 | 29538 |
| 읽기/구조 오류 (집계 제외) | 0 |

원본 851장은 모두 TRAIN이었다. EVAL 수는 검토 복사본의 저장된 split을 센다.
새 프레임은 기본 EVAL이며 기존 TRAIN은 v로 바꾼 뒤 s로 저장한다.
단순 열람·미저장 변경은 집계하지 않는다. EVAL 지정은 라벨 품질 검증이나
독립 테스트 적격성 확인을 뜻하지 않으며, 정식 평가 편입은 별도 절차다.
100장은 임의의 확정 목표로 추가하지 않았다. 촬영 세션별 프레임 수이며 SHA 중복 제거 전 수다.

| 촬영 세션 | 전체 | 라벨 | 신규 | EVAL |
|---|---:|---:|---:|---:|
| forklift_v4_173507 | 3729 | 19 | 0 | 19 |
| forklift_v4_174126 | 757 | 4 | 0 | 4 |
| forklift_v4_174342 | 2501 | 13 | 0 | 13 |
| forklift_v4_174925 | 1923 | 31 | 0 | 31 |
| forklift_v4_20260903_190408 | 28 | 28 | 0 | 28 |
| forklift_v4_20260903_190743 | 16 | 16 | 0 | 16 |
| forklift_v4_20260903_192118 | 2 | 2 | 0 | 2 |
| forklift_v4_20260903_192254 | 5 | 5 | 0 | 5 |
| forklift_v4_20260904_102339 | 476 | 7 | 0 | 7 |
| forklift_v4_20260904_102504 | 493 | 6 | 0 | 6 |
| forklift_v4_20260904_102630 | 192 | 0 | 0 | 0 |
| forklift_v4_20260904_103429 | 1573 | 16 | 0 | 16 |
| forklift_v4_20260904_103739 | 787 | 26 | 0 | 26 |
| forklift_v4_20260904_104212 | 441 | 12 | 0 | 12 |
| forklift_v4_20260904_105241 | 671 | 2 | 0 | 2 |
| forklift_v4_20260904_105508 | 614 | 1 | 0 | 1 |
| forklift_v4_20260904_105615 | 1589 | 5 | 0 | 5 |
| forklift_v4_20260904_142318 | 3338 | 65 | 0 | 65 |
| forklift_v4_20260904_142958 | 9 | 9 | 0 | 9 |
| forklift_v4_20260904_144221 | 1 | 1 | 0 | 1 |
| forklift_v4_20260904_144614 | 13 | 13 | 0 | 13 |
| forklift_v4_20260904_144733 | 66 | 66 | 0 | 66 |
| forklift_v4_20260904_145924 | 60 | 60 | 0 | 60 |
| forklift_v4_20260904_150335 | 8 | 8 | 0 | 8 |
| forklift_v4_20260904_150816 | 5 | 5 | 0 | 5 |
| forklift_v4_20260904_150944 | 19 | 19 | 0 | 19 |
| forklift_v4_20260904_captured | 4849 | 68 | 0 | 68 |
| capture_20260902 | 5945 | 65 | 11 | 65 |
| capture_20260902_kimjihoon | 290 | 290 | 0 | 290 |

<!-- GREEN_REVIEW_PROGRESS_END -->
