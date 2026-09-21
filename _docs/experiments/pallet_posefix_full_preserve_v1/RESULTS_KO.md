# FULL + preservation 결정 실험 결과

**판정: INCONCLUSIVE**. FP PCK10 50.35% (FULL 대비 +0.00pp).

FULL의 B3 복구 6개 중 FP가 그대로 유지한 것은 5개이고 FP의 총 B3 복구는 5개다. BASE 정답 손실은 17 → 16, N2 정답 손실은 31 → 31.

## 실험 계약

PRIOR1에서 FULL과 동일한 Conv/Transpose/head 학습, BN 통계·affine 고정. seed1, TFAdam 1e-4, 300 step, real8/source8/micro2, 각2400노출. 기존253개 실사 OCC tensor/target/mask/order 및 source corruption을 전 step hash로 확인했다. calibration10step은 기존 FULL과 update별 weight SHA까지 일치했고, final 학습 전 state와 optimizer를 초기화했다.

합성 TRAIN normal 1412개 중 유효 코너 11201개, BASE 오차≤5px인 9840개만 T=1 spatial KL로 보호. 채널별 선택 수 [1256, 1254, 1232, 1249, 1237, 1234, 1181, 1197, 0]. 고정 native whole-object identity를 사용하며 점별 재매칭 없음. λ=1.0143272: TRAIN-only 10step warmup 뒤 step11 gradient ratio0.25로 단 한 번 결정. 보존용 normal forward가 추가되므로 compute-matched 실험은 아니다.

신규 본학습 418.6초. 중간 평가·checkpoint 선택·rescue·LoRA·student self-training 없음. 최종 checkpoint 및 모든278개 prediction freeze 후 GT scoring. 기존 FULL/BASE/N2는 재학습·변경하지 않았다.

## 모집단별 결과

PCK는 매칭 실패를 포함한다. 중앙값/P90은 matched corner만. PRIMARY 93장/713코너 중 실패8장/54코너에800px 벌점을 유지했다. GREEN manual과 비초록 legacy reference는 별도 보고한다.

### PLASTIC194 — 실제 128장

| method | PCK10% | PCK20% | med px | P90 px | >20px | B3 복구 | BASE 유지/손실/이득 | N2 유지/손실/이득 | 검출/매칭 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BASE | 54.82 | 75.13 | 8.353 | 40.653 | 245/985 | 0/118 | 540/0/0 | 511/57/29 | 128/120 |
| N2 | 57.66 | 76.45 | 8.311 | 40.793 | 232/985 | 0/118 | 511/29/57 | 568/0/0 | 128/120 |
| FULL | 57.56 | 75.53 | 7.526 | 41.265 | 241/985 | 6/118 | 508/32/59 | 512/56/55 | 128/120 |
| FULL_PRESERVE | 58.38 | 76.24 | 7.594 | 40.275 | 234/985 | 5/118 | 517/23/58 | 520/48/55 | 128/120 |

### GREEN150_MANUAL — 실제 150장

| method | PCK10% | PCK20% | med px | P90 px | >20px | B3 복구 | BASE 유지/손실/이득 | N2 유지/손실/이득 | 검출/매칭 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BASE | 80.18 | 89.43 | 3.953 | 10.172 | 72/681 | 0/3 | 546/0/0 | 537/17/9 | 150/134 |
| N2 | 81.35 | 89.28 | 4.122 | 9.749 | 73/681 | 0/3 | 537/9/17 | 554/0/0 | 150/134 |
| FULL | 79.15 | 88.84 | 4.042 | 10.387 | 76/681 | 0/3 | 527/19/12 | 525/29/14 | 150/134 |
| FULL_PRESERVE | 79.44 | 88.40 | 3.898 | 10.396 | 79/681 | 1/3 | 528/18/13 | 527/27/14 | 150/134 |

### DEV72_REFERENCE_UNKNOWN — 실제 72장

| method | PCK10% | PCK20% | med px | P90 px | >20px | B3 복구 | BASE 유지/손실/이득 | N2 유지/손실/이득 | 검출/매칭 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BASE | 57.84 | 75.68 | 7.684 | 50.524 | 135/555 | 0/79 | 321/0/0 | 310/20/11 | 72/72 |
| N2 | 59.46 | 75.14 | 7.831 | 50.910 | 138/555 | 0/79 | 310/11/20 | 330/0/0 | 72/72 |
| FULL | 60.90 | 75.68 | 7.244 | 52.362 | 135/555 | 6/79 | 308/13/30 | 308/22/30 | 72/72 |
| FULL_PRESERVE | 61.26 | 75.68 | 7.189 | 52.000 | 135/555 | 5/79 | 311/10/29 | 311/19/29 | 72/72 |

### PRIMARY_OCC96 — 실제 93장

| method | PCK10% | PCK20% | med px | P90 px | >20px | B3 복구 | BASE 유지/손실/이득 | N2 유지/손실/이득 | 검출/매칭 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BASE | 47.69 | 67.04 | 9.638 | 52.649 | 235/713 | 0/99 | 340/0/0 | 321/29/19 | 93/85 |
| N2 | 49.09 | 68.02 | 9.448 | 51.193 | 228/713 | 0/99 | 321/19/29 | 350/0/0 | 93/85 |
| FULL | 50.35 | 67.88 | 8.554 | 53.178 | 229/713 | 6/99 | 323/17/36 | 319/31/40 | 93/85 |
| FULL_PRESERVE | 50.35 | 67.88 | 8.645 | 53.174 | 229/713 | 5/99 | 324/16/35 | 319/31/40 | 93/85 |

### CLEAN_NONCAD69 — 실제 17장

| method | PCK10% | PCK20% | med px | P90 px | >20px | B3 복구 | BASE 유지/손실/이득 | N2 유지/손실/이득 | 검출/매칭 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BASE | 92.65 | 97.79 | 3.991 | 8.636 | 3/136 | 0/4 | 126/0/0 | 125/2/1 | 17/17 |
| N2 | 93.38 | 97.06 | 3.750 | 8.642 | 4/136 | 0/4 | 125/1/2 | 127/0/0 | 17/17 |
| FULL | 90.44 | 98.53 | 4.221 | 9.434 | 2/136 | 0/4 | 122/4/1 | 122/5/1 | 17/17 |
| FULL_PRESERVE | 91.91 | 99.26 | 3.802 | 8.943 | 1/136 | 0/4 | 124/2/1 | 124/3/1 | 17/17 |

### CAD18 — 실제 18장

| method | PCK10% | PCK20% | med px | P90 px | >20px | B3 복구 | BASE 유지/손실/이득 | N2 유지/손실/이득 | 검출/매칭 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BASE | 54.41 | 94.85 | 9.829 | 18.584 | 7/136 | 0/15 | 74/0/0 | 65/26/9 | 18/18 |
| N2 | 66.91 | 100.00 | 8.690 | 13.891 | 0/136 | 0/15 | 65/9/26 | 91/0/0 | 18/18 |
| FULL | 62.50 | 92.65 | 7.861 | 18.544 | 10/136 | 0/15 | 63/11/22 | 71/20/14 | 18/18 |
| FULL_PRESERVE | 66.91 | 97.06 | 8.057 | 16.551 | 4/136 | 0/15 | 69/5/22 | 77/14/14 | 18/18 |

### SELECTED_CLEAN8 — 실제 8장

| method | PCK10% | PCK20% | med px | P90 px | >20px | B3 복구 | BASE 유지/손실/이득 | N2 유지/손실/이득 | 검출/매칭 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BASE | 43.75 | 93.75 | 10.671 | 19.299 | 4/64 | 0/6 | 28/0/0 | 24/17/4 | 8/8 |
| N2 | 64.06 | 100.00 | 9.065 | 14.356 | 0/64 | 0/6 | 24/4/17 | 41/0/0 | 8/8 |
| FULL | 73.44 | 93.75 | 7.391 | 18.431 | 4/64 | 0/6 | 28/0/19 | 38/3/9 | 8/8 |
| FULL_PRESERVE | 71.88 | 100.00 | 7.729 | 14.911 | 0/64 | 0/6 | 28/0/18 | 38/3/8 | 8/8 |

## FULL → FP 전이 검산

GOOD→GOOD 350, GOOD→BAD 9, BAD→GOOD 9, BAD→BAD 345. 순정답 변화 = 9 − 9 = 0.

## 합성 heldout256 (학습과 분리)

| method | clean PCK10% | stress PCK10% | clean P90 | stress P90 |
|---|---:|---:|---:|---:|
| BASE | 94.02 | 19.54 | 6.655 | 49.928 |
| FULL | 91.35 | 67.71 | 9.123 | 25.453 |
| FULL_PRESERVE | 92.24 | 67.41 | 7.878 | 26.628 |

## 사전 gate

| gate | PASS |
|---|---|
| PCK10 | True |
| B3_recovery | True |
| BASE_preservation | True |
| N2_preservation | True |
| source_clean | False |
| P90 | True |
| GREEN | True |
| integrity | True |

성공은 모든 gate 충족이다. B3 gate는 총복구 기준이며 동일 코너 유지 수는 별도로 보고했다. single seed/reused DEV 파일럿으로 통계적 유의성이나 일반화 증명을 주장하지 않는다.

## 사후분석

| 사건 | corner 수 | band 분포 | corner ID 분포 | session 분포 |
|---|---:|---|---|---|
| preservation_wins | 6 | {'B1': 5, 'B2': 1} | {'4': 1, '5': 2, '3': 2, '2': 1} | {'eval_night09': 1, 'eval_pallet07': 5} |
| preservation_failures | 16 | {'B2': 4, 'B1': 11, 'B0': 1} | {'0': 3, '5': 2, '3': 1, '4': 5, '6': 3, '7': 2} | {'eval_night08': 1, 'eval_night09': 4, 'eval_outside': 1, 'eval_pallet07': 7, 'eval_pallet09': 3} |
| new_preservation_damage | 5 | {'B2': 1, 'B1': 4} | {'5': 1, '4': 3, '6': 1} | {'eval_night09': 1, 'eval_outside': 1, 'eval_pallet07': 2, 'eval_pallet09': 1} |
| lost_FULL_recovery | 1 | {'B3': 1} | {'2': 1} | {'eval_pallet07': 1} |
| retained_FULL_recovery | 5 | {'B3': 5} | {'0': 1, '3': 1, '5': 1, '6': 1, '1': 1} | {'eval_pallet07': 5} |
| new_recovery | 0 | {} | {} | {} |

### PRIMARY의 R0 band별 gain/loss

| band | 코너 | FULL 정답 | FP 정답 | FULL BASE gain/loss | FP BASE gain/loss |
|---|---:|---:|---:|---|---|
| B0 | 116 | 115 | 115 | 0/1 | 0/1 |
| B1 | 192 | 166 | 166 | 6/12 | 5/11 |
| B2 | 155 | 72 | 73 | 24/4 | 25/4 |
| B3 | 99 | 6 | 5 | 6/0 | 5/0 |
| B4 | 97 | 0 | 0 | 0/0 | 0/0 |
| B5_MATCH_FAILURE | 54 | 0 | 0 | 0/0 | 0/0 |

### 보정 이동량 (원래 R0 대비, 정확도 아님)

| method | median px | P90 px | max px |
|---|---:|---:|---:|
| R0 | 0.000 | 0.000 | 0.000 |
| BASE | 2.681 | 6.012 | 21.946 |
| N2 | 2.306 | 6.282 | 8.000 |
| FULL | 4.350 | 10.620 | 25.977 |
| FULL_PRESERVE | 3.964 | 10.301 | 43.480 |

## 한계와 사람 검토

실사253장은 한 recording의 인접 프레임이고 pseudo target 자체가 오답일 수 있다. Replay teacher의 동일 recording 수동학습 노출도 기존과 같다. 평가 recording 제외·기존 center/box/score 유지. 아래 GT는 평가·그림에만 사용했으며 추론·보존 mask·λ에는 미사용. 가림 이미지를 평가했지만 human visibility review 없이는 외부가림 코너 복원이라고 주장하지 않는다. 최종 논문 모델/표는 교체하지 않았다.

[다음 분기 및 질문별 답](NEXT_STAGE_PLAN.md) · [데이터 다양성 감사](DATA_DIVERSITY_INVENTORY.md) · [사후분석](POSTMORTEM_KO.md) · [전체 HTML](GALLERY.html)

## 실제 이미지 비교

초록 x=평가 정답/reference, 청록=모델 출력. 표시는 evaluator의 전체-object symmetry로 정렬. 좋은 사례·실패·FULL 복구 손실 전체·새 복구 전체와 고정 무작위10장을 모두 포함한다.

### preservation_wins

`eval_night09:1779449631842893312`

![preservation_wins](figures/frame_009.jpg)

`eval_pallet07:1778652124170068224`

![preservation_wins](figures/frame_013.jpg)

`eval_pallet07:1778652128369383168`

![preservation_wins](figures/frame_015.jpg)

`eval_pallet07:1778652156557165568`

![preservation_wins](figures/frame_020.jpg)

`eval_pallet07:1778652158404883456`

![preservation_wins](figures/frame_021.jpg)


### preservation_failures

`eval_night09:1779449591514485504`

![preservation_failures](figures/frame_007.jpg)

`eval_pallet07:1778652130452698368`

![preservation_failures](figures/frame_016.jpg)

`eval_night08:1779449470423201536`

![preservation_failures](figures/frame_004.jpg)

`eval_night09:1779449593782795264`

![preservation_failures](figures/frame_008.jpg)

`eval_night09:1779449634044385536`

![preservation_failures](figures/frame_010.jpg)


### lost_FULL_recovery

`eval_pallet07:1778652152626116352`

![lost_FULL_recovery](figures/frame_019.jpg)


### retained_FULL_recovery

`eval_pallet07:1778652127361815808`

![retained_FULL_recovery](figures/frame_014.jpg)

`eval_pallet07:1778652140531310080`

![retained_FULL_recovery](figures/frame_018.jpg)

`eval_pallet07:1778652152626116352`

![retained_FULL_recovery](figures/frame_019.jpg)


### new_recovery

해당 사례 없음.

### random_primary_seed1

`eval_outside:1778651530691638016`

![random_primary_seed1](figures/frame_012.jpg)

`eval_pallet07:1778652132535256576`

![random_primary_seed1](figures/frame_017.jpg)

`eval_pallet09:1778653713962971904`

![random_primary_seed1](figures/frame_023.jpg)

`eval_night08:1779449479095927552`

![random_primary_seed1](figures/frame_005.jpg)

`eval_pallet09:1778653823253508096`

![random_primary_seed1](figures/frame_025.jpg)

`eval_pallet07:1778652140531310080`

![random_primary_seed1](figures/frame_018.jpg)

`eval_night09:1779449575470221824`

![random_primary_seed1](figures/frame_006.jpg)

`eval_pallet09:1778653659065885952`

![random_primary_seed1](figures/frame_022.jpg)

`eval_pallet09:1778653806958839552`

![random_primary_seed1](figures/frame_024.jpg)

`eval_night09:1779449643284402176`

![random_primary_seed1](figures/frame_011.jpg)


### random_GREEN_seed1

`capture_20260902_kimjihoon__006169`

![random_GREEN_seed1](figures/frame_002.jpg)

`capture_20260902_kimjihoon__005869`

![random_GREEN_seed1](figures/frame_001.jpg)

`capture_20260902_kimjihoon__008369`

![random_GREEN_seed1](figures/frame_003.jpg)

