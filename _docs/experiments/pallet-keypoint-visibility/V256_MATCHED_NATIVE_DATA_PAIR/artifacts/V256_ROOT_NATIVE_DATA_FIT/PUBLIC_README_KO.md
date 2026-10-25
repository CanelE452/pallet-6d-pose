# V256: 같은 head에서 새 coherent SYN 자료를 추가한 SOURCE 비교

같은 76098-parameter head를 같은 seed17 초기값에서 각각 1000회 학습했습니다. CONTROL은 원 SOURCE FIT, TREAT는 여기에 새 own-scene non-BEFORE known FIT 12,495행을 추가했습니다. encoder·구조·문턱을 바꾸지 않았습니다. 원 YCB 학습 자료와 1000회 draw는 두 팔에서 동일하며, 공유 head 학습 결과의 YCB logit까지 같다는 뜻은 아닙니다.

원 SYN DEV E recall은 92.86%에서 98.21%로 올랐고, D recall은 98.14%에서 97.83%로 낮아졌습니다. E precision도 68.42%에서 66.27%로 낮아졌습니다. YCB DEV E는 두 팔 모두 0/57입니다. 부분적인 SYN 클래스 변화이며 공개 실사 전이 문제나 최종 pallet visibility가 해결됐다는 증거는 없습니다. 모델 미채택, current/held 새 실행 0입니다.

## 원 SOURCE DEV: 같은 분모

CM 행은 실제 D/E, 열은 예측 D/E/U입니다. D0/E1, 선택한 가족 E-logit ≥ 0이면 E이며 문턱 탐색은 없습니다.

| 교사 namespace | 클래스 | 원 분모 | CONTROL 정답/분모 | TREAT 정답/분모 | CONTROL recall | TREAT recall | CONTROL precision | TREAT precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYN 등록 witness | D | 1288 | 1264/1288 | 1260/1288 | 98.14% | 97.83% | 99.68% | 99.92% |
| SYN 등록 witness | E | 56 | 52/56 | 55/56 | 92.86% | 98.21% | 68.42% | 66.27% |
| YCB associated-r AUX | D | 340 | 332/340 | 334/340 | 97.65% | 98.24% | 85.35% | 85.42% |
| YCB associated-r AUX | E | 57 | 0/57 | 0/57 | 0.00% | 0.00% | 0.00% | 0.00% |

SYN CM은 CONTROL `[[1264, 24, 0], [4, 52, 0]]`, TREAT `[[1260, 28, 0], [1, 55, 0]]`입니다. YCB CM은 CONTROL `[[332, 8, 0], [57, 0, 0]]`, TREAT `[[334, 6, 0], [57, 0, 0]]`입니다. 네 클래스 recall의 산술평균은 72.16% → 73.57%, 차이는 +1.408705 pp이며 최악의 클래스 recall 감소는 0.310559 pp입니다. 두 namespace를 하나의 물리적 과업 정확도로 합친 수치가 아닙니다.

원 known SOURCE 7,617행에서 모델 U는 0/7617과 0/7617입니다. 원 full15040의 S/U 등 masked7423은 known7617에서의 모델 abstention 분모에 들어가지 않습니다. 원 DEV 1741행은 반복 사용된 exposed SOURCE DEV이며 fresh blind 검증이 아닙니다. FIT 지표는 [METRICS.csv](tables/METRICS.csv), 개별 회복·악화는 [PAIRED_CHANGES.csv](tables/PAIRED_CHANGES.csv)에 별도 기록했습니다.

![원 DEV 클래스별 recall](figures/DEV_RECALL.png)

## 새 SOURCE FIT counterfactual: 별도 분모

후보 688개에서 target-alone S 86개를 보존했고 602개를 구성했습니다. 이 중 네 장 모두 원 input-support/locality 조건으로 known인 것은 595개입니다. 아래 모델 D/E/D/D 패턴은 이 all-four-known 분모에서만 셉니다. 변형 이름으로 정답을 추론하지 않고 각 장의 새 point0 teacher/known target을 직접 확인했습니다. BEFORE는 `(base, native k)`로 결합하고 child의 intervention k와 query j가 같은 경우만 봅니다.

| 등록 주소 k | 후보 | target-alone S | 구성 | 4장 known | CONTROL D/E/D/D | TREAT D/E/D/D |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 86 | 0 | 86 | 85 | 76 | 73 |
| 1 | 86 | 0 | 86 | 85 | 60 | 52 |
| 2 | 86 | 0 | 86 | 85 | 59 | 56 |
| 3 | 86 | 0 | 86 | 85 | 67 | 67 |
| 4 | 86 | 0 | 86 | 85 | 69 | 70 |
| 5 | 86 | 0 | 86 | 85 | 68 | 64 |
| 6 | 86 | 46 | 40 | 40 | 32 | 25 |
| 7 | 86 | 40 | 46 | 45 | 30 | 29 |
| 전체 | 688 | 86 | 602 | 595 | 461 | 436 |

이 표는 학습용 FIT 분할 소속 자료에 대한 저장 출력 진단입니다. TREAT의 loss 후보로는 모든 known non-BEFORE 12,495행을 넣었고, 개별행 실제 draw 노출은 별도 ledger/QA에 기록됐습니다. 모든 장·모든 주소가 실제 loss에 뽑혔다고 이 표만으로 주장하지 않습니다. 새 BEFORE known 595행은 중복 augmentation loss에서 제외했습니다. 새 full-map 15,136행, known 13,090행, masked 2,046행과 원 U를 보존했습니다. 원 q는 저장된 SYN initializer 좌표이며 fresh v61 추론이나 실사 keypoint annotation이 아닙니다. 좌표 보정도 하지 않았습니다. V254의 strict FIT87과는 후보·교사·known 조건이 달라 직접적인 성공률 증가로 비교하지 않습니다.

전체 all-four-known 595개에서 모델 D/E/D/D는 CONTROL **461/595 (77.48%)**, TREAT **436/595 (73.28%)**로 감소했습니다. 구성된 602개의 fresh point0 상태는 모두 D/E/D/D였지만 그중 7개는 local input/known 조건 때문에 이 모델 패턴 분모에 들어가지 않습니다. COVER−BEFORE와 COVER−FAR E-logit 양수는 두 팔 모두 595/595, COVER−NEAR 양수는 569/595 → 576/595입니다. 방향성 반응과 threshold0에서 네 예측을 모두 맞추는 것은 서로 다른 관측입니다.

![새 FIT selected-k 패턴](figures/NEW_FIT_CF_PATTERN.png)

## 해석과 재현 범위

SYN 가족0은 등록된 whole-target의 intended native-address first witness point0 교사입니다. physical native corner·전체 영역·literal query pixel gold로 격상하지 않습니다. YCB 가족1은 associated-r AUX이며 native/SELF 의미는 UNKNOWN입니다. 모든 native 주소가 후보에 포함됐어도 새 실제 known 수와 교사 기하 분모는 주소별로 다릅니다. 원 q/L/U와 input 조건은 보존됐고 teacher/reference/variant 정보는 모델 forward 입력이 아닙니다.

두 final1000 checkpoint/log/ledger를 봉인한 뒤 DEV 출력을 계산한 sealed-code 순서와 저장 barrier를 독립 QA가 확인했습니다. 파일 시각만으로 실행 chronology를 독립 증명했다는 뜻은 아닙니다. 이 보고서는 승인된 저장 CSV/JSON 산술을 재계산하며 checkpoint·feature·encoder·RGB를 읽거나 forward를 재실행하지 않습니다. 새 SOURCE training/독립 QA 완료는 런타임·저장자료 검증이고 모델 채택이나 current/v61 비교 결과가 아닙니다. 사전 recipe는 자동 deployment gate를 정의하지 않았으며 네 recall/평균/최악 감소를 그대로 보고합니다. 군집 불확실성은 원 group 메타데이터가 이 readout의 최소 입력에 없어 산출하지 않았습니다.

이 보고서 CODE는 승인된 4개 CSV와 새 full-map/summary에 대해 직접 CM·P/R·paired 변화·selected-k 패턴을 산술 재계산하고 생산자 SUMMARY 및 독립 QA와 대조했습니다. [상세 수치](tables/SUMMARY.json), [selected-k 전체 join](tables/SELECTED_K_PAIRS.jsonl), [주소별 DEV](tables/NATIVE_DEV.csv)와 입력/출력 바인딩을 함께 보존합니다. 게시 bundle에는 별도의 독립 QA/replay 경로를 포함합니다.

- [CONTROL_ORIGINAL_FIT 원 SOURCE CSV](matched_fit/CONTROL_ORIGINAL_FIT_ORIGINAL_ALL_SOURCE_ROWS.csv)
- [CONTROL_ORIGINAL_FIT 새 FIT CSV](matched_fit/CONTROL_ORIGINAL_FIT_NEW_KNOWN_SOURCE_FIT_ROWS.csv)
- [TREAT_ORIGINAL_PLUS_NATIVE_CHILDREN 원 SOURCE CSV](matched_fit/TREAT_ORIGINAL_PLUS_NATIVE_CHILDREN_ORIGINAL_ALL_SOURCE_ROWS.csv)
- [TREAT_ORIGINAL_PLUS_NATIVE_CHILDREN 새 FIT CSV](matched_fit/TREAT_ORIGINAL_PLUS_NATIVE_CHILDREN_NEW_KNOWN_SOURCE_FIT_ROWS.csv)
- [학습 COMPLETE](matched_fit/COMPLETE.json)
- [독립 결과 QA](artifacts/V256_ACTUAL_RESULT_QA_B/actual_review_v1/REVIEW.json)
- [최종 barrier](matched_fit/BOTH_FINAL1000_FROZEN_BEFORE_ANY_SOURCE_DEV_PREDICTION.json)
- [고정 사전 recipe](artifacts/V256_ROOT_NATIVE_DATA_FIT/PROSPECTIVE_DATA_ONLY_PAIR_POLICY.json)
- [V255 고정 16-case / 64-scene 자료 갤러리](references/V255_GALLERY/PORTABLE_GALLERY_KO.html)

## 공개 파일 검증과 원본 복원

코드·라벨·특징·두 모델의 최종 가중치·학습 log·표·그림을 함께 보존합니다. 큰 NPY는 헤더를 포함한 원본 바이트 그대로 gzip으로 보관합니다. 아래 두 번째 명령은 새 폴더에 모든 원본을 복원하고 각 SHA와 크기를 대조합니다. NumPy나 모델 실행 없이 검증할 수 있습니다. 메타데이터의 원 컴퓨터 절대 경로는 복원에 사용하지 않습니다.

```bash
python -B tools/restore_artifacts.py --package . --verify-only
python -B tools/restore_artifacts.py --package . --output-dir /ABS/FRESH_RESTORED
```

파일·가중치와 저장 보고 수치의 검산을 제공합니다. 다른 GPU 환경의 전체 재학습까지 동일한 결과가 난다고 입증한 것은 아닙니다.
