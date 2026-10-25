# V229: SOURCE 선택 후보의 고정 실사 비교 — 채택하지 않음

공개 SOURCE 결과를 바탕으로 선택한 V227 `CONTROL_FIXED_Q`를 원 실사 223장에 고정 상태로 적용했습니다. 새 모델의 D recall은 **955/1,147 = 83.26%**였지만, 외부 가림 E recall은 **27/90 = 30.00%**였습니다. 보이는 점을 덜 지우는 방향의 변화와 가림을 놓치는 문제가 동시에 나타났으므로 **실사 적용 모델로 채택하지 않습니다**. 이 결과를 보고 모델·문턱을 바꾸지 않았습니다.

공개 SOURCE DEV303 약라벨에서 이 후보의 D recall은 258/267=96.63%, E recall은 34/36=94.44%였습니다. FIT는 D57/E8의 65행이며, 그 SOURCE 결과를 현재 실사 성능으로 해석할 수 없습니다. 이번 실사 balanced recall은 56.63%로 legacy 74.66%보다 낮습니다.

이 선택은 이번 실사 GT·baseline 점수를 열기 전에 사용자 요청과 SOURCE 결과로 고정했습니다. V227의 attention 추가 효과 미확인 판정은 그대로이며, 후보 선택이 배포 채택을 뜻하지 않습니다. V229의 새 학습·정규화 fitting·PnP·GT 좌표 보정·instance/nearest/permutation 탐색은 모두 0입니다.

## 같은 primary에서 D와 E를 함께 비교

Primary는 기존 `paired_same_query_RGB_D_vs_E=True`이면서 human D/E인 **1,237행(D 1,147 / E 90, 222장)**입니다. 원 등록 입력은 **223장 × 8 = 1,784행**이며, 별도 same-slot D/E 1,248행과 혼동하지 않습니다. Primary에 새 좌표 오차·owner·availability 필터를 추가하지 않았습니다. U도 해당 GT 분모에 남깁니다.

혼동행렬의 행은 실제 **D/E**, 열은 판정 **D/E/U**입니다. P/R은 precision/recall이며, recall 분모는 U를 포함한 각 GT 클래스 전체입니다.

| 범위 (D/E) | 모델 | 2×3 혼동행렬 `[[D→D,E,U],[E→D,E,U]]` | D P / R | E P / R | U 비율 |
|---|---|---|---|---|---|
| 전체 (1147/90) | v61 | `[[260,887,0],[2,88,0]]` | 99.24% / 22.67% | 9.03% / 97.78% | 0.00% |
| 전체 (1147/90) | legacy | `[[655,319,173],[3,83,4]]` | 99.54% / 57.11% | 20.65% / 92.22% | 14.31% |
| 전체 (1147/90) | V229 CONTROL_FIXED_Q | `[[955,192,0],[63,27,0]]` | 93.81% / 83.26% | 12.33% / 30.00% | 0.00% |
| WOOD (616/29) | v61 | `[[145,471,0],[0,29,0]]` | 100.00% / 23.54% | 5.80% / 100.00% | 0.00% |
| WOOD (616/29) | legacy | `[[448,65,103],[2,23,4]]` | 99.56% / 72.73% | 26.14% / 79.31% | 16.59% |
| WOOD (616/29) | V229 CONTROL_FIXED_Q | `[[544,72,0],[23,6,0]]` | 95.94% / 88.31% | 7.69% / 20.69% | 0.00% |
| SQUARE (531/61) | v61 | `[[115,416,0],[2,59,0]]` | 98.29% / 21.66% | 12.42% / 96.72% | 0.00% |
| SQUARE (531/61) | legacy | `[[207,254,70],[1,60,0]]` | 99.52% / 38.98% | 19.11% / 98.36% | 11.82% |
| SQUARE (531/61) | V229 CONTROL_FIXED_Q | `[[411,120,0],[40,21,0]]` | 91.13% / 77.40% | 14.89% / 34.43% | 0.00% |

전체에서 v61은 D 260행·E 88행을 맞혔고, legacy는 D 655행·E 83행을 맞혔으며 U는 177행입니다. 새 후보는 D 955행·E 27행을 맞혔고 primary U는 0입니다. 새 후보의 단순 정답률 **982/1,237 = 79.39%**만 보면 높아 보이지만 D가 분모의 대부분이며, **외부 가림 63/90행을 D로 판정**한 실패를 가려서는 안 됩니다. WOOD의 E recall 20.69%, SQUARE의 E recall 34.43%에서도 같은 약점이 보입니다.

[Primary 원표](results/PRIMARY.csv) · [D/E/U 혼동행렬](results/D_VS_EXTERNAL_E_2X3.csv) · [같은 query 회복·악화](results/PAIRED_RECOVERY_AND_DAMAGE.csv) · [native별 primary](results/PRIMARY_NATIVE8.csv)

## 원 availability와 SELF를 분리

원 1,784행 중 입력 unavailable은 **45행**입니다. 이번 실제 후보 출력도 D 1,389 / E 350 / U 45였고, 원 unavailable 45행이 U로 남았습니다. 일반 실행 계약은 원 query가 available이어도 context/local 특징을 지원할 수 없으면 이유가 있는 추가 U를 허용합니다. U는 E로 바꾸거나 confidence 문턱으로 만든 판정이 아닙니다. 전체 등록 슬롯의 상태 분포를 primary 성능 분모로 사용하지 않습니다.

Human SELF는 원 전체 **136행**, paired 플래그가 있는 부분은 **21행**입니다. 아래 E는 SELF 행에서 나온 binary absent 응답일 뿐, 외부 가림 원인이나 SELF 분류를 새로 학습했다는 증거가 아닙니다. SELF를 primary 외부 E 90행에 합치지 않았습니다.

| 모델 | SELF 전체 136행의 D/E/U | paired SELF 21행의 D/E/U |
|---|---|---|
| v61 | 115/21/0 | 9/12/0 |
| legacy | 13/118/5 | 13/3/5 |
| V229 CONTROL_FIXED_Q | 132/4/0 | 19/2/0 |

[전체 슬롯 D/E/U](results/ALL1784_MODEL_INVENTORY.csv) · [SELF binary 응답](results/HUMAN_SELF_BINARY_RESPONSE.csv)

## 수동 클릭 reference 근처에서도 남는 오삭제

기존 수동 클릭 좌표와 원 query의 오차가 3px 이하이고 human D인 **376행**에서 새 후보는 D 297행(**78.99%**)을 유지했지만, **79행을 E로 오삭제**했습니다. WOOD는 126/153 D 유지·27 E, SQUARE는 171/223 D 유지·52 E입니다. 비교하면 v61은 D 85 / E 291 / U 0, legacy는 D 205 / E 108 / U 63입니다.

이 376행은 기존 reference 오차 ≤3px인 human D 전체 461행과 다른 집합입니다. 기존 좌표 진단 구간 **≤3px / >3px / reference missing**만 유지했으며 새 8px 구간, 좌표 정확도 필터 또는 좌표 정답을 만들지 않았습니다. Reference와 visibility의 의미를 동일시하지 않습니다.

[Manual-near-D 원표](results/MANUAL_NEAR_D.csv) · [기존 좌표 구간별 표](results/COORDINATE_ERROR_BINS.csv)

## 입력·모델·실행 계보와 재현 범위

- 모델은 V227 SOURCE 학습의 `CONTROL_FIXED_Q_FINAL300.pt`를 그대로 사용했습니다. checkpoint SHA256 `d1dacc959cb9cca1008250ac4fbfd050f6442ed7d0d40e63f844f577fd6292e2`, head 코드 SHA256 `042ffd9c7bced899e0de2546109cdaafff54db27b47d7cd24113eb056570c5dc`입니다. [기존 공개 checkpoint](../V227_CONTEXT_READ_SOURCE_PROBE/models/CONTROL_FIXED_Q_FINAL300.pt) · [head 코드](../V227_CONTEXT_READ_SOURCE_PROBE/code/head.py) · [SOURCE 학습 COMPLETE](../V227_CONTEXT_READ_SOURCE_PROBE/results/TRAINING_COMPLETE.json).
- 고정 encoder는 DINOv2 ViT-S/14 reg4이며, 공식 사전학습 weight의 기록된 SHA256은 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`입니다. Encoder weight 본문은 이번 결과 package에 새로 포함하지 않습니다. 입력은 원 fullRGB와 원 initializer q8/native/availability, 원 q/L의 local96 및 원 q8 min/max context224입니다. GT reference/pose/ROI/마스크는 배포 특징으로 넣지 않았습니다.
- 현재 실사 처리 전 공개 SOURCE의 기존 첫 frame·6 query를 같은 batch 구성으로 replay한 저장 영수증은 descriptor exact, logit 최대 차이 **0**, 부호 exact, 반복 exact를 기록했습니다. 이는 제한된 구현 일치 검산이며 모든 영상의 의미 대응이나 실사 성능을 보증하지 않습니다. 실사 처리 전후 encoder/head state hash도 같고 첫 supported current batch 반복이 exact였습니다.
- 실사 추론 COMPLETE SHA256 `37a34f091a75a36c4f92ce4e2203922c17a188fb6ab6fc2e132c5a8df12f2824`, 예측 동결 SHA256 `bf63b51f7f2d092a82597213457b0cdb3bc470bceb3094c7f01ba598ebe09c8c` 후에만 기존 human/baseline을 읽었습니다. 평가 COMPLETE SHA256 `cc82e57d2b9c389f5366aef46825d36d954844e2c270f1e13b4af8df049b8c86`의 저장 결과가 이 보고서의 수치 출처입니다.
- 기존 v61/legacy의 저장 응답·GT join·primary·좌표 구간·재료·SELF 계약을 재사용했습니다. 입력 schema와 portable U guard의 사전 교정 이력은 원본과 교정본을 함께 보존하며, 초기 준비본을 최종 성공본으로 바꾸어 해석하지 않습니다. 최종 ledger에는 원 ID/native/q/L/availability와 reference provenance를 나란히 남겼습니다.
- [저장 ledger](results/ALL1784_LEDGER.csv)와 [표 산술 검산기](VERIFY_SAVED_CURRENT.py)는 모델 없이 primary CM/P/R/U·행 순서·availability·logit 부호를 다시 셉니다. 이 검산은 원 GT 권위, 원 이미지, encoder forward를 독립 재인증하는 절차가 아닙니다. 추론·평가 코드는 별도 원 실행 기록으로 보존합니다.

## 해석과 남은 한계

이번 실사에서는 보이는 점을 덜 지우는 효과와 외부 가림을 지나치게 D로 남기는 문제가 함께 나타났습니다. SOURCE 약라벨의 대상 범위, native 4/5에 집중된 E 감독, 실사에서 요구하는 D/E 의미·분포의 차이는 가능한 불일치 요인입니다. 그러나 이 평가만으로 특정 원인, 순수 transfer 문제, GT 전체 오류 또는 backbone 부족을 확정하지 않습니다.

현재 실사는 이 후보의 학습·teacher 생성·정규화 fitting에 사용하지 않았습니다. 다만 여러 과거 실험에서 반복 평가된 셋이며 fresh blind 검증이 아니고, legacy의 과거 CAL 노출 중첩도 유지됩니다. 작은 공개 SOURCE 점수를 실사 검증 없이 성공으로 일반화하지 않습니다. 새 모델·문턱·배포 채택은 없으며 목표는 미해결 상태입니다.

## 고정한 성공·실패 crop 8개

[갤러리와 8개 이미지](figures/PORTABLE_GALLERY_KO.html) · [영상 열기 전 고정 ID](figures/FIXED8_CASE_IDS_BEFORE_RGB.json) · [원 q/L·pixel SHA·정답/출력](figures/CASE_ROWS_AND_DISPLAY_FACTS.json) · [crop 관찰](VISUAL_READOUT_KO.md)

WOOD/SQUARE × D 정답·D→E·E 정답·E→D에서 사전 정책대로 8개 서로 다른 프레임을 골랐습니다. D→E는 수동 클릭 reference 오차≤3px를 우선했습니다. 정답/오답이 잘 보이는 사진으로 바꿔 고르지 않았습니다. 원 RGB와 q/L에서 local96·context224·center32를 표시했으며, GT reference로 다시 자르지 않았습니다.

## 보존한 준비·오류와 실행 범위

[실행하지 않은 V228 공개 YCB query 관측 준비](V228_PREP_ONLY/README_KO.md)도 함께 공개합니다. V228 GT/depth/mask 실제 sidecar 생성·학습은 실행하지 않았습니다. V229 실행 전 manifest 키 연결 오류를 발견했고, 첫 SOURCE preflight에서는 NumPy 반환값을 Torch로 취급한 adapter 오류가 났습니다. 최종 V3는 그 연결만 수리했으며, 현재 실사 이미지에 도달하기 전 두 문제를 해결했습니다. [실패 기록](protocols/V2_SOURCE_PREFLIGHT_ADAPTER_FAILURE_AUDIT.json) · [최종 추론 코드](code/infer_fixed_context_v3.py) · [독립 수치 QA](qa/ACTUAL_METRICS_QA.json). 이 수리는 성능 실험·재학습·문턱 조정이 아닙니다.
