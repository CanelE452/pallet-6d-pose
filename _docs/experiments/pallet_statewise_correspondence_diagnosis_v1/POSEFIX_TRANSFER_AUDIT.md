# PoseFix 원문·현재 비교모델·과거 시도 전달 감사

현재 `POSEFIX_RAW`는 **합성 데이터로 학습한 팔레트용 PoseFix 포트의 seed1, 6,000번째 마지막 checkpoint를 한 번 적용한 uncapped 출력**이다. 실사 replay로 추가학습한 last300도, synthetic selection에서 고른 1% cap 출력도 아니다. Gaussian 입력과 좌표 손실은 이미 구현되어 있다. 이번 감사에서는 새 학습·모델 forward·자세 solve를 실행하지 않았다.

기준 소스는 `96360c61418c435602875f5f34ca8ea4c694fbdc`이다. 원문 접근 기록, 직접 읽은 범위, 짧은 인용, 로컬 코드·결과 40개의 SHA는 [LITERATURE_SOURCE_AUDIT.json](LITERATURE_SOURCE_AUDIT.json)에 있다. 소스·과거 기록에 더해 봉인 후 실제19경로·CAL 잔차 분석을 아래에 연결했다. 현재 수치 재집계 완료와 독립 검산·최종 게시 완료는 구분하며, 선행연구로 현재 효과를 대신하지 않는다.

## 원문에서 가져오는 원리와 구현 차이

[PoseFix 저자 공개본](https://arxiv.org/pdf/1812.03595)의 §4, §5.1–5.2는 이미지와 초기 pose를 함께 사용하고, 초기 pose를 Gaussian으로 표현한 뒤 heatmap과 좌표를 감독한다. 학습 입력은 good/jitter/inversion/swap/miss를 합성한다. inversion은 같은 사람의 좌우, swap은 다른 사람의 부위 혼동이다. 이 분류를 팔레트의 W/D 오류·배경선 오검출과 자동으로 동일시하지 않는다. good 입력을 포함한다는 사실은 정상점의 무손상 보장이나 가려진 점 복원 보장이 아니다. Gaussian도 교정된 신뢰구간을 뜻하지 않는다.

공식 구현은 [PoseFix_RELEASE](https://github.com/mks0601/PoseFix_RELEASE)의 고정 commit `5556364bb0f43b0743a5fcd820de48f34b3d4360`으로 확인했다. [model.py](https://github.com/mks0601/PoseFix_RELEASE/blob/5556364bb0f43b0743a5fcd820de48f34b3d4360/main/model.py)는 Gaussian 입력, spatial softmax·expectation, heatmap cross entropy와 coordinate L1을 구현한다. [gen_batch.py](https://github.com/mks0601/PoseFix_RELEASE/blob/5556364bb0f43b0743a5fcd820de48f34b3d4360/main/gen_batch.py)의 `synthesize_pose`와 `generate_batch`는 사람 관절별 오류합성 및 scale/rotation/flip을 담당한다. 이 공식 loader는 BGR→RGB 변환이 주석 처리되어 있으며, 현재 팔레트 port의 RGB 사용은 명시적인 목적 적응이다.

| 항목 | 원문·공식 구현 | 현재 synthetic-only 팔레트 port / RAW comparator |
|---|---|---|
| 대상·채널 | 사람 관절, 관절 종류·근접 사람·좌우 짝 사용 | 팔레트9점. RGB3+Gaussian9 입력, 팔레트 코너8개 감독. 중심8은 출력 복원 시 원점 유지 |
| 학습 입력 pose | 사람 오류 통계에 따른 합성 pose | 이미 고정된 R0의 예측점·예측 bbox. 별도 사람 오류합성 없음 |
| crop | aspect 보정·1.25배 crop, 학습시 기하증강 | 예측 bbox aspect 보정·1.25배, 384×288, bilinear, 상수0 밖 채움. GT bbox 대체 없음 |
| 색·입력 표현 | 공식 코드의 평균 제거 및 Gaussian | RGB 평균 `[123.68,116.78,103.94]` 제거. Gaussian sigma9 crop px, amplitude255 |
| 구조·출력 | ResNet152 옵션, upsampling, soft-argmax | ResNet152 계열, deconv3단, 96×72×9 logits, expectation×4 |
| 손실 | heatmap CE+coordinate L1; 공식 trainer의 convolution L2 | CE+좌표L1+convolution L2. 중심8 손실 제외. crop 밖·비유한 GT 감독 제외, 평가 분모에서는 별도 실패 처리 |
| augmentation | 사람 pose 오류, scale/rotation/좌우 flip | trainer에 새 pose noise/flip/scale/rotation 없음. 고정 source row 순서·고정 이미지 crop |
| 학습량 | 공식 config 140epoch | 과거3seed×6,000update, batch16, seed별96,000노출. fixed last6000. 140epoch 수렴 비교 아님 |
| 초기화 | 공식 README의 ImageNet backbone 경로 | TF-slim ImageNet ResNet152 계보를 검산한 포트. 사람 pose 가중치 사용 주장 없음 |
| 현재 RAW 후처리 | 해당 팔레트 F는 원문 구성 아님 | RAW PASS1 출력→기존 W/D selector→SQPnP+RefineLM. 추가 cap·SubPix·B/I 없음 |

로컬 구현 근거는 [prior_model.py](../../../scripts/research/pallet_sensors_submission_v1/prior_model.py)의 `gaussian`, `PoseFixPallet9.losses`, [prior_data.py](../../../scripts/research/pallet_sensors_submission_v1/prior_data.py)의 `SourceRGB.item`, [train_prior.py](../../../scripts/research/pallet_sensors_submission_v1/train_prior.py), [posefix_contract_math.py](../../../scripts/research/pallet_sensors_submission_v1/posefix_contract_math.py)의 `axis_aligned_crop_matrix`다. `train_prior.py`는 checkpoint의 loader RNG를 “No augmentation/random worker state”로 기록한다. 최적화는 TFAdam, lr0.0005, update3858·5143에서 각각1/10, clipping 없음이다. 이들은 과거 실행의 설정이며 이번 새 학습량은0이다.

## 현재 238장 comparator의 정확한 계보

- 초기 checkpoint: `data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR1/last.pt`.
- checkpoint SHA256: `b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581`.
- 저장 neural cache: `data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed1_REAL_DEV.npz`, SHA256 `683b7395860865f7e10cba5e1dfb7c04e84234dd296340416c4759b671c95f06`.
- 사용 tensor: `points[selected_rows,0,1]`: chain0=RAW, pass1=첫 보정. 결측과 중심점은 R0에서 복원한다.
- CAL46+TEST49의 기존95장과 REMAINDER143의 저장된 F pose를 연결한다. [143장 연결 코드](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/compare_remainder_posefix_cache_v4_14.py)는 원95와 pose/Q/치수·RGB·K가 같음을 검사하며 새 모델 forward나 PnP 없이 저장 결과를 추출했다.
- 현재 RAW에는 source selection의 `lambda=1, cap=.01×image diagonal`을 적용하지 않았다. 이 wrapper는 [PRIOR_SELECTION.json](../pallet_sensors_submission_v1/PRIOR_SELECTION.json)의 다른 출력 계약이다. RAW를 “cap 때문에 움직이지 않는 PoseFix”로 설명하면 틀린다.

[95장 PoseFix+SubPix protocol](../pallet_visibility_curriculum_20261013_v1/nonsevere_v1/posefix_sub_comparison_v1/PROTOCOL.json)은 PF-anchor와 YOLO-anchor의 두 1% cap을 채점 전 함께 고정했고 winner를 선택하지 않았다. SubPix는 window5×5, maxCount40, epsilon.001이다. **두 결과가 존재한다는 사실은 전체192용 단일 규칙이 이미 채택됐다는 뜻이 아니다.** 전체 확장 때 PRIMARY 점수로 둘 중 유리한 cap을 선택하지 않는다. 현재 소스 근거에서의 판정은 `POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED`이며, 추가 단일 규칙 근거가 확인될 때만 변경한다.

## 과거에 실제 실행한 것

다음은 이번 실험 결과가 아니라 연결된 과거 기록이다. 서로 다른 모집단·교사·감독·지표를 현재 PRIMARY192에 합쳐 집계하지 않는다. “replay”는 학습 데이터 재사용을 뜻할 수도 있고 이름이 비슷한 반복추론 진단을 뜻할 수도 있으므로 구분한다.

| 실제 시도 | 입력·학습·후처리 | 관측된 결과와 제한 | 정확한 출처 |
|---|---|---|---|
| Synthetic-only PRIOR | frozen R0+RGB,3seed×6,000update, last only | 현재 RAW의 원모델. 사람 오류합성 미적용. matched exposure이지 수렴 우월성 비교 아님 | [protocol](../pallet_sensors_submission_v1/PRIOR_PROTOCOL_LOCK.json), [training audit](../pallet_sensors_submission_v1/TRAINING_AUDIT.json) |
| RAW/CAPPED 반복추론 | 같은 PRIOR1/2/3, 각 PASS0–3. 학습0 | DEV319의 초기>20px matched444코너 중 첫 RAW ≤10px 회복은 seed별1/1/2개. 초기 큰 오류의 cap-hit4.05%, 같은 입력에서 cap만 회복을 막은 수는 seed평균.333개. cap 해제만이 주병목이라는 근거 없음. 반복의 손익은 분할·지표별로 다름 | [진단](../pallet_posefix_replay_diagnosis_v1/FINAL_DIAGNOSIS.md), [원수치](../pallet_posefix_replay_diagnosis_v1/FINAL_DIAGNOSIS.json), [실제 추론 호출](../../../scripts/research/pallet_posefix_replay_diagnosis_v1/run.py) |
| 실사 large-error | PRIOR1 복사본,9장/38 manual점,300update.50% R0,50% manual+예측 bbox대각선5–15% 방향 교란 | TRAIN38점·304교란 probe PCK10 100%. DEV72 PCK10 기존57.838→45.225%, good138중35손상. 학습사진에서의 이동 능력은 확인, 평가 전이는 악화. 감독·warm start·교란 효과를 단독 분리한 실험 아님 | [결과](../pallet_posefix_large_error_v1/RESULTS_KO.md), [fit](../pallet_posefix_large_error_v1/FIT.json), [교란 코드](../../../scripts/research/pallet_posefix_large_error_v1/train.py) |
| 실사+합성 GT replay | 같은 초기 PRIOR1, 실사 index/교란/2,400노출 유지+합성2,400노출 추가,300update, BN통계 고정 | DEV72 회복4/156·손상1/138. GREEN150 손상14/344, source clean PCK10 94.016→90.950%. 진입 gate6개 실패, 후속 선택적 self-training 미시작. 계산량 증가 때문에 compute-matched 아님 | [결과](../pallet_posefix_replay_v1/RESULTS_KO.md), [판정](../pallet_posefix_replay_v1/DECISION.json), [실제 계약](../../../scripts/research/pallet_posefix_replay_v1/core.py) |
| pseudo DENOISE | 기존 Replay 복사본 CLEAN/DENOISE 각300update, 같은 실사217장·합성 replay·목표. DENOISE는50% R0/25% 한 신뢰점/25% 전체 신뢰점 교란 | 자연 오류 전체194장: 기존Replay/CLEAN/DENOISE 회복7/5/14(분모281), 손상5/7/13(분모511). DENOISE P90 50.52px, >40px오류 회복0. 인위적 교란 노출3,416회는 자연 오류3,416개 복구가 아님 | [해석](../pallet_type_selftrain_v1/selftrain_recovery_v1/pseudo_denoise/INTERPRETATION_KO.md), [코드](../../../scripts/research/pallet_type_selftrain_v1/recovery_pseudo_denoise.py) |
| Clean19·재료별 replay | 기존9/38과 다른19장/87점, mixed300 또는 재료별300 | Wood116에서 PRIOR/mixed/type PCK10 73.41/67.83/68.93%. 감독·재료별 노출량·모집단이 현재와 달라 재료 자체의 원인이나 현재192의 효과로 사용하지 않음 | [mixed](../pallet_replay_clean19_v1/RESULTS_KO.md), [type](../pallet_replay_by_type_v1/RESULTS_KO.md), [기존 감사](../pallet_oracle_mechanism_followup_v1/PRIOR_ATTEMPTS.md) |

반복추론의 작은 수치 비결정성도 이미 분리했다. 동일 입력10회×3seed의 최대 차이는 crop.0000610352px, 원좌표.000166554px이며 기존 허용치.0003 crop px 안이었다. 이것을 큰 회전·위치 실패의 설명으로 쓰지 않는다. 위444개는 코너 분모이고, 현재192장 pose 성공 수와 같은 단위가 아니다.

## 오류지원과 정상 입력 보존에서 아직 말할 수 없는 것

현재 synthetic-only trainer는 R0가 합성 이미지에서 실제 낸 오류를 본다. 별도 사람 inversion/swap, 실제 팔레트 W/D의 오류빈도, 상태별 S/O/T 비율을 학습 입력에 맞춰 넣었다는 증거는 없다. 반대로 Gaussian sigma9가9px까지만 움직일 수 있는 hard radius라는 뜻도 아니다. 출력은 crop 전체 heatmap의 expectation이며, sigma는 입력 표현이다.

실사 large-error/replay는 5–15% bbox대각선의 위치 교란을 이미 사용했다. 그러나 이것이 자연의 다른 물리 코너 대응, 잘못된 선 owner, partial→full 승격, W/D assignment 오류를 대표한다는 검증은 없다. pseudo-target denoising은 틀릴 수 있는 기존 목표를 복원하는 능력과 실제 정답 정확도를 분리해야 한다.

CAL46의 기존 참조와 상태·owner 계보를 결합한 현재 오류지원 범위는 **`COVERAGE_NOT_VERIFIED`**다. 고정 예측·후보 봉인 후 원 N3, 원 e, 후단을 대상으로 같은 모델 물리 코너의 잔차와 owner/가설 변화를 분석했다. 원자료는 아래 진단 파일에 전량 보존했다. S/B/I/O/T/U는 관측 가능성이고 good/jitter/inversion/swap/miss는 예측 오류이므로 일대일 매핑하지 않는다. 유효 대응의 정답이 없는 점은 `SEMANTIC_OWNER_UNVERIFIED`로 남긴다. 허용 C2 동등성은 오류로 세지 않는다.

“좋은 입력 손상”도 단순히 이동했다는 사실과 구분한다. 기존<5px→>10px와 같은 고정 사건은 해당 코너 분모를 함께 쓰고, 현재 pose 성공→실패는 이미지 분모로 따로 센다. PoseFix가 회귀한 S/O/T 좌표를 실제 B/I 영상관측이라고 승격하지 않는다. 전체 crop을 보는 CNN이 S와 무관하다고 주장하지 않는다.

## 이번 실행의 판단

새 학습이 필요하다는 결론은 아직 없다. 우선 원래 관측 종류·owner·활성 상태의 전달과 동일 관측의 자세 선택을 진단한다. 이 단계에서 이미 구현된 Gaussian·좌표 손실·무조건 cap 해제·반복추론·같은 교란/replay를 새로운 해결책으로 재명명하지 않는다. 새 학습안은 채택하지 않았고 실제 optimizer update는0이다. 후속 제안이 필요해지면 확인된 오류지원 공백 하나, 기존 실패와 달라지는 조건, 분리된 학습/선택/평가, B/I 한정 사용과 S/O/T 정책, 동일 예산 대안 및 good 입력 손상을 함께 명시한 **하나의 변경안**으로만 제시한다.

## 봉인 후 확인한 현재 결과

| 공통분모 | RAW | PF-anchor cap | YOLO-anchor cap | N0 | 기존 e |
|---|---:|---:|---:|---:|---:|
| CAL | 32/46 | 32/46 | 32/46 | 34/46 | 37/46 |
| TEST | 34/49 | 39/49 | 38/49 | 40/49 | 49/49 |
| COMMON95 | 66/95 | 71/95 | 70/95 | 74/95 | 86/95 |

PRIMARY192에서는 PF RAW 80/192, 기존 e 114/192다. e 기준 PF 대비 회복 49장·손상 15장이고, 둘 다 실패 63장을 그대로 보존했다. e의 위치 평균은 14.112109cm, PF RAW는 13.835732cm이며 회전 평균은 각각 13.001138°/11.639767°다. 공동 성공 수의 우세가 모든 평균의 우세는 아니다.

두 cap은 공통95에서만 각각 재집계했고 winner를 고르지 않았다. 전체192용 PF+SubPix 규칙은 여전히 `POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED`다. [두 cap 수치·전이·CI](metrics/POSEFIX_SUBPIX_COMMON95.json), [전19경로 결과](metrics/RESULTS.json), [전체 행별 오차](metrics/ROW_METRICS.csv)를 함께 확인한다.

PRIMARY192×8=1536개 R0→PF 입력 코너에서 입력<5px 684점 중 출력>10px 0점, 입력>20px 274점 중 출력≤10px 0점이었다. 이 고정 사건이0이라는 사실은 모든 좋은 점의 무손상이나 pose 무손상 증명이 아니다. PF와 e의 입력·후단이 달라 이미지 단위 PF/e 손상과 직접 등치하지 않는다.
CAL 원 e의 활성 complete 192점에서는 N3 입력<5px 57점 중 출력>10px 9점, 입력>20px 21점 중 출력≤10px 1점이었다. 잔차는 기존 기하 재구성 참조의 투영과 초기점 묶음의 영상별 전역 C2 phase에 대한 값이다. 독립 2D 클릭 정답·의미 owner 정답이 아니다.

CAL 상태별 잔차·물리 vertex 운반·점 이동·원녹화별 네 pose 패턴은 [DIAGNOSTICS.json](diagnostics/DIAGNOSTICS.json), [CAL owner 감사](diagnostics/CAL_OWNER_AND_STATE_AUDIT.json), [PF 입력 이동 전량](diagnostics/PRIMARY_POSEFIX_GOOD_INPUT_MOVEMENTS.csv)에 있다. 현재 합성 학습의 semantic/W-D/상태별 오류지원은 여전히 `COVERAGE_NOT_VERIFIED`다. Gaussian·좌표 loss·기존 replay를 새로운 해결책으로 반복 제안하지 않는다.
