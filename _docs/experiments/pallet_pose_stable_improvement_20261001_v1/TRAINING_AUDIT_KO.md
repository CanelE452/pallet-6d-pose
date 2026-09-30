# 안정적인 T/R 개선을 위한 학습 경로 감사

작성일: 2026-10-01. 최초 감사는 파일·코드·저장된 학습 기록만 읽었다. 이후 동일한 새 목표 아래 CPU 타깃 준비와 고정 R0의 GPU 가림 입력 준비를 추가로 수행했다. 이 파일 담당자는 optimizer update와 checkpoint 수정을 하지 않았다. 학습 결과는 root가 별도로 기록한다.

**최종 준비 완료: SINGLE251 대 DIVERSE251, 같은 조건쌍의 seed1/2/3 반복을 위한 입력을 동결했다.** [INPUTS.json](INPUTS.json), [최종 데이터 검증](DATA_PREPARATION_FINAL.json), [학습 전 수정 계약](PROTOCOL_AMENDMENT_01.json). 모델 성능은 아직 이 데이터 준비로 입증되지 않는다.

253→252 변경: 독립 데이터 감사에서 DAY253의 `DAY264__005516` 이미지 SHA가 PAPER319의 `plastic_day_01:005516`과 동일함을 확인하여 사전 제외했다. 252→251 변경: 아래 가림 입력 검사의 한 쌍 실패를 그대로 보존하고, 대조군은 `(실제 image SHA,id)` 사전순 최대인 `DAY264__005627` 한 개를 제거했다. 다른 프레임으로 대체하거나 기준을 완화하지 않았다. 이 수정은 평가 출력·정답을 열기 전 TRAIN 품질 검사에만 근거했다. 아래 최초253개 제안과 원래252개 실패 계약을 보존하며, 기존 FULL253의 정확한 재현 실험이라고 부르지 않는다. root의 후속 학습 계약은 동일 조건쌍을 seed1/2/3에서 반복하는 최대6fit이다.

CPU 실물 확인: 캐시259개에 같은 hidden-only PnP/all8LOO를 적용해250개가 적격이었다. DAY252와 합친 후보에서 사전 정의한 hash 순서와 recording max-min 규칙으로 DIVERSE252를 선택했다. 선택 수는 REC_00150, REC_01149, REC_01729, REC_01949, REC_02035, REC_0237, REC_02412, REC_02821이다. 새 가림 R0 추론 대상은202개이며 나머지DAY50개는 기존 paired tensor를 재사용한다. 실제 완료·실패와 최종 입력은 [CPU 영수증](DATA_PREPARATION_CPU.json), [데이터 준비 계약](DATA_PREPARATION_PROTOCOL.json), GPU 영수증 및 `INPUTS.json`으로 확인한다. 선택한 쌍이 실패하면 다른 프레임으로 대체하지 않는 계약이다.

**GPU 준비 결과: 사전 선택252개 중251쌍만 통과했다.** REC_024의 `PLASTIC__10957de42de4bd188099b48c43743677039e60825001704bf19e1ca5e7ff3ff5`는 clean/가림 bbox IoU가0.461980으로 기존0.5 기준에 미달했다. 실패를 보존하고 재선택하지 않았으며 원래252개 계약의 성공 lock을 작성하지 않았다. 가림202회+고정 clean 재현1회=203회 R0 forward, stage wall19.687초, optimizer update0회다. 최초 재현은 keypoint/bbox/confidence/score의 저장값과 최대차0이었다. [GPU 영수증](DATA_PREPARATION_GPU.json), [재현 확인](DATA_PREPARATION_GPU_SMOKE.json). 이후 root가 명시적251개 수정 계약을 작성한 뒤 CPU에서 최종 입력만 동결했다. 수정 뒤 추가 모델 forward는0회다.

최종 SINGLE251은 REC_001251개, DIVERSE251은 REC_00150/REC_01149/REC_01729/REC_01949/REC_02035/REC_0237/REC_02411/REC_02821개다. 두 군의 전체502개 항목에 대해 image/pair hash, CLEAN/OCC 공통 감독 mask, index8 제외, inverse crop target 오차≤1e-4px, pair IoU≥0.5를 확인했다. seed별 `default_rng(6400+seed)`의300×8 index 배열을 두 군이 공유한다.

| 최종 TRAIN 입력 | SINGLE251 | DIVERSE251 |
|---|---:|---:|
| 유효 감독 코너 수 | 2008 | 1913 |
| RGB 가림 적용 이미지 | 184 | 181 |
| clean/가림 bbox IoU 최솟값 | 0.79634 | 0.59825 |
| 입력의 고정 pseudo target 오차 중앙값 px | 1.972 | 3.481 |
| 같은 오차 P90 px | 7.814 | 14.659 |
| 같은 오차 >20px 코너 수 | 50 | 122 |

위 난도 수치는 타깃 추종 난도이며 실제 GT 정확도나 평가 개선이 아니다. 결과를 보고 고른 hard mining은 없었다. 동일 이미지 수·업데이트 수이지만 유효 감독 코너 수는 다르므로 순수한 recording 개수만의 효과라고 해석하지 않는다. 최종 최대6fit의 코드상 예산은1,800update, real14,400+source14,400=28,800학습 이미지 노출이다. 실제 학습 시간·메모리·성능은 root의 실행 영수증으로 확인한다.

PAPER66은 REC_001/REC_002이며 기존 Replay teacher가 수동 감독38코너/9이미지를 받은 recording과 겹친다. 새 학습의 exact image 한 장을 제외해도 독립 확인셋이 되지 않는다. 현재 full128은 새 실사 TRAIN과 recording/image가 분리된다. teacher의 기존 실사 감독 계보를 포함하므로 "실사 GT를 전혀 사용하지 않은 방법"이라고 부르지 않는다.

**가장 근거 있는 학습 개입은 단일 recording 구성과 여러 recording 구성을 같은 수로 비교하는 것이다.** 이미 수행한 32장·2 recording 실험의 단순 반복으로 설명해서는 안 된다. 최초 아래 설계는253개를 제안했고, 위 최종 계약이252개로 수정했다. 학습만 가능하다는 사실은 자연99의 안정적인 T/R 개선을 보장하지 않는다.

## 1. 현재 모델과 작은 head 학습의 한계

`PoseFixPallet9`는 RGB 3채널과 입력 9점 Gaussian을 연결하는 12채널 입력, ResNet-152 계열 bottleneck (3,8,36,3), deconvolution 3개, 9채널 heatmap 출력으로 구성된다. 입력 crop은 288×384, 출력 heatmap은 72×96이며 expectation으로 원 crop 좌표를 복원한다. 전체 파라미터는 68,661,641개다.

FULL125의 이름은 crop 배율 1.25를 가리킨다. 실제 학습량은 TRAIN253/REC_001, PRIOR1 초기화, seed1, 300 update, real8+source8, microbatch2, TFAdam 1e-4다. 모든 BN 통계와 affine을 동결하고 convolution 68,508,681개를 학습했다. 타깃은 고정 Replay+self-occlusion PnP이며 중심 index8은 감독하지 않는다. 좌표·RGB·crop과 별개로 confidence, invalid, 원래 검출 정보 보존이 필요하다.

CPU에서 output head만 학습하는 코드는 기술적으로 만들 수 있다. 기존 `HEAD_ONLY` wrapper의 학습 대상은 9×256+9=2,313개뿐이며, 공간 위치별 1×1 channel mixing이다. 이는 원래 backbone이 가림에서 놓친 RGB 증거를 새로 학습하는 경로와 다르다. head 학습이 빨리 끝난다는 이유만으로 현재 목표의 적절한 대안이라고 판단할 근거는 없다. 기존 학습 entrypoint는 CUDA를 요구하므로 CPU 전환은 새 실행 계약과 수치 검증을 필요로 한다. 이번 감사는 CPU backward 시간을 측정하지 않았다.

주요 근거: [모델 코드](../../../scripts/research/pallet_sensors_submission_v1/prior_model.py), [학습 코드](../../../scripts/research/pallet_posefix_limited_adaptation_pilot_v1/train.py), [학습 파라미터 계약](../pallet_posefix_limited_adaptation_pilot_v1/TRAINABLE_CONTRACTS.json), [FULL 학습 영수증](../pallet_posefix_limited_adaptation_pilot_v1/FIT_FULL.json).

## 2. 반복하면 안 되는 기존 시도

| 기존 시도 | 실제 관찰 | 이번 선택에 주는 의미 |
|---|---|---|
| FULL vs SAME_LAYER vs LoRA | 제한 학습 두 군 모두 사전 continue gate 실패. LoRA는 FULL 대비 PRIMARY PCK10 −2.525pp, hard 복구 −6개 | 파라미터를 줄이는 것만으로 안정성을 얻었다고 할 수 없다. |
| FULL+source preservation | PRIMARY PCK10 50.35%, FULL 대비 +0.00pp, 판정 INCONCLUSIVE | 같은 preservation loss 재실행은 새 원인 검증이 아니다. |
| 구조적 좌표 오류 | radial/near-corner/pair-swap 300step은 DEV72 PCK10·hard복구 사전 gate 실패 | 방향·오류 합성만 다시 늘리는 선택을 우선하지 않는다. |
| crop1.50 | 새롭게 도달 가능한 14코너 복구 0개, PRIMARY 정답 +27/손실27 상쇄 | crop 확대를 다시 주 개입으로 잡지 않는다. |
| SINGLE32 vs MULTI32 | DAY32 대 DAY16+night01 16, 같은 PRIOR1/300step/source/loss. PRIMARY +0.701pp. night 두 세션 정답 +7, 나머지 −2 | 다중 촬영 분포를 넓히는 약한 신호가 있다. 그러나 2 recording·32장 결과로 자연99의 안정적 T/R 개선을 주장할 수 없다. |
| 최근 4코너30px stress | FULL125 복구 independent79/116, correlated81/116 | 동반 방향 상관성 자체를 주요 실패 원인으로 전제하는 학습은 지지되지 않는다. |

근거: [limited adaptation 판정](../pallet_posefix_limited_adaptation_pilot_v1/DECISION.json), [preservation 결과](../pallet_posefix_full_preserve_v1/RESULTS_KO.md), [구조적 오류 결과](../pallet_posefix_structured_error_v1/RESULTS_KO.md), [crop 결과](../pallet_posefix_crop_completion_v2/REPORT_KO.md), [32장 다양성 결과](../pallet_posefix_heatmap_diversity_v1/RESULTS_KO.md), [최근 진단](../pallet_pose_diagnosis_20260930_v1/REPORT_KO.md).

## 3. 정확히 재사용 가능한 입력

아래 상대 경로는 저장소 루트 기준이다. 큰 비공개 캐시·checkpoint는 GitHub에 올라와 있지 않을 수 있다.

| 역할 | 정확한 파일/경로 | 이번 감사에서 확인한 상태 |
|---|---|---|
| 초기 checkpoint | `data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR1/last.pt` | 기존 lock의 SHA `b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581`. 새 실험에서 다시 실물 검증 필요 |
| 기존 FULL checkpoint | `data/pallet/results/pallet_posefix_limited_adaptation_pilot_v1/FULL_last300.pt` | 기존 학습 영수증 SHA `beab7db0785fbe27c0e4333d12bb67e2370071d273a5f2ab977db64a6c362790` |
| DAY253 paired tensors | `data/pallet/results/pallet_occlusion_refiner_transfer_v2/paired_inputs/DAY264__*.pt` | INPUT_LOCK이 가리키는 253개 실물 존재. 첫 파일을 읽어 CLEAN/OCC 각각 RGB·points·valid·matrix·target·target_valid 구조 확인 |
| DAY 입력 lock | `_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/INPUT_LOCK.json` | 253개 binding, target/mask/augmentation 연결 포함 |
| 기존 real 순서 | `data/pallet/results/pallet_occlusion_refiner_transfer_v2/REAL_ORDER.pt` | 실물 SHA `730c569adf9869a48df11c29426f9a7fb43c0e4476d1798624ce2f723bedb038` 확인. (300,8) index 순서를 두 군에서 공유 가능 |
| source 순서/heldout | `data/pallet/results/pallet_posefix_replay_v1/ORDERS.npz` | 실물 SHA `8f2307ba3cb17bd12905987af63013bd2709ca5b278ab6f494756e3584adcbbb` 확인 |
| source RGB/좌표/정답 | `data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json`, `cache/CACHE_MANIFEST.json`, `cache/CACHE_COMPLETE.json` | 기존 `SourceData` adapter가 prediction bbox crop, TRAIN/heldout 분리, 입력과 synthetic target 분리를 수행 |
| 다중 recording 후보 | `_docs/experiments/pallet_type_selftrain_v1/POOL.json` | 총2000 중 표준110×130×11 팔레트1000개, 8 recording. 다른 규격 GREEN1000은 제외 |
| 후보별 동결 raw/Replay | `data/pallet/results/pallet_type_selftrain_v1/pseudo_frames/<POOL.id>.json` | 표준1000개 전부 읽음. raw/판정/사전 flip-LOO 점수, 통과 시 refined가 저장됨 |
| 과거 2 recording pairs | `data/pallet/results/pallet_posefix_heatmap_diversity_v1/paired_inputs/*.pt` 및 `TRAIN_INPUT_LOCK.json` | MULTI32가 가리키는 32개 존재와 real/source 순서 SHA 확인. 같은 2 recording 실험으로 대체하지 않음 |
| 원 촬영 분리 | `data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json` | 현재 평가 기록·alias·partial-overlap와 새 입력을 다시 교차 검사할 근거 |

기존 `SourceData`는 합성 입력의 원래 R0 box를 사용하며 GT box로 crop을 만들지 않는다. Source train/heldout 및 원래 source row 순서를 유지해야 한다. source 좌표 정답이 있다는 이유로 실사 pseudo target의 정확도를 입증했다고 해석하지 않는다.

표준 팔레트 1000개의 실제 캐시 집계:

| Recording | 기존 Replay까지 통과하여 refined 좌표 존재 | 기존 Replay-only 최종 통과 |
|---|---:|---:|
| REC_011 | 99 | 94 |
| REC_019 | 53 | 52 |
| REC_020 | 38 | 35 |
| REC_017 | 29 | 29 |
| REC_028 | 21 | 21 |
| REC_024 | 12 | 12 |
| REC_023 | 7 | 6 |
| REC_049 | 0 | 0 |
| **합계** | **259** | **249** |

1000개 중 raw confidence 탈락728, raw flip/LOO 탈락13, 기존 최종 Replay all8LOO 탈락10, 통과249다. **249개는 FULL125와 동일한 최종 타깃이 아니다.** 이 캐시는 Replay-only 좌표이며 FULL125는 hidden-only PnP를 추가했다. 따라서 249를 그대로 DAY253의 타깃과 혼합하면 데이터 구성과 타깃 생성법이 동시에 바뀐다.

새 모델 forward 없이 가능한 첫 단계는 259개 stored refined에 고정된 기존 hidden-only PnP를 적용하고 all8LOO를 다시 평가하는 것이다. 과거 최종 탈락10개도 포함해야 FULL125와 같은 단계 순서가 된다. raw/flip 출력을 새로 만들었다고 표현해서는 안 되며, saved stage1 점수의 protocol SHA와 R0/Replay checkpoint 연결을 검증해야 한다. 기존 캐시에 원 flip 좌표는 저장되지 않아 원영상 flip 계산을 독립 재현한 검사는 아니다. 단계 연결에 문제가 있으면 해당 clean forward를 명시적으로 재실행해야 한다.

기존 E2 geometry helper는 치수 `1.1,.11,1.3`을 사용한다. GREEN110×110×15를 개수 확보용으로 섞으면 이 계약을 위반한다. 원 이미지 크기480×640·K·corner 순서를 점검하고, 원본 image hash 및 recording alias 중복을 현재 full128, 과거 평가/수동 감독 이력과 분리해 기록한다.

## 4. 권고하는 단 하나의 두 조건 비교

| 항목 | SINGLE253 | MULTI253 |
|---|---|---|
| 실사 구성 | 기존 DAY253/REC_001 | DAY와 다른 적격 recording에서 합계253장, 사전 고정된 recording 균형 선택 |
| 초기화 | 동일 PRIOR1 | 동일 PRIOR1 |
| 모델·loss·BN | FULL, 기존 CE+coord+원래 L2, BN 전부동결 | 동일 |
| update·순서 | seed1/300step, real8+source8, 같은 real index 순서·source row 순서·source corruption RNG7103 | 동일 |
| 실사 target | 동결 Replay+hidden-onlyPnP | 같은 recipe, 평가 reference 입력 없음 |
| 실사 augmentation | 기존 E2 rectangle recipe, 실제 OCC R0 | 같은 recipe와 고정 seed/계획, 실제 OCC R0 |
| checkpoint | last300 고정 | last300 고정 |
| 선택·평가 | 중간 DEV best 선택 없음, 실패 후 LR/epoch/비율 sweep 없음 | 동일 |

253개는 샘플 수를 늘리는 실험과 촬영 구성을 바꾸는 실험을 구별하기 위한 기존 예산이다. 후보별 최종 E2 target·paired support 적격을 먼저 결정한 뒤 recording별 최대 수를 균등하게 채우는 deterministic max-min 방식으로253장을 선택할 수 있다. recording 정렬·동률 규칙과 프레임 선택은 결과를 보기 전에 잠근다. 프레임은 각 recording의 경로 정렬 등간격 선택으로 가까운 프레임 집중을 줄이되 timestamp가 확인되지 않으면 시간 간격이라고 부르지 않는다.

기존 Replay-only 통과 수가 그대로 유지된다고 가정하면 DAY50+REC_011 50+REC_019 50+REC_020 35+REC_017 29+REC_028 21+REC_024 12+REC_023 6=253이 가능하다. **이는 새 PnP·crop-support 통과 전의 산술 예시이며 최종 학습 구성은 아니다.** 적격 수가 달라지면 사전 정의한 같은 선택 알고리즘을 적용한다. 최소3개 recording·253 unique image가 충족되지 않으면 같은 실험을 더 작은 수로 몰래 바꾸지 않는다.

오래된 FULL을 직접 대조로 쓰려면 새 runtime에서 입력·모델·실행 수치 재현을 증명해야 한다. 가장 해석하기 쉬운 방식은 두 군을 같은 현재 환경에서 처음부터 독립 학습하고, SINGLE253의 기존 FULL 대비 차이를 별도로 보고하는 것이다. 학습군끼리 target_valid 개수가 같다는 보장은 없으므로 실제 감독 코너 수·가림 적용 수·recording별 노출을 기록한다. 이는 동일 샘플 수/recipe/update 비교이지 순수한 recording 개수만의 인과 실험은 아니다.

## 5. 계산량과 검증 범위

코드로 확정되는 두 fit의 예산은 **600 optimizer update, 실사4,800회+source4,800회=9,600 학습 이미지 노출**이다. micro2인 기존 loop는 한 fit당 2,400 forward/backward microbatch 호출이다. 여기에 신규 selected frame의 가림 R0 추론, 새 모델별 평가128장, 필요한 source 보존 검사가 더해진다. 캐시 raw/Replay 재사용이 검증되면 신규 clean R0/Replay forward를 생략할 수 있다. 259개 후보에 PnP를 적용하는 것은 image forward가 아니다.

저장된 과거 GPU FULL 영수증은 313.699초, peak allocated 2,289,320,448bytes다. 이는 당시 한 fit의 관측값이며 현재 호스트의 완료 시간 추정이 아니다. 현 CPU/GPU의 실제 step 시간을 측정하지 않고 분/시간을 약속하지 않는다. CPU로 68.5M 파라미터 전체 backward를 수행하는 것은 head-only와 자원 요구가 다르다. 현재 root의 독립 호스트 검사에서는 GPU 사용 가능성이 확인되었으므로 sandbox의 CUDA 비노출을 물리 GPU 부재로 해석하지 않는다.

학습 후에는 current full128/natural99/clean29의 동일 T/R·GEO와 baseline-W/D 고정 결과를 함께 평가한다. 전체 분모/검출 실패/미매칭을 보존하고, 2D pseudo target 추종을 실제 T/R 개선과 분리한다. natural99의 recording별 T/R, paired recording CI, LORO, P90, clean 손상, source 보존을 보고하며 집계 중앙값만으로 승격하지 않는다.

한 seed·반복 사용 DEV에서 좋은 결과가 나와도 새 촬영에 대한 안정적 일반화가 확정되지 않는다. seed 반복과 독립 recording 확인은 실제 결과에 따른 후속 검증 과제다. 후보 다양성을 늘려도 pseudo target의 물리 정확도, 자연 가림의 paired clean reference 부재, 검출 pool 밖 실패는 별도 한계로 남는다. 최근 T 꼬리10장 중8장이 box 미매칭이므로, boxes를 고정한 refiner 학습만으로 전체 tail을 해결했다는 주장은 불가능하다.
