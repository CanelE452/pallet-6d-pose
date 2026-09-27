# Wood 기존 산출물의 재사용 범위와 과거 실험 역할

이번 작업은 Plastic main의 **동일 frozen Replay9장/38코너 교사와 pose-only RAW/REF 학생 계약**을 Wood에 확장한다. 과거 Wood 전용 보정기나 S0/S1/S2를 신규 matched control로 대체하지 않는다. 이 문서는 구현·출처 감사이며 새 성능 평가 또는 학습 결과가 아니다.

## 재사용 판정

| 산출물 | 분류 | 이번 사용 범위 |
|---|---|---|
| SHARED_REPLAY_9_IMAGES_38_CORNERS | REUSABLE_INFRASTRUCTURE | SHARED_METHOD_TEACHER |
| FROZEN_R0_WOOD_PREDICTION_CACHE | REUSABLE_BASELINE | BASELINE_PREDICTIONS_SUBSET |
| MAIN_SYN_LR5_SOURCE_ONLY_CONTROL | REUSABLE_BASELINE | FROZEN_SOURCE_CONTROL_NEW_WOOD_INFERENCE_ONLY |
| PLASTIC_RAW_REF_TARGET_AND_TRAINING_HELPERS | REUSABLE_INFRASTRUCTURE | SHARED_LABEL_SUPPORT_AND_POSE_ONLY_TRAINER |
| PLASTIC_FROZEN_PSEUDO_FILTER | REUSABLE_INFRASTRUCTURE | UNCHANGED_WOOD_FILTER |
| WOOD_DIMENSIONS_AND_CAMERA_CACHE | REUSABLE_INFRASTRUCTURE | REGISTERED_GEOMETRY_AND_INFERENCE_ONLY_METADATA |
| COMMON_D9_AND_2D_SCORER | REUSABLE_INFRASTRUCTURE | IDENTICAL_DEPLOYABLE_COMMON_SELECTOR_AND_METRICS |
| HISTORICAL_WOOD_SPECIFIC_REPLAY_9_39 | SUPPLEMENTARY_ONLY / NOT_COMPARABLE_TO_MAIN | HISTORICAL_CONTEXT_ONLY |
| HISTORICAL_PLASTIC_SPECIFIC_REPLAY | SUPPLEMENTARY_ONLY / NOT_COMPARABLE_TO_MAIN | HISTORICAL_CONTEXT_ONLY |
| HISTORICAL_WOOD_S0_S1_S2 | SUPPLEMENTARY_ONLY / NOT_COMPARABLE_TO_MAIN | HISTORICAL_OCCLUSION_DISTILLATION_CONTEXT_ONLY |
| HISTORICAL_WOOD116_POPULATION | LEAKAGE_RISK / SUPPLEMENTARY_ONLY | PROVENANCE_AUDIT_NOT_UNFILTERED_NEW_MAIN |
| HISTORICAL_NO_CALIBRATED_DISJOINT_POOL_BLOCK | OBSOLETE | OBSOLETE_FOR_45_PANEL_SCOPE_ONLY |
| PORTRAIT_WOOD_20260618_132917 | NOT_COMPARABLE_TO_MAIN | STILL_BLOCKED_UNCALIBRATED_DO_NOT_USE |

각 판정의 원문 경로·SHA256·바이트 수는 [WOOD_REUSE_MATRIX.json](WOOD_REUSE_MATRIX.json)의 `source_bindings`에 기록했다. checkpoint 항목은 기존 FIT/PROTOCOL의 frozen binding을 함께 기록했으며 실행 단계에서 실제 파일 hash를 검증한다.

## 동일 교사가 Wood를 처리할 수 있는가

Replay9/38의 입력은 native RGB, R0의 예측9점과 예측bbox다. material label·치수·GT를 입력받지 않으며 Wood 특화 head나 좌표 분기가 없다. `pallet_posefix_large_error_v1.core.predict`는 고정 affine crop(288×384) 후 native 좌표로 역변환한다. 코너0..7은 cap 없이 수정하고 중심8·bbox·score·keypoint confidence·selected index는 보존한다. 따라서 구현상 Wood 처리 불가 근거는 발견하지 못했다. 정확도가 낮더라도 교사 교체의 근거로 삼지 않는다.

현재 교사의 수동 감독은 Wood6장 + Plastic3장, 총9장/38코너다. Wood는 `wood_day_01`/REC_001, Plastic은 `plastic_night_01`/REC_002다. 같은 녹화에서 재료별 이름으로 나눈 session도 recording은 같다.

## Wood116과 새45장 평가의 역할 구분

기존 Wood116은 `wood_day_01`20장, `wood_night_01`51장, `wood_183705`25장, `wood_184309`20장이다. REC_001/002에는 교사의 manual-training recording이 포함된다. 특히 Plastic teacher의 `plastic_night_01`도 Wood `wood_night_01`과 같은 REC_002이므로 재료가 다르다고 누출 위험이 사라지지 않는다.

이번 main Wood population은 결과를 보기 전에 **REC_039의25장 + REC_042의20장 =45장**으로 고정한다. 기존116 수치·목록은 보존한다. 이는 성능으로 나쁜 subset을 제거하는 것이 아니라 현재 교사/학생 역할과 recording provenance에 따른 별도 DEV population 고정이다. 정확한 지원점·난도·평가자격은 별도 inventory/audit가 최종 근거다.

REC_001/002의 미주석 Wood RGB는 이45장 평가와 recording-disjoint한 후보로 감사할 수 있다. 모든 기존319개 GT 프레임의 exact image ID/SHA는 후보에서 제외하고, material 및 K를 확인한 RGB에만 동일 raw confidence/flip/LOO → 동일 Replay → corrected LOO를 적용한다. 기존 reference/annotation 파일을 학습 좌표로 쓰지 않는다. 후보 허용은 최종 pool sufficiency 또는 pair 성립을 미리 선언하는 것이 아니다.

## 과거 block은 어디까지 해제되는가

기존 `pallet_type_selftrain_v1/POOL.json`의 `BLOCKED_NO_CALIBRATED_DISJOINT_POOL`은 모든 과거 평가 recording을 보존한 당시 계약에서는 타당했다. 현재의45장 범위에서 REC_001/002를 후보로 다시 감사할 수 있으므로 **그 전체 block 결론만 현재 범위에 그대로 적용할 수 없다**(`OBSOLETE_FOR_45_PANEL_SCOPE`).

별도 portrait `data/pallet/raw_data/wood/selected/20260618_132917`225장의 K/hfov 부재는 여전히 차단이다. 새 평가 subset이 calibration을 해결한 것은 아니다. 카메라 추정값을 만들어 넣거나 필터를 생략하지 않는다.

## 학습·평가 재사용 시 주의점

- `recovery_pose.paired_labels`, `train.label`, `PoseOnlyTrainer`를 재사용한다. 기존 entrypoint의 Plastic249/217 hard-code는 실행 orchestration만 분리한다.
- LR은 기존 ARGS 기본1e-4가 아닌 main arm override **1e-5**다. RAW/REF 각5epoch/320update, seed42, batch/nbs16, source512 + real512 slots를 유지한다.
- RAW/REF common support 교집합·bbox·순서·source replay를 같게 하고 유효 pseudo 좌표만 다르게 한다. real reflect101 border100은 한 번만 적용한다.
- source-only `SYN_LR5`는 material-independent한 기존320update 모델을 재사용한다. Wood frozen 추론만 추가하며 새 SYN fit은 하지 않는다. 초기 full-model `SYN_ONLY`와는 다른 모델이다.
- D9는 동일 prediction-only W/D selector와8코너 SQPnP/LM이다. selector residual에9점이 들어가는 기존 계약도 보존한다. oracle/학습selector/severity routing을 추가하지 않는다.
- `INFERENCE_METADATA.json`의 native K/등록 치수를 사용한다. `Pose.metadata('REAL_DEV')`는 GT도 읽으므로 prediction lock 이전 호출은 금지한다.
- whole-object symmetry 2D의 full denominator와 matched-only median/P90를 구별한다. annotation-derived 6D reference를 독립 물리 pose 정답으로 부르지 않는다.

## 과거 material-routed 결과의 위치

과거 Wood 전용 Replay는9장/39수동코너, Plastic 전용 Replay는 별도의10장 감독이며 이번 공동 Replay9/38과 다르다. 과거 Wood S0/S1/S2는 Clean19/type-specific teacher와 가림 증류·augmentation 질문을 포함한다. 따라서 teacher/refiner의 재료별 행동·실패 이력 및 평가 인프라 검증을 위한 supplement로만 유지한다.

> Historical material-routed experiments used different supervision/intervention and are not the matched self-training control.

학생을 material별로 나누는 경우 **material type is externally provided for routed evaluation**이라고 명시한다. 자동 material classifier는 존재한다고 주장하지 않으며 새로 만들지 않는다. unknown material 처리와 모든 pallet/material 일반화는 이번 결과의 범위 밖이다. Plastic/Wood 절대 수치 차이보다 각 material 내부의 corrected−raw delta가 주 질문이다.

## recording/session 별칭

다음 별칭은 `SOURCE_RECORDING_GROUPS.json`에서 그대로 읽었다. 동일 recording의 복사/승격 경로를 독립 세션으로 세면 안 된다.

### REC_001

- `data/evaluation/pallet_eval_v1/incoming/sessions/real_unlabeled_day_20260830`
- `data/evaluation/pallet_eval_v1/final/positive/sessions/plastic_day_01`
- `data/evaluation/pallet_eval_v1/incoming/annotations/real_unlabeled_day_20260830__plastic`
- `data/evaluation/pallet_eval_v1/final/positive/sessions/wood_day_01`
- `data/evaluation/pallet_eval_v1/incoming/annotations/real_unlabeled_day_20260830__wood`

### REC_002

- `data/evaluation/pallet_eval_v1/incoming/sessions/real_unlabeled_night_20260830`
- `data/evaluation/pallet_eval_v1/final/positive/sessions/plastic_night_01`
- `data/evaluation/pallet_eval_v1/final/positive/sessions/wood_night_01`
- `data/evaluation/pallet_eval_v1/incoming/annotations/real_unlabeled_night_20260830__wood`
- `data/evaluation/pallet_eval_v1/incoming/annotations/real_unlabeled_night_20260830__plastic`
- `data/evaluation/pallet_eval_v1/final/positive/annotations/plastic_night_01`

### REC_039

- `data/pallet/raw_data/wood/_annotate_pallet_20260618_183705`
- `data/pallet/raw_data/wood/selected/pallet_20260618_183705`
- `data/evaluation/pallet_eval_v1/dev_existing/sessions/wood_183705`
- `challenge/data/01_real/manual_gt/wood_pallet_20260618_183705_manual_gt`
- `data/evaluation/pallet_eval_v1/dev_existing/annotations/wood_183705`

### REC_042

- `data/pallet/raw_data/wood/_annotate_pallet_20260618_184309`
- `data/pallet/raw_data/wood/selected/pallet_20260618_184309`
- `data/evaluation/pallet_eval_v1/dev_existing/sessions/wood_184309`
- `challenge/data/01_real/manual_gt/wood_pallet_20260618_184309_manual_gt`
