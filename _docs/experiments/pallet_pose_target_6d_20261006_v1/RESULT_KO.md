# 최종 6D target matched 실험 결과

[확인] 후보를 만드는 6D 기하와 원 2D 감독/최종 6D 평가 사이의 차이에서, 이번에는 감독만 hard final ADDsym 첫 argmin action CE로 바꿨어. 후보·특징·치수·모델·초기값·순서·최종6,000 update checkpoint·hard J·prediction-only F를 실제 receipt로 대조해. 논문·LaTeX·PDF·참고문헌을 쓰거나 빌드하지 않았어.

[확인] 요청된 단일 target 교체는 soft 2D CE→hard 6D CE야. target component 안에서 비용 정렬과 label hardening이 함께 달라지므로 둘의 효과는 이 비교로 따로 식별하지 않아. 이를 나누는 추가 대조를 실행하지 않았어.

[확인] 실행한 정식 fit은 1개, optimizer update는 6,000회야. 추가 PERM/hyperparameter/후속 모델 실험은0이야.

## 실제 실행과 seed1 screen

[확인] seed1 screen은 STOP이야. all three paired pose means worsen and RAW canonical good5-to-bad10 damage increases REAL_DEV는 판단에 쓰지 않았어. seed2/3 실행은 미실행야.

[확인] CPU pose-cost와 GPU fit/evaluation의 실제 수는 EXECUTION_COUNTS/POSE_COST_CACHE_MANIFEST/TRAIN_RECEIPTS 및 per-seed 원행 receipt가 기준이야. 아래는 현재 영수증의 그대로인 실행 내역이야.

```json
{
  "execution_counts": {
    "schema": "pose_target_actual_execution_counts_v1",
    "pose_cost_F_attempted": 11238915,
    "pose_cost_F_completed": 11238915,
    "pose_cost_F_failures": 9,
    "parity_F_attempted": 1206,
    "parity_F_completed": 1206,
    "evaluation_F_attempted": 2304,
    "evaluation_F_completed": 2304,
    "actual_fits": 1,
    "optimizer_updates": 6000,
    "training_forward_batches": 6000,
    "training_forward_examples": 96000,
    "excluded_target_exposures": 0,
    "evaluation_forward_batches": 444,
    "evaluation_forward_examples": 2304,
    "PnP_counts": {
      "solvePnP": 33727266,
      "solvePnPGeneric": 0,
      "solvePnPRefineLM": 33727266
    },
    "CPU_pose_cost_wall_seconds": 4890.596119888127,
    "CPU_pose_cost_worker_seconds_sum": 74524.12222577282,
    "CPU_parity_wall_seconds": 14.327230076072738,
    "TRAIN_auxiliary_wall_seconds": 52.138198104919866,
    "fit_wall_seconds": 386.74860430299304,
    "evaluation_wall_seconds": 17.545598553027958,
    "timing_scope": "Stage wall times; fit/evaluation include GPU work plus CPU/I/O, not isolated CUDA kernel time",
    "bank_generation": 0,
    "backbone_detector_forwards": 0,
    "real_training": 0,
    "PERM_fits": 0,
    "auxiliary_refiner_forwards": 0,
    "auxiliary_F": 0,
    "hyperparameter_search": 0,
    "cache_reuse": "first16 TRAIN3216 completed F reused inside full cost; existing bank/features/orders/checkpoints/old metrics read-only",
    "pipeline_logs": "/tmp/pallet-pose-target-6d-cache",
    "manuscript_writes": 0,
    "all_final_F_attempted": 11242425
  },
  "pose_cache_execution": {
    "status": "PASS",
    "full_TRAIN_rows": 55915,
    "completed_rows": 55915,
    "available_targets": 55915,
    "excluded_all_F_invalid": 0,
    "candidate_F_attempted": 11238915,
    "candidate_F_completed": 11238915,
    "candidate_F_available": 11238906,
    "candidate_F_failures": 9,
    "PnP_counts": {
      "solvePnP": 33716736,
      "solvePnPGeneric": 0,
      "solvePnPRefineLM": 33716736
    },
    "chunk_seconds_sum": 74524.12222577282,
    "seconds_wall": 4858.772394289961,
    "workers": 16
  },
  "fit_seconds": [
    {
      "seed": 1,
      "updates": 6000,
      "seconds": 386.74860430299304
    }
  ]
}
```

## TRAIN 2D/6D target index 차이

[확인] source TRAIN 감독에서만 계산한 부수 진단이야. 새 학습·feature forward·F 호출을 추가하지 않고 현재 고정 target을 서로 비교했어. 정의 없는 2D target과 모든 F 실패인 6D target은 별도 분모로 남겼어.

| 항목 | 실제 수 |
|---|---|
| full_usable_TRAIN | 55915 |
| common_defined | 55915 |
| undefined_2d | 0 |
| all_F_invalid_6d | 0 |
| exact_same_index | 11874 |
| both_NoOp | 2818 |
| two_d_NoOp_six_d_move | 6879 |
| two_d_move_six_d_NoOp | 317 |
| both_move_different_action | 36845 |
| 정확히 같은 index % (공동정의 분모) | 21.2358 |

## matched OLD/NEW 계약의 실제 나란한 값

### seed1

| 고정/변경 항목 | OLD soft2D | NEW hard6D |
|---|---|---|
| candidate_bank_sha256 | "c73c1c775b8f1d3b09d2a87fc5f76238bb0628810f98b3b2d40ed324135890e2" | "c73c1c775b8f1d3b09d2a87fc5f76238bb0628810f98b3b2d40ed324135890e2" |
| architecture_sha256 | "ea708f49c237afc8b7466aaaf21a3e55f40f6b985a5b4177017e37ee91da2028" | "ea708f49c237afc8b7466aaaf21a3e55f40f6b985a5b4177017e37ee91da2028" |
| parameter_count | 20259 | 20259 |
| feature_source | "unchanged original frozen p3/p4 source cache" | "unchanged original frozen p3/p4 source cache" |
| dimension_input | "unchanged original five-dimensional normalized context" | "unchanged original five-dimensional normalized context" |
| optimizer | {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999]} | {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999]} |
| schedule | {"warmup": 100, "cosine_steps": 5900, "final_lr_fraction": 0.1} | {"warmup": 100, "cosine_steps": 5900, "final_lr_fraction": 0.1} |
| gradient_clipping | 5 | 5 |
| batch | 16 | 16 |
| updates | 6000 | 6000 |
| order_sha256 | "c868d5156096f2f25939f8b32eebd50732ea922aa3a168a3a4f130fcaa74daca" | "c868d5156096f2f25939f8b32eebd50732ea922aa3a168a3a4f130fcaa74daca" |
| initial_state_sha256 | "abf96528870a331b0071d61d3606a38df6fb0e3d4de02013025991ad88edef0b" | "abf96528870a331b0071d61d3606a38df6fb0e3d4de02013025991ad88edef0b" |
| inference_readout | "unchanged hard joint J/action_scores" | "unchanged hard joint J/action_scores" |
| final_F_code_sha256 | "4e8c1e6b4c4e885fb671af233ea2d90c416fd7b232b63c75ce9c3ffdeb45b1d7" | "4e8c1e6b4c4e885fb671af233ea2d90c416fd7b232b63c75ce9c3ffdeb45b1d7" |
| numeric | {"dtype": "FP32", "TF32_matmul": false, "TF32_cudnn": true, "cudnn_benchmark": false} | {"dtype": "FP32", "TF32_matmul": false, "TF32_cudnn": true, "cudnn_benchmark": false} |
| training_target | "soft2D_target_CE" | "hard_final_ADDsym_argmin_CE" |

[확인] 실제 원 영수증과 별도 대조한 계약 검산: {"candidate_bank": true, "bank_binding": true, "architecture": true, "parameters": true, "config": true, "optimizer": true, "schedule": true, "gradient_clip": true, "batch": true, "updates": true, "initializer": true, "order": true, "numeric": true, "final_F": true, "original_feature_dimension_loader": true, "original_scorer_reducer": true, "original_J_decoder": true, "frozen_input_preflight": true, "changed_components": true, "nested_only_target_diff": true, "final_checkpoint_only": true, "real_training_zero": true}

## SYNTH_HELDOUT

[확인] repeated-use synthetic development diagnostic

| 방법/seed | 코너 med/P90 px | PCK10 분율 | gross20 분율 | T med/P90 cm | R med/P90 deg | ADD med/P90 m | F coverage | NoOp |
|---|---|---|---|---|---|---|---|---|
| RAW | 1.9784 / 7.0566 | 0.936390 | 0.027270 | 2.5036 / 21.2224 | 1.2631 / 178.2558 | 0.034781 / 1.414363 | 1985/1985 | NA |
| N3_seed1 | 1.7638 / 6.7106 | 0.939967 | 0.026440 | 2.4172 / 20.7454 | 1.1913 / 178.3449 | 0.032058 / 1.423361 | 1985/1985 | NA |
| N3_seed2 | 1.7962 / 6.8658 | 0.940478 | 0.026632 | 2.4917 / 21.1036 | 1.1794 / 178.3611 | 0.033313 / 1.421097 | 1985/1985 | NA |
| N3_seed3 | 1.7541 / 6.7893 | 0.941308 | 0.026312 | 2.3508 / 20.4078 | 1.1944 / 178.3151 | 0.032014 / 1.424865 | 1985/1985 | NA |
| PoseFix_seed1 | 1.3893 / 6.3952 | 0.943032 | 0.025865 | 1.8903 / 19.3337 | 0.9779 / 178.5652 | 0.027033 / 1.454860 | 1985/1985 | NA |
| PoseFix_seed2 | 1.3797 / 6.4974 | 0.944246 | 0.025674 | 1.9256 / 18.3490 | 1.0052 / 178.7288 | 0.026517 / 1.464280 | 1985/1985 | NA |
| PoseFix_seed3 | 1.4087 / 6.4985 | 0.943032 | 0.026057 | 1.9686 / 19.3673 | 1.0030 / 178.6880 | 0.027484 / 1.464917 | 1985/1985 | NA |
| FIT_GEO_J_seed1 | 2.1889 / 7.8229 | 0.931153 | 0.027526 | 2.6695 / 20.2316 | 1.3843 / 178.2491 | 0.038514 / 1.418639 | 1985/1985 | 1036 |
| FIT_GEO_J_seed2 | 2.0791 / 7.6301 | 0.932942 | 0.027845 | 2.5830 / 21.1290 | 1.3431 / 178.2558 | 0.036243 / 1.418850 | 1985/1985 | 1435 |
| FIT_GEO_J_seed3 | 2.0934 / 7.7058 | 0.932942 | 0.027590 | 2.5969 / 20.8368 | 1.3163 / 178.0844 | 0.036656 / 1.414350 | 1985/1985 | 1387 |
| GEO_6D_ORACLE | 2.4989 / 9.0588 | 0.916975 | 0.026504 | 1.1290 / 10.0675 | 0.9190 / 168.9424 | 0.016024 / 1.357334 | 1985/1985 | 145 |
| POSE_TARGET_GEO_J_seed1 | 2.7440 / 10.1018 | 0.897113 | 0.027717 | 2.8951 / 27.0489 | 1.6824 / 175.1280 | 0.043770 / 1.428497 | 1985/1985 | 1093 |

### matched OLD→NEW가 우선인 paired 비교

| NEW−control | 지표 | paired 평균 | paired 중앙값 | frame95%CI | scenario/session95%CI | 공동성공 |
|---|---|---|---|---|---|---|
| POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1 | translation_cm | 1.98467338 | 0.00000000 | [1.285731942875196, 2.7406822181290584] | [1.2818228274499894, 2.7307905515917335] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1 | rotation_deg | 0.65844718 | 0.00000000 | [0.1745137136831661, 1.146219865278331] | [0.18372101245038022, 1.1510594054014018] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1 | ADDsym_m | 0.01898940 | 0.00000000 | [0.011984918597026264, 0.02617678352023601] | [0.012157706629110983, 0.0263158173587074] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_N3_seed1 | translation_cm | 2.17418788 | 0.20018510 | [1.4016263062329208, 2.9674125366017354] | [1.3906296630779258, 2.9870438324952064] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_N3_seed1 | rotation_deg | 0.59744915 | 0.08317200 | [-0.12120340840517616, 1.3062310945614621] | [-0.11467359105842564, 1.2939887275269788] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_N3_seed1 | ADDsym_m | 0.01948905 | 0.00205818 | [0.009939259546965019, 0.028810123273560327] | [0.009987487317399674, 0.028883696818204296] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_PoseFix_seed1 | translation_cm | -0.39765046 | 0.42144571 | [-1.868082634913972, 0.9982061888729681] | [-1.8912834639427518, 1.0215209938087004] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_PoseFix_seed1 | rotation_deg | -0.35186709 | 0.17953398 | [-1.3684999725572835, 0.583939437846142] | [-1.3183547167170682, 0.610942206626736] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_PoseFix_seed1 | ADDsym_m | -0.00310023 | 0.00421473 | [-0.018432566359933664, 0.011182671872139604] | [-0.018474450035358376, 0.011794616648275276] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_RAW | translation_cm | 2.17461730 | 0.00000000 | [1.4925917683171221, 2.9214680931260233] | [1.4872752344074827, 2.9216376096120094] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_RAW | rotation_deg | 0.78539348 | 0.00000000 | [0.29338082039328667, 1.2932153953632552] | [0.298787650161309, 1.2854369740574518] | 1985 |
| POSE_TARGET_GEO_J_seed1_minus_RAW | ADDsym_m | 0.02155177 | 0.00000000 | [0.014607151125298458, 0.028764900849286307] | [0.014662783138556408, 0.02888631770301486] | 1985 |

[확인] 동일ID의 차이를 먼저 계산하고 seed평균도 같은ID에서 수행했어. 10,000회/seed20260917 draw를 모든 seed·방법·지표에 공유했어. synthetic은 frame이 주, 원 scenario가 보조야. REAL은 session과 frame을 둘 다 보고했고 독립 실험/다중비교 교정이 아니야. 자세 실패는4분모에 남기고 conditional pose값의 공동성공과 전체 분모를 구분했어.

### oracle gap 회수

| seed | 공동성공 | headroom med m | OLD회수 med m | NEW회수 med m | OLD gap med/P90 m | NEW gap med/P90 m | OLDratio med | NEWratio med | headroom≤1e−7 제외 | NEW 음수ratio | NEWratio>1 | OLD exactaction분율 | NEW exactaction분율 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1985 | 0.01247555 | 0.00000000 | 0.00000000 | 0.01442058 / 0.08467143 | 0.01693445 / 0.13145132 | 0.000000 | 0.000000 | 145 | 598 | 0 | 0.077078 | 0.083627 |

[확인] ratio는 같은 reference/bank의 개발 진단이고 음수·1초과를 clamp하지 않았어. 전체 분포·별도 결측/F실패 제외·oracle보다 좋은 수치오차 범위는 ORACLE_RECOVERY에 남겼어.

### 최종 W/D hypothesis correction

| seed | oracle전환 frame | OLD최종hyp 회수 | NEW최종hyp 회수 | OLD exactaction | NEW exactaction |
|---|---|---|---|---|---|
| 1 | 68 | 4 | 11 | 0 | 1 |

[확인] 생성 perturbation 이름과 최종 F 선택을 구분했어. oracle가 전환하지 않는 집단/전환하는 집단/불가 집단의 RAW·OLD·NEW·oracle T/R/ADD와 frame수도 WD_HYPOTHESIS_ANALYSIS에 모두 있어. REAL GT는 이 사후 표에만 사용돼.

### 2D/6D tradeoff와 canonical damage

| NEW−control | good<5→bad>10 코너 | bad>20→good<10 코너 | canonical코너 분모 | frame사분면 | corner사분면 |
|---|---|---|---|---|---|
| POSE_TARGET_GEO_J_seed1−RAW | 450 | 0 | 15658 | {'BOTH_WORSEN': 586, 'HAS_NUMERIC_TIE': 1093, 'SIX_D_WORSENS_TWO_D_IMPROVES': 29, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 62, 'SIX_D_IMPROVES_TWO_D_WORSENS': 208, 'UNAVAILABLE': 7} | {'BOTH_WORSEN': 3916, 'HAS_NUMERIC_TIE': 8742, 'SIX_D_WORSENS_TWO_D_IMPROVES': 927, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 679, 'SIX_D_IMPROVES_TWO_D_WORSENS': 1394} |
| POSE_TARGET_GEO_J_seed1−FIT_GEO_J_seed1 | 417 | 3 | 15658 | {'BOTH_WORSEN': 565, 'HAS_NUMERIC_TIE': 800, 'SIX_D_WORSENS_TWO_D_IMPROVES': 111, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 263, 'SIX_D_IMPROVES_TWO_D_WORSENS': 239, 'UNAVAILABLE': 7} | {'BOTH_WORSEN': 3938, 'HAS_NUMERIC_TIE': 6393, 'SIX_D_WORSENS_TWO_D_IMPROVES': 1407, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 1929, 'SIX_D_IMPROVES_TWO_D_WORSENS': 1991} |
| POSE_TARGET_GEO_J_seed1−N3_seed1 | 536 | 0 | 15658 | {'BOTH_WORSEN': 1047, 'SIX_D_WORSENS_TWO_D_IMPROVES': 141, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 284, 'SIX_D_IMPROVES_TWO_D_WORSENS': 506, 'UNAVAILABLE': 7} | {'BOTH_WORSEN': 6884, 'SIX_D_WORSENS_TWO_D_IMPROVES': 2586, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 2534, 'SIX_D_IMPROVES_TWO_D_WORSENS': 3654} |
| POSE_TARGET_GEO_J_seed1−PoseFix_seed1 | 573 | 1 | 15658 | {'BOTH_WORSEN': 1217, 'SIX_D_WORSENS_TWO_D_IMPROVES': 77, 'SIX_D_IMPROVES_TWO_D_WORSENS': 525, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 159, 'UNAVAILABLE': 7} | {'BOTH_WORSEN': 8095, 'SIX_D_WORSENS_TWO_D_IMPROVES': 2234, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 1773, 'SIX_D_IMPROVES_TWO_D_WORSENS': 3556} |

[확인] 손상 수는 frame 수가 아니라 같은 canonical GT 코너 수야. ADDsym와 native mean-L2/각 canonical코너의 방향을 결합했고 수치동률·결측/F실패도 별도 남겼어. 2D 또는 ADD 한 지표만으로 전체 센싱 개선을 주장하지 않아.

## REAL_DEV

[확인] 319 repeated REAL_DEV / 13 sessions; same2D+geometry reconstructed reference; exploratory, no decision or independent metrology

| 방법/seed | 코너 med/P90 px | PCK10 분율 | gross20 분율 | T med/P90 cm | R med/P90 deg | ADD med/P90 m | F coverage | NoOp |
|---|---|---|---|---|---|---|---|---|
| RAW | 6.7207 / 43.8900 | 0.634254 | 0.199280 | 7.8969 / 40.5302 | 2.5389 / 86.5273 | 0.087137 / 1.198890 | 319/319 | NA |
| N3_seed1 | 5.7446 / 42.3204 | 0.688275 | 0.175670 | 7.0392 / 38.1358 | 2.1104 / 85.9305 | 0.080549 / 1.192205 | 319/319 | NA |
| N3_seed2 | 5.7677 / 41.8721 | 0.682673 | 0.177671 | 6.7902 / 37.4121 | 2.0727 / 85.8778 | 0.077328 / 1.188082 | 319/319 | NA |
| N3_seed3 | 5.8222 / 42.2087 | 0.686675 | 0.173669 | 7.3735 / 37.8786 | 2.0281 / 85.9500 | 0.082613 / 1.188644 | 319/319 | NA |
| PoseFix_seed1 | 5.5626 / 43.5346 | 0.685474 | 0.179672 | 7.1854 / 43.8098 | 1.9540 / 85.9950 | 0.078512 / 1.185009 | 319/319 | NA |
| PoseFix_seed2 | 5.5284 / 44.0303 | 0.692277 | 0.180872 | 6.8438 / 40.7264 | 1.9871 / 85.9282 | 0.079554 / 1.189018 | 319/319 | NA |
| PoseFix_seed3 | 5.5911 / 44.1571 | 0.683473 | 0.181673 | 6.8250 / 40.4926 | 2.1422 / 86.0480 | 0.076208 / 1.190524 | 319/319 | NA |
| FIT_GEO_J_seed1 | 6.4291 / 44.7775 | 0.648259 | 0.186475 | 7.2888 / 39.2921 | 2.4026 / 86.1368 | 0.082638 / 1.188831 | 319/319 | 82 |
| FIT_GEO_J_seed2 | 6.4614 / 43.0952 | 0.649060 | 0.188876 | 7.6630 / 38.5353 | 2.4674 / 86.1368 | 0.088597 / 1.187672 | 319/319 | 100 |
| FIT_GEO_J_seed3 | 6.4908 / 43.9856 | 0.642257 | 0.190876 | 7.8845 / 41.3039 | 2.4918 / 86.2393 | 0.086492 / 1.189845 | 319/319 | 129 |
| GEO_6D_ORACLE | 6.9304 / 44.5959 | 0.649460 | 0.184074 | 2.9126 / 24.8766 | 1.9250 / 84.1527 | 0.037372 / 1.153861 | 319/319 | 0 |
| POSE_TARGET_GEO_J_seed1 | 8.3764 / 42.5522 | 0.563826 | 0.204482 | 8.1259 / 41.7765 | 3.0948 / 85.9181 | 0.090390 / 1.196645 | 319/319 | 91 |

### matched OLD→NEW가 우선인 paired 비교

| NEW−control | 지표 | paired 평균 | paired 중앙값 | frame95%CI | scenario/session95%CI | 공동성공 |
|---|---|---|---|---|---|---|
| POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1 | translation_cm | 1.12034702 | 0.00000000 | [-0.024832400596277997, 2.6537311249803595] | [0.08268320044601402, 2.4049179309549977] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1 | rotation_deg | 0.20420995 | 0.00000000 | [-0.5096996009001608, 0.9021139716272801] | [-0.5751056179483985, 0.954322946927582] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1 | ADDsym_m | 0.01061949 | 0.00012652 | [-0.0021812976461884938, 0.02609395243360368] | [-0.0026752219250384423, 0.02532736914329354] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_N3_seed1 | translation_cm | 0.87125411 | 0.37713013 | [-1.5161186047286075, 3.075187559640684] | [-2.8137701239786685, 3.3587800622418413] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_N3_seed1 | rotation_deg | 2.80824840 | 0.21986939 | [0.6496034465787649, 5.0239619945218985] | [1.0043819050785274, 4.236777731330817] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_N3_seed1 | ADDsym_m | 0.02793773 | 0.00641456 | [-0.005679724762490105, 0.06149708817920746] | [-0.00798902331403929, 0.051203853460675686] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_PoseFix_seed1 | translation_cm | 0.97876468 | 0.28229720 | [-1.2161988400945392, 3.236446416504936] | [-1.3718780154849692, 2.930977783610806] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_PoseFix_seed1 | rotation_deg | 3.38354633 | 0.26764776 | [1.0878727078040422, 5.657895293616082] | [1.355968776687874, 5.459877319119141] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_PoseFix_seed1 | ADDsym_m | 0.03832411 | 0.00562229 | [0.006150101636674946, 0.07090656682966043] | [0.005152745124735553, 0.06632165925548188] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_RAW | translation_cm | 0.90066528 | 0.00000000 | [-0.2589930941935582, 2.446682599245044] | [-0.03563044552297667, 2.114497447299129] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_RAW | rotation_deg | 0.15624588 | 0.00000000 | [-0.5644532217688333, 0.8602998349795928] | [-0.6479664570235074, 0.9632373582606888] | 319 |
| POSE_TARGET_GEO_J_seed1_minus_RAW | ADDsym_m | 0.00721640 | 0.00000000 | [-0.005587925528691892, 0.0229174038714153] | [-0.006305673332120143, 0.022190733904740596] | 319 |

[확인] 동일ID의 차이를 먼저 계산하고 seed평균도 같은ID에서 수행했어. 10,000회/seed20260917 draw를 모든 seed·방법·지표에 공유했어. synthetic은 frame이 주, 원 scenario가 보조야. REAL은 session과 frame을 둘 다 보고했고 독립 실험/다중비교 교정이 아니야. 자세 실패는4분모에 남기고 conditional pose값의 공동성공과 전체 분모를 구분했어.

### oracle gap 회수

| seed | 공동성공 | headroom med m | OLD회수 med m | NEW회수 med m | OLD gap med/P90 m | NEW gap med/P90 m | OLDratio med | NEWratio med | headroom≤1e−7 제외 | NEW 음수ratio | NEWratio>1 | OLD exactaction분율 | NEW exactaction분율 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 319 | 0.03124382 | 0.00000000 | 0.00000000 | 0.02906926 / 0.14258612 | 0.03431234 / 0.13783336 | 0.000000 | 0.000000 | 0 | 133 | 0 | 0.034483 | 0.050157 |

[확인] ratio는 같은 reference/bank의 개발 진단이고 음수·1초과를 clamp하지 않았어. 전체 분포·별도 결측/F실패 제외·oracle보다 좋은 수치오차 범위는 ORACLE_RECOVERY에 남겼어.

### 최종 W/D hypothesis correction

| seed | oracle전환 frame | OLD최종hyp 회수 | NEW최종hyp 회수 | OLD exactaction | NEW exactaction |
|---|---|---|---|---|---|
| 1 | 16 | 0 | 1 | 0 | 0 |

[확인] 생성 perturbation 이름과 최종 F 선택을 구분했어. oracle가 전환하지 않는 집단/전환하는 집단/불가 집단의 RAW·OLD·NEW·oracle T/R/ADD와 frame수도 WD_HYPOTHESIS_ANALYSIS에 모두 있어. REAL GT는 이 사후 표에만 사용돼.

### 2D/6D tradeoff와 canonical damage

| NEW−control | good<5→bad>10 코너 | bad>20→good<10 코너 | canonical코너 분모 | frame사분면 | corner사분면 |
|---|---|---|---|---|---|
| POSE_TARGET_GEO_J_seed1−RAW | 52 | 1 | 2499 | {'SIX_D_IMPROVES_TWO_D_IMPROVES': 47, 'HAS_NUMERIC_TIE': 98, 'SIX_D_WORSENS_TWO_D_IMPROVES': 24, 'BOTH_WORSEN': 105, 'SIX_D_IMPROVES_TWO_D_WORSENS': 45} | {'SIX_D_IMPROVES_TWO_D_IMPROVES': 362, 'HAS_NUMERIC_TIE': 774, 'BOTH_WORSEN': 691, 'SIX_D_WORSENS_TWO_D_IMPROVES': 317, 'SIX_D_IMPROVES_TWO_D_WORSENS': 355} |
| POSE_TARGET_GEO_J_seed1−FIT_GEO_J_seed1 | 61 | 0 | 2499 | {'SIX_D_IMPROVES_TWO_D_IMPROVES': 43, 'BOTH_WORSEN': 132, 'SIX_D_WORSENS_TWO_D_IMPROVES': 24, 'SIX_D_IMPROVES_TWO_D_WORSENS': 48, 'HAS_NUMERIC_TIE': 72} | {'SIX_D_IMPROVES_TWO_D_IMPROVES': 327, 'SIX_D_IMPROVES_TWO_D_WORSENS': 388, 'BOTH_WORSEN': 847, 'SIX_D_WORSENS_TWO_D_IMPROVES': 376, 'HAS_NUMERIC_TIE': 561} |
| POSE_TARGET_GEO_J_seed1−N3_seed1 | 105 | 0 | 2499 | {'BOTH_WORSEN': 188, 'SIX_D_IMPROVES_TWO_D_WORSENS': 75, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 27, 'HAS_NUMERIC_TIE': 8, 'SIX_D_WORSENS_TWO_D_IMPROVES': 21} | {'SIX_D_WORSENS_TWO_D_IMPROVES': 415, 'BOTH_WORSEN': 1225, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 309, 'SIX_D_IMPROVES_TWO_D_WORSENS': 496, 'HAS_NUMERIC_TIE': 54} |
| POSE_TARGET_GEO_J_seed1−PoseFix_seed1 | 120 | 1 | 2499 | {'SIX_D_IMPROVES_TWO_D_WORSENS': 71, 'BOTH_WORSEN': 182, 'SIX_D_WORSENS_TWO_D_IMPROVES': 25, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 33, 'HAS_NUMERIC_TIE': 8} | {'SIX_D_IMPROVES_TWO_D_WORSENS': 499, 'SIX_D_IMPROVES_TWO_D_IMPROVES': 325, 'SIX_D_WORSENS_TWO_D_IMPROVES': 440, 'BOTH_WORSEN': 1181, 'HAS_NUMERIC_TIE': 54} |

[확인] 손상 수는 frame 수가 아니라 같은 canonical GT 코너 수야. ADDsym와 native mean-L2/각 canonical코너의 방향을 결합했고 수치동률·결측/F실패도 별도 남겼어. 2D 또는 ADD 한 지표만으로 전체 센싱 개선을 주장하지 않아.

## 고정 사항과 supervision 변경

[확인] MATCHED_COMPARISON.json의 OLD/NEW 영수증과 matched_contract에 bank/architecture/params/features/dimensions/optimizer/LR/batch/updates/order/initializer/readout/F/numerical 설정을 나란히 남겼어. initializer 검사는 첫 optimizer 전 원 digest와 같아야 해. hard 6D target의 분모 제외가 있으면 원 55,915 및 order 노출과 분리해 밝혀. 새 checkpoint를 매 update 저장한 운영 차이는 원500 update 간격보다 촘촘한 중단 계수 보존이며 모델/optimizer 산술 차이가 아니다. 따라서 wall시간 자체를 원 방법의 속도 효과로 해석하지 않았어.

## 테스트

[확인] 실제 계약 테스트 명령·PASS 수는 CONTRACT_TEST_RESULTS.json에 기록하고 아래 최종 재집계에서 연결해. 테스트는 임의 재학습이나 새 모델/F 실행이 아니야.

## 판정

POSE_TARGET_NOT_SUPPORTED

[확인] OLD 감독 변경의 판정과 N3보다 6D를 더 잘 예측하는 질문은 별도야. N3_6D_question: {"verdict": "POSE_TARGET_NOT_SUPPORTED", "reference_family": "N3", "improved_pointestimate_metrics": [], "negative_CI_and_reproduced_metrics": [], "per_seed_paired_means": {"translation_cm": [2.1741878811471165], "rotation_deg": [0.597449149961804], "ADDsym_m": [0.019489048499910976]}, "opposing_pose_metrics": ["translation_cm", "rotation_deg", "ADDsym_m"], "per_seed_damage_or_coverage_increase": [true], "stable_evidence_requires_three_seeds": true, "REAL_used": false}

[확인] 이 판정은 고정 bank·참조·모델·예산·반복 개발 자료에 한정돼. 실제 DEV의 reference는 같은2D 주석+치수 재구성이며 독립 physical pair는0/BLOCKED_DATA야. 이후 실험을 자동 실행하지 않았어.

[확인] 실제 테스트: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_target_6d_20261006_v1.contract_tests --artifacts` — 30 PASS / 0 FAIL.

## 실행·검증 시간 범위와 I/O 개선

[확인] fit 내부 wall은 386.748604초, 전체 subprocess는 391.544992초다. 평가 내부 wall 합은 17.545599초, 전체 subprocess는 33.467681초다. aux 내부/전체는 52.138198/55.934849초다. EXECUTION_COUNTS의 기존 timing_scope 문구는 내부 receipt 시간이며, 초기화·입력 검사까지 포함한 전체 시간과 구분해야 한다. GPU·CPU·I/O가 섞여 있으므로 isolated CUDA kernel 시간은 측정하지 않았다. 정확한 대응은 EXECUTION_TIME_SCOPE.json에 있다. CPU 비용 전체 wall은 최초 timing 포함 4,890.596120초이며 worker wall 합 74,524.122226초는 CPU core 사용 시간이 아니다.

[확인] 사용자 속도 요청 후 원 O_DSYNC 병목을 개선했다. 원 과학 비용/F·bank·target 코드를 유지하고 완료 2,071,104 F를 읽기 전용 재사용했다. 빠른 단계의 나머지 9,167,811 F는 native 검산 포함 1,469.714952초/초당 6,237.8157회로, 느린 전체 단계 wall 처리량의 10.2231배다. 두 단계 사이 비계산 준비 대기 421.897403초는 별도이며 모델 추론의 속도 효과로 해석하지 않는다. FAST_IO_EXECUTION 및 audit/FAST_IO_SPEED_FINAL에 실제 수가 있다.

[확인] I/O 테스트 명령은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_target_6d_20261006_v1.fast_cost_tests`이며 12 PASS/0 FAIL, mock 비용 호출5, 실제 F/PnP/NN/update0이다. 기존 산출물 계약 테스트는 위 명령의 30 PASS/0 FAIL이다. 독립 검산 명령은 동일 Python의 `-B -m scripts.research.pallet_pose_target_6d_20261006_v1.verification`이며 전체 11,238,915 WAL/874chunks/2,304 평가행과 134,959,817 계수 검산 및 710개 원입력 76,031,296,408bytes 종료 SHA를 PASS했다(57.929923초). 원입력 종료 SHA 부분은 51.854121초다. 별도 `final_io_verification`도 pretrain/fast writer/auditor/transition/memory/final receipt SHA 연결 PASS이며 과학 재계산은0이다.
