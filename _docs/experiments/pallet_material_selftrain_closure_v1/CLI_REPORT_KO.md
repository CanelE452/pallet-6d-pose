# Material closure CLI 측정 결과 보고

측정 snapshot에 최종 실행 상태를 반영했다. 실험·원고13페이지 build·계약검사30개·자동감사는 완료했다. 사용자가 재현 메타데이터와 평가 오버레이6장 공개를 승인했다. 최종 commit·원격 확인은 PUSH_VERIFICATION.json을 참조한다.

## STATUS

EXPERIMENT_PAPER_AUDIT_COMPLETE_PUBLICATION_AUTHORIZED

## HEAD_START

f45d9c0849e545270a3a7639a2ace3174bc32527

## HEAD_AT_REPORT

47fdd24b5fc56753fb546ff5ec9ef267d54acebc

## BRANCH

main

## MATERIAL_GOAL

동일한 보정 타깃 자기학습 효과가 Plastic/Wood에서 같은 방향인지 검증

## PLASTIC_MAIN

```json
{
  "EVAL_N": 128,
  "RAW_PCK10": 0.4751269035532995,
  "CORR_PCK10": 0.5147208121827411,
  "DELTA_PCK10_PP": 3.9593908629441623,
  "RAW_AUC": 0.33472265625000003,
  "CORR_AUC": 0.359015625,
  "DELTA_AUC": 0.024292968749999977
}
```

## WOOD_DATA

```json
{
  "CANDIDATE_POOL": 1000,
  "ACCEPTED_SHARED": 676,
  "TRAIN_UNIQUE": 361,
  "TRAIN_RECORDINGS": [
    "REC_001",
    "REC_002"
  ],
  "EVAL_N": 45,
  "EVAL_RECORDINGS": [
    "REC_039",
    "REC_042"
  ],
  "TEACHER_IMAGE_OVERLAP": 0,
  "TEACHER_SESSION_OVERLAP": [],
  "TRAIN_EVAL_IMAGE_OVERLAP": 0,
  "TRAIN_EVAL_RECORDING_OVERLAP": []
}
```

## WOOD_PSEUDO_QUALITY

```json
{
  "STATUS": "UNRESOLVED",
  "POINTS": 0,
  "RAW_PCK10": null,
  "CORR_PCK10": null,
  "MEDIAN_DELTA": null,
  "LIMITATION": "No provenance-eligible direct-visible Wood45 points; legacy quality not verified quality"
}
```

## WOOD_MATCHED_PAIR

```json
{
  "STATUS": "COMPLETED",
  "NEW_FITS": 2,
  "PAIR_INTEGRITY": "PASS",
  "runtime": {
    "WOOD_RAW_LR5": {
      "checkpoint": {
        "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_RAW_LR5/weights/last.pt",
        "sha256": "7999e78fb6ff162e60645adc1f05f525a68d658ee1c73d624822b5e274219963",
        "bytes": 6541159
      },
      "seconds": 46.698198260972276,
      "epochs": 5,
      "optimizer_updates": 320,
      "epoch_updates": [
        64,
        128,
        192,
        256,
        320
      ],
      "exact_R0_initialization": true,
      "protected_state_exact": true,
      "results_csv": {
        "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_RAW_LR5/results.csv",
        "sha256": "4bf523484c4db628959e4a824a5867305af68dd991cf0208a952f9e14204a79a",
        "bytes": 975
      }
    },
    "WOOD_REF_LR5": {
      "checkpoint": {
        "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_REF_LR5/weights/last.pt",
        "sha256": "56934125bbbfcfce393fb7560ad5ff186a7ccf96fb74f066c9e827479906451e",
        "bytes": 6541159
      },
      "seconds": 46.443398994160816,
      "epochs": 5,
      "optimizer_updates": 320,
      "epoch_updates": [
        64,
        128,
        192,
        256,
        320
      ],
      "exact_R0_initialization": true,
      "protected_state_exact": true,
      "results_csv": {
        "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_REF_LR5/results.csv",
        "sha256": "c353c5a76aae7f8eba75f8277e69bcc7a4811f2dab10ad2522a12070173f1100",
        "bytes": 975
      }
    }
  }
}
```

## WOOD_2D

```json
{
  "R0_PCK10": 0.48265895953757226,
  "RAW_PCK10": 0.47109826589595377,
  "CORR_PCK10": 0.476878612716763,
  "CORR_MINUS_RAW_PP": 0.5780346820809246,
  "P90_RAW": 67.83939479274221,
  "P90_CORR": 68.98473632608628
}
```

## WOOD_6D

```json
{
  "R0": {
    "frames": 45,
    "available": 45,
    "axis_accuracy": 0.8888888888888888,
    "axis_correct_count": 40,
    "ADDsym_AUC": 0.6705,
    "rotation_deg": {
      "median": 1.5445593884896238,
      "P90": 9.189686803137151
    },
    "yaw_deg": {
      "median": 0.4711965156682254,
      "P90": 3.583353412229407
    },
    "translation_cm": {
      "median": 2.0951296876805823,
      "P90": 8.011072482822469
    },
    "IoU3D": {
      "median": 0.7897904381767441,
      "P90": 0.8943194553902166
    },
    "ADDsym_normalized": {
      "median": 0.02237274507380266,
      "P90": 0.08541676236819437
    },
    "pose_coverage": 1.0
  },
  "WOOD_RAW_LR5": {
    "frames": 45,
    "available": 45,
    "axis_accuracy": 0.8888888888888888,
    "axis_correct_count": 40,
    "ADDsym_AUC": 0.6564333333333333,
    "rotation_deg": {
      "median": 1.598353248562002,
      "P90": 9.084432283415154
    },
    "yaw_deg": {
      "median": 0.5360608876200104,
      "P90": 3.4688596715371784
    },
    "translation_cm": {
      "median": 2.3485363456637542,
      "P90": 8.258958371705063
    },
    "IoU3D": {
      "median": 0.7703981200515015,
      "P90": 0.8856116601366407
    },
    "ADDsym_normalized": {
      "median": 0.024916860390335287,
      "P90": 0.08719489861610533
    },
    "pose_coverage": 1.0
  },
  "WOOD_REF_LR5": {
    "frames": 45,
    "available": 45,
    "axis_accuracy": 0.8888888888888888,
    "axis_correct_count": 40,
    "ADDsym_AUC": 0.6650333333333333,
    "rotation_deg": {
      "median": 1.6048666560555571,
      "P90": 9.337193557519054
    },
    "yaw_deg": {
      "median": 0.4987539489569599,
      "P90": 3.5882840551484336
    },
    "translation_cm": {
      "median": 2.0714261981613196,
      "P90": 7.9851529684904845
    },
    "IoU3D": {
      "median": 0.7739795988778422,
      "P90": 0.867193479050286
    },
    "ADDsym_normalized": {
      "median": 0.022567642013314095,
      "P90": 0.08507699691712269
    },
    "pose_coverage": 1.0
  }
}
```

## WOOD_SEVERITY

```json
{
  "CLEAN": 38,
  "MODERATE": 7,
  "SEVERE_AVAILABLE": false
}
```

## MATERIAL_DECISION

MATERIAL_GENERAL_SIGNAL

## MATERIAL_INTERPRETATION

반복 DEV의 ordinary plastic과 Wood에서 corrected-vs-raw 자기학습의 두 주 지표가 같은 개선 방향을 보였지만, 이는 평가한 두 재료 범주에 한정된 신호다.

## NOT_SUPPORTED

```json
[
  "Every material/pallet improves",
  "Every metric or severity improves",
  "Wood verified pseudo quality established",
  "Material alone causally explains absolute performance differences",
  "Automatic material classification",
  "Estimator architecture generalization"
]
```

## HISTORICAL_TYPE_SPECIFIC_RESULTS

SUPPLEMENTARY_ONLY / NOT_COMPARABLE_TO_MAIN

## DOPE_NEXT

NOT_RUN

## MANUSCRIPT

_docs/paper/selftraining_submission_v1/manuscript.tex

## PDF

_docs/paper/selftraining_submission_v1/manuscript.pdf

## USER_ACTION_REQUIRED

NO — 공개 범위를 설명한 뒤 사용자가 push를 요청했다. 원본RGB파일·원시좌표배열·카메라행렬·가중치파일은 제외한다.

## FINAL_LOCAL_VERIFICATION

TESTS: PASS / 30. AUDIT: PASS. BUILD: PASS / 13pages. PDF시각확인: 1·5·6페이지 및 실제악화사례. 방법개발STOP, 추가fit없음. 사전감사47fdd24b·학습완료def8fb20·결과/이미지7e26d791·원고/PDF4d68f301은origin/main반영확인. `git ls-remote`에서 서버main과 로컬HEAD 및 origin/main의 4d68f301 일치를 확인했다. 이 확인기록을 추가하는 후속 메타데이터 커밋은 push 후 최종CLI응답에서 확인한다. 상세는 PUSH_VERIFICATION.json을 참조하며, 아래 COMMIT_AT_REPORT는 과거 보고서 생성 시점 snapshot이다. 기존미추적파일은보존했다.

## REMAINING_EVIDENCE_LIMITS

```json
[
  "Wood verified/direct-visible pseudo quality",
  "Independent confirmation",
  "Independent physical6D",
  "Wood severe and unseen materials"
]
```

## COMMIT_AT_REPORT

47fdd24b5fc56753fb546ff5ec9ef267d54acebc

## TRACKING_REMOTE_HEAD_AT_REPORT

47fdd24b5fc56753fb546ff5ec9ef267d54acebc

## GIT_STATUS_AT_REPORT

## main...origin/main
A  _docs/experiments/pallet_material_selftrain_closure_v1/FIT_WOOD_RAW_LR5.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/FIT_WOOD_REF_LR5.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/POOL_DECISION.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/POOL_DECISION_CORRECTION.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/POOL_DECISION_V2.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/PREFLIGHT_PUBLIC.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/PSEUDO_COMPLETE.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/PSEUDO_PROTOCOL.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/PUBLICATION_POLICY.md
A  _docs/experiments/pallet_material_selftrain_closure_v1/WOOD_DATA_INVENTORY_PUBLIC.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/WOOD_PAIR_PREFLIGHT.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/WOOD_PSEUDO_PROTOCOL.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/WOOD_TRAIN_PROTOCOL_PUBLIC.json
A  _docs/experiments/pallet_material_selftrain_closure_v1/WOOD_TRUE_IGNORE_TRAIN_FIXTURE_TEST.json
M  scripts/research/pallet_material_selftrain_closure_v1/pseudo_pool.py
A  scripts/research/pallet_material_selftrain_closure_v1/public_summary.py
A  scripts/research/pallet_material_selftrain_closure_v1/test_pair.py
A  scripts/research/pallet_material_selftrain_closure_v1/train_pair.py

## FINAL_SENTENCE

반복 DEV의 ordinary plastic과 Wood에서 corrected-vs-raw 자기학습의 두 주 지표가 같은 개선 방향을 보였지만, 이는 평가한 두 재료 범주에 한정된 신호다.
