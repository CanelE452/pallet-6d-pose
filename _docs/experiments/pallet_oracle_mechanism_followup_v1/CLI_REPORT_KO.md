# 최종 CLI 보고 — 실행 수치 및 공개 범위



STATUS: THREE_CYCLES_COMPLETED_NO_METHOD_PROMOTION

HEAD_START: cf52dc624fb5c83e305ff3b951f3c362b9ca1842

HEAD_AT_REPORT_RENDER: 6fb81b1eb23d4af717cfacff865a956ba5938f5d

REMOTE_TRACKING_AT_RENDER: 6fb81b1eb23d4af717cfacff865a956ba5938f5d

HEAD_END / REMOTE_HEAD: 최종 push 후 실제 조회 결과는 사용자 최종 응답 및 private RELEASE_VERIFICATION.json에 기록한다. 이 파일을 포함하는 commit은 git log -1 -- 이파일경로로 확인한다.



RESEARCH_QUESTION: 보정 자기학습의 제한된 이득을 타깃/학습/선택/참조로 구분하고 한정된 실제 개입으로 회수할 수 있는가?

WHAT_PRIOR_RESULTS_ACTUALLY_RULED_OUT: 같은 낮은LR 연장·기존 local/융합/필터 설정의 제한된 효과. 모든 최적화/표현/selector의 불가능성은 아님; PRIOR_ATTEMPTS 30항목 참조.



## CURRENT_BASELINES



PLASTIC_R0_RAW_REF: 484/468/507 of985; AUC .337964844/.334722656/.359015625.

WOOD_R0_RAW_REF: 167/163/165 of346; AUC .670500/.656433333/.665033333.



## ORACLE_RESULTS



FIXED_CANDIDATE_POSE_BY_MATERIAL: REF Plastic .359015625→.450093750; Wood .665033333→.667400000.

WHOLE_OUTPUT_TEACHER_STUDENT_ORACLE: whole pose Plastic .441824219 / Wood .710688889; whole2D589/985,210/346.

PER_POINT_ORACLE: 613/985,228/346; verified66 whole/point53/66; coordinate mixtures not reported as rigid6D.

LOCAL_MOVE_CEILING_IF_RELEVANT: GT-direction8px REF700/985,252/346; not image-based attainability.

REFERENCE_INPUT_GEOMETRY_CHECK: legacy xy→pose AUC1 circular; exactsynthetic64 production .905797; privileged renderer-correspondence solvermaxreprojection8.13e-6px.

PRIVILEGED_TRAIN_CAPABILITY_OR_NA: C3 existing manual9/38; see FINAL_TABLES and C3 REPORT.

ORACLE_NOT_INTERPRETABLE_ITEMS: synthetic physical-axis/camera-facing mismatch NA; Wood verified-visible NA; no GTcrop or unsupported pointcompletion.



## DIAGNOSIS



TARGET_QUALITY: verified66 teacher50 vsREF43, teacher-only10/student-only3 at10px; Wood trustedvisible unresolved.

TRAINING_SIGNAL_TRANSFER: partial; correctedTRAIN residual Plastic RAW4.078→REF3.070px, Wood3.689→3.325px; not physicalGT accuracy.

AUGMENTATION_OR_SOURCE_CONFLICT: source/real gradient signs mixed; huge affineprobeoutlier is instance switch. C2 primary improvement absent.

CANDIDATE_GENERATION: existing set only; no claim about all possible representations.

CANDIDATE_SELECTION: Plastic W/D gap meaningful, Wood REF gap too small to exceedR0; C1 recovered0%.

REFERENCE_OR_IDENTIFIABILITY: repeatedDEV/legacygeometry, missingWooddirectclickprovenance; C3 unconfirmedphysicalsignedaxes.



## LITERATURE



PRIMARY_SOURCES_READ: {'FULLTEXT_METHOD': 7, 'FULLTEXT_PARTIAL': 5, 'ABSTRACT_AND_OFFICIAL_CODE_README': 4, 'OFFICIAL_METHOD_DOCUMENTATION': 3}

ABSTRACT_ONLY_OR_BLOCKED: R3, R7, R9, R11

SELECTED_PRINCIPLES: fixed robust residual; controlled augmentation; direct same-budget TRAIN capability.

REJECTED_AND_WHY: no calibrated localization-reliability evidence, mixed gradient evidence, CAD/render/depth/extra framework prerequisites; not blanket paper failure.



## EXECUTED_CYCLES



C1: 0fits; same candidates, scoreonly, deltaAUC0.

C2: 4fits×320; paired Plastic/Wood real-affine-off.

C3: 2fits×320; RAW9/MANUAL9 existing38point support.

PAIRED_NEW_FITS: 6; OPTIMIZER_UPDATES: 1920

GPU_HOURS: 0.134182; WALL_HOURS_AT_LEDGER: 1.072293

TECHNICAL_FAILURES_AND_RETRIES: EXPERIMENT_LOG and C3 technical records; no result-driven refit. Sandbox failure before model/optimizer0step distinguished from training.



## RESULT



GAIN_VS_RAW: C2 Plastic+34/985,+.026992188AUC; Wood−3/346,+.007511111AUC.

GAIN_VS_OLD_REF: C2 Plastic−4/985,−.000281250; Wood−1/346,−.002466667.

GAIN_VS_R0: C2 Plastic+19/985,+.020769531; Wood−3/346,−.007933333.

PLASTIC: R0 484/985, AUC 0.337964844 / OLD_REF 507/985, AUC 0.359015625 / RAW9 482/985, AUC 0.334085938 / MANUAL9 487/985, AUC 0.346542969

WOOD: R0 167/346, AUC 0.670500000 / OLD_REF 165/346, AUC 0.665033333 / RAW9 160/346, AUC 0.661300000 / MANUAL9 138/346, AUC 0.618033333

COST_AND_HARMS: sixfits; sameexistingmanualbudget butC3 directstudentpath; allseverity/tail/recordingretained. No hiddenbestcheckpoint/threshold/seed selection.

ATTAINABLE_RECOVERY_DEMONSTRATED: C1 fixedset0%; C2 no gain overoldREF primary. C3 is a distinct finite supervised control; interpretation in REPORT.

REMAINING_HEADROOM_AND_MISSING_INFORMATION: GT-free discriminating cue and out-of-sample targetquality/independentreference remain unresolved.

GENERALIZATION_STATUS: REUSED_DEV_ONLY.



## FINAL_FOUR_AXES



EVIDENCE_VALIDITY: LIMITED — paired computations valid, repeatedDEV and reference scope limited.

HEADROOM: MEASURED — within stated sets/metrics only.

RECOVERY: NOT_DEMONSTRATED against existingREF; C3 TRAIN fitting partial, DEV effects material-mixed. No automatic best-run promotion.

CAUSE: MULTIPLE_EXPLANATIONS.



WHAT_WAS_RESOLVED: currentoracleheadroom, partialTRAINimitation, testedHuber/affineeffects, finite9/38capability.

WHAT_WAS_NOT_RESOLVED: physicaltargettruth, robustGT-freeselection, independentgeneralization, separatecapacity/optimization causality.

WHAT_MUST_NOT_BE_CLAIMED: oracleattainability, allmaterialgeneralization, allaugmentationfailure, physical6Dclosure.

NEXT_ONE_DECISION: STOP this3-cyclebatch; anynextstudy asks whether availableRGBcan discriminate complementaryfrozenoutputs on eligibleTRAINwithoutDEVlabels.

HUMAN_ACTION_REQUIRED: NO.

REPORT: REPORT_KO.md; FINAL_TABLES.md; AUDIT.json; REPRODUCE.md.

COMMITS_AND_PUSH: diagnosis4bc8e41a and C2 6fb81b1e already verifiedremote; finalrelease verifiedafter commit.

GIT_STATUS: original untracked files preserved; only followupnamespace tracked/staged changes intended.



현재 REF의 W/D 선택 단계에 Plastic AUC .091078 / Wood .002367의 여지가 관측됐고, Huber12 방법은 그중0%를 회수했다. 정답을 모르는 입력만으로 이 후보를 고르는 방법과 독립 실사 일반화는 아직 확인하지 못했다.
