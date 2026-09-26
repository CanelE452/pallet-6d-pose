# 지시문 항목별 최종 검증

상태: **PASS_WITH_USER_APPROVED_AMENDMENTS**. 62개 명명 항목: {'PASS': 60, 'AMENDED_USER_APPROVED': 2}. 단위검사 23개.

PASS는 아래 기재한 증거 범위의 통과다. 과거 실행의 모든 파일 접근을 소급 추적했다는 뜻은 아니다. 사용자 승인으로 바뀐 수동 bbox·PnP 조건은 원문 그대로 통과했다고 표시하지 않는다.

|원문 검사명|상태|검증 방법|근거|
|---|---|---|---|
|test_head_recorded|PASS|artifact recheck|INPUT_BINDINGS.json|
|test_input_hashes|PASS|artifact recheck|5742 upstream hashes; changed=[]|
|test_existing_results_unchanged|PASS|artifact recheck|FINALIZATION_INPUT_LOCK.json: 35 completed scientific artifacts; all original upstream hashes also checked|
|test_adaptation_pool_only|PASS|historical inventory audit + current manifest|PREPARATION_TESTS.json + CANDIDATE_POOL_AUDIT.json|
|test_eval_excluded|PASS|artifact recheck|EXCLUSION_IDENTITIES_PRIVATE.json; SHA + entire heldout/reserved recordings|
|test_anchor_excluded|PASS|artifact recheck|EXCLUSION_IDENTITIES_PRIVATE.json; SHA + entire heldout/reserved recordings|
|test_clean10_excluded|PASS|artifact recheck|EXCLUSION_IDENTITIES_PRIVATE.json; SHA + entire heldout/reserved recordings|
|test_h10_excluded|PASS|artifact recheck|EXCLUSION_IDENTITIES_PRIVATE.json; SHA + entire heldout/reserved recordings|
|test_final_reserved_excluded|PASS|artifact recheck|EXCLUSION_IDENTITIES_PRIVATE.json; SHA + entire heldout/reserved recordings|
|test_sha_overlap_zero|PASS|artifact recheck|current eligible pool vs protected SHA sets|
|test_near_duplicate_zero|PASS|verified historical full-queue MAD audit; not claimed rerun|PREPARATION_TESTS.json: full queue368 MAD audit; bound by HARD_SELECTION_AUDIT.json|
|test_queue_built_without_model_outputs|PASS|historical sampling provenance|CANDIDATE_POOL_AUDIT.json|
|test_gui_no_model_overlay|PASS|static imports + historical GUI smoke|raw difficulty GUI import graph; later annotation PnP is a separately approved amendment|
|test_fixed_round_order|PASS|artifact recheck|queue reconstruction from frozen eligible pool|
|test_human_tags_locked_before_model_open|PASS|artifact recheck|tag -> selection -> label -> teacher locks|
|test_hard_only_from_human_tags|PASS|artifact recheck|locked human tags, not prediction scores|
|test_three_recordings|PASS|artifact recheck|HARD_SELECTION_LOCK.json|
|test_max3_per_recording|PASS|artifact recheck|initial + reserve max3|
|test_initial8_reserve2|PASS|artifact recheck|HARD_SELECTION_LOCK.json|
|test_selection_locked_before_teacher_or_student_output|PASS|artifact recheck|ordered immutable locks|
|test_manual_bbox_present_role_confident|AMENDED_USER_APPROVED|artifact recheck|ANNOTATION_PROTOCOL_AMENDMENT.json: user approved shared PnP association box instead of manual visible-envelope bbox|
|test_visible_xy_required|PASS|artifact recheck|private frozen labels; range/duplicate/LR/TB QA|
|test_nonvisible_xy_none|PASS|artifact recheck|private frozen labels; unobserved physical visibility not inferred|
|test_role_uncertain_excluded|PASS|artifact recheck|user confirmation; ROLE_UNCERTAIN exclusion also covered by unit test|
|test_P8_not_primary|PASS|artifact recheck|labels contain corners0..7 only; tensor mask rechecked below|
|test_no_PnP_completion|AMENDED_USER_APPROVED|artifact recheck|User enabled PnP display/fill, but no completed corner enters manual support; original PnP-free UI condition amended|
|test_public_no_private_coordinates|PASS|artifact recheck|public label provenance/coverage vs private coordinate snapshot|
|test_teacher_after_label_lock|PASS|artifact recheck|HARD_LABEL_LOCK.json / TEACHER_HARD_PREDICTION_LOCK.json|
|test_teacher_checkpoint_frozen|PASS|artifact recheck|R0 and refiner hashes rechecked|
|test_same_support_manual_pseudo|PASS|artifact recheck|all320 shared hard-cache masks + P8 true-ignore|
|test_manual_pseudo_same_mask|PASS|artifact recheck|all320 shared hard-cache masks + P8 true-ignore|
|test_same_bbox_manual_pseudo|PASS|artifact recheck|one shared RGB/box/affine cache per hard occurrence, not separate arm transforms|
|test_manual_pseudo_same_aug|PASS|artifact recheck|one shared RGB/box/affine cache per hard occurrence, not separate arm transforms|
|test_same_init_hash|PASS|saved runtime assertion|train on_start compared every initial-state digest to original S1|
|test_same_trainable_inventory|PASS|artifact recheck|actual named trainable inventories|
|test_same_optimizer|PASS|runtime configuration + implementation|same args and fresh optimizer construction; no separate initial optimizer-state snapshot was saved|
|test_seed42|PASS|artifact recheck|actual args.yaml|
|test_320_updates|PASS|artifact recheck|optimizer post-step hooks in FIT records|
|test_5epochs|PASS|artifact recheck|FIT records|
|test_512synthetic_per_epoch|PASS|artifact recheck|frozen5120 occurrence plan + completion actual trace audit|
|test_hard64_per_epoch|PASS|artifact recheck|frozen5120 occurrence plan + completion actual trace audit|
|test_clean448_per_epoch|PASS|artifact recheck|frozen5120 occurrence plan + completion actual trace audit|
|test_512real_per_epoch|PASS|artifact recheck|frozen occurrence plan|
|test_hard_total320|PASS|artifact recheck|frozen occurrence plan|
|test_synthetic_replay_identical|PASS|artifact recheck|completion audit compares every actual RGB/box/target trace digest|
|test_manual_pseudo_same_RGB|PASS|artifact recheck|completion audit compares every actual RGB/box/target trace digest|
|test_manual_pseudo_coordinate_source_only_difference|PASS|artifact recheck|all320 target arrays plus actual5120 trace audit|
|test_last_checkpoint_only|PASS|artifact recheck|FIT/checkpoint paths|
|test_no_real_DEV_validation|PASS|actual args + validation-list paths|val=False; both actual val lists checked: 32 synthetic images each; no DEV; last checkpoint only|
|test_RAW_lock_before_GT|PASS|artifact recheck|raw lock before first scoring-stage reference read|
|test_POSE_lock_before_GT|PASS|artifact recheck|pose lock before scoring-stage reference read|
|test_same_frozen_GEO_LINEAR_all_arms|PASS|artifact recheck|single frozen scorer hash rechecked|
|test_D9_supplement_only|PASS|artifact recheck|RESULTS.json: separate D9 metrics; primary is current|
|test_clean29|PASS|artifact recheck|fixed DEV group counts|
|test_moderate21|PASS|artifact recheck|fixed DEV group counts|
|test_severe78|PASS|artifact recheck|fixed DEV group counts|
|test_anchor_FINAL_V2|PASS|artifact recheck|verified anchor hash + reference_version|
|test_oracle_posthoc_only|PASS|artifact recheck|pose locks and scoring contract|
|test_primary_BASE_vs_HMANUAL|PASS|artifact recheck|decision deltas independently recomputed|
|test_decision_rule_predeclared|PASS|artifact recheck|original immutable PROTOCOL_LOCK; named case maps to frozen DECISION|
|test_no_more_labeling_on_no_gain|PASS|artifact recheck|stop_more_hard_labeling; reserves not activated|
|test_no_threshold_sweep|PASS|run inventory + fixed args; no broader historical claim|only the two fixed arm run directories; seed42/5epoch/320updates|

## 별도 마무리 확인

- 지시문 지정 그림14개 존재 및 manifest 기록.
- REPORT_KO.md 1~16절 + 부록에 표/이미지/해석 정리.
- 원래 실험 결과·checkpoint·raw prediction·pose decision35개 보존.
- CLI final output 항목 및 중간 상태 resume 회귀검사.
- 추가 학습0, 추론0, 어노테이션0.
- 선택 항목인 bootstrap 미실행; 유의성 주장 없음.

[상세 JSON](DIRECTIVE_AUDIT.json) · [재개 테스트](RESUME_TESTS.json) · [그림 manifest](FIGURE_MANIFEST.json)
