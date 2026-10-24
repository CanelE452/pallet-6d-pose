import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
V=Path(__file__).resolve().parent; P=V.parent
C12=P/'query_visibility_material_augmented_query_head_20261023_v114'
S=P/'query_visibility_same_material_foreground_source_20261023_v116'
def b(p):
 p=Path(p); return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
old=json.loads((C12/'ROOT_C12_DESIGN_BEFORE_SOURCE_FORWARD.json').read_text())
d={
 'created_UTC':datetime.now(timezone.utc).isoformat(),
 'candidate':'C13_SAME_MATERIAL_FOREGROUND_AUGMENTED_SPATIAL_QUERY',
 'prospective_before_new_feature_or_head_forward':True,
 'rationale':'PUBLIC source C12 five new VAL regressions all wood16 foreground; old TRAIN controlledwoodforeground0. BCE and sign are correct; relative cover-sham margin can pass despite absolute E errors. Add controlled samewood TRAIN role rather than change backbone, initial q, labels, threshold or PnP.',
 'source_rationale':b(C12/'public_source_regression_inventory/COMPACT_RESULT.json'),
 'public_visual_rationale':b(C12/'ROOT_PUBLIC_REGRESSION_ACTUAL_VISUAL_RECONCILIATION.json'),
 'parent_C12_design':b(C12/'ROOT_C12_DESIGN_BEFORE_SOURCE_FORWARD.json'),
 'parent_C12_trainer':b(C12/'source_fit_preparation/train_source_once.py'),
 'task':old['task'], 'base_source':old['source'], 'features':old['features'], 'head':old['head'],
 'new_source':{
  'corpus_seal':b(S/'corpus_build_v1/CORPUS_OUTPUT_SEAL.json'),
  'root_actual_execution_and_conditional_AUX_admission':b(S/'ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json'),
  'input_occurrences':b(S/'corpus_build_v1/OCCURRENCES924_INPUT_ONLY.jsonl'),
  'GT_occurrences':b(S/'corpus_build_v1/OCCURRENCES924_GT_ONLY.jsonl'),
  'feasibility_attempts':b(S/'corpus_build_v1/FEASIBILITY448_GT_ONLY.jsonl'),
  'input_RGB96':b(S/'corpus_build_v1/RGB96_AND_VALID924_INPUT_ONLY.npz'),
  'occurrences':924,'stored_feature_units':152,'fresh_changed_feature_units':124,'bitexact_old_BEFORE_units':28,
  'labels':{'D':90,'E':62,'S':0,'U':772},'original_split':'TRAIN_ONLY',
  'accepted_pair_count':62,'failed_pair_count':386,'failures_never_negative':True,
  'conditional_AUX_only':True,'native_mapping':'UNKNOWN','human_gold':False,
  'limitation':'All accepted PALLOX, LOCO0; same physical bin and previously source-exposed RGB; obvious2D paste/light/shadow/fringe limitations. New donor role, not independent new images or natural visibility gold.'
 },
 'combined':{'occurrences':46108,'stored_feature_unit_rows':15004,'known_TRAIN':3233,'known_VALIDATION':636,'raw_states':{'D':2944,'E':925,'S':3521,'U':38718},'all_rows_and_original_folds_retained':True},
 'feature_policy':'Exact frozen C11 1928 schema. V116152 units (28bitexact BEFORE +124fresh full/native-context/local/central views). May reuse failed unchanged rows via their BEFORE unit. Every changed context freshly encoded from that changed native image at original fixed box. No masks/depth/GT/native slot/reference enter head or encoder.',
 'sampling':{
  'updates':2000,'batch':128,'D_E_per_batch':[64,64],
  'base96_exact_C12':old['sampling'],
  'base64_draw_SHA256_required':'8d6ec2eecbdd0937e710173328cea8e0d4211d21b11d722faea5d5c6fb8cf680',
  'base96_draw_SHA256_required':'77e99c76f7c0f797aaadd00efaddd73824dd564defc84535114c3e7d24b1484a',
  'new_draw':'Uniform with replacement16 of accepted62 TRAIN cover/sham pairs; append coverE andshamD for each pair. Original28 BEFORE D retained/reported, not inserted as extra sampled duplicates.',
  'new_pair_generator_seed':216,'final128_shuffle_seed':217,
  'matched_hinge_pairs_per_batch':24,
  'old_pair_mapping':'Map old8 cover/sham positions through inverse final128 shuffle. New16 use actual cover/sham batch positions; no fabricated BEFORE index.'
 },
 'optimizer':{'name':'AdamW','lr':0.001,'weight_decay':0.001,'gradient_clip_norm':5,'loss':'Mean absolute BCE over all128 known available rows +0.25*mean max(0,2-(z_cover-z_sham)) over24 matched pairs (old8+new16).'},
 'selector':old['selector'], 'source_gates':old['source_gates'],
 'source_gates_count':15,'threshold_logit':0,'D_rule':'z<0; z>=0 hidden; factual unavailableU',
 'mandatory_additive_reports_not_selector':[
  'All old194 accepted pairs by original TRAIN/VALIDATION and donor objectID, BEFORE/D, COVER/E, SHAM/D, all3correct and cover-sham margin>=2. Include native numericaldenominators and U.',
  'Old VAL wood16covers35, otherobjects4/6/12covers84; report absolute E recall and individual five public regressions, no cherry picking. Report newTRAIN62pairs separately; do not call them newVAL.',
  'Source confusion matrices/precision/recall/U by domain and original split, with all46108original occurrences includingU/S retained.',
  'No source or current success claim from relative-margin pass or strong D alone.'
 ],
 'control_limitations':'One bounded augmentation candidate, not isolated data-only causal proof: mean BCE old-row mass becomes.75 and old hinge mass becomes1/3. Old96 draw IDs, head, initialization, optimizer, selector10 and threshold0 unchanged. One modelseed, no private-selected alternatives.',
 'new_fit_requires_actual_feature_integrity_and_root_entry':True,
 'current223_training_normalizer_threshold_checkpoint_arm_selection':False,
 'V112_training_or_selection_or_new_inference':False,
 'current_report_only_after_source_freeze':'If required15sourcegates pass, compare selected head to frozen v61/legacy/C11/C12 on same saved1784 original feature/query units using fixed z0 and existing cohort definitions. No current result alters C13. No GT recrop/detection/permutation selection.',
 'PnP_or_q_L_correction':False,'fully_solved':False
}
target=V/'ROOT_C13_DESIGN_BEFORE_SOURCE_FORWARD.json'
assert not target.exists()
target.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(b(target)))
