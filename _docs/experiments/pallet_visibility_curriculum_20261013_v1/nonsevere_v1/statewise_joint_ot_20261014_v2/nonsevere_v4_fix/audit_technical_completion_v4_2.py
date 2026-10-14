from pathlib import Path
import hashlib,json,re,subprocess
ROOT=Path('/home/minjae/Documents/github/pallet-pose')
D=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1'
F=D/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix'
W=Path('/tmp/pallet-visibility-publish-20261013')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
score=read(F/'evaluation_interval_v4_2b/POSTSEAL_SCORING.json')
seal=read(F/'evaluation_interval_v4_2b/PREDICTION_SEAL.json')
protocol=read(F/'evaluation_interval_v4_2b/PROTOCOL.json')
contracts=read(F/'evaluation_interval_v4_2b/CONTRACT_CHECKS.json')
assert score['target_achieved'] and all(score['target_checks'].values())
assert seal['rows']==98 and seal['primary_rows']==95 and seal['crop_rows']==3 and seal['severe_rows']==0
assert seal['before_scoring'] and not seal['annotation_targets_accessed']
assert sha(F/'evaluation_interval_v4_2b/PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
assert sha(F/'evaluation_interval_v4_2b/PROTOCOL.json')==seal['protocol_sha256']
assert sha(F/'evaluation_interval_v4_2b/PREDICTION_SEAL.json')==score['prediction_seal']['sha256']
for group in ('source_sha256','input_and_model_sha256','protected_user_sha256'):
 for path,h in protocol[group].items():assert sha(ROOT/path)==h,path
runtime=read(F/'independent_review/runtime_audit_interval_v4_2b_verified/AUDIT.json')
assert runtime['passed'] and runtime['numeric_target_achieved'] and not runtime['failures']
report_review=read(F/'independent_review/FINAL_REPORT_REVIEW_V4_2.json')
assert report_review['passed']
# Review must bind the current report rather than an earlier draft.
assert sha(F/'README_KO.md') in json.dumps(report_review)
cli=read(F/'RUNTIME_CLI_V4_2_RECEIPT.json'); assert cli['passed']
sproof=read(F/'actual_s_smoke_v4_2/COMPLETION_RECEIPT.json');assert sproof['passed'] and sproof['actual_RGB_full_API_calls']==8
peerS=read(F/'independent_review/ACTUAL_S_PROOF_PEER_AUDIT_V4_2.json');assert peerS['passed']
cal=read(F/'truncation_pose_validation_v4_2/CAL_POSE_VALIDATION_SCORE.json')
assert cal['all_four_known_in_canvas_T_contradictions_removed'] and cal['CAL_total_success_count_not_lower']
assert not cal['success_loss_frame_ids'] and not cal['material_translation_increases_ge1cm']
model=read(F/'truncation_interval_model_CAL_v4_2/MODEL.json')
assert model['fit_partition']=='CAL' and model['eligible_point_count']==264 and model['radius_ratio']==.544750754221928
unit=read(F/'truncation_interval_implementation_v4_2/IMPLEMENTATION_RECEIPT_V3.json')
assert unit['tests']['passed']==18 and unit['tests']['failed']==0
result=read(F/'FINAL_RESULTS_V4_2.json');assert result['all_numeric_targets_met'] and result['report_sha256']==sha(F/'README_KO.md')
assert all(counts.get('T',0)==0 for counts in result['primary_state_counts'].values())
for target in re.findall(r'\]\(([^)]+)\)',(F/'README_KO.md').read_text()):
 if not target.startswith(('http://','https://')):assert (F/target).exists(),target
figure_count=0
for folder in ('figures_interval_v4_2','figures_interval_effect_v4_2'):
 for row in read(F/folder/'FIGURE_RECEIPT.json')['records']:
  for ext in ('png','svg'):
   assert sha(F/folder/(row['id'].replace(':','__')+'.'+ext))==row[ext+'_sha256']
  figure_count+=1
assert figure_count==9
prior=read(D/'PUBLICATION_MANIFEST_V5.json')
for path,record in prior['files'].items():
 assert sha(ROOT/path)==record['sha256'] and sha(W/path)==record['sha256'],path
fix_protocol=read(F/'FIX_PROTOCOL_V1.json')
for path,h in fix_protocol['protected_user_sha256'].items():assert sha(ROOT/path)==h,path
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()=='7e92fdcefb0a37bdee0aef95d3e86c572e23b967'
requirements=[
 ('five_keypoint_roles_and_explicit_abstention','README_KO.md'),
 ('actual_S_erasure_and_projection_only_four_real_pairs','actual_s_smoke_v4_2/COMPLETION_RECEIPT.json'),
 ('BI_coordinates_covariance_nativeQ_and_center_preserved','independent_review/runtime_audit_interval_v4_2b_verified/AUDIT.json'),
 ('all_O_and_retained_T_joint_constraints_preserved','independent_review/runtime_audit_interval_v4_2b_verified/AUDIT.json'),
 ('CAL_only_untrimmed_interval_full_and_session_excluded_fit','truncation_interval_model_CAL_v4_2/MODEL_SEAL.json'),
 ('predeclared_CAL_acceptance_no_success_loss','truncation_pose_validation_v4_2/CAL_POSE_VALIDATION_SCORE.json'),
 ('actual98_RGB_with_no_severe_in_primary95_and_seal_before_scoring','evaluation_interval_v4_2b/PREDICTION_SEAL.json'),
 ('same49_baselines_all_fallbacks_and_fixed_numeric_targets','evaluation_interval_v4_2b/POSTSEAL_SCORING.json'),
 ('why_improvement_and_exact_no_OT_pose_effect','postseal_effect_audit_v4_2b/OT_EXACT_POSE_AUDIT.json'),
 ('actual_image_visualizations_and_all_old_OT_effect_cases','figures_interval_effect_v4_2/FIGURE_RECEIPT.json'),
 ('usable_actual_image_CLI_matches_final_runtime','RUNTIME_CLI_V4_2_RECEIPT.json'),
 ('focused_fit_gate_API_tests','truncation_interval_implementation_v4_2/IMPLEMENTATION_RECEIPT_V3.json'),
 ('bounded_final_report_independently_checked','independent_review/FINAL_REPORT_REVIEW_V4_2.json')]
output={
 'schema':'STATEWISE_INTERVAL_TECHNICAL_COMPLETION_AUDIT_V4_2',
 'objective':'Implement and evaluate the user-defined S/B/I/O/T method, fix observed failure mechanisms and compare fairly with existing methods',
 'technical_completion_passed':True,'final_report_review_passed':True,
 'requirements':[{'requirement':name,'passed':True,'evidence':binding(F/path)} for name,path in requirements],
 'fixed_targets':score['required_target'],'target_checks':score['target_checks'],
 'TEST_metrics':score['summary']['TEST']['pose']['SELECTED_V4_ALL_OT'],
 'independent_runtime_checks':runtime['total_checks'],'real_image_figures':figure_count,
 'prior1649_files_unchanged':True,'protected_user6_unchanged':True,'main_HEAD_unchanged':True,
 'limits':['reused exploratory TEST; no general/statistical superiority claim',
           'all natural conditional T38 withheld; real crop T detection0; no T recall success claim',
           'TEST O constraints cause no pose change; no O contribution claim',
           'state classifier still makes visible/occluded errors',
           'initial S geometry conditional; fixed-coarse perturb proof',
           'historical annotation-derived C2 reference; fixed N3 backbone; no cross-backbone generalization'],
 'remaining_required_work':['isolated research branch publication and remote blob verification'],
 'goal_completion_before_remote_verification':False,
 'report':binding(F/'README_KO.md'),'final_results':binding(F/'FINAL_RESULTS_V4_2.json'),
 'audit_source_sha256':sha(Path(__file__))}
p=F/'CORE_COMPLETION_AUDIT_V4_2.json';assert not p.exists()
p.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'technical_completion_passed':True,'requirements':len(requirements),'core_sha256':sha(p)},ensure_ascii=False))
