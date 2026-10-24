from pathlib import Path
import hashlib,json,math,datetime
V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_separated_frozen_current_benchmark_20261024_v156')
H=V/'current_result_summary_agent';Q=V/'independent_saved_evaluation_review'
def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
s=read(H/'CURRENT_RESULTS_SUMMARY_KO.json');r=read(Q/'INDEPENDENT_RECALCULATED_RESULTS.json');p=read(V/'saved_seven_current_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json')
def same(a,b):
 if isinstance(a,float) or isinstance(b,float):assert a is not None and b is not None and math.isclose(a,b,rel_tol=0,abs_tol=1e-15),(a,b)
 else:assert a==b,(a,b)
def cmcheck(cm,o):
 same(cm,o['CM_rows_D_E_columns_D_E_U']);N=sum(map(sum,cm));same(N,o['N'])
 d,e=map(sum,cm);same(d,o['D_N']);same(e,o['E_N']);u=cm[0][2]+cm[1][2]
 for k,x in [('D_recall',cm[0][0]/d),('E_recall',cm[1][1]/e),('D_precision',cm[0][0]/(cm[0][0]+cm[1][0]) if cm[0][0]+cm[1][0] else None),('E_precision',cm[1][1]/(cm[0][1]+cm[1][1]) if cm[0][1]+cm[1][1] else None),('U',u),('U_rate',u/N),('decided_coverage',1-u/N)]:same(x,o[k])
verified=0
for cohort,gs in s['visibility'].items():
 for group,models in gs.items():
  for model,o in models.items():
   z=[a for a in r['primary_D_E_by_group'] if a['cohort']==cohort and a['model']==model and (group=='ALL' or a['group']==group)]
   assert len(z)==(2 if group=='ALL' else 1)
   cm=[[sum(a['CM_D_E_rows_D_E_U_columns'][i][j] for a in z) for j in range(3)] for i in range(2)];cmcheck(cm,o);verified+=1
for group,models in s['manual_near_D'].items():
 for model,o in models.items():
  z=[a for a in r['manual_D376'] if a['model']==model and (group=='ALL' or a['group']==group)]
  for k in ['N','pred_visible','pred_binary_hidden','unresolved']:same(sum(a[k] for a in z),o[k])
  same(o['pred_visible']/o['N'],o['D_recall']);same(o['pred_binary_hidden']/o['N'],o['false_E_rate'])
same(s['full_registered_model_inventory'],r['all1784_inventory']);same(s['teacher_inventory'],r['human_state_inventory'])
same(s['human_self_binary_response'],p['tables']['human_self_binary_response'])
same(s['paired_recovery_damage_by_material'],r['paired_recovery_damage'])
keys=['cohort','truth_class','baseline','new_model'];counts=['N','both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved']
for row in s['paired_recovery_damage_ALL']:
 z=[a for a in r['paired_recovery_damage'] if all(a[k]==row[k] for k in keys)];assert len(z)==2
 for k in counts:same(sum(a[k] for a in z),row[k])
q=s['secondary_fixed_reference_quality'];qr=r['quality_reference_only']
for k,rk in [('new_primary_gold_known','primary_known'),('new_primary_gold_unknown','primary_unknown'),('near','secondary_near'),('far','secondary_far'),('unknown','secondary_unknown')]:same(q[k],qr[rk])
same(q['known'],415);same(q['original_input_U_among_known'],3)
for row in p['tables']['secondary_fixed_reference_quality_diagnostic']:
 x=dict(row);x['CM_rows_far_near_columns_far_near_U']=json.loads(x['CM_rows_far_near_columns_far_near_U']);same(x,q['metrics'][x['scope']][x['group']][x['model']])
same(s['primary_cohort'],{'name':'paired_same_query_RGB','N':1237,'D':1147,'E':90,'WOOD_N':645,'SQUARE_N':592})
same(s['all_registered_slots'],1784);same(s['original_input_valid'],1739);same(s['original_input_U'],45)
assert s['historical_three_all_table_fields_exact'] and s['selected_model'] is None and s['adopted_model'] is None and not s['fully_solved']
for x in s['source_qualification_unchanged'].values():assert x=={'source_checks_all_pass':False,'source_current_allowed':False,'adopted':False}
z=s['visibility']['paired_same_query_RGB']['ALL'];b=z['C16']
claims=[]
for m in ['C35','C36','C37','C38']:
 a=z[m];assert a['D_recall']<b['D_recall'] and a['E_recall']>b['E_recall'];claims.append({'model':m,'D_correct':a['CM_rows_D_E_columns_D_E_U'][0][0],'D_N':1147,'E_correct':a['CM_rows_D_E_columns_D_E_U'][1][1],'E_N':90,'joint_improvement_vs_C16':False})
assert any('legacy' in x and 'CAL' in x and '독립 평가' in x for x in s['limits_KO'])
assert any('gold는0' in x and 'reference' in x for x in s['limits_KO'])
assert 'not independently certified' in s['manual_near_D_scope']
plots=[H/x for x in ['CURRENT_ALL1237_CONFUSIONS_KO.png','CURRENT_MATERIAL_AND_NEAR_DIAGNOSTICS_KO.png','CURRENT_AUXILIARY_AUTHORITY_DIAGNOSTICS_KO.png']]
review={'completed':True,'PASS':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'schema':'V156_INDEPENDENT_SUMMARY_AND_CHART_REVIEW_V1','summary':bind(H/'CURRENT_RESULTS_SUMMARY_KO.json'),'summary_code':bind(H/'summarize_saved_current.py'),'independent_saved_rowloop_review':bind(Q/'REVIEW.json'),'independent_numeric_reference':bind(Q/'INDEPENDENT_RECALCULATED_RESULTS.json'),'producer_saved_results':bind(V/'saved_seven_current_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json'),'review_code':bind(Path(__file__)),'charts':[dict(bind(x),actually_viewed=True,legible=True,numerical_source_checked=True) for x in plots],'aggregate_comparisons':{'visibility_cohort_group_model_packets':verified,'manual_near_material_and_all_packets':21,'all1784_inventory_rows':21,'human_self_response_rows':28,'paired_by_material_rows':192,'paired_ALL_rows':len(s['paired_recovery_damage_ALL']),'secondary_fixed_reference_metric_rows':24,'all1784_teacher_inventory_exact':True,'historical_three_replay_retained':True},'claims_checked':{'C16_reference':{'D_correct':1137,'D_N':1147,'E_correct':35,'E_N':90},'new_models':claims,'legacy_prior_real_CAL_training_and_development_overlap_disclosed':True,'legacy_comparison_is_not_independent_test_or_synthetic_only_model':True,'primary_new_quality_gold_known':0,'primary_new_quality_gold_unknown':1784,'near376_is_frozen_reference_conditional_only':True,'secondary415_not_physical_or_pixel_quality_gold':True,'entity_negative_is_prediction_response_not_accuracy':True,'all_source_failed_no_adoption_selectednull':True,'different_training_budgets_preclude_branch_only_causal_claim':True},'scope_and_limits':['No new metric evaluation from labels/scores; compared completed aggregate JSON to existing independently recalculated aggregates.','Full-labelled-DE and auxiliary-response summaries were inspected by source-expression/provenance continuity; no fresh rowloop of current ledger was performed in this review.','The three chart PNGs were actually viewed; these are plots, not original current RGB or learned input crops.','No model, checkpoint, feature values, original RGB, private raw annotation, or V112 read in this review.'],'material_issues':[],'selected_model':None,'goal_solved':False}
out=Q/'SUMMARY_AND_FIGURE_REVIEW.json';out.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'PASS':True,'review':bind(out),'visibility_packets':verified,'charts':len(plots)},ensure_ascii=False))
