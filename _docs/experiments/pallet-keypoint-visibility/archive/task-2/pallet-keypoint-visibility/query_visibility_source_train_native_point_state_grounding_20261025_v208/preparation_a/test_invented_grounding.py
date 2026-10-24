"""Meaningful authored metadata fixtures only; no source data or GT opened."""
from pathlib import Path
import importlib.util,json,copy
H=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('lookup',H/'lookup_saved_train_native_point_states.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
spec=importlib.util.spec_from_file_location('selection',H/'prepare_grounding_selection.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
checks=[]
def check(n,x):assert x,n;checks.append({'name':n,'PASS':True})
def rejected(fn):
 try:fn()
 except AssertionError:return True
 return False
original={'base_index':0,'role':'TRAIN','native_slot':0,'declared_CF_to_canonical':4,'query_xy':[20.,30.],'L':96.,'original_query_available':True}
rec={'CF_index':0,'canonical_index':4,'native_canonical_index':4,'target_witness_states':['D']*37,'combined_witness_states':['E']*37,'state':'U','target_alone_region_state':'D','projected_reference_xy':[20.,30.]}
r=v.extract_one(original,rec,None)
check('point0_external_E_separate_from_region_U_and_original_q',r['point0_target_alone_state']=='D'and r['point0_combined_state']=='E'and r['region_state']=='U'and r['query_xy']==[20.,30.])
bad=copy.deepcopy(rec);bad['target_witness_states']=[];bad.pop('combined_witness_states');bad.pop('projected_reference_xy');z=v.extract_one(original,bad,None)
check('missing_point_reference_UNKNOWN_no_region_D_alias',z['point0_target_alone_state']==z['point0_combined_state']=='UNKNOWN'and z['region_state']=='U'and z['saved_anchor_projected_reference_xy']is None)
check('canonical_mismatch_fails_closed',rejected(lambda:v.extract_one(original,dict(rec,canonical_index=3),None)))
check('F64_crop_inverse_center_and_canvas_vs_center_extents',v.locality([20.,30.],96.,[20.,30.],True)['raw96_reference_index_xy']==[47.5,47.5]and v.locality([0.,0.],96.,[-48.,0.],True)['inside_canvas_extent']and not v.locality([0.,0.],96.,[-48.,0.],True)['inside_sample_center_extent'])
check('inputU_finite_reference_stays_U_not_promoted',v.locality([20.,30.],96.,[20.,30.],False)['original_input_U_preserved'])
records=[dict(rec,CF_index=k,canonical_index=k,native_canonical_index=k)for k in range(8)]
doc={'rows':[{'scene_index':0,'variant':'BEFORE','records_CF':records},{'scene_index':1,'variant':'COVER','records_CF':records}]}
cf,reason=v.cf_records(doc,0);check('bounded_before_scene_and_all8_CF_records',reason is None and list(sorted(cf))==list(range(8)))
cf,reason=v.cf_records({'unrecognized':'no fields'},0);check('unknown_container_fails_closed_UNKNOWN_not_projection',cf is None and reason is not None)
bad=copy.deepcopy(doc);bad['rows'][0]['records_CF'][1]['CF_index']=0
check('duplicate_native_field_rejected',rejected(lambda:v.cf_records(bad,0)))
cache=[];roles=[]
for k in range(8):
 x={'row_id':'train%d'%k,'control_join_key':['SYNTH_FINITE_END','a','BEFORE',k],'domain':'SYNTH_FINITE_END','cache_index':700+k,'original_global_index':k,'native_slot':k,'query_xy':[1.,2.],'L':8.,'original_query_available':True,'root_admitted':False,'source_status':'UNKNOWN'};cache.append(x);roles.append(dict(x,role='TRAIN',cache_index=7-k,stored_weak_direct_target=-1))
x=dict(cache[0],row_id='held',control_join_key=['SYNTH_FINITE_END','h','BEFORE',0]);cache.append(x);roles.append(dict(x,role='HOLDOUT',stored_weak_direct_target=1))
selected=s.select_train(list(reversed(cache)),roles)
check('TRAIN_controlkey_join_ignores_stale_index_and_known_label',len(selected)==8 and all(r['root_admitted']is False and z['stored_weak_direct_target']==-1 for r,z in selected))
check('V112_lexical_before_any_filesystem',rejected(lambda:v.lexical('/invented/V112/SCENES_GT_ONLY.json')))
result={'completed':True,'PASS':True,'invented_only':True,'checks':checks,'number_checks':len(checks),'actual_GT_config_binary_pixels_or_source_rows_read':0}
with (H/'INVENTED_LOOKUP_CHECKS.json').open('x',encoding='utf-8')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'PASS':True,'checks':len(checks)}))
