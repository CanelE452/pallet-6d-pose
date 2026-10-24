"""Independent saved JSONL/NPZ/scalar audit; no producer helpers or raw source IO."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,sys,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DONE=V/'source_end_supervision_train4096_v1/COMPLETE.json'
ORIGINAL=P/'query_visibility_full_source_query_causal_census_20261024_v173/census_preparation/ALL6144_ORIGINAL_QUERY_METADATA_BEFORE_MASKS.jsonl'
EXPECTED_COMPLETE_SHA='19fc0d43afde5e8ff942cecf35e1caf1ea916bb21798e09f0c7bbf7507544bbb'
CLASSES=['OUTSIDE_END_REGION','END_VISIBLE','END_FOREIGN_FRONT','UNKNOWN_SELF_HIDDEN','UNKNOWN_AMBIGUOUS','UNKNOWN_REGISTRATION_OR_CHANGE','INVALID_SUPPORT']
FLAGS=['query_available','factual_query_available','feature_constructed','crop_constructed','context_available','Q_inside','RGB_central4_available','native_continuous_valid','native_nearest_valid']
# Set before saved result bodies: per-row unit-mass and scalar aggregate tolerances.
ROW_TOL=2e-13;AGG_TOL=5e-10
checks={};bindings={};maxdiff=0.
def need(v,name):
    if not v:raise AssertionError(name)
def record(v,name):need(v,name);checks[name]=True
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=digest(p))
def read(p):return json.loads(Path(p).read_text())
def verified(binding):
    p=Path(binding['path']).resolve();need(p.is_relative_to(V),'saved artifact only');need(b(p)==binding,'artifact binding');bindings[str(p)]=binding;return p
def close(a,z,tol=AGG_TOL):
    global maxdiff
    d=abs(float(a)-float(z));maxdiff=max(maxdiff,d);need(d<=tol,'numeric tolerance '+str(d))
def original_train_prefix():
    prefix=hashlib.sha256();rows=[]
    with ORIGINAL.open('rb') as f:
        for i in range(4096):
            line=next(f);prefix.update(line);r=json.loads(line);need(r['original_global_index']==i and r['split']=='TRAIN','TRAIN original prefix');rows.append(r)
    return rows,prefix.hexdigest()
def expected_targets(r):
    m=r['full96_average_mass']
    if m is None:return (-1,-1)
    if not r['original_input_flags']['query_available'] or r['projected_region_out_of_frame']:return (-1,-1)
    v,f=m[1],m[2];u=math.fsum(m[3:6]);e=math.fsum([v,f,u])
    if e==0:return (0,-1)
    return (1 if v>0 else -1 if u>0 else 0,1 if f>0 else -1 if u>0 else 0)
def run():
    record(digest(DONE)==EXPECTED_COMPLETE_SHA,'root_completed_receipt_SHA')
    complete=read(DONE);record(complete['completed'] and complete['PASS'] and not complete['training_admission'] and complete['selected_model'] is None and not complete['fully_solved'],'complete_success_without_adoption')
    bindings[str(DONE)]=b(DONE)
    plan=[json.loads(x) for x in verified(complete['plan']).read_text().splitlines() if x]
    rows=[json.loads(x) for x in verified(complete['outputs']['rows']).read_text().splitlines() if x]
    source,prefix=original_train_prefix()
    record(len(plan)==len(rows)==len(source)==4096 and [r['original_global_index'] for r in rows]==list(range(4096)),'all4096_unique_original_sequence')
    record(len({r['original_occurrence_id'] for r in rows})==4096 and len({r['scene_index'] for r in rows})==512 and len({r['base_index'] for r in rows})==128,'IDs_512scenes_128bases')
    for i,(r,p,s) in enumerate(zip(rows,plan,source)):
        for key,val in s.items():need(r[key]==p[key]==val,f'original source/plan/output field {i}:{key}')
        for key,val in p.items():need(r[key]==val,f'complete plan provenance {i}:{key}')
        need(r['native_slot']==i%8 and r['scene_index']==i//8 and r['base_index']==i//32 and r['split']=='TRAIN','native input enumeration')
        need(set(r['original_input_flags'])==set(FLAGS),'all9flags')
        need(r['native_visibility_target']==r['point_quality_target']==-1 and not r['training_admission'],'no native/quality/admission')
        m=r['full96_average_mass'];pair=expected_targets(r)
        need(pair==(r['visible_support_target'],r['foreign_front_support_target']),'independent target rule')
        if m is not None:
            need(r['class_order']==CLASSES and len(m)==7 and all(math.isfinite(x) and x>=0 for x in m),'mass7 classes/nonnegative')
            close(math.fsum(m),1,ROW_TOL)
            close(r['visible_support_mass'],m[1],ROW_TOL);close(r['foreign_front_support_mass'],m[2],ROW_TOL)
            close(r['unknown_end_mass'],math.fsum(m[3:6]),ROW_TOL);close(r['sampled_end_region_mass'],math.fsum(m[1:6]),ROW_TOL)
        else:need(r['query_xy'] is None or r['L'] is None,'missing mass reflects original qL')
    record(True,'all_original_fields_qL_CF_native_semantics_flags_and_plan_preserved')
    record(True,'mass7_and_independent_nonexclusive_target_rules')
    inputU=sum(not r['original_input_flags']['query_available'] for r in rows);missing=sum(r['full96_average_mass'] is None for r in rows)
    record(inputU==80 and missing==48 and all((r['visible_support_target'],r['foreign_front_support_target'])==(-1,-1) for r in rows if not r['original_input_flags']['query_available']),'original80U_48missing_targets_unknown')
    # Build each saved NPZ column independently from the audited original rows.
    expected={}
    def column(key,values,dtype=None):expected[key]=np.asarray(values,dtype=dtype)
    for key in ('original_global_index','scene_index','base_index'):column(key,[r[key] for r in rows],np.int32)
    for key in ('native_slot','declared_canonical_index'):column(key,[r[key] for r in rows],np.int8)
    for key in ('original_occurrence_id','split','variant','original_semantic_state'):column(key,[r[key] for r in rows])
    column('query_xy',[[np.nan,np.nan] if r['query_xy'] is None else r['query_xy'] for r in rows],np.float64)
    column('L',[np.nan if r['L'] is None else r['L'] for r in rows],np.float64)
    column('selected_index',[-1 if r['selected_index'] is None else r['selected_index'] for r in rows],np.int32)
    for key,sourcekey in [('query_xy_missing','query_xy'),('L_missing','L'),('selected_index_missing','selected_index')]:column(key,[r[sourcekey] is None for r in rows],bool)
    column('nominal_reference_xy',[r['nominal_reference_xy'] for r in rows],np.float64)
    column('correspondence_boundary',[r['correspondence_boundary'] for r in rows],bool)
    column('primary_part_id',[r['primary_part_id'] for r in rows],np.int32)
    for key in FLAGS:column(key,[r['original_input_flags'][key] for r in rows],bool)
    for key in ('material','aspect','geometry_family','foreign_family','signed_quarter'):column(key,[r['factors'][key] for r in rows])
    for key in ('original_query_metadata','original_input_flags','factors','incident_part_ids','active_witness_indices'):column(key+'_json',[json.dumps(r[key],sort_keys=True,separators=(',',':'),allow_nan=False) for r in rows])
    column('jointmass7',[[np.nan]*7 if r['full96_average_mass'] is None else r['full96_average_mass'] for r in rows],np.float64)
    column('jointmass7_missing',[r['full96_average_mass'] is None for r in rows],bool)
    for key in ('visible_support_target','foreign_front_support_target'):column(key,[r[key] for r in rows],np.int8)
    column('source_end_class_order',CLASSES)
    for key in ('new_native_visibility_target','new_point_quality_target'):column(key,[-1]*4096,np.int8)
    column('training_admission',False,bool);column('reasons_json',[json.dumps(r['reasons']) for r in rows])
    npz=verified(complete['outputs']['sidecar'])
    with np.load(npz,allow_pickle=False) as z:
        record(set(z.files)==set(expected),'exact_NPZ_schema')
        for key,ref in expected.items():
            val=z[key];need(val.shape==ref.shape and val.dtype==ref.dtype,'NPZ shape dtype '+key)
            need(val.tobytes()==ref.tobytes(),'NPZ exact independently rebuilt bytes '+key)
    record(True,'all_NPZ_columns_exact_independent_rebuild')
    summary=read(verified(complete['outputs']['summary']));record(summary['completed'] and not summary['training_admission'] and summary['selected_model'] is None and not summary['fully_solved'],'summary_no_adoption')
    selectors={'ALL':list(range(4096))}
    for key in ('native_slot','variant','original_semantic_state'):
        for value in sorted({r[key] for r in rows}):selectors[key+'::'+str(value)]=[i for i,r in enumerate(rows) if r[key]==value]
    for mat,asp in sorted({(r['factors']['material'],r['factors']['aspect']) for r in rows}):selectors['material_aspect::'+mat+'::'+asp]=[i for i,r in enumerate(rows) if (r['factors']['material'],r['factors']['aspect'])==(mat,asp)]
    record(len(selectors)==len(summary['groups'])==21 and set(selectors)=={g['group'] for g in summary['groups']},'all21_groups_present')
    leaves=0
    for g in summary['groups']:
        rr=[rows[i] for i in selectors[g['group']]];ms=[r['full96_average_mass'] for r in rr if r['full96_average_mass'] is not None]
        ints={'N':len(rr),'original_inputU':sum(not r['original_input_flags']['query_available'] for r in rr),'missing_mass':len(rr)-len(ms)}
        for key,val in ints.items():need(g[key]==val,'group count '+key);leaves+=1
        for j in range(7):close(g['sum_jointmass7'][j],math.fsum(m[j] for m in ms));leaves+=1
        for field,target in [('visible_counts','visible_support_target'),('foreign_counts','foreign_front_support_target')]:
            need(g[field]=={str(k):sum(r[target]==k for r in rr) for k in (-1,0,1)},'group target counts');leaves+=3
        need(g['reasons']==dict(Counter(reason for r in rr for reason in r['reasons'])),'group reasons');leaves+=len(g['reasons'])
    record(True,'all21_group_scalar_fsum_counts_mass_targets_reasons')
    for path,bindval in list(bindings.items()):need(b(Path(path))==bindval,'saved inputs immutable')
    allg=next(g for g in summary['groups'] if g['group']=='ALL')
    both=sum(r['visible_support_target']==r['foreign_front_support_target']==1 for r in rows)
    stale=sum(r.get('new_supervision_not_generated') is True for r in rows)
    return dict(completed=True,PASS=True,review_type='independent saved numerical/identity audit only',code=b(__file__),bindings=bindings,original_TRAIN_metadata_prefix=dict(path=str(ORIGINAL),actual_rows_read=4096,remaining_VALIDATION_rows_read=0,prefix_bytes_sha256=prefix),checks=checks,number_checks=len(checks),NPZ_columns_checked=len(expected),aggregate_groups=21,aggregate_scalar_leaves_checked=leaves,predeclared_tolerances=dict(row_mass_absolute=ROW_TOL,aggregate_absolute=AGG_TOL),maximum_numeric_absolute_difference=maxdiff,
       counts=dict(rows=4096,scenes=512,bases=128,original_inputU=inputU,original_qL_missing_and_mass_missing=missing,finite_mass_inputU=inputU-missing,visible_counts=allg['visible_counts'],foreign_front_counts=allg['foreign_counts'],both_positive=both),
       metadata_observations=[dict(field='new_supervision_not_generated',retained_true_rows=stale,meaning='This historical input-plan flag remains true in copied plan metadata despite completed new output measures. Use COMPLETE.json and the actual support fields for execution status; do not claim generation was absent or mutate original provenance.')],
       limitations='Numerical consistency does not independently validate physical/raster GT, synthetic registration, native identity, real transfer or full-goal success. No raw source RGB/mask/depth, SAM result, model/feature/weight or private/held payload was read.',raw_source_or_SAM_or_model_payload_reads=0,producer_helper_imports_or_calls=0,new_fit_forward_selection_or_qL_modification=0,training_admission=False,selected_model=None,fully_solved=False)
if __name__=='__main__':
    try:
        result=run();out=H/'REVIEW.json'
        with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(PASS=True,review=b(out),counts=result['counts'],max_abs_diff=maxdiff,checks=len(checks))))
    except BaseException:
        with (H/'FAILED_AUDIT_PRESERVED.json').open('x') as f:json.dump(dict(PASS=False,checks=checks,traceback=traceback.format_exc()),f,indent=2)
        raise
