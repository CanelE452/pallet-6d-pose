"""Independent invented-only functional probes; imports no model or data I/O."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np

H=Path(__file__).absolute().parent
code=H.parent/'source_evaluation_preparation_b/evaluate_saved_source_direct_end.py'
spec=importlib.util.spec_from_file_location('v202_saved_source_functions_only',code)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
checks={};findings=[]

t=np.array([0,1,-1,-1,0,1],np.int8);av=np.array([1,1,1,0,1,1],bool)
pa=np.array([0,0,1,-1,-1,1],np.int8);pb=np.array([0,1,0,-1,1,1],np.int8)
m=e.metric(t,pa,av,list(range(6)))
assert m['CM_D_E_by_D_E_U']==[[1,0,1],[1,1,0]]
assert (m['rows'],m['known'],m['target_UNKNOWN'],m['original_inputU'],m['available_target_UNKNOWN'])==(6,4,2,1,1)
assert (m['D_precision'],m['E_precision'],m['D_recall'],m['E_recall'],m['known_prediction_U_rate'])==(.5,1.,.5,.5,.25)
assert m['UNKNOWN_target_prediction_counts']=={'-1':1,'0':0,'1':1}
checks['scalar_CM_PR_UNKNOWN_U_conservation']=True
z=e.metric(t,pa,av,[1,5]);assert z['D_recall'] is None and z['E_recall']==.5
checks['empty_class_denominator_None']=True
try:e.metric(np.array([0],np.int8),np.array([-1],np.int8),np.array([False]),[0]);raise RuntimeError('Known target on inputU accepted')
except AssertionError:pass
checks['unavailable_input_cannot_acquire_known_target']=True
p=e.paired(t,{e.ARMS[0]:pa,e.ARMS[1]:pb},list(range(6)))
assert p==dict(known=4,both_correct=2,both_wrong=1,recovered_by_XY=1,damaged_by_XY=0,net_correct_change=1)
checks['paired_row_recovery_damage_net_separate_from_counts']=True

metadata=[];split=[]
for k in range(8):
    key=['REAL_SEMANTIC_END_WEAK','INVENTED','BEFORE',k]
    r=dict(row_id='invented/native'+str(k),original_global_index=k,native_slot=k,query_xy=[float(k),3.],L=20.,original_query_available=True,control_join_key=key,domain=key[0],attempt_id='INVENTED',variant='BEFORE',cache_index=k,source_packet={'path':'NO_FILE_READ','sha256':'a'*64},authority='INVENTED_WEAK',root_admitted=k==0)
    metadata.append(r);split.append(dict(r,cache_index=700+k,stored_weak_direct_target=0 if k==0 else -1,role='TRAIN',component_id='invented_component'))
aligned=e.join_roles(metadata,list(reversed(split)),np.array([0]+[-1]*7,np.int8))
assert [r['cache_index']for r in aligned]==list(range(8)) and [r['source_split_cache_index']for r in aligned]==list(range(700,708))
checks['canonical_controlkey_join_ignores_stale_plan_cacheindex_and_split_order']=True
bad=[dict(r)for r in split];bad[0]['query_xy']=[900.,900.]
try:e.join_roles(metadata,bad,np.array([0]+[-1]*7,np.int8));raise RuntimeError('Altered q accepted')
except AssertionError:pass
checks['split_q_drift_negative']=True

packet={k:np.asarray([metadata[0][k]])for k in e.META}
packet['query_xy']=np.array([[np.nan,np.nan]],np.float64);packet['L']=np.array([np.nan],np.float64)
e.assert_meta(dict(metadata[0],query_xy=None,L=None),packet,0)
checks['META6_null_q_L_nan_equivalence']=True

mm=[];rr=[];yy=[]
for variant,target in [('BEFORE',0),('COVER',1),('SHAM',0)]:
    for k in range(2):
        mm.append(dict(domain='SYNTH_FINITE_END',attempt_id='INVENTED_COMPLETE',native_slot=k,variant=variant));rr.append(dict(role='TRAIN',component_id='C'));yy.append(target if k==0 else -1)
pp={e.ARMS[0]:np.array([0,0,1,0,0,0],np.int8),e.ARMS[1]:np.array([0,0,0,0,0,0],np.int8)}
summary,records=e.triplets(mm,rr,np.array(yy,np.int8),pp)
assert summary['FULL']['known_D_E_D_triplets']==1 and summary['FULL']['UNKNOWN_triplets']==1
assert summary['FULL']['paired']==dict(both_correct=0,both_wrong=0,recovered_by_XY=0,damaged_by_XY=1,net_correct_change=-1)
checks['complete_known_DED_and_other_unknown_triplet_pairing']=True

# Existing admitted input population includes geometry-failed BEFORE-only U.
mm.append(dict(domain='REAL_SEMANTIC_END_WEAK',attempt_id='INVENTED_BEFORE_ONLY_UNKNOWN',native_slot=0,variant='BEFORE'))
rr.append(dict(role='TRAIN',component_id='C2'));yy.append(-1)
pp={a:np.append(v,np.int8(0))for a,v in pp.items()}
try:
    summary,records=e.triplets(mm,rr,np.array(yy,np.int8),pp)
    assert summary['FULL']['known_D_E_D_triplets']==1
    checks['BEFORE_only_UNKNOWN_preserved_without_forced_COVER_SHAM_or_label']=True
except AssertionError as error:
    checks['BEFORE_only_UNKNOWN_preserved_without_forced_COVER_SHAM_or_label']=False
    findings.append(dict(blocking=True,function='triplets',message=str(error),fixture='Complete known D/E/D plus complete other-slot UNKNOWN and one BEFORE-only UNKNOWN native attempt. No model/input/target file exists.',reason='Latest DIRECT input contract retains geometry-failed BEFORE-only all8 UNKNOWN; complete triplets cannot be required for these rows.'))

out=dict(completed=True,PASS=not findings,invented_only=True,code={'path':str(code),'sha256':hashlib.sha256(code.read_bytes()).hexdigest()},checks=checks,findings=findings,actual_source_metadata_target_savedprediction_fitlog_checkpoint_NPZ_header_or_body_reads=0,model_forward=0)
p=H/'INITIAL_FUNCTIONAL_PROBE.json'
with p.open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(completed=True,PASS=out['PASS'],checks=len(checks),findings=findings,probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),ensure_ascii=False))
