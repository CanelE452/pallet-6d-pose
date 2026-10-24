"""Tiny invented scalar/identity cases only; no producer/model/data imports."""
from pathlib import Path
import ast,copy,importlib.util
import numpy as np
H=Path(__file__).absolute().parent

def rejects(f):
    try:f()
    except (AssertionError,ValueError):return True
    return False

def run_checks():
    s=importlib.util.spec_from_file_location('saved_source_numpy_fixture',H/'evaluate_saved_source_direct_end.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e);checks={}
    y=np.array([0,0,0,1,1,-1,-1],np.int8);pred=np.array([0,1,-1,1,0,-1,0],np.int8);av=np.array([True]*5+[False,True])
    m=e.metric(y,pred,av,list(range(7)))
    checks['2x3_CM_precision_recall_U_and_UNKNOWN_separate']=m['CM_D_E_by_D_E_U']==[[1,1,1],[1,1,0]]and m['D_precision']==m['E_precision']==.5 and m['D_recall']==1/3 and m['E_recall']==.5 and m['known_prediction_U_rate']==.2 and m['target_UNKNOWN']==2 and m['original_inputU']==m['available_target_UNKNOWN']==1
    z=e.metric(np.array([-1,-1],np.int8),np.array([-1,0],np.int8),np.array([False,True]),[0,1]);checks['empty_known_denominators_None_all_UNKNOWN_retained']=z['known']==0 and z['D_precision']is None and z['E_recall']is None and z['known_prediction_U_rate']is None and z['UNKNOWN_target_prediction_counts']=={'-1':1,'0':1,'1':0}
    checks['inputU_known_or_imputed_prediction_rejected']=rejects(lambda:e.metric(np.array([0]),np.array([-1]),np.array([False]),[0]))and rejects(lambda:e.metric(np.array([-1]),np.array([0]),np.array([False]),[0]))
    yy=np.array([0,1,0,0,1,0,-1,-1,-1],np.int8);pp={e.ARMS[0]:np.array([0,1,1,0,1,0,-1,0,1],np.int8),e.ARMS[1]:np.array([0,1,0,1,1,0,-1,1,0],np.int8)}
    pair=e.paired(yy,pp,list(range(9)));checks['paired_recovery_damage_net_counts_known_only']=pair==dict(known=6,both_correct=4,both_wrong=0,recovered_by_XY=1,damaged_by_XY=1,net_correct_change=0)
    metadata=[];roles=[]
    for i in range(9):
        group=i//3;variant=('BEFORE','COVER','SHAM')[i%3];native=0 if group<2 else7
        r=dict(row_id='AUTHORED%d'%i,original_global_index=i,native_slot=native,query_xy=[.25+i,1.5],L=48.,original_query_available=i!=6,domain='SYNTH_FINITE_END',attempt_id='CASE%d'%group,variant=variant,cache_index=i,authority='AUTHORED_WEAK_END',source_packet=dict(path='/authored/%d.npz'%i,sha256=str(i)*64),root_admitted=True)
        r['control_join_key']=[r['domain'],r['attempt_id'],variant,native];metadata.append(r);roles.append(dict(r,role='TRAIN'if group<2 else'HOLDOUT',component_id='COMPONENT%d'%group,stored_weak_direct_target=int(yy[i])))
    summaries,rows=e.triplets(metadata,roles,yy,pp);checks['known_DED_triplet_AND3_vs_unknown_native_preserved']=summaries['FULL']['physical_all_native_triplets']==3 and summaries['FULL']['known_D_E_D_triplets']==2 and summaries['FULL']['UNKNOWN_triplets']==1 and all(summaries['FULL']['models'][a]['all3_correct']==1 for a in e.ARMS)and summaries['FULL']['paired']['recovered_by_XY']==summaries['FULL']['paired']['damaged_by_XY']==1 and summaries['HOLDOUT']['known_D_E_D_triplets']==0
    incomplete=yy.copy();incomplete[2]=-1;checks['partial_known_triplet_not_silently_admitted']=rejects(lambda:e.triplets(metadata,roles,incomplete,pp))
    order=[3,4,5,0,1,2,6,7,8];reordered=[dict(metadata[j],cache_index=i)for i,j in enumerate(order)];aligned=e.join_roles(reordered,roles,yy[order]);checks['control_key_join_not_stale_cacheindex']=all(r['source_split_cache_index']==order[i]and r['cache_index']==i for i,r in enumerate(aligned))
    bad=copy.deepcopy(reordered);bad[0]['query_xy'][0]+=.01;checks['qL_or_controlkey_or_authority_drift_rejected']=rejects(lambda:e.join_roles(bad,roles,yy[order]));bad=copy.deepcopy(reordered);bad[0]['authority']='OWNER_LABEL';checks['qL_or_controlkey_or_authority_drift_rejected']&=rejects(lambda:e.join_roles(bad,roles,yy[order]));bad=copy.deepcopy(reordered);bad[0]['control_join_key']=bad[1]['control_join_key'];checks['qL_or_controlkey_or_authority_drift_rejected']&=rejects(lambda:e.join_roles(bad,roles,yy[order]))
    openold=e.os.open;access=[]
    try:
        e.os.open=lambda *args,**kwargs:access.append(args)or (_ for _ in ()).throw(RuntimeError('Unexpected filesystem'))
        checks['v112_lexical_rejection_before_filesystem']=rejects(lambda:e.opened(H/'V112_FORBIDDEN.json'))and access==[]
    finally:e.os.open=openold
    reader,bound=e.read,e.read_bound
    try:
        e.read=lambda _:dict(approved_saved_source_direct_evaluation=False,source_only=True,automatic_retry=False);e.read_bound=lambda _:(_ for _ in ()).throw(RuntimeError('Premature bound body read'))
        checks['false_entry_stops_before_any_fit_target_or_output_body']=rejects(lambda:e.evaluate(Path('/authored_false_entry.json')))
    finally:e.read,e.read_bound=reader,bound
    tree=ast.parse((H/'evaluate_saved_source_direct_end.py').read_text());imports=[n for n in ast.walk(tree)if isinstance(n,(ast.Import,ast.ImportFrom))];checks['NumPy_stdlib_only_no_model_or_producer_import']=all(not any(t in ast.unparse(n)for t in ('torch','train_spatial','candidate','context_adapter','cv2','PIL'))for n in imports)
    body=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='evaluate'));checks['actual_guard_both4000_before_any_NPZ_target_or_prediction_open']=body.index('barrier = read_bound')<body.index('target_packet = arrays')and 'fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs'] is True'in body
    assert all(checks.values()),checks
    return {k:bool(v)for k,v in checks.items()}
