"""V175 saved-only evaluator; preparation reads text metadata and invented arrays."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, sys, traceback
import numpy as np

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_DIRECT_OWNER_SOURCE_FIT.json'
SCOPE_SHA='565637c2a01955448cd8f33126155084b64957b46126433a426d5cb3a46e9ade'
BASE=P/'query_visibility_saved_query_owner_source_benchmark_20261024_v174'
PURE=BASE/'saved_evaluation_preparation/evaluate_saved_owner_outputs.py'
PURE_SHA='fe108e313736194975774006bb85bcdcba4bb0c857bf3783a4334fc9d684172a'
REFERENCE=BASE/'saved_owner_source_benchmark_v1/COMPLETE.json'
REFERENCE_SHA='b0ef94f49e074b3273ff9590be783cd72ff16d151352b9fc9858b53237d98a11'
PROTOCOL=H/'PROTOCOL_BEFORE_NEW_OWNER_SCORE_OR_TRUTH_BODIES.json'
TESTS=H/'GENERATED_INVENTED_CHECKS.json'; OUT=V/'saved_direct_owner_evaluation_v1'
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
ARMS=('C47_PUBLIC_WEAK253_ONLY','C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC')
OWNER3=('TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED')
OWNER6=('INCIDENT_TARGET_DECLARED_CF','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT')
META=('original_global_index','original_occurrence_id','scene_index','base_index','native_slot','query_xy','L','split','variant','material','aspect','original_semantic_state','query_xy_missing','L_missing','selected_index_missing','query_available','feature_constructed')

def require(ok,why):
    if not bool(ok):raise AssertionError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def checked(b):
    path=Path(b['path']).resolve();require(path.is_relative_to(P),'Project-only binding')
    require(not any(x in str(path).lower() for x in ('_v112','_v169','current223','/private_evaluation/')),'Forbidden binding')
    require(sha(path)==b['sha256'],'Binding SHA mismatch '+str(path));return path
def pure_metrics():
    require(sha(PURE)==PURE_SHA,'Pinned V174 pure source')
    wanted={'require','owner_predictions','truth_mass','metrics','same_rows'}
    nodes=[n for n in ast.parse(PURE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    require({n.name for n in nodes}==wanted,'Pure AST function set')
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PURE)+'::PURE_METRICS_ONLY','exec'),ns)
    return ns
def groups_for(d):
    # Exact original V174 grouping: global levels retain empty strata in each split.
    result={}
    for cohort in ('ALL','TRAIN','VALIDATION'):
        mask=np.ones(len(d['split']),bool) if cohort=='ALL' else d['split'].astype(str)==cohort
        result[cohort]=mask
        for factor in ('native_slot','material','aspect','variant','original_semantic_state'):
            for value in np.unique(d[factor]):result[cohort+'::'+factor+'::'+str(value)]=mask&(d[factor]==value)
    return result
def fit_packets(fit):
    require(fit.get('completed') is True and fit.get('PASS') is True,'Actual fit incomplete/failed')
    require(fit.get('all2_frozen_before_outputs') is True,'Both heads must freeze before outputs')
    require(set(fit['arms'])==set(ARMS),'Exactly frozen C47/C48')
    for arm in ARMS:require(fit['arms'][arm]['updates']==8000,'Final8000 only')
    return {arm:fit['arms'][arm]['outputs']['OLD6144'] for arm in ARMS}
def invented(m):
    c={}; logits=np.array([[2,2,1],[0,1,1],[np.nan]*3],np.float32)
    pred,_=m['owner_predictions'](logits,np.array([1,1,0],bool));c['original_first_tie_argmax_and_inputU']=pred.tolist()==[0,1,-1]
    w=np.array([[.2,.3,.1,.25,.1,.05],[0,0,.5,.5,0,0],[np.nan]*6],np.float64)
    k,a,iv,missing=m['truth_mass'](w,np.array([0,0,1],bool));c['soft_T_merge_no_cutoff_unknown_separate']=np.array_equal(k[0],np.array([.5,.1,.25])) and missing.tolist()==[0,0,1]
    metric=m['metrics'](k,a,iv,missing,np.array([0,-1,2]),np.ones(3,bool),3)
    c['weighted_CM_predictionU_recall_denominator']=np.allclose(metric['expected_soft_CM'],[[.5,0,0,0],[.1,0,0,.5],[.25,0,0,.5]])
    c['known_precision_and_unknown_full_inventory']=abs(metric['expected_precision_on_known_truth_mass'][0]-.5/.85)<1e-15 and abs(metric['known_mass']+sum(metric['unknown_mass_by_prediction'])-3)<1e-15
    binary=np.column_stack((k[:,0],k[:,1:].sum(1)));r=m['metrics'](binary,a,iv,missing,np.array([0,-1,1]),np.ones(3,bool),2)
    c['F_BG_binary_collapse_without_native_E_relabel']=np.allclose(r['expected_soft_CM'],[[.5,0,0],[.35,0,1]])
    q=np.array([[1,2],[np.nan,np.nan]],np.float64);c['q_null_exact_metadata_and_changed_row_rejection']=m['same_rows'](q,q.copy()).all() and not m['same_rows'](q,q+1)[0]
    d=dict(split=np.array(['TRAIN']*32+['VALIDATION']*32),native_slot=np.arange(64)%8,material=np.array(['WOOD','SQUARE']*32),aspect=np.array(['SQUARE','RECTANGULAR']*32),variant=np.array(['BEFORE','COVER','NEAR_SHAM','FAR_SHAM']*16),original_semantic_state=np.array(list('DESU')*16))
    g=groups_for(d);c['same63_strata_full_split_and_empty_levels']=len(g)==63 and int(g['ALL'].sum())==64 and int(g['TRAIN'].sum())==32
    e=m['metrics'](k,a,iv,missing,np.array([0,-1,2]),np.zeros(3,bool),3);c['empty_strata_null_PR']=e['rows']==0 and e['expected_precision_on_known_truth_mass']==[None]*3
    c['paired_conjunction_keeps_no_truth_gate']=(np.array([1,1,0],bool)&np.array([1,0,1],bool)).tolist()==[True,False,False]
    fake=dict(completed=True,PASS=True,all2_frozen_before_outputs=True,arms={x:dict(updates=8000,outputs={'OLD6144':{'path':'INVENTED','sha256':'INVENTED'}}) for x in ARMS})
    c['dynamic_two_final_arm_packet_contract']=set(fit_packets(fake))==set(ARMS)
    fake['all2_frozen_before_outputs']=False
    try:fit_packets(fake)
    except AssertionError:c['reject_pre_freeze_fit_complete']=True
    else:c['reject_pre_freeze_fit_complete']=False
    try:m['truth_mass'](np.array([[.9,0,0,0,0,0]]),np.array([False]))
    except AssertionError:c['nonunit_truth_rejected_never_normalized']=True
    else:c['nonunit_truth_rejected_never_normalized']=False
    require(all(c.values()),'Invented checks');return c
def prepare():
    require(sha(SCOPE)==SCOPE_SHA and sha(REFERENCE)==REFERENCE_SHA,'Frozen text metadata')
    scope=read(SCOPE);ref=read(REFERENCE);base_protocol=read(BASE/'saved_evaluation_preparation/PROTOCOL_BEFORE_SAVED_SCORE_OR_SOFT_TRUTH_BODIES.json')
    sides={'TRAIN':scope['input_bindings']['TRAIN_owner_sidecar'],'VALIDATION':scope['input_bindings']['VALIDATION_owner_sidecar_POST_FREEZE_EVALUATION_ONLY']}
    require(sides==base_protocol['sidecars'],'Same original point truth bindings')
    tests=invented(pure_metrics());write(TESTS,dict(completed=True,PASS=True,code=bind(__file__),checks=tests,number_checks=len(tests),actual_score_truth_feature_weight_or_model_body_reads=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),pure_metric_source=bind(PURE),pure_AST_functions=['require','owner_predictions','truth_mass','metrics','same_rows'],tests=bind(TESTS),sidecars=sides,reference_complete=bind(REFERENCE),reference_reports=ref['reports'],reference_ledger=ref['ledger'],reference_original_protocol=bind(BASE/'saved_evaluation_preparation/PROTOCOL_BEFORE_SAVED_SCORE_OR_SOFT_TRUTH_BODIES.json'),arms=list(ARMS),owner3=list(OWNER3),owner6=list(OWNER6),runtime=dict(executable=PYTHON,numpy='1.26.4',torch_imported=False),output=str(OUT),fit_complete='CLI --fit-complete exact root-entry binding, completed/PASS/all2_frozen_before_outputs; arms[name].updates8000/outputs.OLD6144 and all2_freeze metadata; no checkpoint/feature body reads',truth='Original q-point owner6: merge incident+other intoT; F/BG unchanged; ambiguity/INVALID/missing separately unknown. No hardcut or GT-normalization.',prediction='Saved inputU first, original first-index owner argmax; verify owner_prediction on matched rows. No nativeE mapping.',metrics='Same V174 known-mass 3x4 owner and2x3 binary confusion/P/R; U retained in recall denominator; unknown mass separately allocated byprediction; full rows retained.',cohorts='ALL/TRAIN4096 apparent/VALIDATION2048 primary, same63strata, FULL_INVENTORY and ALL3_MATCHED_AVAILABLE; old v61 reference copied only with exact original6056 paired-mask equality.',v61='Copy bound V174 v61 records, with explicit provenance, never reload v61 raw packet or refit threshold. V174 CSV supplies exact ID/availability/prediction reference for paired-mask verification.',public='Only existing public253 training-apparent diagnostic metadata if present in fit completion; no MR held owner labels/metrics invented.',no_selection='Both final arms reported; selected_model null/adopted false/fully_solved false; original native/source qualification unchanged; reused source VAL is not new blind/external validation.',actual_score_truth_header_body_hash_reads_during_preparation=0)
    write(PROTOCOL,p);write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_direct_owner_evaluation=False,source_only=True,code=p['code'],protocol=bind(PROTOCOL),fit_complete=None,no_forward_fit_or_selection=True))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),tests=p['tests'],source_score_truth_bodies_read=0,actual_evaluation=0))
    print(json.dumps(dict(prepared=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['tests'])),flush=True)
def run(entry_path,fit_path):
    e=read(entry_path);require(e.get('approved_saved_direct_owner_evaluation') is True and e.get('source_only') is True and e.get('no_forward_fit_or_selection') is True,'Root saved-only entry required')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Frozen evaluator entry')
    p=read(PROTOCOL);require(p['code']==bind(__file__) and e['fit_complete']==bind(fit_path),'Dynamic exact fit-complete entry')
    for key in ('scope','pure_metric_source','tests','reference_complete','reference_original_protocol'):checked(p[key])
    require(read(checked(p['tests']))['PASS'],'Prepared tests')
    require(str(Path(sys.executable).resolve())==str(Path(PYTHON).resolve()) and np.__version__=='1.26.4','Pinned runtime')
    fit_path=Path(fit_path).resolve();require(fit_path.is_relative_to(V/'runs'),'Fit complete outside V175 runs')
    fit=read(fit_path);packets=fit_packets(fit);checked(fit['all2_freeze'])
    require(not OUT.exists(),'Preserve any previous result/failure; no automatic retry')
    OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry_path),fit_complete=bind(fit_path),protocol=bind(PROTOCOL)))
    try:
        m=pure_metrics();pieces=[]
        for split,N in [('TRAIN',4096),('VALIDATION',2048)]:
            with np.load(checked(p['sidecars'][split]),allow_pickle=False) as z:
                require(z['owner_class_order'].tolist()==list(OWNER6),'Truth owner order')
                require(not z['training_admission'].item() and not z['initializer_physical_native_identity_certified'].item(),'No admission/physical identity certificate')
                d={k:z[k].copy() for k in META};d['owner6']=z['scene_owner_point'].copy()
                require(len(d['original_global_index'])==N and (d['split'].astype(str)==split).all(),'Original fixed split')
                pieces.append(d)
        d={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]};order=np.argsort(d['original_global_index']);d={k:x[order] for k,x in d.items()}
        require(np.array_equal(d['original_global_index'],np.arange(6144)),'Full6144 once')
        known,a,iv,missing=m['truth_mass'](d['owner6'],d['query_xy_missing']);models={};alignment=[]
        for arm in ARMS:
            with np.load(checked(packets[arm]),allow_pickle=False) as z:
                ok=np.ones(6144,bool);reasons=[[] for _ in range(6144)]
                for src,dst in [('global_index','original_global_index'),('occurrence_id','original_occurrence_id'),('scene_index','scene_index'),('base_index','base_index'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L'),('split','split'),('feature_constructed','feature_constructed'),('query_available','query_available'),('no_detection','selected_index_missing')]:
                    same=m['same_rows'](z[src],d[dst]);ok &= same
                    for i in np.flatnonzero(~same):reasons[i].append(src)
                require(z['owner_class_order'].tolist()==list(OWNER3),'Newhead owner order')
                raw=z['raw_owner_three_logits'].copy();av=z['query_available'].astype(bool).copy();saved=z['owner_prediction'].copy()
                require(raw.dtype==np.float32 and raw.shape==(6144,3) and saved.dtype==np.int8,'Newhead saved schema')
                pred,finite=m['owner_predictions'](raw,av&ok);require(np.array_equal(pred[ok],saved[ok]),'Saved owner argmax/U parity')
            models[arm]=dict(pred=pred,raw=raw,available=av,matched=ok,finite=finite,reasons=reasons)
            alignment.append(dict(arm=arm,matched_rows=int(ok.sum()),mismatch_rows=int((~ok).sum()),original_inputU=int((~av).sum()),available_nonfinite=int((ok&av&~finite).sum())))
        with checked(p['reference_ledger']).open(newline='') as f:refrows=list(csv.DictReader(f))
        require(len(refrows)==6144,'V174 reference full inventory');oldpaired=[]
        for i,r in enumerate(refrows):
            require(int(r['original_global_index'])==i and r['original_occurrence_id']==str(d['original_occurrence_id'][i]),'V174 reference exact IDs')
            oldpaired.append(all(r[x+'_matched']=='True' and r[x+'_original_available']=='True' for x in ('C45','C46','v61')))
        oldpaired=np.array(oldpaired,bool)
        paired=oldpaired.copy()
        for x in models.values():paired &= x['matched']&x['available']&x['finite']
        same_paired=bool(np.array_equal(paired,oldpaired) and int(oldpaired.sum())==6056)
        ref=read(checked(p['reference_reports']));require(ref['primary']=='FULL_INVENTORY/VALIDATION','V174 reference definition')
        reports={};binary=np.column_stack((known[:,0],known[:,1:].sum(1)));groups=groups_for(d);require(len(groups)==63,'Original63strata')
        for cohort,gate in [('FULL_INVENTORY',np.ones(6144,bool)),('ALL3_MATCHED_AVAILABLE',paired)]:
            reports[cohort]={}
            for group,mask in groups.items():
                mm=mask&gate;record={}
                for arm,x in models.items():
                    bp=np.where(x['pred']<0,-1,(x['pred']!=0).astype(np.int8))
                    record[arm]=dict(owner3=m['metrics'](known,a,iv,missing,x['pred'],mm,3),binary_target_presence=m['metrics'](binary,a,iv,missing,bp,mm,2))
                record['v61']=ref['reports'][cohort][group]['v61'] if cohort=='FULL_INVENTORY' or same_paired else dict(binary_target_presence=None,unavailable_reason='New paired cohort differs from frozen V174 reference; no substitute baseline or conditional claim')
                reports[cohort][group]=record
        ledger=OUT/'ALL6144_SAVED_DIRECT_OWNER_AND_EXACT_POINT_MASS.csv'
        fields=['original_global_index','original_occurrence_id','split','native_slot','material','aspect','variant','original_semantic_state','query_available','query_xy_missing','L_missing','w_TARGET','w_FOREIGN','w_BG','ambiguous_mass','invalid_mass','missing_q_inventory_unknown']+[arm+'_'+key for arm in ARMS for key in ('prediction','original_available','matched','mismatch_fields')]
        with ledger.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i in range(6144):
                r={k:d[k][i].item() for k in fields[:11]};r.update(w_TARGET=known[i,0],w_FOREIGN=known[i,1],w_BG=known[i,2],ambiguous_mass=a[i],invalid_mass=iv[i],missing_q_inventory_unknown=missing[i])
                for arm,x in models.items():r.update({arm+'_prediction':int(x['pred'][i]),arm+'_original_available':bool(x['available'][i]),arm+'_matched':bool(x['matched'][i]),arm+'_mismatch_fields':'|'.join(x['reasons'][i])})
                writer.writerow(r)
        report=OUT/'EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json';write(report,dict(completed=True,primary='FULL_INVENTORY/VALIDATION',TRAIN_apparent_only=True,reused_source_VAL_not_blind=True,owner3_class_order=['T','F','BG','predictionU'],binary_class_order=['T','nonT','predictionU'],reports=reports,v61_reference_copied_from=p['reference_reports'],v61_reference_no_raw_re_read=True,paired_reference_mask_exact=same_paired))
        for b in list(packets.values())+list(p['sidecars'].values()):checked(b)
        passed=same_paired and all(x['mismatch_rows']==0 and x['available_nonfinite']==0 and x['original_inputU']==88 for x in alignment)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=passed,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry_path),fit_complete=bind(fit_path),all2_freeze=fit['all2_freeze'],saved_packets=packets,sidecars=p['sidecars'],rows=6144,TRAIN=4096,VALIDATION=2048,original_inputU=88,missing_q=int(missing.sum()),paired_rows=int(paired.sum()),same6056_reference_mask=same_paired,alignment=alignment,ledger=bind(ledger),reports=bind(report),v61_reference=p['reference_reports'],public253_apparent_metadata={arm:fit['arms'][arm].get('public253_final_apparent_diagnostic') for arm in ARMS},no_native_label_or_qualification_changes=True,new_forward_fit_feature_weight_private_current_V169_V112=0,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False))
    except BaseException:write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);ap.add_argument('--fit-complete',type=Path);args=ap.parse_args()
    if args.prepare and args.entry is None and args.fit_complete is None:prepare()
    elif not args.prepare and args.entry is not None and args.fit_complete is not None:run(args.entry,args.fit_complete)
    else:raise SystemExit('--prepare OR --entry ROOT_ENTRY --fit-complete ACTUAL_FINAL_COMPLETE')
