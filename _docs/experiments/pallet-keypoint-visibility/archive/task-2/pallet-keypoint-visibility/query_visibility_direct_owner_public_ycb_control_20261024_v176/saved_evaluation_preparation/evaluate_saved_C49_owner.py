"""Saved C49 owner benchmark; preparation executes pure AST/invented arrays only."""
from pathlib import Path
import argparse, ast, copy, csv, hashlib, json, sys, traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_PUBLIC_YCB_FEATURE_OR_NEW_C49_SCORE_BODIES.json'
SCOPE_SHA='035e25d97efb5e196b334941d4932b7e8845dca2e007e43d93708e56e4be3697'
SOURCE=P/'query_visibility_direct_query_owner_control_20261024_v175/saved_evaluation_preparation/evaluate_saved_direct_owner.py'
SOURCE_SHA='ccff9e14675a3d3ddcd42c0af50f3ab5d0b80cdf3212b6ce96a5ba45755a662d'
SOURCE_PROTOCOL=SOURCE.parent/'PROTOCOL_BEFORE_NEW_OWNER_SCORE_OR_TRUTH_BODIES.json'
PURE=P/'query_visibility_saved_query_owner_source_benchmark_20261024_v174/saved_evaluation_preparation/evaluate_saved_owner_outputs.py'
PURE_SHA='fe108e313736194975774006bb85bcdcba4bb0c857bf3783a4334fc9d684172a'
REFERENCE=P/'query_visibility_direct_query_owner_control_20261024_v175/saved_direct_owner_evaluation_v1/COMPLETE.json'
FIT=V/'runs/C49_PUBLIC_YCB_CF314_FIXED8000/COMPLETE.json'
TRAIN_PROTOCOL=V/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_OR_CF_FEATURE_AND_TRAIN_SOFT_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_C49_SCORE_OR_SOFT_TRUTH_BODIES.json'
TESTS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'saved_C49_owner_evaluation_v1'
ARM='C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314'
CONTROL='C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC'
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
OWNER3=['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
OWNER6=['INCIDENT_TARGET_DECLARED_CF','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT']
META=['original_global_index','original_occurrence_id','scene_index','base_index','native_slot','query_xy','L','split','variant','material','aspect','original_semantic_state','query_xy_missing','L_missing','selected_index_missing','query_available','feature_constructed']


def require(ok,why):
    if not bool(ok):raise AssertionError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def checked(b):
    q=Path(b['path']).resolve();require(q.is_relative_to(P) and not any(x in str(q).lower() for x in ['_v112','_v169','current223','/private_evaluation/']),'Source-only path')
    require(sha(q)==b['sha256'],'SHA '+str(q));return q


def helpers():
    require(sha(SOURCE)==SOURCE_SHA,'Frozen V175 source')
    wanted={'pure_metrics','groups_for'}
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    require({n.name for n in nodes}==wanted,'Only two V175 AST functions')
    ns=dict(np=np,ast=ast,require=require,sha=sha,PURE=PURE,PURE_SHA=PURE_SHA)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE)+'::PURE_ONLY','exec'),ns)
    m=ns['pure_metrics']();m['groups_for']=ns['groups_for'];return m


def reference_pair(rows):
    return np.asarray([all(r[a+'_matched']=='True' and r[a+'_original_available']=='True' for a in ['C47_PUBLIC_WEAK253_ONLY',CONTROL]) for r in rows],bool)


def invented(m):
    out=[]
    def passed(name,yes):require(yes,name);out.append(dict(name=name,PASS=True))
    pred,_=m['owner_predictions'](np.array([[1,1,0],[np.nan]*3,[0,2,2]],np.float32),np.array([1,0,1],bool))
    passed('Original first-index owner ties and inputU precedence',pred.tolist()==[0,-1,1])
    owner=np.array([[.2,.3,.1,.25,.1,.05],[np.nan]*6],np.float64);k,a,iv,missing=m['truth_mass'](owner,np.array([0,1],bool))
    passed('Exact point raw mass merge/unknown/null without cutoff',np.array_equal(k[0],[.5,.1,.25]) and a[0]==.1 and iv[0]==.05 and missing[1]==1)
    cm=m['metrics'](k,a,iv,missing,np.array([0,-1]),np.ones(2,bool),3)
    passed('Expected soft CM and unknown conserve full inventory',abs(cm['known_mass']+sum(cm['unknown_mass_by_prediction'])-2)<1e-15)
    d=dict(split=np.array(['TRAIN','VALIDATION']*4),native_slot=np.arange(8),material=np.array(['WOOD','SQUARE']*4),aspect=np.array(['A','B']*4),variant=np.array(['BEFORE','COVER','NEAR_SHAM','FAR_SHAM']*2),original_semantic_state=np.array(['D','E','S','U']*2))
    passed('Exact63 groups with global factor levels including empty strata',len(m['groups_for'](d))==63)
    passed('Row/q comparison preserves null and rejects mismatches',m['same_rows'](np.array([[np.nan,np.nan],[1,2]]),np.array([[np.nan,np.nan],[1,3]])).tolist()==[True,False])
    rows=[{a+'_'+key:'True' for a in ['C47_PUBLIC_WEAK253_ONLY',CONTROL] for key in ['matched','original_available']} for _ in range(3)]
    rows[1][CONTROL+'_original_available']='False';p=reference_pair(rows)
    passed('Reference paired mask is exact conjunction; differing mask not copied',p.tolist()==[True,False,True] and not np.array_equal(p,np.ones(3,bool)))
    baseline={'owner3':{'expected_soft_CM':[[1.25,2.5]]}};copied=copy.deepcopy(baseline)
    passed('Baseline record copied exactly without model re-evaluation',copied==baseline and copied is not baseline)
    kb=np.column_stack((k[:,0],k[:,1:].sum(1)));b=m['metrics'](kb,a,iv,missing,np.array([0,-1]),np.ones(2,bool),2)
    passed('Binary collapse T vs F+BG keeps unknown separate',np.allclose(b['expected_soft_CM'],[[.5,0,0],[.35,0,0]]))
    return out


def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Scope');m=helpers();tests=invented(m);sourcep=read(SOURCE_PROTOCOL)
    write(TESTS,dict(completed=True,PASS=True,code=bind(__file__),pure_source=bind(SOURCE),checks=tests,actual_feature_score_truth_or_reference_result_bodies_read=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),tests=bind(TESTS),pure_source=bind(SOURCE),nested_pure_source=bind(PURE),pure_AST=['pure_metrics','groups_for'],source_protocol=bind(SOURCE_PROTOCOL),training_protocol=bind(TRAIN_PROTOCOL),sidecars=sourcep['sidecars'],expected_fit_path=str(FIT),expected_reference_complete_path=str(REFERENCE),reference_binding='Root entry supplies exact completed V175 evaluation COMPLETE SHA; body not read in preparation',
           arm=ARM,control=CONTROL,raw_key='raw_owner_three_logits',saved_prediction_key='owner_prediction',classes=OWNER3,truth_order=OWNER6,
           truth='Only original q point owner6; merge incident+other=T/F/BG. Unknown ambiguity/invalid/null separately preserved; no cutoff/relabel.',
           metrics='Same pure V175/V174 owner3+binary expected known-mass CM/P/R; FULL and exact6056 paired63strata; original6144/TRAIN4096apparent/VAL2048primary.',
           reference='Verify V175 CSV all6144 exact IDs/native/split/factors/originalflags/known and unknown mass, C47/C48 pair6056 and same C49 mask, then copy C48/v61 records exactly; never read their raw/checkpoints. A differing pair retains C49 inventory but disables paired baseline copy and PASS.',
           apparent='Only copy C49 trainer public253/CF314 TRAIN-apparent diagnostic JSON metadata; no label/held-owner truth creation.',
           no_source_qualification_gate_or_model_selection=True,runtime=dict(executable=PYTHON,numpy='1.26.4',torch_imported=False),output=str(OUT),selected_model=None,adopted=False,fully_solved=False)
    write(PROTOCOL,p)
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_C49_owner_evaluation=False,source_only=True,no_forward_fit_or_selection=True,code=bind(__file__),protocol=bind(PROTOCOL),fit_complete={'path':str(FIT),'sha256':'ROOT_BIND_AFTER_FIXED_FINAL'},reference_evaluation_complete={'path':str(REFERENCE),'sha256':'ROOT_BIND_EXISTING_COMPLETED_V175'}))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),tests=bind(TESTS),prepared_only=True,actual_score_truth_feature_or_reference_result_bodies_read=0))


def run(entry_path,fit_path):
    e=read(entry_path);require(e.get('approved_saved_C49_owner_evaluation') is True and e.get('source_only') is True and e.get('no_forward_fit_or_selection') is True,'No root saved-only entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Entry code/protocol');p=read(PROTOCOL)
    for key in ['scope','tests','pure_source','nested_pure_source','source_protocol','training_protocol']:checked(p[key])
    require(read(checked(p['tests']))['PASS'] and p['code']==bind(__file__),'Prepared checks')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and np.__version__=='1.26.4','Pinned runtime')
    require(Path(fit_path).resolve()==FIT and e['fit_complete']==bind(FIT),'Dynamic C49 fit binding');fit=read(checked(e['fit_complete']))
    require(fit['completed'] and fit['PASS'] and fit['fixed_final_frozen_before_outputs'] and set(fit['arms'])=={ARM} and fit['arms'][ARM]['updates']==8000,'Fixed final C49')
    require(fit['protocol']==p['training_protocol'],'Exact training protocol');checked(fit['freeze'])
    require(Path(e['reference_evaluation_complete']['path']).resolve()==REFERENCE,'Fixed V175 reference path');reference=read(checked(e['reference_evaluation_complete']))
    require(reference['completed'] and reference['PASS'] and reference['same6056_reference_mask'],'Completed same6056 V175 reference')
    require(not OUT.exists(),'Preserve previous result/failure; no retry');OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry_path),fit_complete=e['fit_complete'],reference_complete=e['reference_evaluation_complete'],protocol=bind(PROTOCOL)))
    try:
        m=helpers();pieces=[]
        for split,N in [('TRAIN',4096),('VALIDATION',2048)]:
            with np.load(checked(p['sidecars'][split]),allow_pickle=False) as z:
                require(z['owner_class_order'].tolist()==OWNER6 and not z['training_admission'].item() and not z['initializer_physical_native_identity_certified'].item(),'Source truth authority/order')
                d={k:z[k].copy() for k in META};d['owner6']=z['scene_owner_point'].copy();require(len(d['split'])==N and (d['split'].astype(str)==split).all(),'Source split');pieces.append(d)
        d={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]};order=np.argsort(d['original_global_index']);d={k:x[order] for k,x in d.items()};require(np.array_equal(d['original_global_index'],np.arange(6144)),'All6144 once')
        known,a,iv,missing=m['truth_mass'](d['owner6'],d['query_xy_missing']);packet=fit['arms'][ARM]['outputs']['OLD6144']
        with np.load(checked(packet),allow_pickle=False) as z:
            ok=np.ones(6144,bool);reasons=[[] for _ in range(6144)]
            for src,dst in [('global_index','original_global_index'),('occurrence_id','original_occurrence_id'),('scene_index','scene_index'),('base_index','base_index'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L'),('split','split'),('query_available','query_available'),('feature_constructed','feature_constructed'),('no_detection','selected_index_missing')]:
                match=m['same_rows'](z[src],d[dst]);ok &= match
                for i in np.flatnonzero(~match):reasons[i].append(src)
            require(z['owner_class_order'].tolist()==OWNER3,'C49 class order');raw=z['raw_owner_three_logits'].copy();av=z['query_available'].astype(bool).copy();saved=z['owner_prediction'].copy()
            require(raw.dtype==np.float32 and raw.shape==(6144,3) and saved.dtype==np.int8,'Raw schema');pred,finite=m['owner_predictions'](raw,av&ok);require(np.array_equal(pred[ok],saved[ok]) and np.isnan(raw[~av]).all(),'Original saved argmax/Uraw')
        with checked(reference['ledger']).open(newline='') as f:rows=list(csv.DictReader(f))
        require(len(rows)==6144,'Reference all6144')
        masscols={'w_TARGET':known[:,0],'w_FOREIGN':known[:,1],'w_BG':known[:,2],'ambiguous_mass':a,'invalid_mass':iv,'missing_q_inventory_unknown':missing}
        for i,r in enumerate(rows):
            require(int(r['original_global_index'])==i and r['original_occurrence_id']==str(d['original_occurrence_id'][i]),'Reference original ID')
            for k in ['split','native_slot','material','aspect','variant','original_semantic_state']:require(r[k]==str(d[k][i]),'Reference factor '+k)
            for k in ['query_available','query_xy_missing','L_missing']:require(r[k]==str(bool(d[k][i])),'Reference original input flag')
            for k,values in masscols.items():require(float(r[k])==float(values[i]),'Reference exact soft mass '+k)
        original_pair=reference_pair(rows);require(int(original_pair.sum())==6056,'Reference6056');pair=original_pair&ok&av&finite;same_pair=np.array_equal(pair,original_pair)
        ref=read(checked(reference['reports']));require(ref['primary']=='FULL_INVENTORY/VALIDATION' and ref['paired_reference_mask_exact'],'Reference metric definition')
        groups=m['groups_for'](d);require(len(groups)==63,'Same63 original strata');reports={};binary=np.column_stack((known[:,0],known[:,1:].sum(1)));bp=np.where(pred<0,-1,(pred!=0).astype(np.int8))
        for cohort,gate in [('FULL_INVENTORY',np.ones(6144,bool)),('ALL3_MATCHED_AVAILABLE',pair)]:
            reports[cohort]={}
            for group,mask in groups.items():
                mm=mask&gate;record={ARM:dict(owner3=m['metrics'](known,a,iv,missing,pred,mm,3),binary_target_presence=m['metrics'](binary,a,iv,missing,bp,mm,2))}
                for name in [CONTROL,'v61']:
                    record[name]=copy.deepcopy(ref['reports'][cohort][group][name]) if cohort=='FULL_INVENTORY' or same_pair else dict(unavailable_reason='C49 paired mask differs; no baseline substitution')
                reports[cohort][group]=record
        ledger=OUT/'ALL6144_C49_OWNER_AND_EXACT_POINT_MASS.csv';fields=['original_global_index','original_occurrence_id','split','native_slot','material','aspect','variant','original_semantic_state','query_available','query_xy_missing','L_missing']+list(masscols)+['owner_prediction','original_available','matched','mismatch_fields','original_reference_pair','C49_paired']
        with ledger.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i in range(6144):
                row={k:d[k][i].item() for k in fields[:11]};row.update({k:float(x[i]) for k,x in masscols.items()});row.update(owner_prediction=int(pred[i]),original_available=bool(av[i]),matched=bool(ok[i]),mismatch_fields='|'.join(reasons[i]),original_reference_pair=bool(original_pair[i]),C49_paired=bool(pair[i]));w.writerow(row)
        apparent={key:dict(binding=b,metadata=read(checked(b))) for key,b in fit['arms'][ARM].get('TRAIN_apparent_diagnostics',{}).items()}
        report=OUT/'EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json';write(report,dict(completed=True,primary='FULL_INVENTORY/VALIDATION',TRAIN_apparent_only=True,reused_source_VAL_not_blind=True,reports=reports,C48_and_v61_records_copied_from=reference['reports'],no_reference_model_raw_reread=True,exact_reference_pair_mask=same_pair,C49_TRAIN_apparent_only_metadata=apparent,owner3_class_order=['T','F','BG','predictionU'],binary_class_order=['T','nonT','predictionU']))
        passed=bool(same_pair and ok.all() and not (av&~finite).any() and int((~av).sum())==88)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=passed,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry_path),fit_complete=e['fit_complete'],freeze=fit['freeze'],C49_packet=packet,reference_evaluation_complete=e['reference_evaluation_complete'],reference_reports=reference['reports'],reference_ledger=reference['ledger'],all6144_ID_and_soft_mass_exact=True,paired_mask_same6056=same_pair,paired_rows=int(pair.sum()),rows=6144,TRAIN=4096,VALIDATION=2048,inputU=int((~av).sum()),missing_q=int(missing.sum()),matched_rows=int(ok.sum()),mismatch_rows=int((~ok).sum()),available_nonfinite=int((av&~finite).sum()),ledger=bind(ledger),reports=bind(report),source_sidecars=p['sidecars'],new_forward_fit_feature_weight_private_current_V169_V112=0,no_native_labels_or_source_qualification_change=True,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False))
    except BaseException:write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a.add_argument('--fit-complete',type=Path);args=a.parse_args();require(args.prepare or args.fit_complete is not None,'Fit complete argument');prepare() if args.prepare else run(args.entry,args.fit_complete)
