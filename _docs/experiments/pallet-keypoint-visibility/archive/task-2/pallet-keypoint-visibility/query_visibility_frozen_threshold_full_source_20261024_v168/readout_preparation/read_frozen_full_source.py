"""V168 saved-output-only retrospective readout; --prepare never opens scores/GT."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V166=P/'query_visibility_native_part_supervision_control_20261024_v166'
V167=P/'query_visibility_public_temporal_threshold_20261024_v167'
RULES=V/'ROOT_RULES_BEFORE_FULL_SOURCE_SCORE_OR_LABEL_BODIES.json'
TRAIN_PROTOCOL=V166/'controlled_fit_preparation/PROTOCOL_BEFORE_SOURCE_FIT_FEATURE_BODIES.json'
TRAIN_SHA='aad1f9533a90de3424ea3a1628ddecb1d81f08f29e1419d3562429919dfd3cd6'
PRIOR_AUDIT=V166/'independent_saved_fit_review_preparation/SAVED_C45_C46_EXECUTION_REVIEW.json'
PRIOR_AUDIT_SHA='029a199336746f752d76a3f98d0472c253d9b91d219befb5ddfbe12385631e87'
SCALARS=V167/'public_calibration_v1/ALL2_THRESHOLDS_FROZEN_BEFORE_ANY_PRIVATE_READ.json'
SCALAR_SHA='f59143616bc9d964b21733c21ee29aafd11310f092090b9ca230b640a51ab833'
CAL_COMPLETE=V167/'public_calibration_v1/COMPLETE.json'
CAL_SHA='ba4d0a96a83deb13f48deb2f0f68956a2aa1384f8c0987028b657331f134f42e'
PROTOCOL=H/'PROTOCOL_BEFORE_FULL_SOURCE_OUTPUT_OR_LABEL_BODIES.json'
INPUTS=H/'FIXED_OUTPUT_HEAD_SCALAR_AND_LABEL_BINDINGS.json'
OUT=V/'retrospective_full_source_v1'
ARMS=('C45_REPLAY_C43_OWNER253_NATIVE0','C46_PART_NATIVE92_OWNER253_NATIVE0')
FAMILIES=('OLD6144','NEW20480','PALLOX512','MR1840','PART1144')
COUNTS=(6144,20480,512,1840,1144)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v112','/private_evaluation/'))
    assert sha(p)==b['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def metric(y,pred,mask):
    known=mask&np.isin(y,[0,1]);cm=[[int((known&(y==i)&(pred==j)).sum()) for j in (0,1,-1)] for i in (0,1)]
    def ratio(n,d):return n/d if d else None
    return dict(CM_rows0_1_columns0_1_U=cm,N=int(known.sum()),negative_recall=ratio(cm[0][0],sum(cm[0])),positive_recall=ratio(cm[1][1],sum(cm[1])),negative_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),positive_precision=ratio(cm[1][1],cm[0][1]+cm[1][1]),U_rate=ratio(cm[0][2]+cm[1][2],int(known.sum())),all_mask_rows=int(mask.sum()),unknown_truth=int((mask&~np.isin(y,[0,1])).sum()))
def shifted(score,threshold):
    assert score.dtype==np.float32
    t=np.asarray(threshold,np.float64);assert t.shape==() and np.isfinite(t)
    z=score.astype(np.float64)-t
    assert np.array_equal(z>=0,score.astype(np.float64)>=t)
    return z
def calls(score,av,t):return np.where(av,(score.astype(np.float64)>=np.float64(t)).astype(np.int8),-1)
def invented():
    c={};x=np.array([-np.finfo(np.float32).max,-11.,-1.,0.,1.,11.,np.finfo(np.float32).max,np.nan],np.float32)
    for name,t in [('fixed0',0.),('C45',-10.97239875793457),('C46',-9.855539321899414),('nextafter_allD',np.nextafter(np.float64(np.finfo(np.float32).max),np.inf))]:
        z=shifted(x,t);c['float64_shift_ge_equivalent_'+name]=np.array_equal(z>=0,x.astype(np.float64)>=np.float64(t)) and z.dtype==np.float64
    t=np.nextafter(np.float64(1.),np.inf);z=shifted(np.array([1.],np.float32),t)
    c['nextafter_scalar_never_FP32_downcast']=bool(z[0]<0 and np.float32(t)==1)
    c['exact_boundary_ge']=calls(np.array([1.,2.],np.float32),[True,True],1.).tolist()==[1,1]
    c['inputU_first_no_new_abstention']=calls(np.array([100.,-100.,np.nan],np.float32),[False,True,False],0).tolist()==[-1,0,-1]
    raw=np.array([[1.,-2.,3.],[-4.,5.,-6.]],np.float32);copy=raw.copy();z=raw.astype(np.float64);z[:,0]=shifted(raw[:,0],-2.)
    c['raw_inputs_and_entity_quality_unchanged']=exact(raw,copy) and np.array_equal(z[:,1:],raw[:,1:].astype(np.float64))
    c['zero_threshold_original_decisions_exact']=np.array_equal(calls(raw[:,0],np.ones(2,bool),0),np.where(raw[:,0]>=0,1,0).astype(np.int8))
    y=np.array([0,0,1,-1],np.int8);pred=np.array([-1,0,1,1],np.int8);r=metric(y,pred,np.ones(4,bool))
    c['CM_unknown_and_factualU_denominators']=r['CM_rows0_1_columns0_1_U']==[[1,0,1],[0,1,0]] and r['N']==3 and r['unknown_truth']==1
    c['nonzero_threshold_never_aux_primary_gate']=calls(np.array([1.,-1.],np.float32),[True,True],-2.).tolist()==[1,1]
    assert all(bool(v) for v in c.values())
    return dict(completed=True,PASS=True,invented_only=True,checks={k:bool(v) for k,v in c.items()},number_checks=len(c),actual_output_GT_label_feature_weight_bodies_read=0,model_or_threshold_fit_calls=0)
def prepare():
    rules=read(RULES);assert rules['fixed_full_PUBLIC_CAL_E_logit_thresholds']==[-10.97239875793457,-9.855539321899414]
    assert sha(TRAIN_PROTOCOL)==TRAIN_SHA and sha(PRIOR_AUDIT)==PRIOR_AUDIT_SHA and sha(SCALARS)==SCALAR_SHA and sha(CAL_COMPLETE)==CAL_SHA
    p=read(TRAIN_PROTOCOL);audit=read(PRIOR_AUDIT);assert audit['PASS'];scalars=read(SCALARS)['thresholds']
    source_bindings={b['path']:b for b in audit['bindings']};models=[]
    for arm,scalar in zip(ARMS,scalars):
        assert scalar['arm']==arm
        folder=V166/'runs/C45_C46_PART_NATIVE_FIXED8000'/arm
        outputs={name:source_bindings[str(folder/f'FINAL_{name}_OUTPUTS.npz')] for name in FAMILIES}
        models.append(dict(arm=arm,threshold_float64=scalar['threshold_float64'],head_checkpoint_binding_inherited=scalar['model_checkpoint_binding'],outputs=outputs,source_results=source_bindings[str(folder/'SOURCE_RESULTS.json')]))
    inp=dict(completed=True,models=models,trainer_protocol=bind(TRAIN_PROTOCOL),parent_report_protocol=p['parent_protocol'],source_labels=p['source_labels'],quality_labels=p['quality_labels'],original_native35=p['original_native35'],
        PART_sidecar=p['CF_part_sidecar'],PART_metadata=p['CF_input_metadata'],PART_statuses=p['CF_statuses'],expected_source_metadata=p['expected_source_metadata'],
        helper=p['helper'],pinned_V155_trainer=p['pinned_V155_trainer'],original_bodies_opened=False)
    write(INPUTS,inp);g=H/'GENERATED_INVENTED_READOUT_CHECKS.json';write(g,invented())
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),rules=bind(RULES),inputs=bind(INPUTS),generated=bind(g),trainer_protocol=bind(TRAIN_PROTOCOL),prior_saved_fit_audit=bind(PRIOR_AUDIT),frozen_scalars=bind(SCALARS),calibration_complete=bind(CAL_COMPLETE),
        output=str(OUT),runtime=dict(executable=sys.executable,numpy=np.__version__),original0_parity_first=True,readout_only_no_threshold_or_model_selection=True,
        shifts='rawFP32 converted to float64, subtract unchanged float64 scalar for existing >=0 helpers; direct>= equivalence checked on every loaded row.',
        scope='Both fixed heads, original28976 plusPART1144; all535+178 factualU; all198status/55missing null; no fake output rows.',
        authority='Retrospective full-source report under two PUBLIC_CAL-fitted working points. MR originalSOURCE_VALIDATION renamed PUBLIC_CAL_APPARENT only in this report; all original files/splits preserved. Source gates unchanged but not independent qualification.',
        forbidden='No private/current/V112/RGB/features/encoder/head forward/newfit/PnP/qL; no model/threshold/winner/gate selection. Checkpoint bindings inherited, weights not opened.',
        actual_saved_output_GT_or_label_bodies_read_during_preparation=0,automatic_retry=False)
    write(PROTOCOL,protocol);template=H/'ROOT_ACTUAL_FULL_SOURCE_READOUT_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_saved_full_source_readout=False,source_only=True,code=protocol['code'],protocol=bind(PROTOCOL),rules=protocol['rules'],inputs=protocol['inputs'],generated=protocol['generated'],forbidden_data_or_model_calls_allowed=False,automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=bind(PROTOCOL),inputs=protocol['inputs'],generated=protocol['generated'],entry_template=bind(template),actual_output_GT_label_feature_or_weight_bodies_read=0,actual_model_or_threshold_fit_calls=0))
    print(json.dumps(dict(completed=True,prepared_only=True,preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def definitions(path,names):
    ns={'np':np};tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::PURE_REPORT_ONLY','exec'),ns);return ns
def load_report_inputs(inp):
    path=check(inp['helper']);spec=importlib.util.spec_from_file_location('v168_frozen_public_report_helpers',path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    parent=read(check(inp['parent_report_protocol']))
    metas=[h.q.read_native_GT(check(parent['native'][0]['GT']))['CF_PRIMARY_DECLARED'],h.h.parse_new_GT(lines(check(parent['native'][1]['GT'])))['CF_PRIMARY_DECLARED']]
    pub=h.public_pools(lines(check(parent['public_labels'])))
    rm=definitions(check(inp['pinned_V155_trainer']),{'labels_and_masks','binary_metric','mr_report'})
    mrrows=lines(check(inp['source_labels']));mr=rm['labels_and_masks'](mrrows)
    quality=[lines(check(b)) for b in inp['quality_labels']];side=lines(check(inp['PART_sidecar']));statuses=lines(check(inp['PART_statuses']))
    assert len(statuses)==198 and sum(not r['constructed_payload'] for r in statuses)==55
    assert all(r['feature_index'] is r['feature_indices'] is r['payload'] is None for r in statuses if not r['constructed_payload'])
    weak=np.array([r['part_native_weak_target'] for r in side],np.int8);assert len(weak)==1144 and np.bincount(weak[weak>=0]).tolist()==[46,46]
    return h,metas,pub,rm,mr,quality,weak,statuses,parent
def report_all(pack,h,metas,pub,rm,mr,quality,weak,limits,t):
    raw=[a['raw_three_logits'] for a in pack];shift=[shifted(a[:,0],t) for a in raw]
    nr=h.native_report(shift[:2],pack[:2],metas,limits);pr=h.public_report(shift[2],pub)
    mrraw=raw[3].astype(np.float64);mrraw[:,0]=shift[3];mrr=rm['mr_report'](mrraw,pack[3],mr)
    qr={}
    for name,a,m,z,rr in zip(FAMILIES[:2],pack,metas,raw,quality):
        y=np.array([-1 if r['point_quality_label'] is None else r['point_quality_label'] for r in rr],np.int8)
        pred=np.where(a['query_available'],(z[:,2]>=0).astype(np.int8),-1);qr[name]={}
        for fold in ('TRAIN','VALIDATION'):
            mask=m['split']==fold;qr[name][fold]=metric(y,pred,mask)
            for factor in ('material','aspect','foreign_family','geometry_family'):
                for value in np.unique(m[factor]):qr[name][fold+'::'+factor+'::'+str(value)]=metric(y,pred,mask&(m[factor]==value))
            slots=np.array([r['native_slot'] for r in m['rows']])
            for k in range(8):qr[name][fold+f'::native{k}']=metric(y,pred,mask&(slots==k))
    cf=metric(weak,calls(raw[4][:,0],pack[4]['query_available'],t),weak>=0)
    held=mrr['SOURCE_VALIDATION'];vals=[held['visibility_inputvalid'][k] for k in ('negative_recall','positive_recall')]+[held['entity_inputvalid'][k] for k in ('negative_recall','positive_recall')]
    sanity=all(v is not None and v>=limit for v,limit in zip(vals,[.90,.80,.80,.80]))
    return dict(native=nr,Pallox=pr,MR=mrr,synthetic_point_quality=qr,CF_weakTRAIN92_response=cf,MR_existing_gate_values_pass=sanity,
        existing_source_gates_all_pass=bool(nr['native35_all_pass'] and pr['sanity']['PASS'] and sanity))
def flat_panels(pack,metas,pub,mr,weak,t):
    panels=[]
    targets=[metas[0]['target'],metas[1]['target'],pub['target'],mr['role'],weak]
    splits=[metas[0]['split'],metas[1]['split'],pub['split'],np.where(mr['training'],'SOURCE_DEVELOPMENT','PUBLIC_CAL_APPARENT'),np.array(['SOURCE_TRAIN_WEAK_RESPONSE']*1144)]
    for i,(name,a,y,split) in enumerate(zip(FAMILIES,pack,targets,splits)):
        pred=calls(a['raw_three_logits'][:,0],a['query_available'],t);factors={'native':a['native_slot']}
        if i<2:
            factors.update({k:metas[i][k] for k in ('material','aspect','foreign_family','geometry_family')})
        if i==3:factors['occluder_or_uncertainty']=mr['occluder']
        for fold in ['ALL']+sorted(set(split.tolist())):
            mask=np.ones(len(y),bool) if fold=='ALL' else split==fold
            panels.append(dict(family=name,original_or_reporting_split=fold,group='ALL',metrics=metric(y,pred,mask)))
            for factor,values in factors.items():
                for value in (range(8) if factor=='native' else np.unique(values)):
                    panels.append(dict(family=name,original_or_reporting_split=fold,group=factor+'::'+str(value),metrics=metric(y,pred,mask&(values==value))))
    return panels
def run(entry):
    e=read(entry);assert e['approved_actual_saved_full_source_readout'] and e['source_only'] and not e['forbidden_data_or_model_calls_allowed'] and not e['automatic_retry']
    assert e['protocol']==bind(PROTOCOL);p=read(check(e['protocol']))
    for k in ('code','rules','inputs','generated'):assert e[k]==p[k];check(p[k])
    assert p['code']==bind(__file__) and read(check(p['generated']))['PASS']
    assert sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'] and not OUT.exists()
    inp=read(check(p['inputs']));assert bind(TRAIN_PROTOCOL)==p['trainer_protocol'] and sha(SCALARS)==SCALAR_SHA and sha(PRIOR_AUDIT)==PRIOR_AUDIT_SHA
    cal=read(check(p['calibration_complete']));assert cal['PASS'] and sha(CAL_COMPLETE)==CAL_SHA
    data=load_report_inputs(inp);h,metas,pub,rm,mr,quality,weak,statuses,parent=data
    limits=read(check(inp['original_native35']))['native35_limits'];original=[];packets=[];checked=[]
    for model in inp['models']:
        pp=[]
        for name,n in zip(FAMILIES,COUNTS):
            b=model['outputs'][name];path=check(b);assert path.name=='FINAL_'+name+'_OUTPUTS.npz';checked.append(b)
            with np.load(path,allow_pickle=False) as z:
                assert 'features' not in z.files;a={k:z[k].copy() for k in z.files}
            assert a['raw_three_logits'].shape==(n,3) and a['raw_three_logits'].dtype==np.float32
            assert exact(calls(a['raw_three_logits'][:,0],a['query_available'],0.),a['primary_visibility_prediction']);pp.append(a)
        assert sum((~a['query_available']).sum() for a in pp[:4])==535 and (~pp[4]['query_available']).sum()==178
        zero=report_all(pp,h,metas,pub,rm,mr,quality,weak,limits,0.)
        old=read(check(model['source_results']));checked.append(model['source_results'])
        for k in ('native','Pallox','MR','synthetic_point_quality'):assert zero[k]==old[k],(model['arm'],k)
        assert all(old['CF_part_native_diagnostic'][k]==v for k,v in zero['CF_weakTRAIN92_response'].items())
        assert zero['MR_existing_gate_values_pass']==old['MR_weak_sanity_PASS'] and zero['existing_source_gates_all_pass']==old['prospective_source_checks_all_pass']
        packets.append(pp);original.append(zero)
    # All original0 parity checks above finish before any thresholded panel is built.
    OUT.mkdir();write(OUT/'ORIGINAL0_PARITY_BEFORE_THRESHOLDED_REPORTS.json',dict(completed=True,PASS=True,arms=list(ARMS),all_native_Pallox_MR_quality_CF_reports_exact=True))
    write(OUT/'STARTED.json',dict(pid=os.getpid(),entry=bind(entry),protocol=bind(PROTOCOL)));started=time.monotonic();results=[]
    try:
        for model,pp,zero in zip(inp['models'],packets,original):
            t=np.float64(model['threshold_float64']);r=report_all(pp,h,metas,pub,rm,mr,quality,weak,limits,t)
            assert r['synthetic_point_quality']==zero['synthetic_point_quality']
            prior_cal=next(x for x in cal['results'] if x['arm']==model['arm'])['full_PUBLIC_CAL_apparent']
            assert r['MR']['SOURCE_VALIDATION']['visibility']==prior_cal['full_known154']
            assert r['MR']['SOURCE_VALIDATION']['visibility_inputvalid']==prior_cal['inputvalid145']
            # Report-only alias: original split metadata and original V166 reports remain unchanged.
            r['MR']={key.replace('SOURCE_VALIDATION','PUBLIC_CAL_APPARENT'):value for key,value in r['MR'].items()}
            r['MR_role_note']='Original SOURCE_VALIDATION sequence was fitted as PUBLIC_CAL inV167; all thresholded MR figures here are apparent/retrospective, not independent qualification.'
            r['CF_weakTRAIN92_not_held_generalization']=True;r['independent_qualification_claimed']=False
            folder=OUT/model['arm'];folder.mkdir();outputs={}
            for name,a in zip(FAMILIES,pp):
                pred=calls(a['raw_three_logits'][:,0],a['query_available'],t)
                shifted_pred=np.where(a['query_available'],(shifted(a['raw_three_logits'][:,0],t)>=0).astype(np.int8),-1);assert exact(pred,shifted_pred)
                path=folder/(name+'_FROZEN_THRESHOLD_READOUT.npz')
                np.savez_compressed(path,**a,retrospective_primary_visibility_prediction=pred,retrospective_secondary_entity_gated_visibility=np.where(a['query_available']&(a['raw_three_logits'][:,1]>=0),pred,-1),frozen_PUBLIC_CAL_E_logit_threshold_float64=t)
                with np.load(path,allow_pickle=False) as z:
                    assert all(exact(z[k],value) for k,value in a.items()) and exact(z['retrospective_primary_visibility_prediction'],pred)
                    assert z['frozen_PUBLIC_CAL_E_logit_threshold_float64'].dtype==np.float64
                outputs[name]=bind(path)
            tables=folder/'FULL_SOURCE_FACTOR_NATIVE8_TABLES.json';write(tables,dict(completed=True,panels=flat_panels(pp,metas,pub,mr,weak,t)))
            report=folder/'RETROSPECTIVE_SOURCE_RESULTS.json';write(report,dict(completed=True,arm=model['arm'],threshold_float64=float(t),threshold_source=p['frozen_scalars'],head_checkpoint_binding_inherited=model['head_checkpoint_binding_inherited'],original0_parity=True,**r,
                outputs=outputs,full_panels=bind(tables),selected_model=None,adopted=False,fully_solved=False))
            results.append(dict(arm=model['arm'],report=bind(report),outputs=outputs,panels=bind(tables),native35_passed=r['native']['native35_passed'],retrospective_existing_source_gates_all_pass=r['existing_source_gates_all_pass'],independent_qualification_claimed=False))
        statusfile=OUT/'ALL198_VARIANT_STATUSES_UNCHANGED.jsonl'
        with statusfile.open('x') as f:
            for row in statuses:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        for b in checked:check(b)
        for b in [inp['source_labels'],inp['PART_sidecar'],inp['PART_metadata'],inp['PART_statuses'],*inp['quality_labels'],parent['public_labels'],*[a['GT'] for a in parent['native']]]:check(b)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),rules=p['rules'],inputs=p['inputs'],
            fixed_scalar_source=p['frozen_scalars'],results=results,all198_statuses=bind(statusfile),original0_parity_first=True,
            original28976_plusPART1144_each=30120,original_U535=535,PART_U178=178,missing55_null_preserved=True,
            raw_logits_entity_quality_owner_and_all_metadata_unchanged=True,MR_original_validation_now_PUBLIC_CAL_apparent=True,independent_qualification_claimed=False,
            threshold_head_or_model_selection=False,new_threshold_fit_or_model_forward_or_fit_calls=0,private_current_V112_RGB_feature_or_weight_bodies_read=0,PnP_qL_changes=0,
            selected_model=None,adopted=False,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Prepare only first; root-approved --entry required for actual readout')
