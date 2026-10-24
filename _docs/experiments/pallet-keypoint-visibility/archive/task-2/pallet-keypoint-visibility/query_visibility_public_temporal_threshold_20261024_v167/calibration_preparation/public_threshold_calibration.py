"""V167 fixed-head MR PUBLIC_CAL threshold diagnosis; prepare is logit/label-free."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_DESIGN_BEFORE_PUBLIC_THRESHOLD_CALIBRATION.json'
DESIGN_SHA='d957b8ed2d354539168a4a3146b9d9a723172dd9a88fcd88149f4a1129ea8a61'
PROTOCOL=H/'PROTOCOL_BEFORE_PUBLIC_LOGIT_OR_CALIBRATION_LABEL_BODIES.json'
INPUTS=H/'FIXED_HEAD_AND_PUBLIC_INPUT_BINDINGS.json'
OUT=V/'public_calibration_v1'
ARMS=('C45_REPLAY_C43_OWNER253_NATIVE0','C46_PART_NATIVE92_OWNER253_NATIVE0')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(x in str(p).lower() for x in ('current223','_v112','/private_evaluation/'))
    assert sha(p)==b['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def metric(y,pred,mask):
    known=mask&np.isin(y,[0,1]);cm=[[int((known&(y==i)&(pred==j)).sum()) for j in (0,1,-1)] for i in (0,1)]
    def ratio(n,d):return n/d if d else None
    return dict(CM_rows0_1_columns0_1_U=cm,N=int(known.sum()),negative_recall=ratio(cm[0][0],sum(cm[0])),positive_recall=ratio(cm[1][1],sum(cm[1])),negative_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),positive_precision=ratio(cm[1][1],cm[0][1]+cm[1][1]),U_rate=ratio(cm[0][2]+cm[1][2],int(known.sum())),all_mask_rows=int(mask.sum()),unknown_truth=int((mask&~np.isin(y,[0,1])).sum()))
def predict(scores,available,threshold):
    x=np.asarray(scores,np.float64);av=np.asarray(available,bool);t=np.asarray(threshold,np.float64)
    assert np.isfinite(x[av]).all() and np.isfinite(t).all()
    return np.where(av,(x>=t).astype(np.int8),-1)
def fit_threshold(scores,y,fit_mask):
    x=np.asarray(scores,np.float64)[fit_mask];target=np.asarray(y)[fit_mask]
    assert np.isfinite(x).all() and np.isin(target,[0,1]).all()
    nd=int((target==0).sum());ne=int((target==1).sum())
    if not nd or not ne:
        return dict(threshold_float64=0.,fallback='MISSING_CLASS_OR_EMPTY_FIT_USE_ORIGINAL0',N_D=nd,N_E=ne,candidates=0,integer_BA_objective=None)
    unique=np.unique(x);candidates=np.r_[unique,np.nextafter(unique[-1],np.float64(np.inf))]
    assert candidates.dtype==np.float64 and np.isfinite(candidates).all()
    best=None
    for t in candidates:
        ep=x>=t;tn=int(((target==0)&~ep).sum());tp=int(((target==1)&ep).sum())
        objective=tn*ne+tp*nd
        if best is None or objective>=best['integer_BA_objective']:
            best=dict(threshold_float64=float(t),fallback=None,N_D=nd,N_E=ne,candidates=len(candidates),integer_BA_objective=objective,TN=tn,TP=tp)
    return best
def crossfit(scores,y,available,frame_numbers,original_fit_eligible):
    block=np.asarray(frame_numbers,np.int64)//5;assert np.isin(block,np.arange(9)).all()
    pred=np.full(len(y),-9,np.int8);threshold=np.full(len(y),np.nan,np.float64);visits=np.zeros(len(y),np.int8);records=[]
    for b in range(9):
        held=block==b;fit=np.asarray(original_fit_eligible,bool)&(np.abs(block-b)>1)
        result=fit_threshold(scores,y,fit);t=np.float64(result['threshold_float64'])
        pred[held]=predict(np.asarray(scores)[held],np.asarray(available)[held],t);threshold[held]=t;visits[held]+=1
        records.append(dict(block_id=b,held_indices=np.flatnonzero(held).tolist(),fit_indices=np.flatnonzero(fit).tolist(),
            purged_indices=np.flatnonzero(np.abs(block-b)<=1).tolist(),
            fit_frame_numbers=sorted(set(np.asarray(frame_numbers)[np.abs(block-b)>1].tolist())),
            held_frame_numbers=sorted(set(np.asarray(frame_numbers)[held].tolist())),
            fit_nonpurged_total_rows=int((np.abs(block-b)>1).sum()),fit_eligible_rows=int(fit.sum()),
            fit_inputU_rows=int(((np.abs(block-b)>1)&~np.asarray(available,bool)).sum()),
            fit_unknown_truth_rows=int(((np.abs(block-b)>1)&~np.isin(y,[0,1])).sum()),
            held_total_rows=int(held.sum()),held_known_D_rows=int((held&(np.asarray(y)==0)).sum()),held_known_E_rows=int((held&(np.asarray(y)==1)).sum()),
            held_inputvalid_D_rows=int((held&np.asarray(available,bool)&(np.asarray(y)==0)).sum()),held_inputvalid_E_rows=int((held&np.asarray(available,bool)&(np.asarray(y)==1)).sum()),
            held_inputU_rows=int((held&~np.asarray(available,bool)).sum()),held_unknown_truth_rows=int((held&~np.isin(y,[0,1])).sum()),**result))
    assert (visits==1).all() and not (pred==-9).any() and np.isfinite(threshold).all()
    return pred,threshold,records
def invented_checks():
    checks={}
    x=np.array([-2.,-1.,1.,2.],np.float32);y=np.array([0,0,1,1]);m=np.ones(4,bool)
    r=fit_threshold(x,y,m);checks['known_separation_and_highest_equivalent_threshold']=r['threshold_float64']==1. and r['integer_BA_objective']==8
    # Inverse ordering has tied all-E/all-D optima: highest threshold must win.
    r=fit_threshold(np.array([0.,1.]),np.array([1,0]),np.ones(2,bool))
    checks['integer_objective_tie_chooses_highest_allD']=r['threshold_float64']==float(np.nextafter(np.float64(1.),np.inf))
    checks['exact_ge_boundary']=predict(np.array([1.,np.nextafter(1.,-np.inf)]),[True,True],1.).tolist()==[1,0]
    r=fit_threshold(np.array([1.,1.],np.float32),np.array([0,1]),np.ones(2,bool))
    checks['nextafter_float64_no_float32_scalar_downcast']=r['threshold_float64']>1 and np.float32(r['threshold_float64'])==1 and predict([1.],[True],r['threshold_float64']).tolist()==[0]
    unequal=fit_threshold(np.array([0.,1.,2.,3.,4.]),np.array([0,0,0,1,1]),np.ones(5,bool))
    checks['unequal_class_integer_BA_weights']=unequal['threshold_float64']==3. and unequal['integer_BA_objective']==12
    checks['singleclass_fallback0']=fit_threshold([1.,2.],np.array([1,1]),np.ones(2,bool))['fallback'] is not None
    checks['emptyclass_fallback0']=fit_threshold([1.,2.],np.array([1,1]),np.zeros(2,bool))['threshold_float64']==0
    checks['inputU_first_even_high_score']=predict([100.,1.],[False,True],0.).tolist()==[-1,1]
    frames=np.repeat(np.arange(45),8);n=len(frames);x=np.linspace(-3.,3.,n);y=(np.arange(n)%3==0).astype(np.int8)
    av=np.arange(n)%11!=0;known=np.arange(n)%13!=0;eligible=av&known;y[~known]=-1
    pred,threshold,folds=crossfit(x,y,av,frames,eligible)
    checks['nine_temporal_blocks_and_all360_once']=len(folds)==9 and sorted(i for f in folds for i in f['held_indices'])==list(range(360))
    checks['purge_held_and_adjacent_excluded']=all(not set(f['fit_indices'])&set(f['purged_indices']) for f in folds)
    checks['allOOF_originalU_preserved']=np.array_equal(pred==-1,~av)
    checks['fit_only_original_available_known']=all(eligible[f['fit_indices']].all() for f in folds)
    changed=y.copy();mutate=np.abs(frames//5-4)<=1;changed[mutate]=1-np.maximum(changed[mutate],0)
    _,_,other=crossfit(x,changed,av,frames,eligible)
    checks['held_and_adjacent_label_mutation_cannot_change_fold_threshold']=folds[4]['threshold_float64']==other[4]['threshold_float64']
    changed=y.copy();changed[~eligible]=1
    _,_,other=crossfit(x,changed,av,frames,eligible)
    checks['unknowntruth_or_inputU_label_mutation_never_admits_fit']=all(a['threshold_float64']==b['threshold_float64'] and a['fit_indices']==b['fit_indices'] for a,b in zip(folds,other))
    yy=np.zeros(n,np.int8);_,_,fallback=crossfit(x,yy,av,frames,av)
    checks['classabsent_fold_fallback_no_row_drop']=all(f['fallback'] and f['threshold_float64']==0 for f in fallback)
    checks['float64_threshold_vector']=threshold.dtype==np.float64
    assert all(bool(v) for v in checks.values()),checks
    return dict(completed=True,PASS=True,invented_only=True,checks=checks,number_checks=len(checks),actual_public_logit_or_label_bodies_read=0,actual_threshold_searches=0)
def prepare():
    assert sha(DESIGN)==DESIGN_SHA;d=read(DESIGN);assert not d['actual_threshold_search_allowed_by_this_file']
    review=V/'independent_design_review_a/REVIEW.json'
    assert sha(review)=='ae5c6cb240f896602752e4de43405da83110fce42985eb63b4375f6cce413393' and read(review)['PASS']
    c=read(check(d['parent_control']));rank=read(check(d['parent_ranking_diagnosis']));tp=read(check(d['parent_trainer_protocol']))
    assert c['completed'] and c['PASS'] and list(c['arms'])==list(ARMS)
    groups=[]
    for arm in ARMS:
        fit=read(check(c['arms'][arm]['fit']));freeze=read(check(fit['model_freeze']))
        output=next(r['output'] for r in rank['metrics'] if r['arm']==arm)
        groups.append(dict(arm=arm,output=output,checkpoint=freeze['checkpoint'],model_freeze=fit['model_freeze'],source_results=fit['source_results']))
    inputs=dict(completed=True,models=groups,source_labels=d['source_labels'],parent_control=d['parent_control'],parent_trainer_protocol=d['parent_trainer_protocol'],
        original_split='SOURCE_VALIDATION',new_reporting_role='PUBLIC_CAL',sequence='val/000001',frames45=list(range(45)),rows=360,known154=154,fit145=145,D93=93,E52=52,known_inputU9=9,
        output_or_checkpoint_or_label_bodies_opened=False)
    write(INPUTS,inputs);generated=H/'GENERATED_INVENTED_THRESHOLD_CHECKS.json';write(generated,invented_checks())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),design=bind(DESIGN),independent_design_review=bind(review),inputs=bind(INPUTS),generated=bind(generated),output=str(OUT),
        runtime=dict(executable=sys.executable,numpy=np.__version__),candidate='unique float64-cast fit logits + float64 nextafter(max,+inf); >= in float64',
        objective='integer TN*N_E+TP*N_D; ties highest threshold',crossfit='9 frame//5 blocks; held +/-1 purge; original frozen fit-eligibility mask; missing-class fallback0; all360 once',
        reporting=['Original fixed0 known154/full360 and inputvalid145 metrics immutable.','Full145 fit/apply is PUBLIC_CAL apparent performance, not held-test accuracy.','OOF is threshold stability within one public sequence; repeated V153–V166 public metric/crop/ranking observation and subsequent design adaptation prevent fresh model-level validation. Nine folds are not independent scenes and154 queries are not154 independent samples.','This fits a decision working point, not calibrated posterior p(E).','No other source panels implemented here; later readout needs separate review.'],
        actual_public_output_or_calibration_label_bodies_read_during_preparation=0,actual_threshold_searches=0,private_current_V112_RGB_models_encoders_newheadfit_PnP_qL_calls=0,automatic_retry=False)
    write(PROTOCOL,p);template=H/'ROOT_ACTUAL_CALIBRATION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_public_threshold_calibration=False,source_only=True,code=p['code'],protocol=bind(PROTOCOL),design=p['design'],inputs=p['inputs'],generated=p['generated'],current_allowed=False,V112_allowed=False,RGB_model_encoder_headfit_PnP_qL_allowed=False,automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),inputs=p['inputs'],generated=p['generated'],entry_template=bind(template),actual_threshold_searches=0,actual_public_logits_labels_weights_bodies_read=0))
    print(json.dumps(dict(completed=True,prepared_only=True,preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def run(entry):
    e=read(entry);assert e['approved_actual_public_threshold_calibration'] and e['source_only']
    for k in ('current_allowed','V112_allowed','RGB_model_encoder_headfit_PnP_qL_allowed','automatic_retry'):assert e[k] is False
    assert e['protocol']==bind(PROTOCOL);p=read(check(e['protocol']))
    for k in ('code','design','inputs','generated'):assert e[k]==p[k];check(p[k])
    assert p['code']==bind(__file__) and sha(DESIGN)==DESIGN_SHA and read(check(p['generated']))['PASS'] and read(check(p['independent_design_review']))['PASS']
    assert sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'] and not OUT.exists()
    inp=read(check(p['inputs']));labels_path=check(inp['source_labels']);labels=[json.loads(x) for x in labels_path.read_text().splitlines()]
    selected=[r for r in labels if r['split']=='SOURCE_VALIDATION'];assert len(labels)==1840 and len(selected)==360
    ids=np.array([r['global_index'] for r in selected]);frames=np.array([int(r['source_id'].rsplit(':',1)[1]) for r in selected])
    assert all(r['source_id'].rsplit(':',1)[0]=='val/000001' for r in selected)
    assert sorted(set(frames))==list(range(45)) and all(int((frames==f).sum())==8 for f in range(45))
    y=np.array([r['role_target'] for r in selected],np.int8);known=np.array([r['role_state'] in ('D','E') for r in selected])
    av=np.array([r['original_input_valid'] for r in selected],bool);eligible=av&known
    assert np.array_equal(known,np.isin(y,[0,1])) and known.sum()==154 and eligible.sum()==145
    assert ((y==0)&eligible).sum()==93 and ((y==1)&eligible).sum()==52 and (known&~av).sum()==9
    OUT.mkdir();write(OUT/'STARTED.json',dict(pid=os.getpid(),entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL)))
    measured=[];frozen=[];checked=[inp['source_labels']];started=time.monotonic()
    try:
        for model in inp['models']:
            output_path=check(model['output']);assert output_path.name=='FINAL_MR1840_OUTPUTS.npz'
            # Checkpoint hash only: no model/checkpoint deserialization or forward.
            check(model['checkpoint']);checked.extend([model['output'],model['checkpoint'],model['model_freeze'],model['source_results']])
            with np.load(output_path,allow_pickle=False) as z:
                assert 'features' not in z.files
                raw=z['raw_three_logits'].copy();saved=z['primary_visibility_prediction'].copy()
                assert z['global_index'][ids].tolist()==ids.tolist() and z['query_id'][ids].tolist()==[r['query_id'] for r in selected]
                assert np.array_equal(z['query_available'][ids],av)
            assert raw.shape==(1840,3) and raw.dtype==np.float32
            scores=raw[ids,0].astype(np.float64);fixed=predict(scores,av,0.)
            assert np.array_equal(fixed,saved[ids])
            original_report=read(check(model['source_results']))['MR']['SOURCE_VALIDATION']
            assert metric(y,fixed,np.ones(360,bool))==original_report['visibility']
            assert metric(y,fixed,av)==original_report['visibility_inputvalid']
            oof,thresholds,folds=crossfit(scores,y,av,frames,eligible)
            full=fit_threshold(scores,y,eligible);apparent=predict(scores,av,np.float64(full['threshold_float64']))
            name=model['arm'];packet=OUT/(name+'_ALL360_THRESHOLD_PREDICTIONS.npz')
            np.savez_compressed(packet,global_index=ids,query_id=np.array([r['query_id'] for r in selected]),frame_number=frames,block_id=frames//5,
                original_input_available=av,original_weak_role_target=y,original_weak_role_state=np.array([r['role_state'] for r in selected]),
                original_raw_E_logit=scores,fixed0_prediction=fixed,purged_temporal_OOF_prediction=oof,full_PUBLIC_CAL_apparent_prediction=apparent,
                OOF_threshold_float64=thresholds,full_threshold_float64=np.float64(full['threshold_float64']),original_fit_eligible=eligible)
            foldfile=OUT/(name+'_PURGED9_FOLDS.json');write(foldfile,dict(completed=True,folds=folds,authority='Within one public sequence threshold stability only; historically exposed public data, not fresh model validation.'))
            measured.append(dict(arm=name,original_fixed0=dict(full_known154=metric(y,fixed,np.ones(360,bool)),inputvalid145=metric(y,fixed,av)),
                purged_temporal_OOF=dict(full_known154=metric(y,oof,np.ones(360,bool)),inputvalid145=metric(y,oof,av),authority='Same-sequence threshold-fitting stability; not unseen-scene validation.'),
                full_PUBLIC_CAL_apparent=dict(full_known154=metric(y,apparent,np.ones(360,bool)),inputvalid145=metric(y,apparent,av),authority='Fit/apply on same145 eligible rows; apparent, not held-test accuracy.'),
                predictions=bind(packet),folds=bind(foldfile),known154=154,inputvalid_known145=145,unknown_truth206=206,all360_retained=True,
                inputU_all360=int((~av).sum()),inputU_known154=int((known&~av).sum()),inputU_preserved=True,
                fallback_fold_count=sum(f['fallback'] is not None for f in folds),fallback_fold_ids=[f['block_id'] for f in folds if f['fallback'] is not None],
                samples_not_independent154=True,folds_not_independent_scenes9=True))
            frozen.append(dict(arm=name,**full,output_binding=model['output'],model_checkpoint_binding=model['checkpoint'],threshold_is_logit_working_point_not_probability_calibration=True))
        thresholds=OUT/'ALL2_THRESHOLDS_FROZEN_BEFORE_ANY_PRIVATE_READ.json';write(thresholds,dict(completed=True,thresholds=frozen,no_model_winner_selected=True,current_reads=0,original_heads_and_fixed0_results_unchanged=True))
        for b in checked:check(b)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,entry=bind(entry),code=p['code'],protocol=bind(PROTOCOL),inputs=p['inputs'],
            thresholds_frozen=bind(thresholds),results=measured,original_fixed0_results_preserved=True,all360_known154_valid145_denominators_preserved=True,
            single_public_sequence=True,historical_public_exposure_not_fresh_model_validation=True,probability_calibration_claimed=False,
            model_or_encoder_or_newheadfit_forward_calls=0,current_private_V112_RGB_PnP_qL_reads_or_changes=0,other_source_panels_executed=False,
            selected_model=None,adopted=False,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Use --prepare first; actual public threshold calculation requires separate approved --entry')
