"""Independent Python brute-force verification of frozen V167 public results.
No producer function imported; no model/checkpoint deserialization or forward.
"""
from pathlib import Path
import hashlib,json,math,datetime,sys
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
BINDINGS=[];CHECKS=[]
def bind(p):
    p=Path(p).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v112','/private_evaluation/'))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    b=dict(path=str(p),sha256=h.hexdigest());BINDINGS.append(b);return b
def verify(b):assert bind(b['path'])==b;return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def ck(name,value=True):assert value,name;CHECKS.append(dict(check=name,PASS=True))
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def brute(scores,y,indices):
    # Scalar Python counting, independent of the producer's NumPy fitting routine.
    xx=[float(scores[i]) for i in indices];yy=[int(y[i]) for i in indices]
    nd=yy.count(0);ne=yy.count(1)
    assert all(math.isfinite(x) for x in xx) and all(t in (0,1) for t in yy)
    if not nd or not ne:return dict(threshold_float64=0.,fallback='MISSING_CLASS_OR_EMPTY_FIT_USE_ORIGINAL0',N_D=nd,N_E=ne,candidates=0,integer_BA_objective=None)
    unique=sorted(set(xx));candidates=unique+[math.nextafter(unique[-1],math.inf)];records=[]
    for t in candidates:
        tn=sum(label==0 and score<t for score,label in zip(xx,yy))
        tp=sum(label==1 and score>=t for score,label in zip(xx,yy))
        records.append((tn*ne+tp*nd,t,tn,tp))
    objective,t,tn,tp=max(records,key=lambda r:(r[0],r[1]))
    return dict(threshold_float64=t,fallback=None,N_D=nd,N_E=ne,candidates=len(candidates),integer_BA_objective=objective,TN=tn,TP=tp)
def decode(scores,av,t):return np.array([-1 if not bool(a) else int(float(x)>=float(t)) for x,a in zip(scores,av)],np.int8)
def metric(y,pred,mask):
    indices=[i for i in range(len(y)) if mask[i] and int(y[i]) in (0,1)]
    cm=[[sum(int(y[i])==truth and int(pred[i])==call for i in indices) for call in (0,1,-1)] for truth in (0,1)]
    def ratio(n,d):return n/d if d else None
    return dict(CM_rows0_1_columns0_1_U=cm,N=len(indices),negative_recall=ratio(cm[0][0],sum(cm[0])),positive_recall=ratio(cm[1][1],sum(cm[1])),
        negative_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),positive_precision=ratio(cm[1][1],cm[0][1]+cm[1][1]),
        U_rate=ratio(cm[0][2]+cm[1][2],len(indices)),all_mask_rows=int(sum(mask)),unknown_truth=int(sum(mask))-len(indices))
def main():
    complete_path=V/'public_calibration_v1/COMPLETE.json'
    assert bind(complete_path)['sha256']=='ba4d0a96a83deb13f48deb2f0f68956a2aa1384f8c0987028b657331f134f42e'
    c=read(complete_path);assert c['completed'] and c['PASS']
    entry=read(verify(c['entry']));assert entry['approved_actual_public_threshold_calibration'] and entry['source_only']
    for k in ('current_allowed','V112_allowed','RGB_model_encoder_headfit_PnP_qL_allowed','automatic_retry'):assert entry[k] is False
    for k in ('code','protocol','inputs'):assert c[k]==entry[k];verify(c[k])
    p=read(c['protocol']['path']);verify(p['design']);verify(p['independent_design_review'])
    inp=read(c['inputs']['path']);frozen=read(verify(c['thresholds_frozen']))
    assert frozen['completed'] and frozen['no_model_winner_selected'] and frozen['current_reads']==0
    lab=list(json.loads(x) for x in verify(inp['source_labels']).read_text().splitlines());selected=[r for r in lab if r['split']=='SOURCE_VALIDATION']
    assert len(lab)==1840 and len(selected)==360
    ids=np.array([r['global_index'] for r in selected]);frames=np.array([int(r['source_id'].rsplit(':',1)[1]) for r in selected])
    assert all(r['source_id'].rsplit(':',1)[0]=='val/000001' for r in selected)
    assert sorted(set(frames))==list(range(45)) and all((frames==f).sum()==8 for f in range(45))
    y=np.array([r['role_target'] for r in selected],np.int8);known=np.isin(y,[0,1]);av=np.array([r['original_input_valid'] for r in selected],bool);eligible=known&av
    ck('full360_single_sequence45frames_native8',len(selected)==360)
    ck('known154_D102_E52_unknown206_and_U108_U9allD',known.sum()==154 and (y==0).sum()==102 and (y==1).sum()==52 and (~known).sum()==206 and (~av).sum()==108 and (known&~av).sum()==9 and not ((y==1)&~av).any())
    ck('only_inputvalid145_D93_E52_eligible',eligible.sum()==145 and ((y==0)&eligible).sum()==93 and ((y==1)&eligible).sum()==52)
    measured=[]
    for model,result,full_saved in zip(inp['models'],c['results'],frozen['thresholds']):
        assert model['arm']==result['arm']==full_saved['arm']
        assert model['output']==full_saved['output_binding'] and model['checkpoint']==full_saved['model_checkpoint_binding']
        verify(model['checkpoint'])  # Byte SHA only, never torch.load or model construction.
        verify(model['model_freeze'])
        with np.load(verify(model['output']),allow_pickle=False) as z:
            assert 'features' not in z.files
            raw=z['raw_three_logits'].copy();oldpred=z['primary_visibility_prediction'][ids].copy()
            assert z['global_index'][ids].tolist()==ids.tolist() and z['query_id'][ids].tolist()==[r['query_id'] for r in selected]
            assert np.array_equal(z['query_available'][ids],av) and z['native_slot'][ids].tolist()==[r['native_slot'] for r in selected]
        assert raw.dtype==np.float32 and raw.shape==(1840,3)
        scores=raw[ids,0].astype(np.float64);assert np.isfinite(scores[av]).all()
        with np.load(verify(result['predictions']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        assert equal(a['global_index'],ids) and np.array_equal(a['frame_number'],frames) and np.array_equal(a['block_id'],frames//5)
        assert a['query_id'].tolist()==[r['query_id'] for r in selected]
        assert equal(a['original_raw_E_logit'],scores) and equal(a['original_weak_role_target'],y)
        assert np.array_equal(a['original_weak_role_state'],[r['role_state'] for r in selected])
        assert np.array_equal(a['original_input_available'],av) and np.array_equal(a['original_fit_eligible'],eligible)
        fixed=decode(scores,av,0.);assert equal(fixed,oldpred) and equal(fixed,a['fixed0_prediction'])
        prior=read(verify(model['source_results']))['MR']['SOURCE_VALIDATION']
        assert metric(y,fixed,np.ones(360,bool))==prior['visibility']==result['original_fixed0']['full_known154']
        assert metric(y,fixed,av)==prior['visibility_inputvalid']==result['original_fixed0']['inputvalid145']
        folds=read(verify(result['folds']))['folds'];assert len(folds)==9
        oof=np.full(360,-9,np.int8);tv=np.full(360,np.nan,np.float64);visits=np.zeros(360,np.int8);records=[]
        for b,fold in enumerate(folds):
            block=frames//5;held=np.flatnonzero(block==b);fit=np.flatnonzero(eligible&(np.abs(block-b)>1));purged=np.flatnonzero(np.abs(block-b)<=1)
            assert fold['block_id']==b and fold['held_indices']==held.tolist() and fold['fit_indices']==fit.tolist() and fold['purged_indices']==purged.tolist()
            assert not set(fit)&set(purged) and eligible[fit].all()
            expected=brute(scores,y,fit)
            for k,v in expected.items():assert fold[k]==v,(model['arm'],b,k)
            assert expected['fallback'] is None
            metadata=dict(fit_frame_numbers=sorted(set(frames[np.abs(block-b)>1].tolist())),held_frame_numbers=sorted(set(frames[held].tolist())),
                fit_nonpurged_total_rows=int((np.abs(block-b)>1).sum()),fit_eligible_rows=len(fit),fit_inputU_rows=int(((np.abs(block-b)>1)&~av).sum()),
                fit_unknown_truth_rows=int(((np.abs(block-b)>1)&~known).sum()),held_total_rows=len(held),held_known_D_rows=int((y[held]==0).sum()),held_known_E_rows=int((y[held]==1).sum()),
                held_inputvalid_D_rows=int(((y==0)&av&(block==b)).sum()),held_inputvalid_E_rows=int(((y==1)&av&(block==b)).sum()),
                held_inputU_rows=int((~av[held]).sum()),held_unknown_truth_rows=int((~known[held]).sum()))
            for k,v in metadata.items():assert fold[k]==v,(b,k)
            # Independent mutation check keeps the original eligibility mask frozen.
            changed=y.copy();changed[purged]=1-np.maximum(changed[purged],0);assert brute(scores,changed,fit)==expected
            changed=y.copy();changed[~eligible]=1;assert brute(scores,changed,fit)==expected
            threshold=expected['threshold_float64'];oof[held]=decode(scores[held],av[held],threshold);tv[held]=threshold;visits[held]+=1
            records.append(dict(block=b,threshold_float64=threshold,N_D=expected['N_D'],N_E=expected['N_E'],objective=expected['integer_BA_objective']))
        assert (visits==1).all() and a['OOF_threshold_float64'].dtype==np.float64 and equal(tv,a['OOF_threshold_float64'])
        assert equal(oof,a['purged_temporal_OOF_prediction']) and np.array_equal(oof==-1,~av)
        full=brute(scores,y,np.flatnonzero(eligible))
        for k,v in full.items():assert full_saved[k]==v,(model['arm'],k)
        scalar=a['full_threshold_float64'];assert scalar.dtype==np.float64 and scalar.shape==() and scalar.item()==full['threshold_float64']
        apparent=decode(scores,av,full['threshold_float64']);assert equal(apparent,a['full_PUBLIC_CAL_apparent_prediction'])
        assert np.array_equal(apparent==-1,~av) and result['fallback_fold_count']==0 and result['fallback_fold_ids']==[]
        for key,pred in [('purged_temporal_OOF',oof),('full_PUBLIC_CAL_apparent',apparent)]:
            assert metric(y,pred,np.ones(360,bool))==result[key]['full_known154'] and metric(y,pred,av)==result[key]['inputvalid145']
        ck(model['arm']+'_independent_brute_all9_and_full_objective_highesttie_float64_ge_nextafter')
        ck(model['arm']+'_purge_once360_frozen145_eligibility_mutation_U_and_report_metrics_exact')
        measured.append(dict(arm=model['arm'],full_threshold_float64=full['threshold_float64'],full_objective=full['integer_BA_objective'],
            fixed0_full=metric(y,fixed,np.ones(360,bool)),OOF_full=metric(y,oof,np.ones(360,bool)),OOF_valid=metric(y,oof,av),
            apparent_full=metric(y,apparent,np.ones(360,bool)),apparent_valid=metric(y,apparent,av),folds=records,fallback_count=0))
    for b in list(BINDINGS):assert bind(b['path'])==b
    ck('all_bound_head_checkpoint_bytes_public_outputs_labels_and_calibration_artifacts_immutable')
    r=dict(completed=True,PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_code=bind(__file__),source_complete=bind(complete_path),
        checks=CHECKS,checks_passed=len(CHECKS),method='Independent scalar Python brute-force candidate thresholds and integer objective; no producer fit function import or invocation.',
        measurements=measured,all360=360,known154=154,fit_valid145=145,D102=102,E52=52,U_known9_allD=True,U_all108=True,unknown_truth206=206,
        fullCAL_apparent_not_held_test=True,purgedOOF_same_sequence_stability_not_fresh_model_validation=True,historical_public_exposure_preserved=True,
        ninefolds_not_independent_scenes=True,queries154_not_independent_samples=True,threshold_workingpoint_not_probability_calibration=True,
        original_fixed0_source_results_unchanged=True,new_threshold_or_model_selection=False,checkpoint_access='SHA bytes only; no deserialization',
        model_encoder_newheadfit_RGB_current_private_V112_PnP_qL_calls_or_changes=0,selected_model=None,adopted=False,fully_solved=False,
        bindings=list({(b['path'],b['sha256']):b for b in BINDINGS}.values()))
    out=H/'REVIEW.json'
    with out.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,checks=len(CHECKS),review=bind(out),measurements=measured),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
