"""Saved-only V148 audit; CPU tensors allowed, model construction/forward absent."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
import torch

H=Path(__file__).resolve().parent;V=H.parent
RUN=V/'runs/C33_C34_PUBLIC_NATIVE_FIXED2000'
CONTROL=RUN/'CONTROL_COMPLETE.json'
EXPECTED='28d23e04c76c11b48b6c84d7a75be8dcd4bb1feda4977d71c5c852189de0c3a2'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for z in iter(lambda:f.read(1<<20),b''):h.update(z)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,r):
    with p.open('x')as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def npz(p,excluded=()):
    with np.load(p,allow_pickle=False)as f:return{k:f[k].copy()for k in f.files if k not in excluded}
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def tensor_state_sha(state):
    h=hashlib.sha256()
    for k,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()
def metric(state,pred,mask):
    target=np.where(state=='D',0,np.where(state=='E',1,-1))
    cm=np.array([[np.sum(mask&(target==t)&(pred==z))for z in(0,1,-1)]for t in(0,1)],np.int64)
    div=lambda n,d:float(n/d)if d else None
    return dict(CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=div(cm[0,0],cm[0].sum()),E_recall=div(cm[1,1],cm[1].sum()),
        D_precision=div(cm[0,0],cm[:,0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),prediction_U_known=int(cm[:,2].sum()),
        prediction_U_rate_known=div(cm[:,2].sum(),cm.sum()),all_original_rows=int(mask.sum()),known_DE=int(cm.sum()),
        teacher_S_outside_external=int(np.sum(mask&(state=='S'))),teacher_U_outside_external=int(np.sum(mask&(state=='U'))),
        teacher_S_binary_responses={str(z):int(np.sum(mask&(state=='S')&(pred==z)))for z in(0,1,-1)},
        teacher_U_binary_responses={str(z):int(np.sum(mask&(state=='U')&(pred==z)))for z in(0,1,-1)})
def group_mask(name,rr):
    parts=name.split('::');fold=parts.pop(0);n=len(rr)
    col=lambda key:np.array([r[key]for r in rr])
    mask=np.ones(n,bool)if fold=='ALL'else col('split')==fold
    boundary=col('correspondence_boundary').astype(bool)
    while parts:
        key=parts.pop(0)
        if key in('NONBOUNDARY','historical_area_agreement'):mask &= ~boundary
        elif key in('BOUNDARY','historical_area_disagreement'):mask &= boundary
        elif key=='SAME_TARGET_MATERIAL':mask &= np.array([r['factors']['foreign_family']=='SAME_TARGET_MATERIAL'for r in rr])
        elif key in('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY'):mask &= col('position_band')==key
        elif key=='material_aspect':
            material,aspect=parts.pop(0),parts.pop(0);mask &= np.array([r['factors']['material']==material and r['factors']['aspect']==aspect for r in rr])
        elif key in('variant','material','aspect','geometry_family','foreign_family'):
            value=parts.pop(0);mask &= np.array([(r[key]if key=='variant'else r['factors'][key])==value for r in rr])
        else:raise AssertionError('Unexpected cohort selector '+name)
    return mask
def validate_packet(a,metadata,public=False):
    n=len(a['prediction']);assert a['raw_head_logit_before_U'].dtype==np.float32 and a['prediction'].dtype==np.int8
    for k,b in metadata.items():
        if k=='feature_constructed':assert eq(a['head_forwarded'],b)
        else:assert eq(a[k],b),k
    assert np.array_equal(a['prediction'],np.where(a['query_available'],(a['raw_head_logit_before_U']>=0).astype(np.int8),-1))
    if public:assert a['head_forwarded'].all()and np.isfinite(a['raw_head_logit_before_U']).all()
    else:assert np.array_equal(np.isfinite(a['raw_head_logit_before_U']),a['head_forwarded'])
    assert a['prediction'].shape==(n,)

def main():
    assert sha(CONTROL)==EXPECTED
    control=read(CONTROL);assert control['completed']and control['PASS']and control['selected_model']is None and not control['full_goal_solved']
    for k in('code','policy','protocol','entry','public_feature_seal','source_inputs_unchanged'):check(control[k])
    p=read(control['protocol']['path']);feature=read(control['public_feature_seal']['path'])
    fixed=read(check(p['fixed_public_pools_and_native35']));limits=fixed['native35_limits'];assert len(limits)==35
    source_limits={k:v for k,v in read(check(p['C32_source_results']))['gates'].items()if k.startswith(('NATIVE::','NEW_NATIVE::'))};assert limits==source_limits
    labels=rows(check(p['public_labels']));ps=np.array([r['semantic_state']for r in labels]);pf=np.array([r['split']for r in labels])
    publicmeta=npz(check(feature['features']),('features',))
    av=publicmeta['query_available'];assert av.sum()==463 and len(av)==512
    pi=np.array(sorted(fixed['TRAIN_D35']+fixed['TRAIN_E8']),np.int64);assert len(pi)==43 and len(np.unique(pi))==43
    assert (pf[pi]=='TRAIN_LAB').all()and av[pi].all()and np.sum(ps[pi]=='D')==35 and np.sum(ps[pi]=='E')==8
    assert np.flatnonzero(np.isin(ps,['D','E'])&~av).tolist()==[402,434]
    metas=[];baselines=[];native_rows=[]
    for n,b in zip((6144,20480),p['native']):
        meta=npz(check(b['features']),('features',));base=npz(check(b['baseline']))
        for key in('occurrence_id','query_xy','L','query_available'):assert eq(meta[key],base[key])
        assert meta['global_index'].dtype.kind in'iu'and base['global_index'].dtype.kind in'iu'
        assert np.array_equal(meta['global_index'],np.arange(n))and np.array_equal(base['global_index'],np.arange(n))
        assert eq(meta['feature_constructed'],base['head_forwarded'])
        allrows=rows(check(b['GT']));rr=[r for r in allrows if r['convention']=='CF_PRIMARY_DECLARED'];assert len(rr)==n and len(allrows)==n*2
        for i,r in enumerate(rr):
            assert r['query_id']==meta['occurrence_id'][i]and r['native_slot']==int(meta['native_slot'][i])
            assert np.array_equal(meta['query_xy'][i],r['query_xy']if r['query_xy']is not None else[np.nan,np.nan],equal_nan=True)
            assert (r['roi_side']is None and np.isnan(meta['L'][i]))or r['roi_side']==meta['L'][i]
        metas.append(meta);baselines.append(base);native_rows.append(rr)
    assert [int(a['query_available'].sum())for a in metas]==[6056,20384]
    initial_payload=torch.load(check(p['anchor_checkpoint']),map_location='cpu',weights_only=True)
    assert initial_payload['update']==8000
    initial_state=initial_payload['state_dict'];initial_hash=tensor_state_sha(initial_state)
    assert sum(t.numel()for t in initial_state.values())==224513 and not initial_state['geometry.weight'].any()
    # Read the fixed prefix once; no random draws or pool reconstruction.
    with check(p['C32_draws']).open()as f:prefix=[json.loads(next(f))for _ in range(2000)]
    saved={};all_results={};initial_public=None;draw_digest=None;first_trace=None
    for coefficient,arm in enumerate(p['arms']):
        directory=RUN/arm;fit=read(check(control['arms'][arm]['fit']));assert fit['completed']and fit['optimizer_updates']==2000
        for b in fit['outputs'].values():check(b)
        report=read(check(fit['source_results']));freeze=read(check(fit['freeze']))
        assert report['selected_model']is None and not report['full_goal_solved']and not report['adopted']
        assert freeze['selected_model']is None and not freeze['current_allowed']and freeze['diagnostic_only']and freeze['original64_not_claimed']and freeze['AUX29_not_retested']
        assert freeze['update']==2000 and freeze['parent_C32_update']==8000 and freeze['threshold_logit']==0
        initial_native=[npz(directory/('INITIAL_'+name+'_SCORES.npz'))for name in('OLD6144','NEW20480')]
        final_native=[npz(directory/('FINAL_'+name+'_SCORES.npz'))for name in('OLD6144','NEW20480')]
        initpub=npz(directory/'INITIAL_PUBLIC512_SCORES.npz');finalpub=npz(directory/'FINAL_PUBLIC512_SCORES.npz')
        for j,(i,f,m,base)in enumerate(zip(initial_native,final_native,metas,baselines)):
            validate_packet(i,m);validate_packet(f,m);assert eq(i['raw_head_logit_before_U'],base['raw_head_logit_before_U'])
        validate_packet(initpub,publicmeta,True);validate_packet(finalpub,publicmeta,True)
        if initial_public is None:initial_public=initpub
        else:assert set(initial_public)==set(initpub)and all(eq(initial_public[k],initpub[k])for k in initpub)
        parity=read(directory/'UPDATE0_C32_EXACT_PARITY.json');assert parity['PASS']and parity['actual_updates']==0 and parity['C32_state_sha256']==initial_hash
        assert parity['public512_botharms_initial_comparison']==('PENDING_SECOND_ARM'if coefficient==0 else'PASS')
        assert report['initial_state_sha256']==initial_hash
        traces=rows(directory/'ALL2000_DRAWS_AND_LOSSES.jsonl');assert len(traces)==2000
        exposure=[np.zeros(6144,np.int64),np.zeros(20480,np.int64)];public_exposure=np.zeros(512,np.int64);digest=hashlib.sha256()
        for step,(t,old)in enumerate(zip(traces,prefix),1):
            assert t['update']==old['update']==step and t['public_weight']==coefficient and t['public_TRAIN_indices']==pi.tolist()
            for key in('native_occurrences','extra_native_occurrences','native_pair_pool_positions','extra_pair_pool_positions'):
                assert t[key]==old[key];values=np.asarray(t[key],'<i8');digest.update(values.tobytes())
                if key.endswith('pair_pool_positions'):assert values.shape==(16,)
            for j,key in enumerate(('native_occurrences','extra_native_occurrences')):
                ix=np.array(t[key],np.int64);assert ix.shape==(64,)
                assert metas[j]['query_available'][ix].all()
                assert all(native_rows[j][i]['split']=='TRAIN'for i in ix)
                states=np.array([native_rows[j][i]['teacher_state']for i in ix]);assert np.array_equal(states,np.r_[['E']*16,['D']*32,['E']*16])
                np.add.at(exposure[j],ix,1)
            np.add.at(public_exposure,pi,1)
            pd,pe,nl,el,pl=map(np.float32,(t['public_D_BCE'],t['public_E_BCE'],t['native_BCE'],t['extra_native_BCE'],t['public_balanced_BCE']))
            assert pl==np.float32(np.float32(.5)*pd+np.float32(.5)*pe)
            expected=np.float32(np.float32(nl+el)+np.float32(coefficient)*pl);assert np.float32(t['loss'])==expected
            assert np.isfinite(t['loss'])and np.isfinite(t['gradient_before_clip'])
            count=t['public_output_gradient_nonzero'];assert count==0 if coefficient==0 else 0<count<=43
        dh=digest.hexdigest();assert dh==fit['draw_sha256']==report['all2000_draw_hash']==control['draw_sha256']
        if draw_digest is None:draw_digest=dh;first_trace=traces
        else:
            assert dh==draw_digest
            for key in('native_BCE','extra_native_BCE','public_D_BCE','public_E_BCE','public_balanced_BCE'):assert traces[0][key]==first_trace[0][key]
        exp=npz(directory/'SAMPLING_EXPOSURE.npz')
        assert eq(exp['old6144'],exposure[0])and eq(exp['new20480'],exposure[1])and eq(exp['public_loss_computed'],public_exposure)
        assert eq(exp['public_loss_weighted'],public_exposure*coefficient)
        assert exposure[0].sum()==exposure[1].sum()==128000 and public_exposure.sum()==86000
        assert np.flatnonzero(public_exposure).tolist()==pi.tolist()and (public_exposure[pi]==2000).all()
        ck=torch.load(check(freeze['checkpoint']),map_location='cpu',weights_only=True)
        assert ck['update']==2000 and ck['parent_C32_update']==8000 and ck['seed']==1480 and ck['arm']==arm and ck['public_coefficient']==coefficient and ck['selected_model']is None
        assert ck['policy']==control['policy']and ck['protocol']==control['protocol']and ck['source_feature_seal']==control['public_feature_seal']
        state=ck['state_dict'];assert tensor_state_sha(state)==report['final_state_sha256']and tensor_state_sha(state)!=initial_hash
        assert set(state)==set(initial_state)and sum(t.numel()for t in state.values())==224513
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all()for t in state.values())and not state['geometry.weight'].any()
        opt=ck['optimizer_state'];assert len(opt['param_groups'])==1 and len(opt['state'])==5
        group=opt['param_groups'][0];assert group['lr']==1e-4 and group['weight_decay']==.001 and group['betas']==(.9,.999)and group['eps']==1e-8
        assert all(float(s['step'])==2000 for s in opt['state'].values())
        assert all(torch.isfinite(s[k]).all()for s in opt['state'].values()for k in('exp_avg','exp_avg_sq'))
        assert not opt['state'][4]['exp_avg'].any()and not opt['state'][4]['exp_avg_sq'].any()
        metrics={};ngates={}
        for j,name in enumerate(('old_native','new_native')):
            rr=native_rows[j];state=np.array([r['teacher_state']for r in rr]);pr=final_native[j]['prediction'];allgroups={}
            for selector,stored in report['native'][name].items():
                result=metric(state,pr,group_mask(selector,rr));assert result==stored,(arm,selector);allgroups[selector]=result
            assert len(allgroups)==(84 if j==0 else 78);metrics[name]=allgroups
        for name,base in limits.items():
            new=name.startswith('NEW_NATIVE::');selector,state=name[len('NEW_NATIVE::')if new else len('NATIVE::'):].rsplit('::',1)
            row=metrics['new_native'if new else'old_native'][selector];value=row[state+'_recall'];support=sum(row['CM_D_E_rows_D_E_U_columns'][int(state=='E')]);limit=base['limit']
            g=dict(value=value,limit=limit,N=support,PASS=bool(support>0 and value is not None and limit is not None and value>=limit))
            assert g==report['native']['gates'][name];ngates[name]=g
        assert sum(g['PASS']for g in ngates.values())==22==report['native']['native35_passed']and not all(g['PASS']for g in ngates.values())
        pubresults={};initialresults={}
        for name,stored in report['public']['groups'].items():
            selector=name.split('::')[0];mask=np.ones(512,bool)if selector=='ALL'else pf==selector
            if 'INPUT_VALID'in name:mask &= av
            result=metric(ps,finalpub['prediction'],mask);assert result==stored;pubresults[name]=result
            initial=metric(ps,initpub['prediction'],mask);assert initial==report['initial_public_C32']['groups'][name];initialresults[name]=initial
        hall=pubresults['VAL_HALL::INPUT_VALID']['CM_D_E_rows_D_E_U_columns']
        sanity=dict(D_correct=hall[0][0],D_N=33,D_minimum=30,E_correct=hall[1][1],E_N=6,E_minimum=5,PASS=bool(hall[0][0]>=30 and hall[1][1]>=5))
        assert sanity==freeze['public_sanity']==report['public']['sanity']
        assert not freeze['native35_all_pass']and not report['prospective_diagnostic_criteria_all_pass']
        all_results[arm]=dict(native35_passed=22,native_groups_recomputed=[84,78],native_gates=ngates,
            old_native_VAL=metrics['old_native']['VALIDATION'],new_native_VAL=metrics['new_native']['VALIDATION'],
            public=pubresults,public_initial=initialresults,public_sanity=sanity,
            exposure_old=128000,exposure_new=128000,public_loss_computed=86000,public_loss_weighted=86000*coefficient,
            all2000_draws_and_loss_arithmetic_exact=True,initial_C32_state_SHA=initial_hash,final_state_SHA=report['final_state_sha256'],optimizer_steps_each_parameter=2000,
            full_native_prediction_U=[int((a['prediction']<0).sum())for a in final_native],public_prediction_U=int((finalpub['prediction']<0).sum()))
        saved[arm]=finalpub
    unchanged=read(control['source_inputs_unchanged']['path']);assert unchanged['PASS']and unchanged['all_original_bound_files_rehashed_after_both_fits']
    for b in unchanged['original_bindings']:check(b)
    arm0,arm1=p['arms'];mask=(pf=='VAL_HALL')&np.isin(ps,['D','E']);y=np.where(ps=='D',0,1)
    before=saved[arm0]['prediction'];after=saved[arm1]['prediction'];recovered=np.flatnonzero(mask&(before!=y)&(after==y)).tolist();damaged=np.flatnonzero(mask&(before==y)&(after!=y)).tolist()
    resultfile=H/'RECALCULATED_SOURCE_RESULTS.json';write(resultfile,dict(completed=True,PASS=True,control_complete=bind(CONTROL),arms=all_results,
        HALL_C34_vs_C33=dict(recovered_indices=recovered,damaged_indices=damaged,known_full41=True,teacherD_inputU_preserved=[402,434]),
        allsource_qualifications_pass=False,selected_model=None,full_goal_solved=False))
    receipt=dict(completed=True,PASS=True,code=bind(__file__),control_complete=bind(CONTROL),results=bind(resultfile),
        checks=dict(all26624_native512_public_metadata_qL_masks_and_rows_exact=True,C32_update0_allraw_bitexact=True,public_botharms_initial_allarrays_bitexact=True,
            all2x2000_native_draws_fixed_C32_prefix_exact=True,public_full43_TRAIN_computed_botharms_HALLzero=True,public_coefficient0_output_gradient_alwayszero=True,
            loss_and_exposure_totals_independently_recomputed=True,finalCPU_parameter_state_checkpoint_and_AdamW2000_verified=True,all84plus78_native_cohorts_and35gates_exact=True,
            all_public_CM_precision_recall_U_and_initial_metrics_exact=True,all_original_bindings_endSHA_exact=True,all512_public_inputavailability463_U49_preserved=True),
        interpretation='C34 improves this tiny public weak-label HALL cohort relative to matched C33, but both remain22/35 registered-native gates, AUX29 was not retested, selected_model stays null. No full-goal or private transfer success established.',
        actual_new_model_objects_forward_backward_optimizer_steps=0,CPU_checkpoint_tensor_reads_only=True,RGB_reads=0,private_current_reads=0,
        saved_repeat_claim_limit='Stored repeated-output receipts and fixed outputs checked; no forward repeated in this audit.',
        previous_first_preflight_exit_preserved=True,automatic_retry=False)
    path=H/'REVIEW_COMPLETE.json';write(path,receipt)
    print(json.dumps(dict(review=bind(path),results=bind(resultfile),public_HALL={a:all_results[a]['public']['VAL_HALL']['CM_D_E_rows_D_E_U_columns']for a in p['arms']},native35=[22,22],recovered=recovered,damaged=damaged)))

if __name__=='__main__':main()
