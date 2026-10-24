"""Saved-only C32 audit: CPU states/RNG and source arithmetic, no model/forward."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys,time
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
RUN=V/'runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000';ARM='C32_AUX_TO_NATIVE_GRAD0_FIXED8000';D=RUN/ARM
OLDHELP=P/'query_visibility_continuous_source_coverage_control_20261024_v137/independent_saved_review/review_saved_controls.py'
STATEHELP=P/'query_visibility_broad_source_geometry_control_20261024_v139/independent_design_review/review_completed_geometry_control.py'
C29=P/'query_visibility_continuous_source_coverage_control_20261024_v137/runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/C29_CONTINUOUS_NATIVE_DATA_ADDITION'
C30=P/'query_visibility_broad_source_geometry_control_20261024_v139/runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000/C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL'
FILES=('FINAL_OLD51004_SOURCE_SCORES.npz','FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz','FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz','SAMPLING_EXPOSURE.npz')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    assert '_v112'not in b['path'].lower()and'/private_'not in b['path'].lower()
    assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def exact(a,b):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def packet_equal(a,b):
    assert set(a)==set(b)
    for key in a:exact(a[key],b[key])


def traces(old,p,op,on,en,om,nm,oldpairs,newpairs,torch):
    # All original six draw arrays, plus independent native pair/uniform RNG.
    source_iters=[iter(old.lines(check(p[key])))for key in('historical_C24_trace','historical_C26_trace','historical_C29_trace')]
    npg,nug=torch.Generator().manual_seed(1334),torch.Generator().manual_seed(1332)
    epg,eug=torch.Generator().manual_seed(1374),torch.Generator().manual_seed(1372)
    nu=old.uniform_pools(om,on['query_available']);eu=old.uniform_pools(nm,en['query_available'])
    exposure=[np.zeros(n,np.int64)for n in(51004,6144,20480)];shared=hashlib.sha256();extra=hashlib.sha256()
    count=0
    for count,row in enumerate(old.lines(D/'TRAIN_DRAWS_AND_LOSSES.jsonl'),1):
        assert row['update']==count
        ix=np.asarray(row['old_occurrences'],np.int64);pp=np.asarray(row['old_matched_cover_sham_positions'],np.int64)
        ni=np.asarray(row['native_occurrences'],np.int64);npi=np.asarray(row['native_pair_pool_positions'],np.int64);ei=np.asarray(row['extra_native_occurrences'],np.int64);epi=np.asarray(row['extra_pair_pool_positions'],np.int64)
        assert ix.shape==(192,)and pp.shape==(24,2)and ni.shape==ei.shape==(64,)and npi.shape==epi.shape==(16,)
        histories=[next(it)for it in source_iters]
        for previous in histories:
            assert previous['update']==count and previous['old_occurrences']==ix.tolist()and previous['old_matched_cover_sham_positions']==pp.tolist()
        for previous in histories[1:]:assert previous['native_occurrences']==ni.tolist()and previous['native_pair_pool_positions']==npi.tolist()
        assert histories[2]['extra_native_occurrences']==ei.tolist()and histories[2]['extra_pair_pool_positions']==epi.tolist()
        expected,positions=old.extra_draw(oldpairs,nu,npg,nug);assert np.array_equal(ni,expected)and np.array_equal(npi,positions)
        expected,positions=old.extra_draw(newpairs,eu,epg,eug);assert np.array_equal(ei,expected)and np.array_equal(epi,positions)
        assert row['extra_source']=='V136_NATIVE'and row['native_hinge_weight']==0 and row['AUX_old192_loss_coefficient']==0 and row['AUX_old192_output_gradient_nonzero']==0
        assert op['query_available'][ix].all()and(op['split'][ix]=='TRAIN').all()and np.bincount(op['external_target'][ix],minlength=2).tolist()==[96,96]
        assert np.array_equal(op['external_target'][ix[pp]],np.tile([1,0],(24,1)))
        for indices,pack,meta in((ni,on,om),(ei,en,nm)):
            assert pack['query_available'][indices].all()and(meta['split'][indices]=='TRAIN').all()and np.array_equal(meta['target'][indices],np.r_[np.ones(16),np.zeros(32),np.ones(16)])
        for arr in(ix,pp,ni):shared.update(arr.astype('<i8').tobytes())
        extra.update(ei.astype('<i8').tobytes());extra.update(epi.astype('<i8').tobytes())
        for acc,indices in zip(exposure,(ix,ni,ei)):np.add.at(acc,indices,1)
        assert all(math.isfinite(row[k])and row[k]>=0 for k in('loss','old_BCE','old_hinge','flat_BCE','native_BCE','extra_native_BCE','gradient_before_clip','AUX_old192_full_loss_value'))
        oldloss=np.float32(np.float32(np.float32(row['old_BCE'])+np.float32(.25)*np.float32(row['old_hinge']))+np.float32(row['flat_BCE']))
        assert float(oldloss)==row['AUX_old192_full_loss_value']
        loss=np.float32(np.float32(np.float32(oldloss*0.)+np.float32(row['native_BCE']))+np.float32(row['extra_native_BCE']))
        assert float(loss)==row['loss'],count
    assert count==8000 and all(next(it,None)is None for it in source_iters)
    saved=old.packet(D/'SAMPLING_EXPOSURE.npz')
    for key,expected,n in zip(('old51004','native6144','new_native20480'),exposure,(192,64,64)):
        exact(saved[key],expected);assert int(expected.sum())==8000*n
    rng_hash=hashlib.sha256(epg.get_state().numpy().tobytes()+eug.get_state().numpy().tobytes()).hexdigest()
    return dict(updates=8000,all6draw_arrays_exact_C29=True,old192_exact_C24_C26_C29=True,native_old64_new64_pair_and_uniform_CPU_RNG_exact=True,
        all_sampled_rows_TRAIN_available_known_DE_no_SU_VAL=True,old192_observed_draws_retained_but_AUX_gradient_coefficient0=True,
        AUX_entire_loss_and_native_only_total_FP32_arithmetic_exact_all8000=True,AUX_zero_output_gradient_trace_all8000=True,
        output_gradient_is_producer_log_not_independent_backward_replay=True,shared_sha256=shared.hexdigest(),extra_sha256=extra.hexdigest(),
        extra_rng_consumption_sha256=rng_hash,exposures_exact=True,exposure_sums=[int(a.sum())for a in exposure])


def paired(meta,left,right,mask):
    target=meta['target'];out={}
    for state in('DE','D','E','S','U'):
        cohort=mask&(np.isin(meta['state'],list('DE'))if state=='DE'else meta['state']==state)
        transitions=[[int((cohort&(left==x)&(right==y)).sum())for y in(0,1,-1)]for x in(0,1,-1)]
        rr=dict(rows=int(cohort.sum()),prediction_transition_baseline_rows_C32_columns_D_E_U=transitions,baseline_U=int((cohort&(left==-1)).sum()),C32_U=int((cohort&(right==-1)).sum()))
        if state in('S','U'):rr['outside_external_binary']=True
        else:
            lc=left==target;rc=right==target;rec=cohort&~lc&rc;reg=cohort&lc&~rc
            base=np.asarray([r['base_index']for r in meta['rows']]);deltas=[int(rec[base==b].sum()-reg[base==b].sum())for b in np.unique(base[cohort])]
            rr.update(correct_both=int((cohort&lc&rc).sum()),wrong_both=int((cohort&~lc&~rc).sum()),recovered=int(rec.sum()),regressed=int(reg.sum()),net_correct_change=int(rec.sum()-reg.sum()),
                base_groups=len(deltas),bases_improved=sum(x>0 for x in deltas),bases_regressed=sum(x<0 for x in deltas),bases_tied=sum(x==0 for x in deltas))
        out[state]=rr
    return out


def run():
    start=time.monotonic();assert not(H/'REVIEW.json').exists()
    control=read(RUN/'CONTROL_COMPLETE.json');assert control['completed']and control['PASS']and control['final_update8000_single_arm']and control['selected_model']is None and control['selected_arm_source_only']is None
    p=read(check(control['protocol']));check(control['code']);check(control['policy'])
    assert sha(OLDHELP)=='8cb5c878c79aaf3a09e74c4163f45f17ae5bd487459f747d41204587a0c0465c'
    assert sha(STATEHELP)=='84ba430addfcc36e58cd4b4da30b415103bb4c2f6e936d002d68ef4df4ddff81'
    old=module(OLDHELP,'C32_independent_prior_arithmetic');state=module(STATEHELP,'C32_CPU_state_only_helpers');a,n,torch=old.a,old.n,old.torch
    torch.set_num_threads(1);assert torch.__version__==p['runtime']['torch']and np.__version__==p['runtime']['numpy']
    fit=read(check(control['arms'][ARM]['fit']));freeze=read(check(control['arms'][ARM]['freeze']));saved=read(check(fit['results']))
    assert fit['completed']and fit['optimizer_updates']==freeze['update']==saved['selected_update']==8000 and freeze['parameters']==224513 and freeze['threshold_logit']==saved['threshold_logit']==0
    assert freeze['current_allowed']is False and freeze['V112_allowed']is False and freeze['selected_model']is None and freeze['diagnostic_only']and freeze['AUX_old192_loss_coefficient']==0
    source=read(check(p['source_binding_policy']));baseline=a.packet(check(source['C16_source_scores']));oldmasks=a.old_masks(baseline);base14=a.old14(baseline,oldmasks);original15=read(a.OLD/'SOURCE_GATE_RESULTS.json')['gates']
    oldseal=read(check(source['native_feature_seal']));oldbase=a.packet(check(oldseal['predictions']));oldmeta=a.native_data()
    newseal=read(check(p['new_feature_seal']));newbase=a.packet(check(newseal['predictions']));newmeta=n.metadata(check(p['new_GT_ledger']))
    old.baseline_packet_checks(oldbase,oldmeta[n.CONV[0]],6144,'V129');old.baseline_packet_checks(newbase,newmeta[n.CONV[0]],20480,'V136')
    om,nm=oldmeta[n.CONV[0]],newmeta[n.CONV[0]]
    oldpairs=old.pairpool(om,oldbase['query_available'],128);newpairs=old.pairpool(nm,newbase['query_available'],512);assert len(oldpairs)==125 and len(newpairs)==473
    op=a.packet(D/FILES[0]);on=old.native_packet(D/FILES[1],oldbase,om,6144);en=old.native_packet(D/FILES[2],newbase,nm,20480)
    c29=[a.packet(C29/f)for f in FILES];c30=[a.packet(C30/f)for f in FILES]
    for left,right in zip(c29,c30):packet_equal(left,right)
    for key in('query_available','input_unit_index','external_target','raw_teacher_state','split','domain','source','variant'):
        assert np.array_equal(op[key],baseline[key]);exact(op[key],c29[0][key])
    assert op['query_logit'].dtype==np.float32 and op['prediction'].dtype==np.int8 and op['query_available'].dtype==bool
    assert op['query_logit'].shape==op['prediction'].shape==(51004,)and np.array_equal(op['global_occurrence_index'],np.arange(51004))
    assert np.isfinite(op['query_logit']).all()and np.array_equal(op['prediction'],np.where(op['query_available'],(op['query_logit']>=0).astype(np.int8),-1))
    for pack,base,size in((on,c29[1],6144),(en,c29[2],20480)):
        assert pack['query_logit'].dtype==pack['raw_head_logit_before_U'].dtype==np.float32 and pack['prediction'].dtype==np.int8
        for key in pack:
            if key not in('query_logit','raw_head_logit_before_U','prediction'):exact(pack[key],base[key])
        assert pack['query_xy'].shape==(size,2)and pack['L'].shape==(size,)
    basen=a.native_metrics(oldbase,om);basee=old.new_metrics(newbase,nm)
    nr=a.native_metrics(on,om);er=old.new_metrics(en,nm);v4=a.native_metrics(on,oldmeta[n.CONV[1]]);ev4=old.new_metrics(en,newmeta[n.CONV[1]])
    for key,value in dict(native_CF=nr,native_V4_sensitivity_only=v4,new_native_CF=er,new_native_obsolete_area_V4_sensitivity_only=ev4).items():a.close(value,saved[key],key)
    gates,old14=a.gate_values(op,baseline,base14,original15,nr,basen,oldmasks);gates.update(old.new_gates(er,basee));assert len(gates)==64 and set(gates)==set(saved['gates'])
    for key,value in gates.items():
        for field,expected in value.items():a.close(expected,saved['gates'][key][field],key+'/'+field)
    a.close(old14,saved['old']['old14'])
    aux={k:v for k,v in gates.items()if k.startswith(('OLD14::','HISTORICAL15::'))};native={k:v for k,v in gates.items()if k not in aux};assert len(aux)==29 and len(native)==35
    split=dict(all64=dict(passed=sum(g['PASS']for g in gates.values()),required=64,all_pass=all(g['PASS']for g in gates.values()),failed=[k for k,g in gates.items()if not g['PASS']]),
        AUX29=dict(passed=sum(g['PASS']for g in aux.values()),required=29,all_pass=all(g['PASS']for g in aux.values()),failed=[k for k,g in aux.items()if not g['PASS']]),
        registered_native35=dict(passed=sum(g['PASS']for g in native.values()),required=35,all_pass=all(g['PASS']for g in native.values()),failed=[k for k,g in native.items()if not g['PASS']]),
        original64_unchanged=True,retrospective_qualification_changed=False,selected_model=None,private_evaluation_authorized=False)
    a.close(split,saved['qualification_split']);a.close(split,read(check(fit['qualification_split'])))
    rank=[v['value']for k,v in gates.items()if not k.startswith('HISTORICAL15::')];assert len(rank)==49
    worst=None if any(v is None for v in rank)else min(rank);eligible=all(v['PASS']for v in gates.values())
    assert worst==saved['worst_required49_class_recall']==fit['worst_required49_class_recall']==control['arms'][ARM]['worst_required49_class_recall']
    assert eligible==saved['source_eligible']==fit['source_eligible']==freeze['source_eligible']==control['arms'][ARM]['source_eligible']and saved['selected_model']is None
    z=op['query_logit'].astype(np.float64);y=np.where(np.isin(op['external_target'],[0,1]),op['external_target'],0);loss=np.logaddexp(0,z)-y*z
    parts={k:dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~op['query_available']).sum()))for k,mask in oldmasks.items()}
    a.close(parts,saved['old']['old14_BCE_parts']);a.close(float(np.mean([v['mean_BCE']for v in parts.values()])),saved['old']['old14_BCE'])
    for key,metric in saved['old']['domain_metrics'].items():
        domain,fold=key.split('::');a.close(a.metrics(op['external_target'],op['prediction'],(op['domain']==domain)&(op['split']==fold),op['raw_teacher_state']),metric)
    trace=traces(old,p,op,on,en,om,nm,oldpairs,newpairs,torch)
    assert trace['shared_sha256']==saved['shared_old192_native64_sha256']==p['historical_C29_draw_hash']==control['shared_draw_sha256']
    assert trace['extra_sha256']==saved['extra_native_draw_sha256']==p['historical_C29_extra_draw_hash']and trace['extra_rng_consumption_sha256']==saved['extra_rng_consumption_sha256']==control['extra_rng_consumption_sha256']
    parity=read(check(fit['initialization_parity']));assert parity['completed']and parity['PASS']and parity['actual_updates']==0 and parity['geometry_weights_zero']and not parity['available_only_compaction']
    assert all(parity[k]for k in('all51004_old_scores_bitexact','all6144_native_raw_scores_bitexact','all20480_new_raw_scores_bitexact','original_constructed_compact512_batches_and_last_lengths_exact'))
    c16=torch.load(check(p['C16_checkpoint']),map_location='cpu',weights_only=True);initial=torch.load(D/'INITIAL_CHECKPOINT.pt',map_location='cpu',weights_only=True);final=torch.load(check(freeze['checkpoint']),map_location='cpu',weights_only=True)
    assert initial['update']==0 and initial['C16_parent_update']==4500 and not initial['state_dict']['geometry.weight'].any()and final['update']==8000 and not final['state_dict']['geometry.weight'].any()
    assert sum(t.numel()for t in initial['state_dict'].values())==224513 and all(t.dtype==torch.float32 and torch.isfinite(t).all()for t in final['state_dict'].values())
    for key,value in c16['state_dict'].items():state.state_exact(initial['state_dict'][key],value,torch)
    original_final=torch.load(C29/'FINAL_RESUME_CHECKPOINT.pt',map_location='cpu',weights_only=True);state.state_exact(final['RNG'],original_final['RNG'],torch)
    assert not initial['optimizer_state']['state']
    for optimizer in(initial['optimizer_state'],final['optimizer_state']):
        for group in optimizer['param_groups']:assert group['lr']==1e-4 and group['weight_decay']==.001
    assert all(int(s['step'])==8000 for s in final['optimizer_state']['state'].values())
    history=list(old.lines(D/'HISTORY.jsonl'));assert [r['update']for r in history]==list(range(0,8001,100))
    for key,expected in dict(old14=base14,native_VAL=basen['VALIDATION'],new_native_VAL=basee['VALIDATION']).items():a.close(history[0][key],expected)
    for key,expected in dict(old14=old14,native_VAL=nr['VALIDATION'],new_native_VAL=er['VALIDATION']).items():a.close(history[-1][key],expected)
    paired_reports={};selected_metrics={}
    for corpus,meta,pack,bi in(('OLD_NATIVE',om,on,1),('CONTINUOUS_NATIVE',nm,en,2)):
        masks=a.native_masks(meta)if bi==1 else old.new_masks(meta)
        val=meta['split']=='VALIDATION';same=meta['foreign_family']=='SAME_TARGET_MATERIAL';wood=meta['material']=='PROCEDURAL_WOOD';le3=meta['bands']=='LE3_PX'
        masks.update(VALIDATION_SAME_TARGET_MATERIAL=val&same,VALIDATION_SAME_TARGET_MATERIAL_LE3=val&same&le3,
            VALIDATION_PROCEDURAL_WOOD_AND_SAME_TARGET=val&wood&same,VALIDATION_PROCEDURAL_WOOD_AND_SAME_TARGET_LE3=val&wood&same&le3)
        paired_reports[corpus]={name:paired(meta,c29[bi]['prediction'],pack['prediction'],mask)for name,mask in masks.items()}
        selected_metrics[corpus]={name:{model:a.metrics(meta['target'],prediction,mask,meta['state'])for model,prediction in(('C16',(oldbase if bi==1 else newbase)['prediction']),('C29_C30',c29[bi]['prediction']),('C32',pack['prediction']))}for name,mask in masks.items()if name in('VALIDATION','VALIDATION::LE3_PX','VALIDATION_SAME_TARGET_MATERIAL','VALIDATION_SAME_TARGET_MATERIAL_LE3','VALIDATION_PROCEDURAL_WOOD_AND_SAME_TARGET','VALIDATION_PROCEDURAL_WOOD_AND_SAME_TARGET_LE3')}
    fixed=read(check(fit['fixed_cases']));near=next(r for r in fixed['cases']if r['global_index']==18897)
    assert near['prediction']=='E'and nm['state'][18897]=='D'and near['raw_hidden_logit_before_U']==float(en['raw_head_logit_before_U'][18897])
    recalculated=H/'RECALCULATED_SOURCE_RESULTS.json';write(recalculated,dict(completed=True,PASS=True,full_old_native_CF=nr,full_new_native_CF=er,old14=old14,gates64=gates,qualification_split=split,metrics=selected_metrics,
        cohort_note='SAME_TARGET_MATERIAL includes wood and polymer. PROCEDURAL_WOOD_AND_SAME_TARGET is the actual wood intersection. LE3 is diagnosis-only, never a runtime filter.',
        teacher_SU_outside_binary=True,prediction_U_retained_in_full_known_DE=True,source_eligible=eligible,worst_required49_class_recall=worst,fixed_18897=near))
    paired_path=H/'PAIRED_C29_C30_TO_C32_TRANSITIONS.json';write(paired_path,dict(completed=True,baseline_aliases=['C29','C30'],all4_saved_NPZ_packets_C29_C30_byteexact=True,cohorts=paired_reports,
        interpretation='Descriptive paired source counts; same-base variants/slots correlated. No private transfer claim or independent-slot p-values.'))
    receipt=dict(completed=True,PASS=True,code=bind(__file__),control_complete=bind(RUN/'CONTROL_COMPLETE.json'),source_fit=control['arms'][ARM]['fit'],
        arithmetic_helpers=bind(OLDHELP),state_helpers=bind(STATEHELP),recalculated=bind(recalculated),paired=bind(paired_path),trace=trace,
        all51004_6144_20480_rows_ID_q_L_availability_teacher_dtypes_noquery_exact=True,all64_gates49_rank_and_AUX29_native35_split_recomputed=True,
        original_C16_initial_weights_plus_zero_G42_exact=True,initial_optimizer_empty_final8000_and_RNG_exact_C29=True,history81_endpoints_exact=True,
        initial_state_SHA=state.state_hash(initial['state_dict']),final_state_SHA=state.state_hash(final['state_dict']),
        update0_raw_score_bitparity_verified_from_producer_receipt_not_new_forward=True,qualification_split=split,selected_model=None,selected_arm_source_only=None,
        source_only=True,application_solved=False,model_instantiations_forward_fit_encoder_RGB_private_V112_PnP_reads_or_calls=0,CPU_checkpoint_state_inspection=True,seconds=time.monotonic()-start)
    path=H/'REVIEW.json';write(path,receipt);print(json.dumps(dict(completed=True,PASS=True,review=bind(path),split=split,main={corpus:metrics['VALIDATION']['C32']for corpus,metrics in selected_metrics.items()},fixed18897=near)))
if __name__=='__main__':run()
