"""Terminal-only saved C30/C31 arithmetic and CPU state replay audit.

No trainer/model instantiation, forward, fit, RGB, visual-feature, private,
V112 or PnP access. torch.load is CPU weights-only state inspection.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
RUN=V/'runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000'
ARMS=('C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL','C31_BROAD_SOURCE_NATIVE_G42')
OLDV=P/'query_visibility_continuous_source_coverage_control_20261024_v137'
OLDHELP=OLDV/'independent_saved_review/review_saved_controls.py'
C29=OLDV/'runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/C29_CONTINUOUS_NATIVE_DATA_ADDITION'
OUT=V/'independent_saved_review'
FILES=('FINAL_OLD51004_SOURCE_SCORES.npz','FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz','FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz','SAMPLING_EXPOSURE.npz')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def exact_array(a,b):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def exact_packet(left,right):
    assert set(left)==set(right)
    for key in left:exact_array(left[key],right[key])
def state_exact(a,b,torch):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor);exact_array(a.detach().cpu().contiguous().numpy(),b.detach().cpu().contiguous().numpy())
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        for k in a:state_exact(a[k],b[k],torch)
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):state_exact(x,y,torch)
    else:assert type(a)==type(b) and a==b
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def paired(meta,p0,p1,mask):
    truth=meta['target'];out={}
    for state in ('DE','D','E','S','U'):
        cohort=mask&(np.isin(meta['state'],list('DE')) if state=='DE' else meta['state']==state)
        if state in ('S','U'):
            cm=np.array([[int((cohort&(p0==x)&(p1==y)).sum()) for y in (0,1,-1)] for x in (0,1,-1)])
            out[state]=dict(rows=int(cohort.sum()),prediction_transition_C30rows_C31columns_D_E_U=cm.tolist(),outside_external_binary=True);continue
        left=p0==truth;right=p1==truth;recovered=cohort&~left&right;regressed=cohort&left&~right
        base_delta={str(int(base)):int(recovered[meta['base']==base].sum()-regressed[meta['base']==base].sum()) for base in np.unique(meta['base'][cohort])}
        out[state]=dict(rows=int(cohort.sum()),correct_both=int((cohort&left&right).sum()),wrong_both=int((cohort&~left&~right).sum()),recovered=int(recovered.sum()),regressed=int(regressed.sum()),net_correct_change=int(recovered.sum()-regressed.sum()),C30_U=int((cohort&(p0==-1)).sum()),C31_U=int((cohort&(p1==-1)).sum()),base_groups=len(base_delta),bases_improved=sum(v>0 for v in base_delta.values()),bases_regressed=sum(v<0 for v in base_delta.values()),bases_tied=sum(v==0 for v in base_delta.values()),correlated_base_net_correct_changes=base_delta)
    return out

def run():
    start=time.monotonic();cp=RUN/'CONTROL_COMPLETE.json'
    # This is the first body read: refuse partial or absent fit outputs.
    c=read(cp);assert c['completed'] and c['PASS'] and c['final_update8000_both'] and c['only_difference_native_geometry_ZERO_vs_G42']
    assert not OUT.exists();p=read(checked(c['protocol']));checked(c['code']);checked(c['policy'])
    assert sha(OLDHELP)=='8cb5c878c79aaf3a09e74c4163f45f17ae5bd487459f747d41204587a0c0465c'
    spec=importlib.util.spec_from_file_location('v139_independent_prior_arithmetic',OLDHELP);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    a,n,torch=old.a,old.n,old.torch;torch.set_num_threads(1)
    assert torch.__version__==p['runtime']['torch'] and np.__version__==p['runtime']['numpy']
    source=read(checked(p['source_binding_policy']));baseline=a.packet(checked(source['C16_source_scores']));oldmasks=a.old_masks(baseline);base14=a.old14(baseline,oldmasks);original15=read(a.OLD/'SOURCE_GATE_RESULTS.json')['gates']
    oldseal=read(checked(source['native_feature_seal']));oldbase=a.packet(checked(oldseal['predictions']));oldmeta=a.native_data();newseal=read(checked(p['new_feature_seal']));newbase=a.packet(checked(newseal['predictions']));newmeta=n.metadata(checked(p['new_GT_ledger']))
    old.baseline_packet_checks(oldbase,oldmeta[n.CONV[0]],6144,'V129');old.baseline_packet_checks(newbase,newmeta[n.CONV[0]],20480,'V136')
    basen=a.native_metrics(oldbase,oldmeta[n.CONV[0]]);basee=old.new_metrics(newbase,newmeta[n.CONV[0]])
    oldpairs=old.pairpool(oldmeta[n.CONV[0]],oldbase['query_available'],128);newpairs=old.pairpool(newmeta[n.CONV[0]],newbase['query_available'],512);assert len(oldpairs)==125 and len(newpairs)==473
    # Complete C30 all-array replay, including NaN payloads and signed zero.
    for filename in FILES:exact_packet(a.packet(RUN/ARMS[0]/filename),a.packet(C29/filename))
    for filename in ('TRAIN_DRAWS_AND_LOSSES.jsonl','HISTORY.jsonl'):assert sha(RUN/ARMS[0]/filename)==sha(C29/filename)
    original=torch.load(checked(p['historical_C29_checkpoint']),map_location='cpu',weights_only=True)
    zero=torch.load(RUN/ARMS[0]/'FINAL_RESUME_CHECKPOINT.pt',map_location='cpu',weights_only=True)
    compared=('state_dict','optimizer_state','RNG','update','C16_parent_update','threshold_logit')
    for key in compared:state_exact(zero[key],original[key],torch)
    C16=torch.load(checked(p['C16_checkpoint']),map_location='cpu',weights_only=True);statehashes={};armresults={};packets={};OUT.mkdir()
    for index,arm in enumerate(ARMS):
        directory=RUN/arm;fit=read(checked(c['arms'][arm]['fit']));freeze=read(checked(c['arms'][arm]['freeze']));saved=read(checked(fit['results']))
        assert fit['completed'] and fit['optimizer_updates']==freeze['update']==freeze['selected_update']==saved['selected_update']==8000 and freeze['parameters']==224513 and freeze['threshold_logit']==saved['threshold_logit']==0
        assert freeze['current_allowed'] is False and freeze['V112_allowed'] is False and freeze['native_original_G42_input']==bool(index) and freeze['all_geometry_input_zero']==(not index)
        parity=read(checked(fit['initialization_parity']));assert parity['completed'] and parity['PASS'] and parity['actual_updates']==0 and parity['geometry_weights_zero'] and not parity['available_only_compaction']
        for key in ('all51004_old_scores_bitexact','all6144_native_raw_scores_bitexact','all20480_new_raw_scores_bitexact','original_constructed_compact512_batches_and_last_lengths_exact'):assert parity[key]
        initial=torch.load(directory/'INITIAL_CHECKPOINT.pt',map_location='cpu',weights_only=True)
        assert initial['update']==0 and initial['C16_parent_update']==4500 and not initial['state_dict']['geometry.weight'].any()
        for key,value in C16['state_dict'].items():state_exact(initial['state_dict'][key],value,torch)
        final=torch.load(checked(freeze['checkpoint']),map_location='cpu',weights_only=True);assert final['update']==8000;state_exact(final['RNG'],original['RNG'],torch)
        if index:assert final['state_dict']['geometry.weight'].abs().sum()>0
        else:assert not final['state_dict']['geometry.weight'].any()
        statehashes[arm]=dict(initial=state_hash(initial['state_dict']),final=state_hash(final['state_dict']))
        op=a.packet(directory/FILES[0]);on=old.native_packet(directory/FILES[1],oldbase,oldmeta[n.CONV[0]],6144);en=old.native_packet(directory/FILES[2],newbase,newmeta[n.CONV[0]],20480)
        for key in ('query_available','input_unit_index','external_target','raw_teacher_state','split','domain','source','variant'):assert np.array_equal(op[key],baseline[key])
        assert np.array_equal(op['global_occurrence_index'],np.arange(51004)) and np.array_equal(op['prediction'],np.where(op['query_available'],op['query_logit']>=0,-1))
        nr=a.native_metrics(on,oldmeta[n.CONV[0]]);v4=a.native_metrics(on,oldmeta[n.CONV[1]]);er=old.new_metrics(en,newmeta[n.CONV[0]]);ev4=old.new_metrics(en,newmeta[n.CONV[1]])
        for key,value in dict(native_CF=nr,native_V4_sensitivity_only=v4,new_native_CF=er,new_native_obsolete_area_V4_sensitivity_only=ev4).items():a.close(value,saved[key],key)
        gates,old14=a.gate_values(op,baseline,base14,original15,nr,basen,oldmasks);gates.update(old.new_gates(er,basee));assert len(gates)==64 and set(gates)==set(saved['gates']);a.close(old14,saved['old']['old14'])
        for key,value in gates.items():
            for field,expected in value.items():a.close(expected,saved['gates'][key][field],key+'/'+field)
        z=op['query_logit'].astype(np.float64);y=np.where(np.isin(op['external_target'],[0,1]),op['external_target'],0);loss=np.logaddexp(0,z)-y*z
        parts={k:dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~op['query_available']).sum())) for k,mask in oldmasks.items()};a.close(parts,saved['old']['old14_BCE_parts']);a.close(float(np.mean([v['mean_BCE'] for v in parts.values()])),saved['old']['old14_BCE'])
        for key,metric in saved['old']['domain_metrics'].items():
            domain,fold=key.split('::');a.close(a.metrics(op['external_target'],op['prediction'],(op['domain']==domain)&(op['split']==fold),op['raw_teacher_state']),metric)
        trace=old.traces(directory,old.ARMS[1],op,on,en,oldmeta[n.CONV[0]],newmeta[n.CONV[0]],oldpairs,newpairs,p)
        assert trace['shared_sha256']==saved['shared_old192_native64_sha256']==c['shared_draw_sha256']==p['historical_C29_draw_hash'] and trace['extra_sha256']==saved['extra_native_draw_sha256']==p['historical_C29_extra_draw_hash']
        history=list(a.lines(directory/'HISTORY.jsonl'));assert [r['update'] for r in history]==list(range(0,8001,100))
        for key,expected in dict(old14=base14,native_VAL=basen['VALIDATION'],new_native_VAL=basee['VALIDATION']).items():a.close(history[0][key],expected)
        for key,expected in dict(old14=old14,native_VAL=nr['VALIDATION'],new_native_VAL=er['VALIDATION']).items():a.close(history[-1][key],expected)
        rank=[g['value'] for k,g in gates.items() if not k.startswith('HISTORICAL15::')];assert len(rank)==49;worst=None if any(v is None for v in rank) else min(rank);eligible=all(g['PASS'] for g in gates.values())
        assert worst==saved['worst_required49_class_recall']==c['arms'][arm]['worst_required49_class_recall'] and eligible==saved['source_eligible']==fit['source_eligible']==freeze['source_eligible']==c['arms'][arm]['source_eligible']
        rec=dict(completed=True,PASS=True,source_eligible=eligible,gates_pass=sum(g['PASS'] for g in gates.values()),gates_total=64,gates64_recomputed=gates,worst_required49_class_recall=worst,failed_gates=[k for k,g in gates.items() if not g['PASS']],native_VAL=nr['VALIDATION'],new_native_VAL=er['VALIDATION'],full_native_CF=nr,full_new_native_CF=er,trace=trace,update0_parity_producer_receipt_not_forward_replayed=fit['initialization_parity'])
        path=OUT/(arm+'_RECALCULATED.json');a.write(path,rec);armresults[arm]=dict(source_eligible=eligible,gates_pass=rec['gates_pass'],worst_required49_class_recall=worst,native_VAL=nr['VALIDATION'],new_native_VAL=er['VALIDATION'],recalculated=bind(path));packets[arm]=(op,on,en)
    assert statehashes[ARMS[0]]['initial']==statehashes[ARMS[1]]['initial']
    eligible=[arm for arm in ARMS if armresults[arm]['source_eligible']];chosen=max(eligible,key=lambda arm:(armresults[arm]['worst_required49_class_recall'],-ARMS.index(arm))) if eligible else None;assert chosen==c['selected_arm_source_only']
    pairedresults={}
    for corpus,metadata,packetindex in (('OLD_NATIVE',oldmeta[n.CONV[0]],1),('CONTINUOUS_NATIVE',newmeta[n.CONV[0]],2)):
        meta=dict(metadata);meta['base']=np.array([r['base_index'] for r in meta['rows']]);groups=old.new_masks(meta) if packetindex==2 else a.native_masks(meta)
        pairedresults[corpus]={group:paired(meta,packets[ARMS[0]][packetindex]['prediction'],packets[ARMS[1]][packetindex]['prediction'],mask) for group,mask in groups.items()}
    pairedpath=OUT/'PAIRED_C30_C31_FULL_COHORT_TRANSITIONS.json';a.write(pairedpath,dict(completed=True,source_only=True,statistical_inference='Descriptive paired counts only; same-base variants and slots are correlated. No per-slot independence p-values or threshold choice.',cohorts=pairedresults))
    receipt=dict(completed=True,PASS=True,code=bind(Path(__file__)),control_complete=bind(cp),prior_independent_arithmetic_helpers=bind(OLDHELP),arms=armresults,selected_arm_source_only=chosen,all64_gates49_rank_recomputed=True,C30_C29_all4_NPZ_packets_byte_exact=True,C30_C29_all8000_draw_loss_trace_and81history_lines_byte_exact=True,C30_C29_final_state_optimizer_RNG_update_parent_threshold_byte_exact=True,both_initial_state_exact_original_C16_plus_zero_geometry=True,state_hashes=statehashes,both_all8000_sampler_draws_seeds_losses_FP32_and_exposures_recomputed=True,paired_full_cohorts=bind(pairedpath),update0_raw_prediction_parity_is_bound_producer_receipt_not_new_model_forward=True,CPU_checkpoint_state_inspection=True,actual_model_instantiations_or_forwards_fit_RGB_visual_features_private_V112_PnP_reads_or_calls=0,source_only=True,application_solved=False,seconds=time.monotonic()-start)
    path=OUT/'REVIEW.json';a.write(path,receipt);print(json.dumps(dict(completed=True,PASS=True,review=bind(path),arms=armresults,selected=chosen,seconds=receipt['seconds'])),flush=True)
if __name__=='__main__':run()
