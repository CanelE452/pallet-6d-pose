"""V137 saved numeric audit. Torch is used solely for CPU RNG regeneration.
No trainer imports, models/checkpoints/optimizer/RGB/features/private/V112 reads.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
RUN=V/'runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000';ARMS=('C28_OLD_NATIVE_EXTRA_REPLAY_CONTROL','C29_CONTINUOUS_NATIVE_DATA_ADDITION')
V136=P/'query_visibility_continuous_native_source_20261024_v136'
HELPER=V136/'independent_saved_review/review_saved_evaluation.py'
spec=importlib.util.spec_from_file_location('V137_independent_numeric_helpers',HELPER);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n);a=n.a
sha,bind,read,check,same,lines,packet,write,close=a.sha,a.bind,a.read,a.check,a.same,a.lines,a.packet,a.write,a.close
MATERIALS=('PROCEDURAL_WOOD','DARK_POLYMER');ASPECTS=('RECTANGULAR','SQUARE');GEOMETRIES=('BOARD_BLOCK_RUNNER','RIB_LEG_RUNNER');FOREIGN=('SAME_TARGET_MATERIAL','CARDBOARD_TAN','BACKGROUND_LIKE','HIGH_CONTRAST')

def new_masks(meta):
    out={}
    for fold in('ALL','TRAIN','VALIDATION'):
        f=np.ones(20480,bool)if fold=='ALL'else meta['split']==fold;out[fold]=f
        out[fold+'::historical_area_agreement']=f&~meta['boundary'];out[fold+'::historical_area_disagreement']=f&meta['boundary']
        for band in n.BANDS:out[fold+'::'+band]=f&(meta['bands']==band)
        for key,values in dict(variant=n.VARIANTS,material=MATERIALS,aspect=ASPECTS,geometry_family=GEOMETRIES,foreign_family=FOREIGN).items():
            for value in values:out[fold+'::'+key+'::'+value]=f&(meta[key]==value)
        for material in MATERIALS:
            for aspect in ASPECTS:out[fold+'::material_aspect::'+material+'::'+aspect]=f&(meta['material']==material)&(meta['aspect']==aspect)
    return out

def new_metrics(scores,meta):return{k:a.metrics(meta['target'],scores['prediction'],mask,meta['state'])for k,mask in new_masks(meta).items()}

def primary_groups():return['VALIDATION']+['VALIDATION::foreign_family::'+x for x in FOREIGN]+['VALIDATION::material_aspect::'+x+'::'+y for x in MATERIALS for y in ASPECTS]+['VALIDATION::geometry_family::'+x for x in GEOMETRIES]

def new_gates(report,baseline):
    gates={}
    for group in primary_groups():
        rr,bb=report.get(group),baseline.get(group)
        for state,row in(('D',0),('E',1)):
            support=0 if rr is None else sum(rr['CM_D_E_rows_D_E_U_columns'][row]);base_support=0 if bb is None else sum(bb['CM_D_E_rows_D_E_U_columns'][row])
            value=None if rr is None else rr[state+'_recall'];baseD=None if bb is None else bb['D_recall'];limit=(None if baseD is None else baseD-.01)if state=='D'else .8
            gates['NEW_NATIVE::'+group+'::'+state]=dict(value=value,limit=limit,N=support,baseline_N=base_support,PASS=bool(support>0 and base_support>0 and value is not None and limit is not None and value>=limit),missing_support=bool(support==0 or base_support==0))
    assert len(gates)==22;return gates

def native_packet(path,baseline,meta,size):
    x=packet(path);assert np.array_equal(x['global_index'],np.arange(size))
    for key in('occurrence_id','query_xy','L','query_available','head_forwarded'):
        left,right=x[key],baseline[key];assert np.array_equal(left,right,equal_nan=True)if left.dtype.kind in'fc'else np.array_equal(left,right)
    raw=x['raw_head_logit_before_U'];assert raw.dtype==np.float32 and raw.shape==(size,)and np.isfinite(raw[x['head_forwarded']]).all()and np.isnan(raw[~x['head_forwarded']]).all()
    assert not(x['query_available']&~x['head_forwarded']).any()
    assert np.array_equal(x['query_logit'],np.where(x['query_available'],raw,0))and np.array_equal(x['prediction'],np.where(x['query_available'],(raw>=0).astype(np.int8),-1))
    for key,expected in dict(raw_teacher_state_CF=meta['state'],external_target_CF=meta['target'],convention_boundary=meta['boundary'],split=meta['split']).items():assert np.array_equal(x[key],expected)
    return x

def pairpool(meta,available,bases):
    pairs=[]
    for base in range(bases):
        for slot in range(8):
            e,d=(base*4+1)*8+slot,(base*4+2)*8+slot
            if meta['state'][e]=='E'and meta['state'][d]=='D'and available[e]and available[d]:
                assert meta['split'][e]==meta['split'][d]=='TRAIN'and meta['rows'][e]['canonical_index']==meta['rows'][d]['canonical_index']
                pairs.append((e,d))
    pairs=np.asarray(pairs,np.int64);assert len(pairs)>0 and len(np.unique(pairs))==2*len(pairs);return pairs

def uniform_pools(meta,available):return[np.flatnonzero((meta['split']=='TRAIN')&available&(meta['target']==t))for t in(0,1)]

def extra_draw(pool,uniform,pg,ug):
    pi=torch.randint(len(pool),(16,),generator=pg).numpy();dd=uniform[0][torch.randint(len(uniform[0]),(16,),generator=ug).numpy()];ee=uniform[1][torch.randint(len(uniform[1]),(16,),generator=ug).numpy()]
    return np.r_[pool[pi,0],pool[pi,1],dd,ee].astype(np.int64),pi

def traces(directory,arm,old,oldnative,newnative,oldmeta,newmeta,oldpairs,newpairs,p):
    oldhist=iter(lines(check(p['historical_C24_trace'])));nativehist=iter(lines(check(p['historical_C26_trace'])))
    pg=torch.Generator().manual_seed(1374);ug=torch.Generator().manual_seed(1372);shared=hashlib.sha256();extra=hashlib.sha256()
    exposure_old=np.zeros(51004,np.int64);exposure_native=np.zeros(6144,np.int64);exposure_new=np.zeros(20480,np.int64)
    addition=arm==ARMS[1];meta=newmeta if addition else oldmeta;target=newnative if addition else oldnative;pool=newpairs if addition else oldpairs;uniform=uniform_pools(meta,target['query_available'])
    count=0
    for count,row in enumerate(lines(directory/'TRAIN_DRAWS_AND_LOSSES.jsonl'),1):
        assert row['update']==count;ix=np.asarray(row['old_occurrences'],np.int64);pp=np.asarray(row['old_matched_cover_sham_positions'],np.int64);ni=np.asarray(row['native_occurrences'],np.int64);npi=np.asarray(row['native_pair_pool_positions'],np.int64);ei=np.asarray(row['extra_native_occurrences'],np.int64);epi=np.asarray(row['extra_pair_pool_positions'],np.int64)
        assert ix.shape==(192,)and pp.shape==(24,2)and ni.shape==ei.shape==(64,)and npi.shape==epi.shape==(16,)
        oh,nh=next(oldhist),next(nativehist);assert oh['update']==nh['update']==count
        for historical in(oh,nh):assert historical['old_occurrences']==ix.tolist()and historical['old_matched_cover_sham_positions']==pp.tolist()
        assert nh['native_occurrences']==ni.tolist()and nh['native_pair_pool_positions']==npi.tolist()
        assert np.array_equal(ni[:16],oldpairs[npi,0])and np.array_equal(ni[16:32],oldpairs[npi,1])
        expected_e,expected_pi=extra_draw(pool,uniform,pg,ug);assert np.array_equal(ei,expected_e)and np.array_equal(epi,expected_pi)
        assert row['extra_source']==('V136_NATIVE'if addition else'OLD_NATIVE')and row['native_hinge_weight']==0
        assert old['query_available'][ix].all()and(old['split'][ix]=='TRAIN').all()and np.bincount(old['external_target'][ix],minlength=2).tolist()==[96,96]
        assert np.array_equal(old['external_target'][ix[pp]],np.tile([1,0],(24,1)))
        for indices,x,m in((ni,oldnative,oldmeta),(ei,target,meta)):
            assert x['query_available'][indices].all()and(m['split'][indices]=='TRAIN').all()and np.array_equal(m['target'][indices],np.r_[np.ones(16),np.zeros(32),np.ones(16)])
        for value in(ix,pp,ni):shared.update(value.astype('<i8').tobytes())
        extra.update(ei.astype('<i8').tobytes());extra.update(epi.astype('<i8').tobytes())
        np.add.at(exposure_old,ix,1);np.add.at(exposure_native,ni,1);np.add.at(exposure_new if addition else exposure_native,ei,1)
        assert all(math.isfinite(row[k])and row[k]>=0 for k in('loss','old_BCE','old_hinge','flat_BCE','native_BCE','extra_native_BCE','gradient_before_clip'))
        loss=np.float32(np.float32(np.float32(row['old_BCE'])+np.float32(.25)*np.float32(row['old_hinge']))+np.float32(row['flat_BCE']))
        loss=np.float32(np.float32(loss+np.float32(row['native_BCE']))+np.float32(row['extra_native_BCE']));assert float(loss)==row['loss'],(count,float(loss),row['loss'])
    assert count==8000 and next(oldhist,None)is None and next(nativehist,None)is None
    saved=packet(directory/'SAMPLING_EXPOSURE.npz');assert np.array_equal(saved['old51004'],exposure_old)and np.array_equal(saved['native6144'],exposure_native)and np.array_equal(saved['new_native20480'],exposure_new)
    assert exposure_old.sum()==8000*192 and exposure_native.sum()+exposure_new.sum()==8000*128
    rng=hashlib.sha256(pg.get_state().numpy().tobytes()+ug.get_state().numpy().tobytes()).hexdigest()
    return dict(rows=8000,shared_sha256=shared.hexdigest(),extra_sha256=extra.hexdigest(),extra_rng_consumption_sha256=rng,
        every_old192_indices24positions_exact_historicalC24_C26=True,every_oldnative64_indices16pairs_exact_historicalC26=True,
        every_extra64_indices_and16pairs_exact_CPU_generators1374_1372=True,all_16k_losses_exact_leftgrouped_FP32=True,
        all_sampled_native_TRAIN_available32D32E_no_SU_VAL=True,sampling_exposure_all51004_6144_20480_exact=True,CPU_rng_only_no_model_or_optimizer=True)

def run():
    assert not(H/'REVIEW.json').exists();torch.set_num_threads(1)
    cp=RUN/'CONTROL_COMPLETE.json';c=read(cp);assert c['completed']and c['PASS']and c['application_solved']is False
    p=read(check(c['protocol']));source=read(check(p['source_binding_policy']));assert torch.__version__==p['runtime']['torch']and np.__version__==p['runtime']['numpy']
    baseline=packet(check(source['C16_source_scores']));oldmasks=a.old_masks(baseline);base14=a.old14(baseline,oldmasks)
    original15=read(a.OLD/'SOURCE_GATE_RESULTS.json')['gates'];oldseal=read(check(source['native_feature_seal']));oldbase=packet(check(oldseal['predictions']));oldmeta=a.native_data()
    newseal=read(check(p['new_feature_seal']));newbase=packet(check(newseal['predictions']));newmeta=n.metadata(check(p['new_GT_ledger']))
    basen=a.native_metrics(oldbase,oldmeta[n.CONV[0]]);basee=new_metrics(newbase,newmeta[n.CONV[0]])
    oldpairs=pairpool(oldmeta[n.CONV[0]],oldbase['query_available'],128);newpairs=pairpool(newmeta[n.CONV[0]],newbase['query_available'],512);assert len(oldpairs)==125
    results={};shared_hash=None;rng_hash=None
    for arm in ARMS:
        directory=RUN/arm;fit=read(check(c['arms'][arm]['fit']));freeze=read(check(c['arms'][arm]['freeze']));saved=read(check(fit['results']))
        assert fit['completed']and fit['optimizer_updates']==8000 and freeze['update']==freeze['selected_update']==8000 and freeze['parameters']==224513 and freeze['threshold_logit']==0
        assert freeze['current_allowed']is False and freeze['V112_allowed']is False and freeze['all_geometry_input_zero']is True
        assert saved['selected_update']==8000 and saved['VAL_checkpoint_choice']is False and saved['threshold_logit']==0
        old=packet(directory/'FINAL_OLD51004_SOURCE_SCORES.npz')
        for key in('query_available','input_unit_index','external_target','raw_teacher_state','split','domain','source','variant'):assert np.array_equal(old[key],baseline[key])
        assert np.array_equal(old['global_occurrence_index'],np.arange(51004))and np.array_equal(old['prediction'],np.where(old['query_available'],(old['query_logit']>=0).astype(np.int8),-1))
        on=native_packet(directory/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz',oldbase,oldmeta[n.CONV[0]],6144)
        en=native_packet(directory/'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz',newbase,newmeta[n.CONV[0]],20480)
        nr=a.native_metrics(on,oldmeta[n.CONV[0]]);v4=a.native_metrics(on,oldmeta[n.CONV[1]]);er=new_metrics(en,newmeta[n.CONV[0]]);ev4=new_metrics(en,newmeta[n.CONV[1]])
        for key,expected in dict(native_CF=nr,native_V4_sensitivity_only=v4,new_native_CF=er,new_native_obsolete_area_V4_sensitivity_only=ev4).items():close(expected,saved[key],key)
        gates,old14=a.gate_values(old,baseline,base14,original15,nr,basen,oldmasks);gates.update(new_gates(er,basee));assert len(gates)==64
        close(old14,saved['old']['old14'])
        for key,expected in gates.items():
            for field,value in expected.items():close(value,saved['gates'][key][field],key+'/'+field)
        assert set(gates)==set(saved['gates'])
        z=old['query_logit'].astype(np.float64);y=np.where(np.isin(old['external_target'],[0,1]),old['external_target'],0);loss=np.logaddexp(0,z)-y*z
        parts={k:dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~old['query_available']).sum()))for k,mask in oldmasks.items()}
        close(parts,saved['old']['old14_BCE_parts']);close(float(np.mean([v['mean_BCE']for v in parts.values()])),saved['old']['old14_BCE'])
        for key,metric in saved['old']['domain_metrics'].items():
            domain,fold=key.split('::');close(a.metrics(old['external_target'],old['prediction'],(old['domain']==domain)&(old['split']==fold),old['raw_teacher_state']),metric)
        trace=traces(directory,arm,old,on,en,oldmeta[n.CONV[0]],newmeta[n.CONV[0]],oldpairs,newpairs,p)
        assert trace['shared_sha256']==saved['shared_old192_native64_sha256']==fit['shared_draw_sha256']==c['shared_draw_sha256']
        assert trace['extra_sha256']==saved['extra_native_draw_sha256']and trace['extra_rng_consumption_sha256']==saved['extra_rng_consumption_sha256']==c['extra_rng_consumption_sha256']
        if shared_hash is None:shared_hash,rng_hash=trace['shared_sha256'],trace['extra_rng_consumption_sha256']
        else:assert shared_hash==trace['shared_sha256']and rng_hash==trace['extra_rng_consumption_sha256']
        parity=read(check(fit['initialization_parity']));assert parity['completed']and parity['PASS']and parity['actual_updates']==0 and parity['geometry_weights_zero']
        assert all(parity[k]for k in('all51004_old_scores_bitexact','all6144_native_raw_scores_bitexact','all20480_new_raw_scores_bitexact','original_constructed_compact512_batches_and_last_lengths_exact'))and parity['available_only_compaction']is False
        history=list(lines(directory/'HISTORY.jsonl'));assert [r['update']for r in history]==list(range(0,8001,100))
        for key,expected in dict(old14=base14,native_VAL=basen['VALIDATION'],new_native_VAL=basee['VALIDATION']).items():close(history[0][key],expected)
        for key,expected in dict(old14=old14,native_VAL=nr['VALIDATION'],new_native_VAL=er['VALIDATION']).items():close(history[-1][key],expected)
        rank=[g['value']for k,g in gates.items()if not k.startswith('HISTORICAL15::')];assert len(rank)==49;worst=None if any(v is None for v in rank)else min(rank);eligible=all(g['PASS']for g in gates.values())
        assert eligible==saved['source_eligible']==fit['source_eligible']==freeze['source_eligible']==c['arms'][arm]['source_eligible']
        assert worst==saved['worst_required49_class_recall']==fit['worst_required49_class_recall']==c['arms'][arm]['worst_required49_class_recall']
        review=dict(completed=True,PASS=True,source_eligible=eligible,gates64_recomputed=gates,gates_pass=sum(g['PASS']for g in gates.values()),failed_gates=[k for k,g in gates.items()if not g['PASS']],worst_required49_class_recall=worst,
            native_VAL=nr['VALIDATION'],new_native_VAL=er['VALIDATION'],new_native_full_CF=er,new_native_obsoleteV4=ev4,trace=trace,update0_bound_receipt_verified_not_model_replayed=fit['initialization_parity'],
            all51004_6144_20480_ID_q_L_teacher_and_predictionU_preserved=True,all_source14_BCE_and64gates_exact=True,history81_endpoints_and_updatecounts_verified=True)
        write(H/(arm+'_RECALCULATED.json'),review);results[arm]=review
    eligible=[arm for arm in ARMS if results[arm]['source_eligible']];chosen=max(eligible,key=lambda arm:(results[arm]['worst_required49_class_recall'],-ARMS.index(arm)))if eligible else None
    assert chosen==c['selected_arm_source_only']
    review=dict(completed=True,PASS=True,code=bind(__file__),independent_math_helper=bind(HELPER),control_complete=bind(cp),scope='Saved NumPy/JSON arithmetic and CPU Torch RNG only. No trainer/model/checkpoint/optimizer/RGB/features/private/V112/PnP reads.',
        arms={arm:dict(source_eligible=r['source_eligible'],gates_pass=r['gates_pass'],gates_total=64,failed_gates=r['failed_gates'],worst_required49_class_recall=r['worst_required49_class_recall'],recalculated=bind(H/(arm+'_RECALCULATED.json')))for arm,r in results.items()},
        selected_arm_source_only=chosen,all64gates_rank49_fullknown_denominators_verified=True,all8000_shared_historical_and_extra_seed_draws_exact=True,all16k_loss_FP32_sums_and_exposures_exact=True,
        update0_model_parity_SU_gradient_and_weights_are_producer_receipts_not_independent_model_replay=True,goal_status='active',fully_solved=False,application_solved=False)
    write(H/'REVIEW.json',review);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),arms=review['arms'],selected=chosen)),flush=True)
if __name__=='__main__':run()
