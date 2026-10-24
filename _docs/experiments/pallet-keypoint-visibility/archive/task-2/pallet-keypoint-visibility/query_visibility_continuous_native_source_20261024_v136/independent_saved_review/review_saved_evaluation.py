"""V136 independent saved arithmetic. No producer/trainer imports or RGB/model I/O."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import collections,hashlib,importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;N=20480
OUT=V/'saved_frozen_visibility_evaluation_v1';DIAG=V/'saved_initializer_diagnosis_v1'
MODELS={'C16':'frozen_C16_actual_initializer_v1','v61':'frozen_v61_actual_initializer_v1'}
CONV=('CF_PRIMARY_DECLARED','V4_SENSITIVITY_ONLY');BANDS=('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY');VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
A133=P/'query_visibility_native_geometry_control_20261024_v133/independent_saved_review/review_saved_results.py'
spec=importlib.util.spec_from_file_location('V136_independent_old_arithmetic',A133);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
sha,bind,read,check,same,lines,packet,write,close=a.sha,a.bind,a.read,a.check,a.same,a.lines,a.packet,a.write,a.close

def metadata(path):
    rr=list(lines(path));assert len(rr)==2*N;out={}
    for j,c in enumerate(CONV):
        rows=rr[j::2];state=np.asarray([r['teacher_state']for r in rows]);assert np.isin(state,list('DESU')).all()
        meta=dict(rows=rows,state=state,target=np.where(state=='D',0,np.where(state=='E',1,-1)),split=np.asarray([r['split']for r in rows]),
            boundary=np.asarray([r['correspondence_boundary']for r in rows],bool),bands=np.asarray([r['position_band']for r in rows]),variant=np.asarray([r['variant']for r in rows]))
        for key in('material','aspect','geometry_family','foreign_family'):meta[key]=np.asarray([r['factors'][key]for r in rows])
        for i,r in enumerate(rows):
            assert r['convention']==c and r['scene_index']==i//8 and r['base_index']==i//32 and r['native_slot']==i%8 and r['query_id']==f'V136_SCENE_{i//8:04d}:native{i%8}'
            assert r['split']==('TRAIN'if i//32<512 else'VALIDATION')and r['variant']==VARIANTS[(i//8)%4]
            q=r['query_xy'];cont=False if q is None else bool((np.asarray(q)>=0).all()and(np.asarray(q)<512).all());near=False if q is None else bool((np.floor(np.asarray(q)+.5)>=0).all()and(np.floor(np.asarray(q)+.5)<512).all())
            assert cont==r['continuous_inbounds']and near==r['nearest_inbounds']
            err=None if q is None else float(np.linalg.norm(np.asarray(q)-np.asarray(r['reference_xy'])))
            assert err==r['error_px'];assert r['position_band']==('NO_QUERY'if err is None else'LE3_PX'if err<=3 else'GT3_LE8_PX'if err<=8 else'GT8_LE20_PX'if err<=20 else'GT20_PX')
        out[c]=meta
    return out

def masks(meta):
    groups={}
    for fold in('ALL','TRAIN','VALIDATION'):
        f=np.ones(N,bool)if fold=='ALL'else meta['split']==fold;groups[fold]=f
        for band in BANDS:groups[fold+'::'+band]=f&(meta['bands']==band)
        for flag in(False,True):
            bb=f&(meta['boundary']==flag);groups[fold+'::boundary'+str(flag)]=bb
            for band in BANDS:groups[fold+'::boundary'+str(flag)+'::'+band]=bb&(meta['bands']==band)
        for variant in VARIANTS:groups[fold+'::'+variant]=f&(meta['variant']==variant)
        for key in('material','aspect','geometry_family','foreign_family'):
            for value in sorted(np.unique(meta[key][f])):groups[fold+'::'+key+'::'+str(value)]=f&(meta[key]==value)
    return groups

def metric(meta,pred,av,mask,denominator=None):
    known=mask&np.isin(meta['target'],[0,1]);used=known if denominator is None else known&denominator
    cm=np.asarray([[np.count_nonzero(used&(meta['target']==t)&(pred==v))for v in(0,1,-1)]for t in(0,1)])
    div=lambda n,d:float(n/d)if d else None
    count=lambda vals:dict(collections.Counter(vals.tolist()))
    names=np.asarray(['U','D','E'])[pred.astype(np.int64)+1]
    return dict(CM_rows_D_E_cols_D_E_U=cm.tolist(),known_DE_original=int(known.sum()),known_DE_in_this_denominator=int(used.sum()),original_rows=int(mask.sum()),
        teacher_states=count(meta['state'][mask]),D_precision=div(cm[0,0],cm[:,0].sum()),D_recall=div(cm[0,0],cm[0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),E_recall=div(cm[1,1],cm[1].sum()),
        prediction_U_known=int(cm[:,2].sum()),prediction_U_rate_known=div(cm[:,2].sum(),used.sum()),teacher_S_prediction_counts=count(names[mask&(meta['state']=='S')]),
        teacher_U_prediction_counts=count(names[mask&(meta['state']=='U')]),all_teacher_states_model_available=int((mask&av).sum()),all_teacher_states_model_U=int((mask&~av).sum()))

def expected_selection(meta,pred,av):
    valid=(meta['split']=='VALIDATION')&(meta['bands']=='LE3_PX')&av
    pools={s:np.flatnonzero(valid&(meta['state']==s)&(pred!=(0 if s=='D'else 1))).tolist()for s in('E','D')}
    failures=[];bases=set();missing=[]
    for rank in range(4):
        for state in('E','D'):
            chosen=next((i for i in pools[state]if i//32 not in bases),None)
            if chosen is not None and len(failures)<4:failures.append(chosen);bases.add(chosen//32)
    cases=[]
    for state in('E','D'):
        if not pools[state]:missing.append('C16_'+state+'_failure_LE3_CF_PRIMARY')
    for i in failures:
        state=meta['state'][i];p={0:'D',1:'E',-1:'U'}[int(pred[i])];base=i//32;slot=i%8
        cases.append(dict(category='C16_'+state+'_AS_'+p+'_LE3_CF_PRIMARY',global_index=i,base_index=base,native_slot=slot,
            obsolete_area_rule_disagreement=bool(meta['boundary'][i]),companion_global_indices_same_base_slot=[(base*4+k)*8+slot for k in range(4)]))
        candidates=[j for j in range(base*32,(base+1)*32)if j%8==slot and meta['variant'][j]!=meta['variant'][i]and av[j]and meta['state'][j]in('D','E')and pred[j]==meta['target'][j]]
        success=next((j for j in candidates if valid[j]),None)
        if success is None:success=next(iter(candidates),None)
        if success is None:missing.append('same_base_slot_success_for_'+str(i))
        else:cases.append(dict(category='C16_SAME_BASE_SLOT_SUCCESS_'+meta['state'][success],global_index=success,base_index=base,native_slot=slot,paired_failure_global_index=i,
            success_also_LE3=meta['bands'][success]=='LE3_PX',obsolete_area_rule_disagreement=bool(meta['boundary'][success])))
    if not failures:
        for state,t in(('D',0),('E',1)):
            ii=np.flatnonzero(valid&(meta['state']==state)&(pred==t))
            if len(ii):
                i=int(ii[0]);cases.append(dict(category='C16_SUCCESS_'+state+'_LE3_CF_PRIMARY',global_index=i,base_index=i//32,native_slot=i%8,obsolete_area_rule_disagreement=bool(meta['boundary'][i])))
    return cases,missing,{s:len(p)for s,p in pools.items()}

def run():
    assert not(H/'REVIEW.json').exists()
    sealpath=OUT/'EVALUATION_SEAL.json';seal=read(sealpath);assert seal['completed']and seal['PASS']and seal['RGB_read']is False
    for b in seal['outputs']:check(b)
    results=read(check(seal['results']));selection=read(check(seal['selection']));check(results['protocol']);check(results['code'])
    ds=read(check(seal['diagnosis_seal']));assert ds['completed']and ds['PASS']and ds['RGB_read']is False
    gt=next(b for b in ds['outputs']if Path(b['path']).name=='ALL20480_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl');metas=metadata(check(gt));arrays={};seals={}
    for model,folder in MODELS.items():
        s=read(check(seal[model+'_prediction_seal']));assert s['completed']and s['PASS']and s['GT_read']is False and s['rows']==N and s['all2560_scenes20480_slots_preserved']
        assert s['no_model_or_threshold_choice']is True if model=='C16'else s['no_model_threshold_or_calibration_choice']is True
        assert same(s['initializer_seal'],ds['prediction_seal']);seals[model]=s;x=packet(check(s['predictions']));av=x['query_available'];forward=x['head_forwarded']
        assert av.dtype==forward.dtype==bool and av.shape==forward.shape==(N,)and not(av&~forward).any()
        for k,expected in dict(global_index=np.arange(N),source_occurrence_index=np.arange(N),scene_index=np.arange(N)//8,base_index=np.arange(N)//32,native_slot=np.arange(N)%8).items():assert np.array_equal(x[k],expected)
        assert np.array_equal(x['occurrence_id'],np.asarray([f'V136_SCENE_{i//8:04d}:native{i%8}'for i in range(N)]))
        raw=x['raw_head_logit_before_U']if model=='C16'else x['raw_hidden_logit'];pred=x['prediction']if model=='C16'else x['prediction_D0_E1_Uminus1']
        expected=np.where(av,(raw>=0).astype(np.int8),-1)if model=='C16'else np.where(av,(x['raw_visible_logit']<0).astype(np.int8),-1)
        assert np.array_equal(pred,expected)and raw.dtype==np.float32 and np.isfinite(raw[forward]).all()and np.isnan(raw[~forward]).all()
        if model=='C16':assert np.array_equal(av,x['factual_query_available']&x['context_available'])and np.array_equal(x['query_logit'],np.where(av,raw,0))
        else:assert np.array_equal(raw,-x['raw_visible_logit'],equal_nan=True)and np.array_equal(av,x['native_query_valid']&x['ROI_any_valid'])
        arrays[model]=dict(data=x,raw=raw,pred=pred,av=av)
    for key in('global_index','source_occurrence_index','scene_index','base_index','native_slot','occurrence_id','query_xy','L','no_detection','native_continuous_valid','native_nearest_valid'):
        x,y=arrays['C16']['data'][key],arrays['v61']['data'][key];assert np.array_equal(x,y,equal_nan=True)if x.dtype.kind in'fc'else np.array_equal(x,y)
    shared=arrays['C16']['av']&arrays['v61']['av'];computed={}
    assert set(results['summary'])==set(metas)
    for c,meta in metas.items():
        assert set(results['summary'][c])==set(masks(meta))
        for i,r in enumerate(meta['rows']):
            q=[np.nan,np.nan]if r['query_xy']is None else r['query_xy'];side=np.nan if r['roi_side']is None else r['roi_side']
            for x in arrays.values():
                assert np.array_equal(x['data']['query_xy'][i],q,equal_nan=True)and(np.isnan(side)and np.isnan(x['data']['L'][i])or side==x['data']['L'][i])
        computed[c]={}
        for group,mask in masks(meta).items():
            computed[c][group]={}
            for model,x in arrays.items():
                d=dict(all_known_DE_original_denominator=metric(meta,x['pred'],x['av'],mask),own_available_only=metric(meta,x['pred'],x['av'],mask,x['av']),shared_available_only=metric(meta,x['pred'],x['av'],mask,shared))
                close(d,results['summary'][c][group][model],c+'/'+group+'/'+model);computed[c][group][model]=d
    meta=metas[CONV[0]];cases,missing,counts=expected_selection(meta,arrays['C16']['pred'],arrays['C16']['av'])
    close(cases,selection['cases'],'selection');close(missing,selection['missing_categories']);close(counts,selection['failure_counts_eligible'])
    assert selection['obsolete_area_disagreement_is_annotation_only']and selection['CF_primary_all_configs_certified']and selection['max_cases']==8
    scores=list(lines(check(seal['scores'])));assert len(scores)==N
    for i,r in enumerate(scores):
        g=meta['rows'][i];assert r['global_index']==i and r['scene_index']==i//8 and r['base_index']==i//32 and r['native_slot']==i%8 and r['raw_state']==g['teacher_state']and r['canonical_index']==g['canonical_index']
        assert r['applied_convention']==CONV[0]and r['query_xy']==g['query_xy']and r['roi_side']==g['roi_side']and r['error_px']==g['error_px']and r['convention_boundary']==g['correspondence_boundary']
        assert r['shared_available']==bool(shared[i])
        for key in('id','base_id','query_id','reference_xy','position_band','input_status','factors','raw_RGB_PNG'):assert r[key]==g[key]
        for model,x in arrays.items():
            mr=r[model];assert mr['prediction']=={-1:'U',0:'D',1:'E'}[int(x['pred'][i])]and mr['available']==bool(x['av'][i])and mr['head_forwarded']==bool(x['data']['head_forwarded'][i])
            assert mr['raw_logit']==(float(x['raw'][i])if np.isfinite(x['raw'][i])else None)
    write(H/'RECALCULATED_ALL_COHORTS.json',computed)
    review=dict(completed=True,PASS=True,code=bind(__file__),helper=bind(A133),evaluation_seal=bind(sealpath),all20480_ID_q_L_availability_threshold_polarity_verified=True,
        all_CF_obsoleteV4_full_own_shared_CM_precision_recall_U_exact=True,teacher_SU_separate_known_predictionU_kept=True,
        gallery_all_CF_no_area_exclusion_exact=True,selection_cases=len(cases),all20480_savedscore_rows_exact=True,
        CF_VAL=computed[CONV[0]]['VALIDATION'],recalculated=bind(H/'RECALCULATED_ALL_COHORTS.json'),model_optimizer_checkpoint_RGB_feature_private_V112_PnP_reads=0,
        goal_status='active',fully_solved=False,limitations='Saved arithmetic only; producer model parity, weights and image fidelity are receipts, not independently replayed.')
    write(H/'REVIEW.json',review);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),CF_VAL=review['CF_VAL'])),flush=True)
if __name__=='__main__':run()
