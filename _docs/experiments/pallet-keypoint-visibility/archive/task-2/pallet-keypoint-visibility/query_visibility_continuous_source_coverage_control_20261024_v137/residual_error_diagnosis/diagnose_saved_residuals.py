"""Fixed saved source residuals; no model/fit/image/private/coordinate filtering."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import collections,hashlib,importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
HELPER=P/'query_visibility_continuous_native_source_20261024_v136/independent_saved_review/review_saved_evaluation.py'
spec=importlib.util.spec_from_file_location('residual_independent_saved_math',HELPER);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n);a=n.a
sha,bind,read,check,lines,packet,write=a.sha,a.bind,a.read,a.check,a.lines,a.packet,a.write
RUN=V/'runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000';ARMS={'C28':'C28_OLD_NATIVE_EXTRA_REPLAY_CONTROL','C29':'C29_CONTINUOUS_NATIVE_DATA_ADDITION'}

def quantile(x):
    v=np.asarray(x);v=v[np.isfinite(v)];return np.quantile(v,[0,.25,.5,.75,.9,.95,1]).tolist()if len(v)else None

def counts(meta,pred,mask):
    known=mask&np.isin(meta['target'],[0,1]);dd=known&(meta['target']==0)&(pred==1);ed=known&(meta['target']==1)&(pred==0);uu=known&(pred==-1);flip=dd|ed;fail=flip|uu
    error=np.asarray([np.nan if r['error_px']is None else r['error_px']for r in meta['rows']]);high=np.isfinite(error)&(error>3);near=np.isfinite(error)&(error<=3)
    fraction=lambda x,y:float(x/y)if y else None
    out=dict(a.metrics(meta['target'],pred,mask,meta['state']),true_D_pred_E=int(dd.sum()),true_E_pred_D=int(ed.sum()),known_prediction_U=int(uu.sum()),D_E_classification_errors=int(flip.sum()),all_known_failures_including_U=int(fail.sum()),
        q_GT3_classification_errors=int((flip&high).sum()),q_GT3_fraction_of_classification_errors=fraction((flip&high).sum(),flip.sum()),
        q_LE3_classification_errors=int((flip&near).sum()),q_unknown_classification_errors=int((flip&~np.isfinite(error)).sum()),
        q_GT3_all_known_failures=int((fail&high).sum()),q_GT3_fraction_of_all_known_failures=fraction((fail&high).sum(),fail.sum()),
        q_LE3_all_known_failures=int((fail&near).sum()),q_unknown_all_known_failures=int((fail&~np.isfinite(error)).sum()),
        DfalseE_q_GT3=int((dd&high).sum()),EfalseD_q_GT3=int((ed&high).sum()),inputU_q_GT3=int((uu&high).sum()))
    return out

def cohorts(meta):
    groups={};va=meta['split']=='VALIDATION'
    for fold in('ALL','TRAIN','VALIDATION'):groups[fold]=np.ones(len(meta['state']),bool)if fold=='ALL'else meta['split']==fold
    for band in n.BANDS:groups['VAL::qerror::'+band]=va&(meta['bands']==band)
    for key in('material','aspect','geometry_family','foreign_family','variant'):
        for value in np.unique(meta[key]):
            group=va&(meta[key]==value);name='VAL::'+key+'::'+str(value);groups[name]=group
            for band in n.BANDS:groups[name+'::'+band]=group&(meta['bands']==band)
    for material in np.unique(meta['material']):
        for aspect in np.unique(meta['aspect']):groups['VAL::material_aspect::'+material+'::'+aspect]=va&(meta['material']==material)&(meta['aspect']==aspect)
    return groups

def transitions(meta,left,right,mask):
    known=mask&np.isin(meta['target'],[0,1]);lc=left==meta['target'];rc=right==meta['target']
    return dict(known_DE=int(known.sum()),correct_both=int((known&lc&rc).sum()),wrong_both=int((known&~lc&~rc).sum()),
        recovered_by_C29=int((known&~lc&rc).sum()),regressed_in_C29=int((known&lc&~rc).sum()),
        source_fixed_threshold_no_new_selection=True)

def pair_motion(meta,models):
    N=len(meta['state']);baseN=N//32;out={}
    for fold in('TRAIN','VALIDATION'):
        allgroups=[];eligible=[]
        for base in range(baseN):
            for slot in range(8):
                e,d=(base*4+1)*8+slot,(base*4+2)*8+slot
                if meta['split'][e]!=fold:continue
                allgroups.append((e,d))
                if meta['state'][e]=='E'and meta['state'][d]=='D':eligible.append((e,d))
        finite=[];inputU=0;bothcorrect={m:0 for m in models};delta={m:[]for m in models};errors=[]
        for e,d in eligible:
            re,rd=meta['rows'][e],meta['rows'][d];assert re['canonical_index']==rd['canonical_index']
            qe,qd=re['query_xy'],rd['query_xy'];Le,Ld=re['roi_side'],rd['roi_side']
            available=all(models[m]['query_available'][e]and models[m]['query_available'][d]for m in models)
            if not available:inputU+=1
            if qe is not None and qd is not None:
                finite.append((float(np.linalg.norm(np.asarray(qe)-np.asarray(qd))),abs(Le-Ld)));errors.extend((re['error_px'],rd['error_px']))
            for m,x in models.items():
                bothcorrect[m]+=int(x['prediction'][e]==1 and x['prediction'][d]==0)
                if np.isfinite(x['raw_head_logit_before_U'][[e,d]]).all():delta[m].append(float(x['raw_head_logit_before_U'][e]-x['raw_head_logit_before_U'][d]))
        v=np.asarray(finite).reshape(-1,2)
        out[fold]=dict(all_original_samebase_slot_COVER_NEAR_groups=len(allgroups),teacher_COVER_E_NEAR_D_groups=len(eligible),finite_query_pairs=len(finite),knownpairs_with_any_inputU=inputU,
            exact_same_q_pairs=int((v[:,0]==0).sum()),q_movement_quantiles=quantile(v[:,0]),L_absolute_change_quantiles=quantile(v[:,1]),endpoint_error_quantiles=quantile(errors),
            both_DE_correct_full_known_pair_denominator=bothcorrect,hidden_score_COVER_minus_NEAR_quantiles={m:quantile(z)for m,z in delta.items()},
            interpretation='Same physical registered corner, different actual q/L/context on each image. Semantic supervised comparison is valid; not an isolated image-only or causal q intervention.')
    return out

def run():
    assert not(H/'RESULTS.json').exists();control=read(RUN/'CONTROL_COMPLETE.json');assert control['completed']and control['selected_arm_source_only']is None
    p=read(check(control['protocol']));source=read(check(p['source_binding_policy']));oldseal=read(check(source['native_feature_seal']));newseal=read(check(p['new_feature_seal']))
    native_meta=a.native_data()[n.CONV[0]];new_meta=n.metadata(check(p['new_GT_ledger']))[n.CONV[0]]
    inputs={};reports={};allrows=[]
    for corpus,meta,baseline,scorefile in(('ORIGINAL_NATIVE6144',native_meta,oldseal['predictions'],'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz'),('CONTINUOUS_NATIVE20480',new_meta,newseal['predictions'],'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz')):
        models={'C16':packet(check(baseline))};inputs[corpus]={'C16':baseline}
        for m,arm in ARMS.items():models[m]=packet(RUN/arm/scorefile);inputs[corpus][m]=bind(RUN/arm/scorefile)
        for m,x in models.items():
            assert len(x['prediction'])==len(meta['state'])and np.array_equal(x['query_available'],models['C16']['query_available'])
            assert np.array_equal(x['prediction'],np.where(x['query_available'],(x['raw_head_logit_before_U']>=0).astype(np.int8),-1))
        groups=cohorts(meta);r={g:{m:counts(meta,x['prediction'],mask)for m,x in models.items()}for g,mask in groups.items()}
        trans={comparison:{g:transitions(meta,models[comparison]['prediction'],models['C29']['prediction'],mask)for g,mask in groups.items()}for comparison in('C16','C28')}
        reports[corpus]=dict(original_slots=len(meta['state']),summary=r,transitions_to_C29=trans,paired_motion_noncausal=pair_motion(meta,models))
        for i,g in enumerate(meta['rows']):
            row=dict(corpus=corpus,global_index=i,scene_index=i//8,base_index=i//32,native_slot=i%8,split=g['split'],variant=g['variant'],query_id=g['query_id'],canonical_index=g['canonical_index'],raw_state=g['teacher_state'],query_xy=g['query_xy'],L=g['roi_side'],reference_xy=g['reference_xy'],error_px=g['error_px'],position_band=g['position_band'],factors=g['factors'])
            for m,x in models.items():row[m]=dict(prediction=int(x['prediction'][i]),available=bool(x['query_available'][i]),raw_hidden_logit=float(x['raw_head_logit_before_U'][i])if np.isfinite(x['raw_head_logit_before_U'][i])else None)
            allrows.append(row)
    assert len(allrows)==26624
    rows_path=H/'ALL26624_FIXED_SOURCE_SCORE_RESIDUALS.jsonl'
    with rows_path.open('x')as f:
        for row in allrows:f.write(json.dumps(row,allow_nan=False)+'\n')
    result=dict(completed=True,PASS=True,source_only=True,goal_status='active',fully_solved=False,control_complete=bind(RUN/'CONTROL_COMPLETE.json'),independent_control_review=bind(V/'independent_saved_review/REVIEW.json'),code=bind(__file__),protocol=bind(H/'PROTOCOL_BEFORE_RESIDUAL_READS.json'),inputs=inputs,all26624_slots_retained=True,reports=reports,rows=bind(rows_path),
        definitions=dict(DfalseE='Teacher-visible registered corner predicted externally hidden.',EfalseD='Teacher-externally-hidden registered corner predicted visible.',inputU='Known teacherD/E but unavailable model input; kept in allknown failure denominator.',q_GT3='GT-coordinate evaluation-only label; never available as a deployment filter.'),
        limitations=['Residual association with coordinate error does not establish causation; image difficulty/foreign overlap can move q and change visibility together.','Actual q/L/context retained. No correction, recrop, confidence threshold, subset admission or candidate selection.','All data are constructed source data; application and private performance remain unmeasured for C28/C29.','TeacherS/U are outside externalD/E; a visible/hidden prediction on them is reported separately, not relabelled.'],
        recommendation='First validate whether saved input-only initializer confidence/geometry carries reliable source-held-out evidence of query failure. A separate query-reliability auxiliary is a hypothesis, not GT-error filtering or solved coordinates. Broad-data G42 conditioning remains an interpretable controlled hypothesis, but V133G42 failed on old data; neither intervention is established and no new fit/protocol is created.',model_fit_optimizer_RGB_feature_private_V112_PnP_or_new_selection=0)
    write(H/'RESULTS.json',result)
    print(json.dumps(dict(completed=True,PASS=True,results=bind(H/'RESULTS.json'),VAL={c:r['summary']['VALIDATION']for c,r in reports.items()},qbands={c:{b:r['summary']['VAL::qerror::'+b]['C29']for b in n.BANDS}for c,r in reports.items()})),flush=True)
if __name__=='__main__':run()
