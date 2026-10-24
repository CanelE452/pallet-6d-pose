"""Saved V136 C16/v61 classification at original q; no model/RGB I/O."""
from pathlib import Path
import argparse, collections, hashlib, json, sys
import numpy as np

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent; V=HERE.parent
OUT=V/'saved_frozen_visibility_evaluation_v1'
PROTOCOL=HERE/'FROZEN_VISIBILITY_EVALUATION_PROTOCOL.json'
PRED_DIRS=dict(C16='frozen_C16_actual_initializer_v1',v61='frozen_v61_actual_initializer_v1')
DIAG=V/'saved_initializer_diagnosis_v1'
POLICY=V/'ROOT_CONTINUOUS_CORPUS_POLICY_BEFORE_PIXELS.json'
POLICY_SHA='546f3d28d4454ba8a28a5eecdbfb0d30d91e6513e2ecd3209fb8945be8dd9ec4'
CONVENTIONS=('CF_PRIMARY_DECLARED','V4_SENSITIVITY_ONLY')
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
ERROR_BANDS=('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def check(b):
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def same(a,b): return a['path']==b['path'] and a['sha256']==b['sha256']
def lines(p): return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,d):
    with Path(p).open('x')as f: json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def finite_float(x): return float(x) if np.isfinite(x) else None
def div(a,b): return a/b if b else None

def load_prediction_seals(entry):
    seals={}
    for model,folder in PRED_DIRS.items():
        path=V/folder/'PREDICTION_SEAL.json';assert same(entry[model+'_prediction_seal'],bind(path))
        s=read(path);assert s['completed'] and s['PASS'] and s['all2560_scenes20480_slots_preserved']
        assert s['GT_read'] is False and s['fit_current_V112_PnP_q_correction']==0
        assert s['no_model_or_threshold_choice']and s['actual_original_initializer']is True and s['source_emulation']is False
        assert s['old_prediction_replay_on_new_data']is False
        assert s['rows']==20480 and s['head_state_unchanged'] and s['head_first_repeat_exact']
        check(s['predictions']);seals[model]=s
    assert same(seals['C16']['initializer_seal'],seals['v61']['initializer_seal'])
    assert Path(seals['C16']['initializer_seal']['path']).resolve()==(V/'actual_initializer_v1/PREDICTION_SEAL.json').resolve()
    check(seals['C16']['initializer_seal'])
    return seals

def load_arrays(seals):
    arrays={}
    for model,s in seals.items():
        with np.load(s['predictions']['path'],allow_pickle=False)as f: a={k:f[k].copy()for k in f.files}
        av=a['query_available'];forward=a['head_forwarded']
        assert av.shape==forward.shape==(20480,) and av.dtype==forward.dtype==bool
        assert np.array_equal(a['global_index'],np.arange(20480))
        assert np.array_equal(a['source_occurrence_index'],np.arange(20480))
        assert np.array_equal(a['scene_index'],np.arange(20480)//8)
        assert np.array_equal(a['base_index'],np.arange(20480)//32)
        assert np.array_equal(a['native_slot'],np.arange(20480)%8)
        assert np.array_equal(a['occurrence_id'],np.array([f'V136_SCENE_{i//8:04d}:native{i%8}'for i in range(20480)]))
        assert int(av.sum())==s['available'] and int(forward.sum())==s['head_forwarded'] and int((~av).sum())==s['input_U']
        assert not (av&~forward).any()
        if model=='C16':
            raw=a['raw_head_logit_before_U'];pred=a['prediction']
            decoded=np.where(av,(raw>=0).astype(np.int8),-1)
            assert np.array_equal(av,a['factual_query_available']&a['context_available'])
            assert np.array_equal(a['query_logit'],np.where(av,raw,0),equal_nan=True)
        else:
            raw=a['raw_hidden_logit'];pred=a['prediction_D0_E1_Uminus1']
            assert np.array_equal(raw,-a['raw_visible_logit'],equal_nan=True)
            # v61 visible>=0 is D, so hidden exactly0 is D (C16 exactly0 is E).
            decoded=np.where(av,(a['raw_visible_logit']<0).astype(np.int8),-1)
            assert np.array_equal(av,a['native_query_valid']&a['ROI_any_valid'])
        assert raw.shape==pred.shape==(20480,) and raw.dtype==np.float32
        assert np.array_equal(pred,decoded) and np.isfinite(raw[forward]).all() and np.isnan(raw[~forward]).all()
        arrays[model]=dict(data=a,raw=raw,pred=pred,available=av)
    for other in ('v61',):
        for k in ('global_index','source_occurrence_index','scene_index','base_index','native_slot','occurrence_id','query_xy','L','no_detection','native_continuous_valid','native_nearest_valid'):
            assert np.array_equal(arrays['C16']['data'][k],arrays[other]['data'][k],equal_nan=True) if arrays['C16']['data'][k].dtype.kind in 'fc' else np.array_equal(arrays['C16']['data'][k],arrays[other]['data'][k]),k
    return arrays

def metric(rows,model,availability_filter=None):
    planned=rows
    known=[r for r in planned if r['raw_state'] in ('D','E')]
    used=known if availability_filter is None else [r for r in known if availability_filter(r)]
    cm=np.zeros((2,3),np.int64)
    for r in used:cm[0 if r['raw_state']=='D'else 1,{'D':0,'E':1,'U':2}[r[model]['prediction']]]+=1
    assert int(cm.sum())==len(used)
    d,e,u=map(int,cm.sum(0));dn,en=map(int,cm.sum(1))
    return dict(CM_rows_D_E_cols_D_E_U=cm.tolist(),known_DE_original=len(known),known_DE_in_this_denominator=len(used),
        original_rows=len(planned),teacher_states=dict(collections.Counter(r['raw_state']for r in planned)),
        D_precision=div(int(cm[0,0]),d),D_recall=div(int(cm[0,0]),dn),
        E_precision=div(int(cm[1,1]),e),E_recall=div(int(cm[1,1]),en),
        prediction_U_known=u,prediction_U_rate_known=div(u,len(used)),
        teacher_S_prediction_counts=dict(collections.Counter(r[model]['prediction']for r in planned if r['raw_state']=='S')),
        teacher_U_prediction_counts=dict(collections.Counter(r[model]['prediction']for r in planned if r['raw_state']=='U')),
        all_teacher_states_model_available=sum(r[model]['available']for r in planned),
        all_teacher_states_model_U=sum(not r[model]['available']for r in planned))

def cohort_masks(rows):
    out={}
    for fold in ('ALL','TRAIN','VALIDATION'):
        fr=[r for r in rows if fold=='ALL'or r['split']==fold];out[fold]=fr
        for band in ERROR_BANDS:out[fold+'::'+band]=[r for r in fr if r['position_band']==band]
        for flag in (False,True):
            bb=[r for r in fr if r['convention_boundary']==flag]
            out[fold+'::boundary'+str(flag)]=bb
            for band in ERROR_BANDS:out[fold+'::boundary'+str(flag)+'::'+band]=[r for r in bb if r['position_band']==band]
        for variant in VARIANTS:out[fold+'::'+variant]=[r for r in fr if r['variant']==variant]
        for factor in ('material','aspect','geometry_family','foreign_family'):
            for value in sorted({r['factors'][factor]for r in fr}):
                out[fold+'::'+factor+'::'+value]=[r for r in fr if r['factors'][factor]==value]
    return out

def gallery_selection(rows):
    valid=lambda r:r['split']=='VALIDATION'and not r['convention_boundary']and r['position_band']=='LE3_PX'and r['C16']['available']
    failure_pools={s:[r for r in rows if valid(r)and r['raw_state']==s and r['C16']['prediction']!=s]for s in ('E','D')}
    failures=[];used_bases=set()
    # Round robin gives both failure types coverage when present; max4 bases.
    for rank in range(4):
        for state in ('E','D'):
            choice=next((r for r in failure_pools[state]if r['base_index']not in used_bases),None)
            if choice and len(failures)<4:failures.append(choice);used_bases.add(choice['base_index'])
    cases=[];missing=[]
    for state in ('E','D'):
        if not failure_pools[state]:missing.append('C16_'+state+'_failure_LE3_nonboundary')
    for failure in failures:
        idx=failure['global_index'];label='C16_'+failure['raw_state']+'_AS_'+failure['C16']['prediction']+'_LE3_NONBOUNDARY'
        cases.append(dict(category=label,global_index=idx,base_index=failure['base_index'],native_slot=failure['native_slot'],
                          companion_global_indices_same_base_slot=[(failure['base_index']*4+i)*8+failure['native_slot']for i in range(4)]))
        success=next((r for r in rows if valid(r)and r['base_index']==failure['base_index']and r['native_slot']==failure['native_slot']and
                      r['variant']!=failure['variant']and r['raw_state']in ('D','E')and r['C16']['prediction']==r['raw_state']),None)
        if success is None:
            success=next((r for r in rows if r['split']=='VALIDATION'and not r['convention_boundary']and r['C16']['available']and
                    r['base_index']==failure['base_index']and r['native_slot']==failure['native_slot']and r['variant']!=failure['variant']and
                    r['raw_state']in ('D','E')and r['C16']['prediction']==r['raw_state']),None)
        if success:cases.append(dict(category='C16_SAME_BASE_SLOT_SUCCESS_'+success['raw_state'],global_index=success['global_index'],
                base_index=success['base_index'],native_slot=success['native_slot'],paired_failure_global_index=idx,
                success_also_LE3=success['position_band']=='LE3_PX'))
        else:missing.append('same_base_slot_success_for_'+str(idx))
    if not failures:
        for state in ('D','E'):
            success=next((r for r in rows if valid(r)and r['raw_state']==state and r['C16']['prediction']==state),None)
            if success:cases.append(dict(category='C16_SUCCESS_'+state+'_LE3_NONBOUNDARY',global_index=success['global_index'],base_index=success['base_index'],native_slot=success['native_slot']))
    assert len(cases)<=8 and len({r['global_index']for r in cases})==len(cases)
    return dict(completed=True,PASS=True,max_cases=8,cases=cases,missing_categories=missing,
        rule='First native-order VAL <=3px nonboundary C16 D/E failures, distinct bases, max4; pair same base/native slot correct other variant, prefer <=3 then any error. No RGB read.',
        failure_counts_eligible={s:len(x)for s,x in failure_pools.items()},
        labels_or_coordinates_changed=False,RGB_read=False,selection_for_diagnosis_only=True)

def build_rows(gt,arrays):
    byconv={name:[]for name in CONVENTIONS}
    assert len(gt)==40960
    for j,g in enumerate(gt):
        i,ci=divmod(j,2);assert g['convention']==CONVENTIONS[ci]
        assert g['scene_index']==i//8 and g['base_index']==i//32 and g['native_slot']==i%8
        assert g['query_id']==f'V136_SCENE_{i//8:04d}:native{i%8}'
        q=np.asarray(g['query_xy']if g['query_xy']is not None else [np.nan,np.nan],np.float64)
        L=np.nan if g['roi_side']is None else g['roi_side']
        for a in arrays.values():
            assert np.array_equal(q,a['data']['query_xy'][i],equal_nan=True)
            assert (np.isnan(L)and np.isnan(a['data']['L'][i]))or L==a['data']['L'][i]
            assert bool(a['data']['native_continuous_valid'][i])==g['continuous_inbounds']
            assert bool(a['data']['native_nearest_valid'][i])==g['nearest_inbounds']
        row=dict(global_index=i,global_occurrence_index=i,scene_index=g['scene_index'],base_index=g['base_index'],native_slot=g['native_slot'],
            id=g['id'],base_id=g['base_id'],query_id=g['query_id'],split=g['split'],variant=g['variant'],
            reference_xy=g['reference_xy'],query_xy=g['query_xy'],roi_side=g['roi_side'],error_px=g['error_px'],position_band=g['position_band'],
            raw_state=g['teacher_state'],convention_boundary=g['correspondence_boundary'],applied_convention=g['convention'],
            canonical_index=g['canonical_index'],input_status=g['input_status'],factors=g['factors'],raw_RGB_PNG=g['raw_RGB_PNG'])
        for model,a in arrays.items():
            row[model]=dict(prediction={-1:'U',0:'D',1:'E'}[int(a['pred'][i])],raw_logit=finite_float(a['raw'][i]),
                           available=bool(a['available'][i]),head_forwarded=bool(a['data']['head_forwarded'][i]),
                           score_meaning='hidden_logit; exact0=>E'if model=='C16'else'negative_visible_logit; exact0=>D')
        row['shared_available']=all(row[m]['available']for m in PRED_DIRS)
        byconv[g['convention']].append(row)
    assert all(len(v)==20480 for v in byconv.values())
    return byconv

def run(entry):
    e=read(entry);assert e['approved_saved_evaluation']is True
    assert all(e[k]is False for k in('model_forward','fit','current223','V112','PnP','q_correction','automatic_retry'))
    for k,path in(('code',Path(__file__)),('protocol',PROTOCOL),('policy',POLICY)):assert same(e[k],bind(path))
    assert sha(POLICY)==POLICY_SHA
    assert not OUT.exists(),'Preserve previous evaluation.'
    protocol=read(PROTOCOL);assert same(protocol['code'],bind(__file__))
    seals=load_prediction_seals(e);arrays=load_arrays(seals)
    # Existing GT-only ledger is read only after both prediction seals/NPZ parity.
    ds=DIAG/'DIAGNOSIS_SEAL.json';assert same(e['diagnosis_seal'],bind(ds))
    diagseal=read(ds);assert diagseal['completed']and diagseal['PASS']and diagseal['RGB_read']is False
    assert same(diagseal['prediction_seal'],seals['C16']['initializer_seal'])
    gtpath=DIAG/'ALL20480_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl'
    diagreport=read(check(next(b for b in diagseal['outputs']if Path(b['path']).name=='SOURCE_RESULTS_KO.json')))
    check(next(x for x in diagseal['outputs']if x['path']==str(gtpath)))
    byconv=build_rows(lines(gtpath),arrays); reports={}
    for convention,rows in byconv.items():
        reports[convention]={}
        for name,cohort in cohort_masks(rows).items():
            reports[convention][name]={model:dict(
                all_known_DE_original_denominator=metric(cohort,model),
                own_available_only=metric(cohort,model,lambda r:r[model]['available']),
                shared_available_only=metric(cohort,model,lambda r:r['shared_available']))for model in PRED_DIRS}
    primary=byconv[CONVENTIONS[0]];selection=gallery_selection(primary)
    OUT.mkdir();scorepath=OUT/'ALL20480_SCORES.jsonl'
    with scorepath.open('x')as f:
        for row in primary:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write(OUT/'GALLERY_SELECTION_BEFORE_RGB.json',selection)
    results=dict(completed=True,PASS=True,goal_status='active',fully_solved=False,source_only=True,original_frames=2560,original_slots=20480,
        primary_convention=CONVENTIONS[0],sensitivity_convention=CONVENTIONS[1],convention_choice_from_predictions=False,
        obsolete_area_disagreement_base_counts=diagreport['obsolete_area_disagreement_base_counts'],CF_normal_facing_source_order_certified=True,GT_DESU_preserved=True,teacher_S_U_excluded_external_binary=True,
        full_known_DE_denominator_retains_prediction_U=True,models=list(PRED_DIRS),summary=reports,
        input_availability={m:dict(available=int(a['available'].sum()),U=int((~a['available']).sum()))for m,a in arrays.items()},
        common_availability=dict(both_available=int((arrays['C16']['available']&arrays['v61']['available']).sum()),
            C16_only=int((arrays['C16']['available']&~arrays['v61']['available']).sum()),v61_only=int((~arrays['C16']['available']&arrays['v61']['available']).sum())),
        shared_denominator='Both frozen models input-available; full original knownDE denominator remains primary.',
        score_rule='C16 hidden z<0 D,z>=0 E; v61 visible z>=0 D,z<0 E; unavailable U first. Raw scores before U preserved; null only nonforward.',
        metrics_scope='Registered synthetic teacher semantic D/E evaluation; v61 trained pixel-presence target differs. Not private real performance.',
        human_gold=False,no_current_model_threshold_admission=True,RGB_model_fit_private_V112_PnP_q_correction=0,
        code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),diagnosis_seal=bind(ds),
        prediction_seals={m:bind(V/folder/'PREDICTION_SEAL.json')for m,folder in PRED_DIRS.items()})
    write(OUT/'SOURCE_RESULTS_KO.json',results)
    paths={k:bind(OUT/f)for k,f in dict(results='SOURCE_RESULTS_KO.json',scores='ALL20480_SCORES.jsonl',selection='GALLERY_SELECTION_BEFORE_RGB.json').items()}
    write(OUT/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,C16_prediction_seal=results['prediction_seals']['C16'],
        v61_prediction_seal=results['prediction_seals']['v61'],diagnosis_seal=bind(ds),**paths,
        outputs=list(paths.values()),code=bind(__file__),RGB_read=False,source_only=True))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'EVALUATION_SEAL.json'),
        CF_VAL=reports[CONVENTIONS[0]]['VALIDATION'],selection_cases=len(selection['cases'])),ensure_ascii=False),flush=True)

def prepare():
    assert sha(POLICY)==POLICY_SHA
    # Invented cases verify primary known-DE denominator keeps unavailable U.
    invented=[]
    for state,pred in zip('DDEESU',('D','U','E','U','D','E')):
        invented.append(dict(raw_state=state,C16=dict(prediction=pred,available=pred!='U'),v61=dict(prediction=pred,available=pred!='U')))
    checked=metric(invented,'C16');assert checked['CM_rows_D_E_cols_D_E_U']==[[1,0,1],[0,1,1]]and checked['D_recall']==checked['E_recall']==.5
    assert checked['known_DE_original']==checked['known_DE_in_this_denominator']==4 and checked['teacher_states']==dict(D=2,E=2,S=1,U=1)
    write(PROTOCOL,dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),models=list(PRED_DIRS),
        expected_prediction_seals={m:str(V/d/'PREDICTION_SEAL.json')for m,d in PRED_DIRS.items()},expected_diagnosis_seal=str(DIAG/'DIAGNOSIS_SEAL.json'),
        output=str(OUT),all20480_retained=True,CF_normal_facing_primary_certified_V4_obsolete_sensitivity_fixed=True,
        raw_scores_no_threshold_choice=True,known_predictionU_in_full_denominator=True,teacher_SU_outside_binary=True,
        separate_own_and_shared_availability=True,invented_knownU_denominator_check_PASS=True,no_predictions_GT_RGB_model_body_reads=True))
    template=dict(approved_saved_evaluation=False,model_forward=False,fit=False,current223=False,V112=False,PnP=False,q_correction=False,automatic_retry=False,
        code=bind(__file__),protocol=bind(PROTOCOL),policy=bind(POLICY),C16_prediction_seal=None,v61_prediction_seal=None,diagnosis_seal=None,
        fill_after_complete='Bind actual V136 two prediction seals and GT-only saved initializer diagnosis seal. Never substitute old seals.')
    write(HERE/'ROOT_EVALUATION_ENTRY_TEMPLATE.json',template)
    print(json.dumps(dict(protocol=bind(PROTOCOL),code=bind(__file__),template=bind(HERE/'ROOT_EVALUATION_ENTRY_TEMPLATE.json'),actual_evaluation=False)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and a.entry is None:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Use --prepare, then root-approved --entry after completed actual predictions/diagnosis.')
