"""Independent NumPy/JSON arithmetic for terminal V135 results.

No trainer/helper imports, Torch, model/checkpoint, RGB, feature, private or
V112 reads. Saved scores, source GT metadata and optimizer trace text only.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
RUN=V/'runs/C26_C27_NATIVE_PAIR_FIXED8000'
ARMS=('C26_NATIVE_PAIR_SAMPLER','C27_NATIVE_PAIR_SAMPLER_AND_MARGIN')
C24=P/'query_visibility_native_geometry_control_20261024_v133/runs/C24_C25_MATCHED_G42_FIXED8000/C24_ZERO_GEOMETRY_CONTROL'
COVERAGE=P/'query_visibility_native_witness_diagnosis_20261024_v134/feature_diagnosis/NATIVE_TRAIN_PAIR_COVERAGE_AND_LOSS_HISTORY.json'
OLD=P/'query_visibility_training_duration_control_20261023_v124/runs/C16_EXACT_C15_DURATION_CONTROL_8000'
NATIVE=P/'query_visibility_actual_initializer_source_20261024_v132'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b;return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def lines(p):
    with Path(p).open()as f:
        for s in f:
            if s.strip():yield json.loads(s)
def packet(p):
    with np.load(p,allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
def write(p,d):
    with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ratio(n,d):return float(n/d)if d else None
def close(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:close(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y)in enumerate(zip(a,b)):close(x,y,path+'/'+str(i))
    elif isinstance(a,float):assert b is not None and abs(a-b)<=1e-12,(path,a,b)
    else:assert a==b,(path,a,b)

def metrics(target,pred,mask,states):
    cm=np.zeros((2,3),dtype=np.int64)
    for true in(0,1):
        for column,label in enumerate((0,1,-1)):cm[true,column]=np.count_nonzero(mask&(target==true)&(pred==label))
    return dict(CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=ratio(cm[0,0],sum(cm[0])),E_recall=ratio(cm[1,1],sum(cm[1])),
        D_precision=ratio(cm[0,0],sum(cm[:,0])),E_precision=ratio(cm[1,1],sum(cm[:,1])),
        prediction_U_known=int(sum(cm[:,2])),prediction_U_rate_known=ratio(sum(cm[:,2]),cm.sum()),
        all_original_rows=int(mask.sum()),known_DE=int(cm.sum()),teacher_S_outside_external=int(np.count_nonzero(mask&(states=='S'))),
        teacher_U_outside_external=int(np.count_nonzero(mask&(states=='U'))),
        teacher_S_binary_responses={str(label):int(np.count_nonzero(mask&(states=='S')&(pred==label)))for label in(0,1,-1)},
        teacher_U_binary_responses={str(label):int(np.count_nonzero(mask&(states=='U')&(pred==label)))for label in(0,1,-1)})

def old_masks(a):
    va=a['split']=='VALIDATION';dd=a['external_target']==0;ee=a['external_target']==1
    f=a['domain'];s=a['source'];v=a['variant'];o=f=='V96_REAL';n=f=='V99_SYNTH';m=f=='V113_MATERIAL_SURFACE';flat=f=='V122_PUBLIC_FLAT_PALLET'
    return {'old_LOCO_BEFORE_D':va&o&dd&(s=='LOCO')&(v=='BEFORE'),'old_PALLOX_BEFORE_D':va&o&dd&(s=='PALLOX')&(v=='BEFORE'),
        'old_SHAM_D':va&o&dd&(v=='CUE_PRESERVING_SHAM'),'old_BEFORE_E':va&o&ee&(v=='BEFORE'),'old_COVER_E':va&o&ee&(v=='OPAQUE_COVER'),
        'new_BEFORE_D':va&n&dd&(v=='BEFORE'),'new_SHAM_D':va&n&dd&(v=='SHAM'),'new_COVER_E':va&n&ee&(v=='COVER'),
        'material_VALIDATION_D':va&m&dd,'material_VALIDATION_E':va&m&ee,
        'LOCO_VALIDATION_D':va&flat&dd&(s=='LOCO'),'LOCO_VALIDATION_E':va&flat&ee&(s=='LOCO'),
        'TOMIE_DEVELOPMENT_D':va&flat&dd&(s=='TOMIE'),'TOMIE_DEVELOPMENT_E':va&flat&ee&(s=='TOMIE')}

def old14(a,masks):
    p=a['prediction'];out={}
    for name,mask in masks.items():
        classes=np.unique(a['external_target'][mask]);assert len(classes)==1 and int(classes[0])in(0,1)
        t=int(classes[0]);out[name]=dict(N=int(mask.sum()),correct=int(np.count_nonzero(p[mask]==t)),recall=float(np.mean(p[mask]==t)),
            target_D0_E1=t,prediction_D_E_U=[int(np.count_nonzero(p[mask]==j))for j in(0,1,-1)],
            binary_metrics=metrics(a['external_target'],p,mask,a['raw_teacher_state']),
            precision_scope='Single-class selector cohort; pooled D/E precision reported additionally by domain/split.')
    return out

def native_data():
    ds=read(NATIVE/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json');assert ds['completed']and ds['PASS']
    gt=next(x for x in ds['outputs']if Path(x['path']).name=='ALL6144_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl')
    rr=list(lines(check(gt)));assert len(rr)==12288;byconv={}
    for c,convention in enumerate(('CF_PRIMARY_DECLARED','V4_SENSITIVITY_ONLY')):
        sr=rr[c::2]
        for i,r in enumerate(sr):assert r['convention']==convention and r['scene_index']==i//8 and r['native_slot']==i%8
        state=np.array([r['teacher_state']for r in sr]);assert np.isin(state,list('DESU')).all()
        byconv[convention]=dict(rows=sr,state=state,target=np.where(state=='D',0,np.where(state=='E',1,-1)),
            split=np.array([r['split']for r in sr]),boundary=np.array([r['correspondence_boundary']for r in sr]),
            bands=np.array([r['position_band']for r in sr]),variant=np.array([r['variant']for r in sr]))
        for factor in('material','aspect','geometry_family','foreign_family'):byconv[convention][factor]=np.array([r['factors'][factor]for r in sr])
    return byconv

def native_masks(n):
    groups={}
    for fold in('ALL','TRAIN','VALIDATION'):
        f=np.ones(6144,bool)if fold=='ALL'else n['split']==fold
        groups[fold]=f;groups[fold+'::NONBOUNDARY']=f&~n['boundary'];groups[fold+'::BOUNDARY']=f&n['boundary']
        groups[fold+'::NONBOUNDARY::SAME_TARGET_MATERIAL']=f&~n['boundary']&(n['foreign_family']=='SAME_TARGET_MATERIAL')
        for band in('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY'):
            groups[fold+'::'+band]=f&(n['bands']==band);groups[fold+'::NONBOUNDARY::'+band]=f&~n['boundary']&(n['bands']==band)
        for key in('variant','material','aspect','geometry_family','foreign_family'):
            for value in np.unique(n[key][f]):groups[fold+'::'+key+'::'+str(value)]=f&(n[key]==value)
    return groups

def native_metrics(a,n):return{k:metrics(n['target'],a['prediction'],mask,n['state'])for k,mask in native_masks(n).items()}

def gate_values(a,baseline,baseline14,original15,n,nbase,masks):
    rr=old14(a,masks);gates={}
    for name,metric in rr.items():
        limit=baseline14[name]['recall']-.01;gates['OLD14::'+name]=dict(value=metric['recall'],limit=limit,N=metric['N'],PASS=metric['recall']>=limit)
    selected=['VALIDATION','VALIDATION::NONBOUNDARY']+sorted(k for k in n if k.startswith('VALIDATION::foreign_family::'))
    assert len(selected)==6
    for group in selected:
        for state in('D','E'):
            value=n[group][state+'_recall'];limit=nbase[group]['D_recall']-.01 if state=='D'else .8
            gates['NATIVE::'+group+'::'+state]=dict(value=value,limit=limit,PASS=value is not None and value>=limit)
    value=n['VALIDATION::NONBOUNDARY::SAME_TARGET_MATERIAL']['E_recall']
    gates['NATIVE::VALIDATION::NONBOUNDARY::SAME_TARGET_MATERIAL::E']=dict(value=value,limit=.8,PASS=value is not None and value>=.8)
    p=a['prediction'];y=a['external_target'];va=a['split']=='VALIDATION';known=np.isin(y,[0,1]);dom=a['domain']
    recall=lambda mask,t:float(np.mean(p[mask]==t))
    u=lambda mask:float(np.mean(p[mask]<0))
    pair={}
    for key in('old_accepted_triplets','new_selected_triplets'):
        triplets=baseline[key];triplets=triplets[va[triplets[:,0]]]
        assert np.array_equal(y[triplets],np.tile([0,1,0],(len(triplets),1)))
        pair[key]=float(np.mean(np.all(p[triplets]==[0,1,0],axis=1)))
    vals=dict(old_LOCO_before_D_recall_min=recall(masks['old_LOCO_BEFORE_D'],0),old_PALLOX_before_D_recall_min=recall(masks['old_PALLOX_BEFORE_D'],0),
        old_natural_before_E_recall_min=recall(masks['old_BEFORE_E'],1),old_sham_D_recall_min=recall(masks['old_SHAM_D'],0),
        old_cover_E_recall_min=recall(masks['old_COVER_E'],1),old_accepted_VAL_pair_all3_min=pair['old_accepted_triplets'],
        old_known_VAL_U_rate_max=u(va&known&(dom=='V96_REAL')),new_BEFORE_D_recall_min=recall(masks['new_BEFORE_D'],0),
        new_SHAM_D_recall_min=recall(masks['new_SHAM_D'],0),new_COVER_E_recall_min=recall(masks['new_COVER_E'],1),
        new_selected_VAL_pair_all3_min=pair['new_selected_triplets'],new_known_VAL_U_rate_max=u(va&known&(dom=='V99_SYNTH')),
        material_VAL_D_recall_min=recall(masks['material_VALIDATION_D'],0),material_VAL_E_recall_min=recall(masks['material_VALIDATION_E'],1),
        material_VAL_known_U_rate_max=u(va&known&(dom=='V113_MATERIAL_SURFACE')))
    assert set(vals)==set(original15)
    for key,value in vals.items():
        limit=original15[key]['limit'];gates['HISTORICAL15::'+key]=dict(value=value,limit=limit,PASS=value<=limit if key.endswith('_max')else value>=limit)
    assert len(gates)==42
    return gates,rr

def check_trace(directory,old,native,pool,arm):
    digest=hashlib.sha256();prefix128=hashlib.sha256();flat64=hashlib.sha256();count=0
    reference=iter(lines(C24/'TRAIN_DRAWS_AND_LOSSES.jsonl'));pair_digest=hashlib.sha256();pair_exposure=np.zeros(125,np.int64);uniform_D_exposure=np.zeros(6144,np.int64);uniform_E_exposure=np.zeros(6144,np.int64)
    for count,row in enumerate(lines(directory/'TRAIN_DRAWS_AND_LOSSES.jsonl'),1):
        assert row['update']==count
        ix=np.asarray(row['old_occurrences'],np.int64);pairs=np.asarray(row['old_matched_cover_sham_positions'],np.int64);ni=np.asarray(row['native_occurrences'],np.int64)
        assert ix.shape==(192,)and pairs.shape==(24,2)and ni.shape==(64,)and pairs.min()>=0 and pairs.max()<128
        assert (ix>=0).all()and(ix<51004).all()and(ni>=0).all()and(ni<6144).all()
        assert (old['split'][ix]=='TRAIN').all()and old['query_available'][ix].all()and np.bincount(old['external_target'][ix],minlength=2).tolist()==[96,96]
        assert np.array_equal(old['external_target'][ix[pairs]],np.tile([1,0],(24,1)))
        assert np.array_equal(ix[176:192],ix[128:144]+1)
        assert (native['split'][ni]=='TRAIN').all()and native['query_available'][ni].all()
        pi=np.asarray(row['native_pair_pool_positions'],np.int64);assert pi.shape==(16,)and pi.min()>=0 and pi.max()<125
        assert np.array_equal(ni[:16],pool[pi,0])and np.array_equal(ni[16:32],pool[pi,1])
        assert np.array_equal(native['external_target_CF'][ni],np.r_[np.ones(16),np.zeros(32),np.ones(16)])
        pair_digest.update(pi.astype('<i8').tobytes());np.add.at(pair_exposure,pi,1);np.add.at(uniform_D_exposure,ni[32:48],1);np.add.at(uniform_E_exposure,ni[48:],1)
        oldrow=next(reference);assert oldrow['update']==count and oldrow['old_occurrences']==ix.tolist()and oldrow['old_matched_cover_sham_positions']==pairs.tolist()
        for item in(ix,pairs,ni):digest.update(item.astype('<i8').tobytes())
        prefix128.update(ix[:128].astype('<i8').tobytes());prefix128.update(pairs.astype('<i8').tobytes());flat64.update(ix[128:].astype('<i8').tobytes())
        vals=[row[k]for k in('loss','old_BCE','old_hinge','flat_BCE','native_BCE','native_pair_hinge','native_pair_hinge_weight','gradient_before_clip')]
        assert all(math.isfinite(x)and x>=0 for x in vals)
        expected=np.float32(np.float32(np.float32(row['old_BCE'])+np.float32(.25)*np.float32(row['old_hinge']))+np.float32(row['flat_BCE']))+np.float32(row['native_BCE'])
        weight=.25 if arm==ARMS[1]else 0.
        assert row['native_pair_hinge_weight']==weight
        expected=np.float32(expected+np.float32(weight)*np.float32(row['native_pair_hinge']))
        assert float(expected)==row['loss'],(count,float(expected),row['loss'])
    assert count==8000 and next(reference,None)is None
    assert pair_exposure.sum()==128000 and pair_exposure.min()>0 and uniform_D_exposure.sum()==uniform_E_exposure.sum()==128000
    eligible=(native['split']=='TRAIN')&native['query_available']
    assert np.array_equal(np.flatnonzero(uniform_D_exposure),np.flatnonzero(eligible&(native['external_target_CF']==0)))
    assert np.array_equal(np.flatnonzero(uniform_E_exposure),np.flatnonzero(eligible&(native['external_target_CF']==1)))
    previous=read(OLD/'SOURCE_RESULTS.json')
    assert prefix128.hexdigest()==previous['old128_all8000_draws_sha256']and flat64.hexdigest()==previous['new64_all8000_draws_sha256']
    return dict(rows=count,combined_old_native_sha256=digest.hexdigest(),old128_sha256=prefix128.hexdigest(),flat64_sha256=flat64.hexdigest(),
        all8000_old192_indices_and24pairs_exact_historical_C24=True,all_native64_TRAIN_available_known_32D32E=True,
        native_pair_position_sha256=pair_digest.hexdigest(),native_pair_exposure125=pair_exposure.tolist(),native_pair_draws=128000,uniform_D_draws=128000,uniform_E_draws=128000,uniform_TRAIN_D1869_E131_pools_all_seen=True,
        pairmember_mean_total_exposure_E=float((pair_exposure+uniform_E_exposure[pool[:,0]]).mean()),pairmember_mean_total_exposure_D=float((pair_exposure+uniform_D_exposure[pool[:,1]]).mean()),
        native_pair_order_E16_D16_uniformD16_uniformE16_and_samephysical125=True,
        teacher_SU_OOB_noquery_VAL_not_sampled=True,all8000_loss_sums_exact_FP32=True)

def run():
    assert not(HERE/'REVIEW.json').exists(),'Preserve previous audit.'
    control_path=RUN/'CONTROL_COMPLETE.json';assert control_path.exists(),'Terminal completion required.'
    c=read(control_path);assert c['completed']and c['PASS']and c['final_update8000_both']and c['application_solved']is False
    check(c['policy']);check(c['protocol']);protocol=read(c['protocol']['path']);policy=read(check(protocol['source_binding_policy']))
    assert sha(COVERAGE)==protocol['pair_coverage']['sha256']
    coverage=read(COVERAGE);assert coverage['PASS']and coverage['TRAIN_only_pair_definition']and coverage['no_modelerror_or_VAL_mining']
    pool=np.asarray([(r['global_indices'][1],r['global_indices'][2])for r in coverage['groups']if r['available_coverE_nearD']],np.int64)
    assert pool.shape==(125,2)and len(np.unique(pool))==250
    savedpool=read(RUN/'NATIVE_TRAIN_PAIR_POOL_BEFORE_FIT.json');assert np.array_equal(pool,np.asarray(savedpool['pairs_COVER_E_NEAR_SHAM_D']))
    baseline=packet(check(policy['C16_source_scores']));masks=old_masks(baseline);base14=old14(baseline,masks)
    original15=read(OLD/'SOURCE_GATE_RESULTS.json')['gates'];ns=read(check(policy['native_feature_seal']))
    native_baseline=packet(check(ns['predictions']));native_baseline['prediction']=native_baseline['prediction']
    nmeta=native_data();nm=nmeta['CF_PRIMARY_DECLARED'];nb=native_metrics(native_baseline,nm);results={};draw_hash=None;pair_hash=None
    for e,d in pool:
        assert nm['state'][e]=='E'and nm['state'][d]=='D'and nm['split'][e]==nm['split'][d]=='TRAIN'
        assert native_baseline['query_available'][e]and native_baseline['query_available'][d]
        assert nm['variant'][e]=='COVER'and nm['variant'][d]=='NEAR_SHAM'
        assert e//32==d//32 and e%8==d%8 and nm['rows'][e]['canonical_index']==nm['rows'][d]['canonical_index']
    for arm in ARMS:
        directory=RUN/arm;fit=read(check(c['arms'][arm]['fit']));freeze=read(check(c['arms'][arm]['freeze']))
        assert fit['completed']and fit['optimizer_updates']==8000 and freeze['completed']and freeze['update']==freeze['selected_update']==8000
        assert freeze['parameters']==224513 and freeze['threshold_logit']==0 and freeze['current_allowed']is False and freeze['V112_allowed']is False and freeze['PnP']==0
        saved=read(check(fit['results']));assert saved['completed']and saved['selected_update']==8000 and saved['VAL_checkpoint_choice']is False
        old=packet(directory/'FINAL_OLD51004_SOURCE_SCORES.npz');native=packet(directory/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz')
        for key in('query_available','input_unit_index','external_target','raw_teacher_state','split','domain','source','variant'):
            assert np.array_equal(old[key],baseline[key]),key
        assert np.array_equal(old['global_occurrence_index'],np.arange(51004))
        assert np.array_equal(old['prediction'],np.where(old['query_available'],(old['query_logit']>=0).astype(np.int8),-1))
        for key in('query_available','head_forwarded','global_index','occurrence_id','query_xy','L'):
            assert np.array_equal(native[key],native_baseline[key],equal_nan=True)if native[key].dtype.kind in'fc'else np.array_equal(native[key],native_baseline[key]),key
        raw=native['raw_head_logit_before_U'];assert raw.shape==(6144,)and np.isfinite(raw[native['head_forwarded']]).all()and np.isnan(raw[~native['head_forwarded']]).all()
        assert np.array_equal(native['query_logit'],np.where(native['query_available'],raw,0))
        assert np.array_equal(native['prediction'],np.where(native['query_available'],(raw>=0).astype(np.int8),-1))
        assert np.array_equal(native['raw_teacher_state_CF'],nmeta['CF_PRIMARY_DECLARED']['state'])and np.array_equal(native['external_target_CF'],nmeta['CF_PRIMARY_DECLARED']['target'])
        assert np.array_equal(native['convention_boundary'],nmeta['CF_PRIMARY_DECLARED']['boundary'])and np.array_equal(native['split'],nmeta['CF_PRIMARY_DECLARED']['split'])
        nr=native_metrics(native,nmeta['CF_PRIMARY_DECLARED']);v4=native_metrics(native,nmeta['V4_SENSITIVITY_ONLY'])
        close(nr,saved['native_CF'],'native_CF');close(v4,saved['native_V4_sensitivity_only'],'native_V4')
        gates,r14=gate_values(old,baseline,base14,original15,nr,nb,masks);close(r14,saved['old']['old14'],'old14')
        assert set(gates)==set(saved['gates'])
        for key,expected in gates.items():
            got=saved['gates'][key]
            for field in('value','limit','PASS'):close(expected[field],got[field],key+'/'+field)
        # Fourteen source BCE cohorts independently from final saved logits.
        z=old['query_logit'].astype(np.float64);eligible=np.isin(old['external_target'],[0,1]);y=np.where(eligible,old['external_target'],0)
        loss=np.logaddexp(0,z)-y*z;parts={}
        for key,mask in masks.items():parts[key]=dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~old['query_available']).sum()))
        close(parts,saved['old']['old14_BCE_parts'],'BCE14');close(float(np.mean([v['mean_BCE']for v in parts.values()])),saved['old']['old14_BCE'],'macro_BCE14')
        for key,metric in saved['old']['domain_metrics'].items():
            domain,fold=key.split('::');mask=(old['domain']==domain)&(old['split']==fold)
            close(metrics(old['external_target'],old['prediction'],mask,old['raw_teacher_state']),metric,key)
        trace=check_trace(directory,old,native,pool,arm)
        assert trace['combined_old_native_sha256']==saved['old_native_all8000_draw_sha256']==fit['draw_sha256']==c['draw_sha256']
        if draw_hash is None:draw_hash=trace['combined_old_native_sha256']
        else:assert draw_hash==trace['combined_old_native_sha256']
        if pair_hash is None:pair_hash=trace['native_pair_position_sha256']
        else:assert pair_hash==trace['native_pair_position_sha256']
        parity=read(check(fit['initialization_parity']));assert parity['completed']and parity['PASS']and parity['all51004_old_scores_bitexact']and parity['all6088_native_raw_scores_bitexact']and parity['missing56_raw_NaN_exact']
        assert parity['same_parameters']==224513 and parity['actual_updates']==0 and parity['geometry_weights_zero']
        assert freeze['all_geometry_input_zero']and saved['actual_geometry_input_used']is False and saved['old192_all8000_exact_historical_C24']
        assert saved['native_pair_pool_count']==125 and saved['native_pair_hinge_margin']==2 and saved['native_pair_hinge_weight']==(.25 if arm==ARMS[1]else 0.)
        hist=list(lines(directory/'HISTORY.jsonl'));assert len(hist)==81 and [r['update']for r in hist]==list(range(0,8001,100))
        close(hist[-1]['old14'],r14,'final_history_old14');close(hist[-1]['native_VAL'],nr['VALIDATION'],'final_history_nativeVAL')
        close(hist[0]['old14'],base14,'initial_history_old14');close(hist[0]['native_VAL'],nb['VALIDATION'],'initial_history_nativeVAL')
        for row in hist:
            assert list(row['old14_BCE_parts'])==list(parts)
            for key,value in row['old14_BCE_parts'].items():assert value['count']==parts[key]['count']and value['unavailable']==parts[key]['unavailable']and math.isfinite(value['mean_BCE'])and value['mean_BCE']>=0
        rank=[v['value']for k,v in gates.items()if not k.startswith('HISTORICAL15::')];assert len(rank)==27
        worst=min(rank);eligible=all(x['PASS']for x in gates.values())
        assert eligible==saved['source_eligible']==fit['source_eligible']==freeze['source_eligible']==c['arms'][arm]['source_eligible']
        assert worst==saved['worst_required27_class_recall']==fit['worst_required27_class_recall']==c['arms'][arm]['worst_required27_class_recall']
        review=dict(completed=True,PASS=True,source_eligible=eligible,gates42_recomputed=gates,gates_pass=sum(v['PASS']for v in gates.values()),
            failed_gates=[k for k,v in gates.items()if not v['PASS']],worst_required27_class_recall=worst,trace=trace,
            all51004_old_and6144_native_metadata_availability_unchanged=True,all_CF_and_V4_group_metrics_exact=True,
            update0_receipt_hash_and_assertions_checked=True,initialization_parity=fit['initialization_parity'],
            all14_source_BCE_recomputed=True,histories81_counts_and_endpoints_verified=True,
            native_VAL=nr['VALIDATION'],native_VAL_nonboundary=nr['VALIDATION::NONBOUNDARY'],
            native_same_material=nr['VALIDATION::foreign_family::SAME_TARGET_MATERIAL'],
            native_same_material_nonboundary=nr['VALIDATION::NONBOUNDARY::SAME_TARGET_MATERIAL'],
            source_results=fit['results'],final_old_scores=bind(directory/'FINAL_OLD51004_SOURCE_SCORES.npz'),final_native_scores=bind(directory/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz'))
        write(HERE/(arm+'_RECALCULATED.json'),review);results[arm]=review
    eligible=[a for a in ARMS if results[a]['source_eligible']]
    chosen=max(eligible,key=lambda a:(results[a]['worst_required27_class_recall'],-ARMS.index(a)))if eligible else None
    assert chosen==c['selected_arm_source_only']
    final=dict(completed=True,PASS=True,goal_status='active',fully_solved=False,scope='Independent saved NumPy/JSON arithmetic; no trainer imports or model/checkpoint/RGB/private/V112 reads.',
        source_only=True,control_complete=bind(control_path),policy=c['policy'],code=bind(__file__),
        arms={a:dict(source_eligible=r['source_eligible'],gates_pass=r['gates_pass'],gates_total=42,failed_gates=r['failed_gates'],
                    worst_required27_class_recall=r['worst_required27_class_recall'],recalculated=bind(HERE/(a+'_RECALCULATED.json')))for a,r in results.items()},
        selected_arm_source_only=chosen,all8000_old192_draws_exact_historical_C24=True,same_native8000_draws_both=True,combined_draw_sha256=draw_hash,native_pair_position_sha256=pair_hash,
        all125_samephysicalTRAIN_pair_indices_and_250_original_members_verified=True,all16k_trace_FP32_loss_groupings_verified=True,update0_receipt_hash_assertions_verified=True,
        all42_gate_values_limits_PASS_and27ranking_checked=True,all_CF_V4_matrices_precision_recall_U_and_14BCE_checked=True,
        old51004_and_native6144_full_denominators_preserved=True,current_V112_PnP_calls=0,
        producer_only_not_independently_replayed=['Model/checkpoint contents and parameter updates','Zero-geometry weights staying zero','Update0 model bit parity and gradients','Intermediate history scores beyond endpoint arithmetic'],
        limitations='Saved arithmetic validates results and sample semantics; it does not independently rerun inference, training, RGB or parameters. No new stress-arm evaluation claimed.')
    write(HERE/'REVIEW.json',final);print(json.dumps(dict(completed=True,PASS=True,review=bind(HERE/'REVIEW.json'),arms=final['arms'],selected=chosen)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if args.run:run()
    else:raise SystemExit('Use --run only after terminal CONTROL_COMPLETE.json exists.')
