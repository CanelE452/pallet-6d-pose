"""Saved source logits/labels/draws only; no learned module/feature/checkpoint read."""
from pathlib import Path
import hashlib,json,collections
import numpy as np

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
R=V/'runs/C35_C36_SEPARATE_QUERY_FIXED2000'
ARMS=('C35_MATCHED_VISIBILITY_ONLY_CONTROL','C36_MATCHED_VISIBILITY_PLUS_ENTITY_AND_POINT_QUALITY')
S=P/'query_visibility_mr6d_full_public_source_20261024_v153'
NAMES=('OLD6144','NEW20480','PALLOX512','MR1840')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def check(b):
    assert '_v112' not in b['path'].lower() and '/private_evaluation/' not in b['path']
    assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def packet(p):
    with np.load(p,allow_pickle=False) as f:
        assert 'features' not in f.files
        return {k:f[k].copy() for k in f.files}
def numbers(x):
    x=np.asarray(x);finite=np.isfinite(x);a=x[finite]
    return dict(N=len(x),finite=int(finite.sum()),undefined=int((~finite).sum()),
        negative=int((a<0).sum()),negative_fraction=float((a<0).mean()) if len(a) else None,
        zero=int((a==0).sum()),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        p10=float(np.quantile(a,.1)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None)
def cm(y,p,m):return [[int((m&(y==t)&(p==v)).sum()) for v in (0,1,-1)] for t in (0,1)]
def corr(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[m],b[m])[0,1]) if m.sum()>1 and a[m].std()>0 and b[m].std()>0 else None
def intlist(x):return [int(v) for v in x]
def clean(x):
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def main():
    control=read(R/'CONTROL_COMPLETE.json');assert control['completed'] and control['PASS'] and control['updates_each']==2000
    assert control['selected_model'] is None and control['new_private_or_encoder_R0_PnP_xy_calls']==0
    protocol=read(check(control['protocol']));previous=read(check(protocol['parent_protocol']))
    seal=read(check(protocol['source_supervision_seal']));assert seal['source_fit_admitted'] and not seal['visibility_scores_used_for_labels']
    mrrows=lines(check(protocol['source_labels']));assert len(mrrows)==1840
    pubrows=lines(check(previous['public_labels']))
    native=[];qualities=[]
    for n,b,q in zip((6144,20480),previous['native'],protocol['quality_labels']):
        rr=[r for r in lines(check(b['GT'])) if r['convention']=='CF_PRIMARY_DECLARED']
        assert len(rr)==n
        qq=lines(check(q));assert len(qq)==n
        for i,(r,v) in enumerate(zip(rr,qq)):assert v['global_index']==i and r['query_id']==v['query_id']
        native.append(rr);qualities.append(qq)
    rows=native+[pubrows,mrrows]
    states=[np.array([r.get('teacher_state',r.get('semantic_state',r.get('role_state'))) for r in rr]) for rr in rows]
    targets=[np.where(s=='D',0,np.where(s=='E',1,-1)).astype(np.int8) for s in states]
    folds=[np.array([r['split'] for r in rr]) for rr in rows]
    quals=[np.array([-1 if r['point_quality_label'] is None else r['point_quality_label'] for r in qq],np.int8) for qq in qualities]
    quals += [np.full(512,-1,np.int8),np.full(1840,-1,np.int8)]
    entity=np.array([r['entity_target'] for r in mrrows],np.int8)
    reference_path=S/'native_reference_preparation/provided_reference_v1/ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl'
    refs=lines(reference_path);assert len(refs)==2424
    refmap={(r['frame_index'],r['gt_instance_index'],r['native_slot']):r for r in refs};assert len(refmap)==2424
    nominal=np.full(1840,np.nan);nominal_L=np.full(1840,np.nan);nominal_supported=np.zeros(1840,bool)
    for i,r in enumerate(mrrows):
        if r['role_correspondence_supported'] and r['selected_gt_instance_index'] is not None:
            rr=refmap[(r['frame_index'],r['selected_gt_instance_index'],r['native_slot'])]
            assert rr['id']==r['id'] and rr['source_id']==r['source_id'] and rr['split']==r['split']
            if r['query_xy'] is not None and rr['reference_xy'] is not None:
                q=np.asarray(r['query_xy'],float);ref=np.asarray(rr['reference_xy'],float)
                if np.isfinite(q).all() and np.isfinite(ref).all() and r['L'] is not None and r['L']>0:
                    nominal[i]=float(np.linalg.norm(q-ref));nominal_L[i]=nominal[i]/r['L'];nominal_supported[i]=True
    packets=[];inputs=[bind(R/'CONTROL_COMPLETE.json'),control['protocol'],protocol['source_supervision_seal'],protocol['source_labels'],previous['public_labels'],bind(reference_path)]+[v['GT'] for v in previous['native']]+protocol['quality_labels']
    for arm in ARMS:
        pp=[]
        for name,rr in zip(NAMES,rows):
            path=R/arm/f'FINAL_{name}_OUTPUTS.npz';a=packet(path);inputs.append(bind(path));pp.append(a)
            assert a['raw_three_logits'].shape==(len(rr),3)
            assert np.array_equal(a['global_index'],np.arange(len(rr)))
            for i,r in enumerate(rr):
                key='occurrence_id' if 'occurrence_id' in a else 'query_id'
                assert str(a[key][i])==r.get('query_id',r.get('id','')+':native'+str(r['native_slot']))
            av=a['query_available'];z=a['raw_three_logits']
            assert np.isfinite(z[av]).all()
            assert np.array_equal(a['primary_visibility_prediction'],np.where(av,(z[:,0]>=0).astype(np.int8),-1))
            assert np.array_equal(a['secondary_entity_gated_visibility'],np.where(av&(z[:,1]>=0),(z[:,0]>=0).astype(np.int8),-1))
        packets.append(pp)
    for a,b in zip(*packets):
        for k in a:
            if k not in ('raw_three_logits','primary_visibility_prediction','secondary_entity_gated_visibility'):
                assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes()
    # Replay every entity and quality draw from fixed TRAIN-only label pools.
    av=packets[0][3]['query_available'];tr=folds[3]=='SOURCE_DEVELOPMENT'
    ep=[np.flatnonzero(tr&av&(entity==t)) for t in (0,1)]
    qp=[[np.flatnonzero((folds[i]=='TRAIN')&packets[0][i]['query_available']&(quals[i]==t)) for t in (0,1)] for i in (0,1)]
    pi=np.flatnonzero((folds[2]=='TRAIN_LAB')&packets[0][2]['query_available']&(targets[2]>=0))
    vi=np.flatnonzero(tr&av&(targets[3]>=0))
    assert len(ep[0])==365 and len(ep[1])==638 and len(pi)==43 and len(vi)==390
    prefix=[]
    with check(previous['C32_draws']).open() as f:
        for step in range(1,2001):
            r=json.loads(next(f));assert r['update']==step
            prefix.append(tuple(np.array(r[k],np.int64) for k in ('native_occurrences','extra_native_occurrences','native_pair_pool_positions','extra_pair_pool_positions')))
    exposure=np.zeros(1840,np.int64);qexp=[np.zeros(6144,np.int64),np.zeros(20480,np.int64)]
    hashes=[];replays=[]
    for arm in ARMS:
        trace=R/arm/'ALL2000_DRAWS_AND_LOSSES.jsonl';inputs.append(bind(trace));rng=np.random.default_rng(1540);dh=hashlib.sha256();count=0
        ownex=np.zeros(1840,np.int64);ownq=[np.zeros(6144,np.int64),np.zeros(20480,np.int64)]
        for step,r in enumerate(lines(trace),1):
            assert r['update']==step and r['auxiliary_weight']==ARMS.index(arm)
            ei=np.r_[rng.choice(ep[0],32,replace=True),rng.choice(ep[1],32,replace=True)].astype(np.int64)
            qi=[np.r_[rng.choice(p[0],16,replace=True),rng.choice(p[1],16,replace=True)].astype(np.int64) for p in qp]
            oi,ni,opos,npos=prefix[step-1]
            for k,a in [('old_indices',oi),('new_indices',ni),('entity_indices',ei),('quality_old_indices',qi[0]),('quality_new_indices',qi[1]),('MR_visibility_indices',vi)]:assert np.array_equal(np.asarray(r[k]),a),(arm,step,k)
            for a in (oi,ni,opos,npos,pi,vi,ei,*qi):dh.update(a.astype('<i8').tobytes())
            np.add.at(ownex,ei,1)
            for j in (0,1):np.add.at(ownq[j],qi[j],1)
            count+=1
        assert count==2000 and dh.hexdigest()==control['draw_sha'];hashes.append(dh.hexdigest())
        if arm==ARMS[0]:exposure=ownex;qexp=ownq
        else:assert np.array_equal(exposure,ownex) and all(np.array_equal(x,y) for x,y in zip(qexp,ownq))
        replays.append(dict(arm=arm,updates=count,draw_sha=dh.hexdigest(),all_draws_exact=True))
    # All-row ledgers remain source evidence, never new labels or filters.
    ledger=H/'ALL28976_SAVED_THREE_OUTPUT_DIAGNOSIS.jsonl'
    with ledger.open('x') as f:
        for j,(name,rr) in enumerate(zip(NAMES,rows)):
            for i,r in enumerate(rr):
                d=dict(corpus=name,global_index=i,query_id=r.get('query_id',r.get('id','')+':native'+str(r['native_slot'])),split=r['split'],native_slot=r['native_slot'],teacher_state=str(states[j][i]),point_quality_truth=int(quals[j][i]),entity_truth=int(entity[i]) if j==3 else None,original_input_valid=bool(packets[0][j]['query_available'][i]),entity_sample_count=int(exposure[i]) if j==3 else None)
                if j==3:d.update(frame_index=r['frame_index'],source_id=r['source_id'],source_reason=r['source_role_review_reason'],entity_reason=r['association_reason'],nominal_reference_distance_px=nominal[i],nominal_reference_distance_over_L=nominal_L[i],nominal_same_native_supported=bool(nominal_supported[i]),reference_is_cm_uncertain_not_pixel_gold=True)
                for k,arm in enumerate(('C35','C36')):
                    a=packets[k][j];d[arm]=dict(raw_three_logits=a['raw_three_logits'][i].tolist(),primary_visibility=int(a['primary_visibility_prediction'][i]),secondary_entity_gated_visibility=int(a['secondary_entity_gated_visibility'][i]))
                f.write(json.dumps(clean(d),ensure_ascii=False,allow_nan=False)+'\n')
    conditionals={}; transitions={}
    for j,name in enumerate(NAMES):
        conditionals[name]={};transitions[name]={}
        for fold in np.unique(folds[j]):
            fm=folds[j]==fold;av=packets[0][j]['query_available'];out={}
            for state in list(np.unique(states[j]))+['ROLE_UNKNOWN_COMBINED']:
                m=fm & ((targets[j]<0) if state=='ROLE_UNKNOWN_COMBINED' else (states[j]==state));d=dict(all_rows=int(m.sum()),input_available=int((m&av).sum()),input_U=int((m&~av).sum()))
                for k,arm in enumerate(('C35','C36')):
                    z=packets[k][j]['raw_three_logits'];d[arm]={n:numbers(z[m&av,c]) for c,n in enumerate(('visibility','entity','quality'))}
                    d[arm]['available_visibility_entity_Pearson']=corr(z[m&av,0],z[m&av,1]);d[arm]['available_entity_quality_Pearson']=corr(z[m&av,1],z[m&av,2])
                out[str(state)]=d
            for target in (-1,0,1):
                m=fm&av&(quals[j]==target)
                out[f'QUALITY_TRUTH_{target}']={arm:numbers(packets[k][j]['raw_three_logits'][m,2]) for k,arm in enumerate(('C35','C36'))}
            conditionals[name][str(fold)]=out
            p35=packets[0][j]['primary_visibility_prediction'];p36=packets[1][j]['primary_visibility_prediction'];gate=packets[1][j]['secondary_entity_gated_visibility'];z35=packets[0][j]['raw_three_logits'];z36=packets[1][j]['raw_three_logits'];y=targets[j]
            transitions[name][str(fold)]=dict(C35_CM=cm(y,p35,fm),C36_CM=cm(y,p36,fm),C36_gated_CM=cm(y,gate,fm),all_rows=int(fm.sum()),unknown_truth=int((fm&(y<0)).sum()),original_inputU=int((fm&~av).sum()),D_correct_to_falseE=int((fm&(y==0)&(p35==0)&(p36==1)).sum()),E_falseD_to_correct=int((fm&(y==1)&(p35==0)&(p36==1)).sum()),E_correct_rejected_by_entity=int((fm&(y==1)&(p36==1)&(gate==-1)).sum()),D_falseE_rejected_by_entity=int((fm&(y==0)&(p36==1)&(gate==-1)).sum()),visibility_logit_delta_by_state={s:numbers((z36-z35)[fm&av&(states[j]==s),0]) for s in np.unique(states[j])})
    z35=packets[0][3]['raw_three_logits'];z36=packets[1][3]['raw_three_logits'];p35=packets[0][3]['primary_visibility_prediction'];p36=packets[1][3]['primary_visibility_prediction'];y=targets[3]
    nominal_report={}
    for fold,fm in [('ALL',np.ones(1840,bool)),('SOURCE_DEVELOPMENT',tr),('SOURCE_VALIDATION',~tr)]:
        nominal_report[fold]={}
        for state in ('D','E','ROLE_UNKNOWN_COMBINED'):
            sm=(y<0) if state=='ROLE_UNKNOWN_COMBINED' else states[3]==state
            for suffix,extra in [('ALL',np.ones(1840,bool)),('ENTITY_NEGATIVE',z36[:,1]<0),('ENTITY_NONNEGATIVE',z36[:,1]>=0),('QUALITY_NEGATIVE',z36[:,2]<0)]:
                allmask=fm&av&sm&extra;m=allmask&nominal_supported;x=nominal[m]
                nominal_report[fold][state+'::'+suffix]=dict(rows=int(allmask.sum()),exact_same_slot_nominal_distance_defined=int(m.sum()),undefined_or_unsupported=int((allmask&~nominal_supported).sum()),distance_px=numbers(x),distance_over_L=numbers(nominal_L[m]),distance_gt20=int((x>20).sum()),distance_gt100=int((x>100).sum()),quality_negative_by_nominal_distance_band={band:numbers(z36[m&bm,2]) for band,bm in [('LE20',nominal<=20),('GT20_LE100',(nominal>20)&(nominal<=100)),('GT100',nominal>100)]})
    expose={}
    for key,m in [('entity_positive_all',tr&av&(entity==1)),('entity_negative_all',tr&av&(entity==0)),('D_entity_positive',tr&av&(entity==1)&(y==0)),('E_entity_positive',tr&av&(entity==1)&(y==1)),('unknown_role_entity_positive',tr&av&(entity==1)&(y<0)),('E_later_entity_rejected',tr&av&(y==1)&(z36[:,1]<0)),('E_later_entity_accepted',tr&av&(y==1)&(z36[:,1]>=0))]:
        x=exposure[m];expose[key]=dict(rows=int(m.sum()),total_draws=int(x.sum()),minimum=int(x.min()) if len(x) else None,median=float(np.median(x)) if len(x) else None,maximum=int(x.max()) if len(x) else None,never_sampled=int((x==0).sum()),samples=numbers(x))
    # Fixed post-hoc diagnostic cases are selected before any RGB pixel read.
    cases=[]
    rules=[('TRAIN_TRUE_E_ENTITY_REJECT',tr&av&(y==1)&(p36==1)&(z36[:,1]<0),z36[:,1],2),('VAL_TRUE_E_ENTITY_REJECT',~tr&av&(y==1)&(p36==1)&(z36[:,1]<0),z36[:,1],2),('VAL_D_DAMAGED',~tr&av&(y==0)&(p35==0)&(p36==1),-(z36[:,0]-z35[:,0]),2)]
    for category,m,key,n in rules:
        ix=sorted(np.flatnonzero(m),key=lambda i:(float(key[i]),int(i)))[:n]
        for i in ix:cases.append(dict(category=category,global_index=int(i)))
    for category,m in [('TRAIN_WRONG_ENTITY_CORRECT',tr&av&(entity==0)&(z36[:,1]<0)),('VAL_WRONG_ENTITY_CORRECT',~tr&av&(entity==0)&(z36[:,1]<0))]:
        i=min(np.flatnonzero(m),key=lambda i:(float(z36[i,1]),int(i)));cases.append(dict(category=category,global_index=int(i)))
    evidence=read(S/'original_query_evidence_full230/ALL230_SAVED_CROP_EVIDENCE.json');er=evidence['records'];assert len(er)==230
    for d in cases:
        i=d['global_index'];r=mrrows[i];d.update(query_id=r['query_id'],frame_index=r['frame_index'],source_id=r['source_id'],native_slot=r['native_slot'],teacher_state=r['role_state'],entity_truth=r['entity_target'],split=r['split'],entity_sample_count=int(exposure[i]),query_xy=r['query_xy'],L=r['L'],nominal_reference_distance_px=nominal[i],nominal_reference_distance_over_L=nominal_L[i],nominal_same_native_supported=bool(nominal_supported[i]),source_role_reason=r['source_role_review_reason'],association_reason=r['association_reason'],C35_logits=z35[i].tolist(),C36_logits=z36[i].tolist(),saved_crop_packet=er[r['frame_index']]['saved_crop_packet'],original_query_evidence=r['original_query_evidence'],expected_RGB96_sha256=er[r['frame_index']]['rows'][r['native_slot']]['RGB96_HWC_sha256'],expected_valid96_sha256=er[r['frame_index']]['rows'][r['native_slot']]['valid96_sha256'])
    selection=dict(completed=True,source_only=True,posthoc_diagnostic_selection_before_any_RGB_read=True,groundtruth_unchanged=True,cases=clean(cases),rule='within each declared category strongest signed error or largest visibility delta; ties original global index. Wrong-entity one TRAIN and one VAL. No model/threshold selection.')
    write(H/'FIXED8_CASE_SELECTION_BEFORE_RGB.json',selection)
    result=dict(completed=True,PASS=True,source_only=True,control_complete=bind(R/'CONTROL_COMPLETE.json'),all_rows=28976,all_MR1840_preserved=True,conditions=conditionals,transitions=transitions,exposure=expose,draw_replays=replays,entity_pools=dict(negative=len(ep[0]),positive=len(ep[1]),positive_D=int((tr&av&(entity==1)&(y==0)).sum()),positive_E=int((tr&av&(entity==1)&(y==1)).sum()),quality_pool_counts=[[len(p) for p in pool] for pool in qp],nominal_reference_distance_diagnostic=nominal_report,nominal_distance_is_not_real_accuracy_gold_or_filter=True,reference_matching='exact frame_index/selected_gt_instance_index/original_native_slot; no nearest owner or permutation',real_point_quality_gold=False,real_quality_logits_are_not_accuracy_measurements=True,shared_hidden64_fact=True,auxiliary_entity_and_quality_added_together_cannot_identify_separate_causal_effects=True,selected_model=None,new_model_feature_weight_private_or_fit_calls=0,ledger=bind(ledger),case_selection=bind(H/'FIXED8_CASE_SELECTION_BEFORE_RGB.json'),inputs=inputs)
    write(H/'RESULTS.json',clean(result));write(H/'DIAGNOSIS_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),results=bind(H/'RESULTS.json'),ledger=bind(ledger),selection=bind(H/'FIXED8_CASE_SELECTION_BEFORE_RGB.json'),source_only=True,no_new_model_or_fit=True,selected_model=None))
    print(json.dumps(dict(completed=True,results=bind(H/'RESULTS.json'),exposure=expose,MR_transitions=transitions['MR1840'],cases=cases),ensure_ascii=False))

if __name__=='__main__':main()
