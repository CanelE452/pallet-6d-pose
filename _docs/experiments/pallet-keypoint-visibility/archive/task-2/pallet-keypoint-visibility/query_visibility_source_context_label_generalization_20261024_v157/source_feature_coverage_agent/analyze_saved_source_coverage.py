"""V157 descriptive source-only coverage; no torch, fit, model or RGB reads."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import csv, hashlib, json, math
from collections import Counter
import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
O = Path(__file__).resolve().parent
V155 = P/'query_visibility_independent_auxiliary_source_control_20261024_v155'
V154 = P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154'
BINDINGS = []

def guard(path):
    path = Path(path)
    low = str(path).lower()
    assert 'v112' not in low and 'current223' not in low and '/private' not in low, path
    assert path.is_absolute() and str(path).startswith(str(P)+'/'), path
    return path

def bind(path, expected=None, kind='read'):
    path = guard(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected: assert sha == expected, path
    b = dict(path=str(path), sha256=sha, kind=kind, prior_hash_verified=bool(expected))
    if b not in BINDINGS: BINDINGS.append(b)
    return path

def read(path, expected=None): return json.loads(bind(path, expected).read_text())
def bound(b): return bind(b['path'], b['sha256'])
def jlines(b): return [json.loads(s) for s in bound(b).read_text().splitlines() if s]
def pack(b):
    with np.load(bound(b), allow_pickle=False) as a: return {k:a[k] for k in a.files}
def clean(v):
    if isinstance(v, np.generic): v=v.item()
    if isinstance(v, float) and not math.isfinite(v): return None
    if isinstance(v, np.ndarray): return [clean(x) for x in v.tolist()]
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    return v
def writej(name, obj): (O/name).write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(name, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict,np.ndarray)) else clean(v) for k,v in r.items()})
def desc(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a), **({k:float(v) for k,v in zip(['min','p10','p25','median','p75','p90','max'],np.quantile(a,[0,.1,.25,.5,.75,.9,1]))} if len(a) else {}))

# The analysis rule artifact exists before any feature or score bodies.
rules=read(O/'ANALYSIS_RULES_BEFORE_FEATURE_AND_SCORE_BODIES.json')
tie_rules=read(O/'TIE_AWARE_DISTANCE_AMENDMENT_BEFORE_TIE_COUNTS.json')
TIE_TOLERANCE=tie_rules['fixed_numerical_tie_tolerance']
scope=read(P/'query_visibility_source_context_label_generalization_20261024_v157/ROOT_SOURCE_ONLY_SCOPE_BEFORE_BODIES.json')
p=read(V155/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json',scope['source_protocol']['sha256'])
old=read(bound(p['parent_protocol']))
for key in ('code','helper','head'): bound(p[key]) # hash source code only, never execute it
p154=read(V154/'head_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json')
bound(p154['code'])
layout_sources=[
 ('query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/prepare_spatial_query_features.py','cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e','36-64: T,Q,F,C,P,8 scalars'),
 ('query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py','cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578','33-42: literal local96/central32 preprocessing'),
 ('query_visibility_public_native_features_20261024_v147/feature_preparation/encode_saved_public512.py','2809691ec4b773c7c336733579b072d67460f3720c71adcdba17b2f9d528b17c','252-294: support fractions and independent availability'),
 ('query_visibility_mr6d_full_public_source_20261024_v153/initializer_feature_preparation/encode_full230.py','0294e12c7dcae7ede4f0d76d6766decf1b2f3e8e626b4e6645d31d1108e4d2d5','141-187: feature construction, support, availability'),
 ('query_visibility_actual_initializer_source_20261024_v132/frozen_visibility_preparation/run_frozen_visibility.py','8241261ca58e6c077594d2de3c4397ec71497565284653a6d4463b979aa72021','253-258,310-333,365-378: shared producer and saved arrays'),
 ('query_visibility_continuous_native_source_20261024_v136/frozen_feature_preparation/run_frozen_C16.py','0d62885528a2bb477f4010ca3bd69d77c84c216da61edc04e284203166d215c2','273-277: reuses V132 run_c16'),
 ('query_visibility_boundary_query_source_20261023_v125/feature_baseline_preparation/run_frozen_boundary_source.py','c59d986a29668c04e19500ac1418062ad1912fdcdcfd5ed0e56209e8225be1e1','44-48: stricter factual synthetic availability')]
for relative,digest,meaning in layout_sources: bind(P/relative,digest,kind='verified_frozen_descriptor_source_code')

datasets={}
mrlabels=jlines(p['source_labels']);mrfacts=jlines(p['MR_inputfacts'])
palabels=jlines(old['public_labels']);pafacts=jlines(p['Pallox_inputfacts'])
for family,bb,labels,facts in [('MR1840',p['MR_features'],mrlabels,mrfacts),('PALLOX512',p['Pallox_features'],palabels,pafacts)]:
    a=pack(bb);n=len(a['features']);rows=[]
    assert n==len(labels)==len(facts)
    for i,(r,f) in enumerate(zip(labels,facts)):
        assert i==r['global_index']==f['global_index']==a['global_index'][i]
        assert r['id']==f['frame_id']==str(a['frame_id'][i])
        assert r['original_input_valid']==bool(a['query_available'][i])
        state=r['role_state'] if family=='MR1840' else r['semantic_state']
        role=r['role_target'] if family=='MR1840' else {'D':0,'E':1}.get(state,-1)
        rows.append(dict(family=family,global_index=i,frame_index=int(a['frame_index'][i]),frame_id=str(a['frame_id'][i]),
            query_id=str(a['query_id'][i]),source_id=str(a['source_id'][i]),split=str(a['split'][i]),native_slot=int(a['native_slot'][i]),
            state=state,role_target=role,entity_target=r.get('entity_target',-1),point_quality_label=r.get('point_quality_label',-1),
            query_available=bool(a['query_available'][i]),feature_constructed=bool(a.get('feature_constructed',np.ones(n,bool))[i]),
            native_continuous_valid=bool(a['native_continuous_valid'][i]),nearest_valid=bool(a['nearest_valid'][i]),
            context_available=bool(a['context_available'][i]),local_central4_available=bool(a['local_central4_available'][i]),
            Q_inside=bool(a['unit_Q_inside'][i]),no_detection=bool(a.get('no_detection',np.zeros(n,bool))[i]),
            L=float(a['L'][i]),support_fraction_target=float(a['support_fractions'][i,0]),
            support_fraction_local96=float(a['support_fractions'][i,1]),support_fraction_central32=float(a['support_fractions'][i,2]),
            human_gold=r.get('human_gold',False),exact_real_point_gold=r.get('exact_real_point_gold',False),
            role_correspondence_supported=r.get('role_correspondence_supported',None),
            frame_association_state=r.get('frame_association_state',r.get('root_selected_entity_status')),
            occluder_or_uncertainty=r.get('occluder_or_uncertainty'),original_label_record=r))
    datasets[family]=dict(a=a,rows=rows)
for family,native,qb in zip(('OLD6144','NEW20480'),old['native'],p['quality_labels']):
    a=pack(native['features']);qr=jlines(qb);n=len(a['features']);rows=[]
    inputfacts_binding=dict(path=str(Path(native['features']['path']).parent/'INPUT_ROWS_AND_CROP_FACTS.json'),
        sha256={'OLD6144':'2a205f713eff6cc14d8ca7da06e66a0ca2cac8d2eea962ff2cf2f865628d6c48','NEW20480':'ea74c43813a3b2d64bc51abd36a6cd3062c70bd3784b0f6cf0971cc6cee84b68'}[family])
    sf=read(bound(inputfacts_binding))['rows']
    assert n==len(qr)
    for i,r in enumerate(qr):
        assert i==r['global_index']==int(a['global_index'][i])
        f=sf[i];assert f['occurrence_id']==r['query_id'] and f['native_slot']==int(a['native_slot'][i])
        query=np.asarray(r['original_query_xy'] if r['original_query_xy'] is not None else [np.nan,np.nan])
        assert np.array_equal(query,a['query_xy'][i],equal_nan=True)
        assert np.isclose(np.nan if r['original_roi_side'] is None else r['original_roi_side'],a['L'][i],equal_nan=True)
        rows.append(dict(family=family,global_index=i,frame_index=int(a['scene_index'][i]),frame_id=r['query_id'].split(':native')[0],
            query_id=r['query_id'],source_id=str(a['occurrence_id'][i]),split=r['split'],native_slot=int(a['native_slot'][i]),
            state=r['original_role_teacher_state_UNCHANGED'],role_target={'D':0,'E':1}.get(r['original_role_teacher_state_UNCHANGED'],-1),
            entity_target=-1,point_quality_label=-1 if r['point_quality_label'] is None else r['point_quality_label'],
            query_available=bool(a['query_available'][i]),feature_constructed=bool(a['feature_constructed'][i]),
            native_continuous_valid=bool(a['native_continuous_valid'][i]),nearest_valid=bool(a['native_nearest_valid'][i]),
            factual_query_available=bool(a['factual_query_available'][i]),context_available=bool(a['context_available'][i]),
            local_central4_available=f['RGB_central4_available'],
            Q_inside=bool(a['Q_inside'][i]),no_detection=bool(a['no_detection'][i]),L=float(a['L'][i]),
            support_fraction_target=float(a['features'][i,1924]),support_fraction_local96=float(a['features'][i,1925]),support_fraction_central32=float(a['features'][i,1926]),
            human_gold=False,exact_real_point_gold=False,quality_reason=r['point_quality_reason'],
            original_visibility_training_mask=r['role_visibility_training_mask_UNCHANGED'],
            original_label_record=r))
    datasets[family]=dict(a=a,rows=rows)

ARM_DIRS=[V154/'runs/C35_C36_SEPARATE_QUERY_FIXED2000/C35_MATCHED_VISIBILITY_ONLY_CONTROL',
          V154/'runs/C35_C36_SEPARATE_QUERY_FIXED2000/C36_MATCHED_VISIBILITY_PLUS_ENTITY_AND_POINT_QUALITY',
          V155/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000/C37_WARM_VISIBILITY_INDEPENDENT_AUX_FIXED8000',
          V155/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000/C38_FRESH_VISIBILITY_INDEPENDENT_AUX_FIXED8000']
saved_errors=[];frames=[];arm_summaries={};score_provenance=[]
metric_reproduction=[]
for ad in ARM_DIRS:
    arm=ad.name.split('_')[0]; complete=read(ad/'FIT_COMPLETE.json');assert complete['completed']
    sr=read(bound(complete['source_results']));assert sr['completed']
    arm_summaries[arm]={}
    for family,d in datasets.items():
        path=bind(ad/f'FINAL_{family}_OUTPUTS.npz',kind='exact_saved_output_filename_from_bound_code_not_prior_NPZ_hash')
        with np.load(path,allow_pickle=False) as z: s={k:z[k] for k in z.files}
        a=d['a'];rows=d['rows'];av=a['query_available'];raw=s['raw_three_logits'];n=len(rows)
        assert raw.shape==(n,3) and np.array_equal(s['global_index'],a['global_index'])
        assert np.array_equal(s['query_available'],av)
        if 'query_id' in a: assert np.array_equal(s['query_id'],a['query_id'])
        pv=np.where(av,(raw[:,0]>=0).astype(np.int8),-1)
        pe=np.where(av,(raw[:,1]>=0).astype(np.int8),-1)
        pq=np.where(av,(raw[:,2]>=0).astype(np.int8),-1)
        assert np.array_equal(pv,s['primary_visibility_prediction'])
        assert np.array_equal(np.where(av&(raw[:,1]>=0),pv,-1),s['secondary_entity_gated_visibility'])
        for i,r in enumerate(rows):
            for task,col,pred in [('visibility',0,pv),('entity',1,pe),('quality',2,pq)]:
                target=r[dict(visibility='role_target',entity='entity_target',quality='point_quality_label')[task]]
                r[f'{arm}_{task}_logit']=float(raw[i,col]);r[f'{arm}_{task}_pred']=int(pred[i])
                r[f'{arm}_{task}_error_all_known']=None if target<0 else int(pred[i]!=target)
        groups={}
        for split in sorted(set(r['split'] for r in rows)):
            sm=np.array([r['split']==split for r in rows])
            groups[split]={}
            for task,pred,key in [('visibility',pv,'role_target'),('entity',pe,'entity_target'),('quality',pq,'point_quality_label')]:
                y=np.array([r[key] for r in rows]);known=sm&(y>=0);valid=known&av
                cm=[[int((known&(y==t)&(pred==v)).sum()) for v in (0,1,-1)] for t in (0,1)]
                rec=dict(arm=arm,family=family,split=split,task=task,rows=int(sm.sum()),known=int(known.sum()),
                    known_available=int(valid.sum()),errors_all_known=int((known&(pred!=y)).sum()),
                    errors_available=int((valid&(pred!=y)).sum()),unavailable_known=int((known&~av).sum()),
                    CM_rows0_1_columns0_1_U=cm)
                groups[split][task]=rec;saved_errors.append(rec)
                if family=='MR1840' and task in ('visibility','entity'):
                    original=sr['MR'][split][task]['CM_rows0_1_columns0_1_U']
                    assert original==cm,(arm,split,task,original,cm)
                    originalvalid=sr['MR'][split][task+'_inputvalid']['CM_rows0_1_columns0_1_U']
                    validcm=[[int((valid&(y==t)&(pred==v)).sum()) for v in (0,1,-1)] for t in (0,1)]
                    assert originalvalid==validcm
                    metric_reproduction.append(dict(arm=arm,split=split,task=task,all_known_CM_exact=True,inputvalid_CM_exact=True))
        arm_summaries[arm][family]=groups
        if family=='MR1840':
            for fi in range(230):
                ii=np.flatnonzero(a['frame_index']==fi);rr=[rows[i] for i in ii]; assert len(ii)==8
                rec=dict(arm=arm,frame_index=fi,frame_id=rr[0]['frame_id'],source_id=rr[0]['source_id'],split=rr[0]['split'],
                    available=sum(r['query_available'] for r in rr),constructed=sum(r['feature_constructed'] for r in rr),
                    states=dict(Counter(r['state'] for r in rr)),L=rr[0]['L'])
                for task,key in [('visibility','role_target'),('entity','entity_target')]:
                    known=[r for r in rr if r[key]>=0]
                    rec[f'{task}_known']=len(known);rec[f'{task}_known_available']=sum(r['query_available'] for r in known)
                    rec[f'{task}_errors_all_known']=sum(r[f'{arm}_{task}_error_all_known'] for r in known)
                    rec[f'{task}_errors_available']=sum(r[f'{arm}_{task}_error_all_known'] for r in known if r['query_available'])
                rec['knownE_available']=sum(r['query_available'] and r['role_target']==1 for r in rr)
                rec['knownE_FN_available']=sum(r['query_available'] and r['role_target']==1 and r[f'{arm}_visibility_pred']==0 for r in rr)
                frames.append(rec)
    score_provenance.append(dict(arm=arm,fit_complete=str(ad/'FIT_COMPLETE.json'),source_results_binding=complete['source_results'],
        saved_NPZ_prior_hash_binding_present=False,layout='raw_three_logits=[external_occlusion,target_entity,coarse_point_quality]',threshold=0))

coverage=[]
for family,d in datasets.items():
    rows=d['rows']
    for split in sorted(set(r['split'] for r in rows)):
        for state in sorted(set(r['state'] for r in rows)):
            rr=[r for r in rows if r['split']==split and r['state']==state]
            if not rr: continue
            coverage.append(dict(family=family,split=split,state=state,rows=len(rr),frames=len(set(r['frame_index'] for r in rr)),
                **{k:sum(bool(r.get(k,False)) for r in rr) for k in ['query_available','feature_constructed','native_continuous_valid','nearest_valid','context_available','local_central4_available','Q_inside','no_detection']},
                role_target_counts=dict(Counter(r['role_target'] for r in rr)),entity_target_counts=dict(Counter(r['entity_target'] for r in rr)),
                point_quality_label_counts=dict(Counter(r['point_quality_label'] for r in rr)),L=desc([r['L'] for r in rr])))

# Exact descriptor blocks verified in frozen source producer; the first block is target crop CLS.
components={'raw_full':(0,1928),'target_context_CLS':(0,384),'query_context_tokens':(384,768),
            'local96_CLS':(768,1152),'central32_CLS':(1152,1536),'central_local_patch_mean':(1536,1920)}
mr=datasets['MR1840'];ma=mr['a'];rr=mr['rows'];n=len(rr)
train=np.array([r['split']=='SOURCE_DEVELOPMENT' for r in rr]);state=np.array([r['state'] for r in rr]);fi=ma['frame_index']
def norm_component(a, sl):
    x=a['features'][:,sl[0]:sl[1]].astype(np.float32);norm=np.linalg.norm(x,axis=1)
    ok=a['query_available']&a.get('feature_constructed',np.ones(len(x),bool))&np.isfinite(x).all(1)&(norm>0)
    z=np.zeros_like(x);z[ok]=x[ok]/norm[ok,None]
    return z,ok
nearest_rows=[];distance_summary=[];component_arrays={}
for comp,sl in components.items():
    z,ok=norm_component(ma,sl);ref=np.flatnonzero(ok&train);q=np.flatnonzero(ok)
    dist=np.clip(1-z[q]@z[ref].T,0,2);dist[fi[q,None]==fi[ref][None,:]]=np.inf
    nr=[]
    for qi,ix in enumerate(q):
        same=state[ref]==state[ix]; different=~same;d=dist[qi]
        def nearest(mask):
            where=np.flatnonzero(mask&np.isfinite(d))
            if not len(where):return (None,None)
            pos=where[np.argmin(d[where])];return int(ref[pos]),float(d[pos])
        nn,nd=nearest(np.ones(len(ref),bool));sn,sd=nearest(same);dn,dd=nearest(different)
        rec=dict(component=comp,global_index=int(ix),query_id=rr[ix]['query_id'],frame_index=int(fi[ix]),split=rr[ix]['split'],state=state[ix],
            nearest_train_index=nn,nearest_train_state=None if nn is None else state[nn],nearest_train_distance=nd,
            same_state_nearest_train_index=sn,same_state_distance=sd,different_state_nearest_train_index=dn,different_state_distance=dd,
            same_over_different_distance_ratio=sd/dd if sd is not None and dd is not None and dd>0 else None)
        rec['nearest_state_tie_caveat']='single row argmin; target_context rows in a frame are duplicates'
        for arm in arm_summaries:
            rec[f'{arm}_visibility_error']=rr[ix][f'{arm}_visibility_error_all_known'];rec[f'{arm}_entity_error']=rr[ix][f'{arm}_entity_error_all_known']
        nr.append(rec)
    nearest_rows.extend(nr);component_arrays[comp]=nr
    for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
        for st in sorted(set(state)):
            ss=[r for r in nr if r['split']==split and r['state']==st]
            if ss:distance_summary.append(dict(component=comp,split=split,state=st,n=len(ss),
                nearest_distance=desc([r['nearest_train_distance'] for r in ss]),same_state_distance=desc([r['same_state_distance'] for r in ss]),
                different_state_distance=desc([r['different_state_distance'] for r in ss]),ratio=desc([r['same_over_different_distance_ratio'] for r in ss]),
                nearest_same_state_fraction=sum(r['state']==r['nearest_train_state'] for r in ss)/len(ss)))

# Auxiliary source coverage: each MR query to original eligible source TRAIN rows, no relabeling.
aux_nearest=[]
for family in ('PALLOX512','OLD6144','NEW20480'):
    ad=datasets[family];ar=ad['rows'];atr=np.array([r['split'].startswith('TRAIN') for r in ar])
    for comp in ('target_context_CLS','local96_CLS','central32_CLS'):
        z,ok=norm_component(ma,components[comp]);az,aok=norm_component(ad['a'],components[comp]);ref=np.flatnonzero(aok&atr);q=np.flatnonzero(ok)
        for start in range(0,len(q),128):
            ix=q[start:start+128];dist=np.clip(1-z[ix]@az[ref].T,0,2);best=np.argmin(dist,axis=1)
            for jj,qq in enumerate(ix):
                nn=int(ref[best[jj]]);aux_nearest.append(dict(component=comp,reference_family=family,global_index=int(qq),query_id=rr[qq]['query_id'],
                    split=rr[qq]['split'],state=rr[qq]['state'],nearest_source_index=nn,nearest_source_query_id=ar[nn]['query_id'],
                    nearest_source_split=ar[nn]['split'],nearest_source_state=ar[nn]['state'],distance=float(dist[jj,best[jj]])))

# D/E overlap. Same-frame duplicated context blocks are shown separately from leave-frame-out neighbors.
overlap=[];de_nearest=[]
for comp,sl in components.items():
    z,ok=norm_component(ma,sl);de=ok&np.isin(state,['D','E']);ix=np.flatnonzero(de)
    d=np.clip(1-z[ix]@z[ix].T,0,2)
    sameframe=fi[ix,None]==fi[ix][None,:];cross=state[ix,None]!=state[ix][None,:]
    vals=d[sameframe&cross]
    sameframe_pairs=int((sameframe&cross).sum()//2)
    rec=dict(component=comp,D_available=int((de&(state=='D')).sum()),E_available=int((de&(state=='E')).sum()),
        same_frame_DE_unordered_pairs=sameframe_pairs,same_frame_DE_cosine_distance=desc(vals),
        same_frame_DE_distance_le1e_6=int((vals<=1e-6).sum()//2))
    ref=np.flatnonzero(de&train);dx=np.clip(1-z[ix]@z[ref].T,0,2);dx[fi[ix,None]==fi[ref][None,:]]=np.inf
    for k,qq in enumerate(ix):
        same=state[ref]==state[qq];other=~same
        ds=float(np.min(dx[k,same]));do=float(np.min(dx[k,other]));nn=int(ref[np.argmin(dx[k])])
        sameix=int(ref[np.flatnonzero(same)[np.argmin(dx[k,same])]])
        otherix=int(ref[np.flatnonzero(other)[np.argmin(dx[k,other])]])
        de_nearest.append(dict(component=comp,global_index=int(qq),query_id=rr[qq]['query_id'],split=rr[qq]['split'],state=state[qq],
            nearest_DE_train_index=nn,nearest_DE_train_state=state[nn],same_DE_state_distance=ds,opposite_DE_state_distance=do,
            nearest_same_DE_state_index=sameix,nearest_same_DE_state_query_id=rr[sameix]['query_id'],
            nearest_opposite_DE_state_index=otherix,nearest_opposite_DE_state_query_id=rr[otherix]['query_id'],
            same_over_opposite_ratio=ds/do if do>0 and np.isfinite(ds+do) else None,
            same_minus_opposite_distance=ds-do,
            tie_aware_support='TIE' if abs(ds-do)<=TIE_TOLERANCE else ('SAME_CLOSER' if ds<do else 'OPPOSITE_CLOSER')))
    for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
        ss=[r for r in component_arrays[comp] if r['split']==split and r['state'] in ('D','E')]
        rec[split]=dict(rows=len(ss),nearest_allstate_is_opposite_DE=sum(r['nearest_train_state'] in ('D','E') and r['nearest_train_state']!=r['state'] for r in ss),
            same_state_distance=desc([r['same_state_distance'] for r in ss]),different_state_distance=desc([r['different_state_distance'] for r in ss]),
            same_over_different_ratio=desc([r['same_over_different_distance_ratio'] for r in ss]))
    overlap.append(rec)

entity_nearest=[];entity_distance_summary=[]
z,ok=norm_component(ma,components['target_context_CLS'])
all_frame_reps=np.array([int(np.flatnonzero((fi==frame)&ok)[0]) for frame in range(230) if ((fi==frame)&ok).any()])
all_ref=all_frame_reps[train[all_frame_reps]];fd=np.clip(1-z[all_frame_reps]@z[all_ref].T,0,2)
fd[fi[all_frame_reps,None]==fi[all_ref][None,:]]=np.inf
frame_context=[]
for k,qq in enumerate(all_frame_reps):
    nn=int(all_ref[np.argmin(fd[k])]);frame_context.append(dict(frame_index=int(fi[qq]),frame_id=rr[qq]['frame_id'],split=rr[qq]['split'],
        representative_global_index=int(qq),nearest_train_frame_index=int(fi[nn]),nearest_distance=float(fd[k].min()),L=rr[qq]['L']))
frame_context_summary=[]
for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
    ss=[r for r in frame_context if r['split']==split]
    frame_context_summary.append(dict(split=split,eligible_frames=len(ss),nearest_distance=desc([r['nearest_distance'] for r in ss]),
        unique_nearest_train_frames=len(set(r['nearest_train_frame_index'] for r in ss))))
frame_reps=[]
for frame in range(230):
    ix=np.flatnonzero((fi==frame)&ok&np.array([r['entity_target']>=0 for r in rr]))
    if len(ix):
        assert len(set(rr[i]['entity_target'] for i in ix))==1
        assert np.array_equal(ma['features'][ix,:384],np.repeat(ma['features'][ix[:1],:384],len(ix),axis=0))
        frame_reps.append(int(ix[0]))
q=np.asarray(frame_reps);ref=q[train[q]];ey=np.array([r['entity_target'] for r in rr]);d=np.clip(1-z[q]@z[ref].T,0,2)
d[fi[q,None]==fi[ref][None,:]]=np.inf
for k,qq in enumerate(q):
    same=ey[ref]==ey[qq];different=~same;sd=float(np.min(d[k,same]));dd=float(np.min(d[k,different]));nn=int(ref[np.argmin(d[k])])
    entity_nearest.append(dict(frame_index=int(fi[qq]),frame_id=rr[qq]['frame_id'],representative_global_index=int(qq),split=rr[qq]['split'],entity_target=int(ey[qq]),
        nearest_train_frame_index=int(fi[nn]),nearest_train_entity_target=int(ey[nn]),nearest_distance=float(d[k].min()),same_entity_distance=sd,different_entity_distance=dd,
        same_over_different_ratio=sd/dd if dd>0 and np.isfinite(sd+dd) else None))
for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
    for target in (0,1):
        ss=[r for r in entity_nearest if r['split']==split and r['entity_target']==target]
        entity_distance_summary.append(dict(split=split,entity_target=target,frames=len(ss),
            nearest_same_entity_fraction=sum(r['nearest_train_entity_target']==target for r in ss)/len(ss) if ss else None,
            nearest_distance=desc([r['nearest_distance'] for r in ss]),same_entity_distance=desc([r['same_entity_distance'] for r in ss]),
            different_entity_distance=desc([r['different_entity_distance'] for r in ss]),same_over_different_ratio=desc([r['same_over_different_ratio'] for r in ss])))

concentration={}
for arm in arm_summaries:
    fr=[r for r in frames if r['arm']==arm and r['split']=='SOURCE_VALIDATION'];assert len(fr)==45
    concentration[arm]={}
    for task in ('visibility','entity'):
        key=f'{task}_errors_available';sr=sorted(fr,key=lambda r:(-r[key],r['frame_index']));total=sum(r[key] for r in fr)
        concentration[arm][task]=dict(all_45_frames_retained=True,available_error_total=total,frames_with_errors=sum(r[key]>0 for r in fr),
            top5_error_count=sum(r[key] for r in sr[:5]),top10_error_count=sum(r[key] for r in sr[:10]),
            top5_error_fraction=sum(r[key] for r in sr[:5])/total if total else None,
            ordered_frames=[dict(frame_index=r['frame_index'],frame_id=r['frame_id'],errors=r[key],known_available=r[f'{task}_known_available']) for r in sr])

association=[]
for arm in arm_summaries:
    for task,label in [('visibility','role_target'),('entity','entity_target')]:
        for comp,nr in component_arrays.items():
            for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
                for err in (0,1):
                    ss=[r for r in nr if r['split']==split and rr[r['global_index']][label]>=0 and r[f'{arm}_{task}_error']==err]
                    association.append(dict(arm=arm,task=task,component=comp,split=split,error=err,n=len(ss),
                        nearest_train_distance=desc([r['nearest_train_distance'] for r in ss]),
                        same_over_different_ratio=desc([r['same_over_different_distance_ratio'] for r in ss]),
                        L=desc([rr[r['global_index']]['L'] for r in ss])))

# Compact row outputs retain each original label record including uncertainty/provenance.
for family,d in datasets.items(): csvout(f'{family}_ALL_ROWS_FLAGS_LABELS_AND_SAVED_PREDICTIONS.csv',d['rows'])
csvout('SOURCE_COVERAGE_BY_SPLIT_STATE.csv',coverage)
csvout('ALL_ARMS_SAVED_ERRORS_BY_FAMILY_SPLIT_TASK.csv',saved_errors)
csvout('MR_ALL230_FRAMES_ALL_ARMS_SAVED_ERRORS.csv',frames)
csvout('MR_ALL1840_ELIGIBLE_FROZEN_NEAREST_TRAIN.csv',nearest_rows)
csvout('MR_TO_AUXILIARY_TRAIN_FROZEN_NEAREST.csv',aux_nearest)
csvout('MR_DE_ONLY_FROZEN_NEAREST_TRAIN.csv',de_nearest)
csvout('MR_FRAME_ENTITY_TARGET_CONTEXT_NEAREST_TRAIN.csv',entity_nearest)
csvout('MR_ALL_ELIGIBLE_FRAME_CONTEXT_NEAREST_TRAIN.csv',frame_context)
writej('FROZEN_DISTANCE_SUMMARIES.json',distance_summary)
writej('FRAME_ENTITY_TARGET_CONTEXT_DISTANCE_SUMMARIES.json',entity_distance_summary)
writej('FRAME_BALANCED_CONTEXT_DISTANCE_SUMMARIES.json',frame_context_summary)
writej('MR_SAVED_METRIC_REPRODUCTION_CHECKS.json',dict(PASS=True,checks=metric_reproduction))
writej('DE_TIE_AWARE_NEAREST_SUMMARIES.json',[dict(component=comp,split=split,state=st,rows=len(ss),
    fixed_numerical_tie_tolerance=TIE_TOLERANCE,tie_aware_support_counts=dict(Counter(r['tie_aware_support'] for r in ss)),
    same_minus_opposite_distance=desc([r['same_minus_opposite_distance'] for r in ss]),
    same_over_opposite_ratio=desc([r['same_over_opposite_ratio'] for r in ss]))
    for comp in components for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION') for st in ('D','E')
    if (ss:=[r for r in de_nearest if r['component']==comp and r['split']==split and r['state']==st])])
writej('DE_CONTEXT_OVERLAP.json',overlap)
writej('HELD45_SAVED_ERROR_CONCENTRATION_ALL_ARMS.json',concentration)
writej('FROZEN_DISTANCE_ERROR_ASSOCIATIONS_DESCRIPTIVE.json',association)
writej('SOURCE_COVERAGE_AND_SAVED_ERRORS.json',dict(completed=True,descriptive_only=True,all_rows_retained=True,
    rows={k:len(v['rows']) for k,v in datasets.items()},coverage=coverage,saved_errors=saved_errors,arm_summaries=arm_summaries,
    components=components,component_policy='Exact unmodified source feature blocks; target_context_CLS is cropped target/context CLS, not whole-image global.',
    family_availability_caveat='MR/Pallox query_available retains original continuous validity; synthetic retains stricter original factual continuous+nearest+central4 and context eligibility. No mask harmonization.',
    saved_output_provenance=score_provenance,no_model_encoder_initializer_forward_fit_probe_calibration_threshold_search_or_private_access=True,
    layout_source_references=[dict(path=str(P/r),sha256=h,line_reference=meaning) for r,h,meaning in layout_sources],
    source_weak_labels_not_gold=True))

def median_for(comp,split):
    return desc([r['nearest_train_distance'] for r in component_arrays[comp] if r['split']==split])['median']
de_summary=[]
for comp in ('target_context_CLS','query_context_tokens','local96_CLS','central32_CLS'):
    for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
        ss=[r for r in de_nearest if r['component']==comp and r['split']==split and r['state']=='E']
        de_summary.append(dict(component=comp,split=split,E_rows=len(ss),tie_aware_support_counts=dict(Counter(r['tie_aware_support'] for r in ss)),
            median_sameE_over_D_distance_ratio=desc([r['same_over_opposite_ratio'] for r in ss])['median']))
crop_support=[]
for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
    for st in ('D','E'):
        ss=[r for r in rr if r['split']==split and r['state']==st and r['query_available']]
        crop_support.append(dict(split=split,state=st,available_rows=len(ss),
            L=desc([r['L'] for r in ss]),**{k:desc([r[k] for r in ss]) for k in ('support_fraction_target','support_fraction_local96','support_fraction_central32')}))
writej('SOURCE_SUPPORTED_DIAGNOSIS.json',dict(completed=True,scope='existing frozen source artifacts only; all four arms; no adoption or new fit',
    retained_rows={k:len(v['rows']) for k,v in datasets.items()},
    frame_balanced_target_context=frame_context_summary,
    query_weighted_median_nearest_TRAIN_cosine_distances=[dict(component=comp,TRAIN_leave_frame_out=median_for(comp,'SOURCE_DEVELOPMENT'),held_to_TRAIN=median_for(comp,'SOURCE_VALIDATION')) for comp in ('target_context_CLS','local96_CLS','central32_CLS')],
    tie_aware_E_source_label_overlap=de_summary,numerical_tie_tolerance=TIE_TOLERANCE,
    target_context_same_frame_DE_unordered_pairs=next(r['same_frame_DE_unordered_pairs'] for r in overlap if r['component']=='target_context_CLS'),
    target_context_same_frame_DE_identical_pairs=next(r['same_frame_DE_distance_le1e_6'] for r in overlap if r['component']=='target_context_CLS'),
    crop_support=crop_support,held45_concentration=concentration,
    source_supported_findings=[
      'Saved C37/C38 visibility has zero TRAIN available D/E errors (390 known rows), while held errors remain46/50 of145. This establishes separation of these weak TRAIN labels by the saved heads, not reliable external visibility.',
      'Held frozen features are farther from original eligible MR TRAIN features in target context, local96 and central32. Frame-balanced context medians .0710 TRAIN vs .2554 held; 42 eligible held contexts choose only9 distinct nearest TRAIN frames.',
      'Local96 E has a closer same-label E TRAIN reference for48/49 TRAIN queries but2/52 held queries; central32 has41/49 vs1/52. This is descriptive overlap of existing source labels and descriptors, not a fitted probe or general capacity result.',
      'Target-context row argmin label cannot establish unique E absence: among heldE52,31 have tied minimum E and D context distance and21 have D closer under fixed1e-7 numerical tolerance. Context is shared by D/E rows within the same frame.',
      'Scalar L ranges for held D/E overlap TRAIN ranges. Every heldE central32 valid fraction is1.0 and local96 is at least.875, so these saved E failures do not require empty central input or out-of-range scalar L. This does not establish semantic crop correctness.',
      'Weak visibility errors span32/34/33/38 of45 held frames for C35/C36/C37/C38; top5frames account for20–26percent, rather than a unique isolated frame.',
      'Synthetic and Pallox features remain farther in cosine distance than nearest MR TRAIN on reported medians; their original masks and weak/constructed label ontology differ, so nearest distances alone do not establish task-equivalent supervision.'
    ],
    interpretive_limits=[
      'All real source role/entity labels remain assistant weak structural labels; exact real point quality labels remain unavailable. S/OOF/U are unchanged and never E.',
      'TRAIN leave-frame-out retains temporal/scene-neighbor source frames while the fixed held group can differ by scene. Repeated queries and frame descriptors are not independent physical examples. Differences are descriptive, not causal domain-shift proof.',
      'No source-only cosine comparison identifies a unique failure cause, guarantees crop semantic correctness, or authorizes an architecture, loss, threshold, checkpoint, arm choice or new fit.',
      'FINAL output NPZ were not individually hash-bound by FIT_COMPLETE/SOURCE_RESULTS. They are linked by exact bound saver code and filenames, their original metadata/decoder is verified, their MR CMs match bound SOURCE_RESULTS exactly, and new SHA256 values are recorded.'
    ],
    concrete_source_supported_next_investigation='Review all52 held E original crop/context packets alongside their deterministic nearest TRAIN E and D source packets in local96 and central32, using exported row IDs and minimum-distance ties, and compare existing weak association/native-role/occluder evidence. Preserve splits, labels and original coordinates. This is source evidence inspection only; no new training design or fit is authorized.',
    next_investigation_pair_ids='HELD52_E_DETERMINISTIC_NEAREST_TRAIN_PAIR_IDS.json',
    key_outputs=['ALL_ARMS_SAVED_ERRORS_BY_FAMILY_SPLIT_TASK.csv','MR_ALL230_FRAMES_ALL_ARMS_SAVED_ERRORS.csv','MR_DE_ONLY_FROZEN_NEAREST_TRAIN.csv','FRAME_BALANCED_CONTEXT_DISTANCE_SUMMARIES.json','MR_SAVED_METRIC_REPRODUCTION_CHECKS.json','SOURCE_COVERAGE_SAVED_ERRORS.png']))
writej('HELD52_E_DETERMINISTIC_NEAREST_TRAIN_PAIR_IDS.json',[r for r in de_nearest if r['split']=='SOURCE_VALIDATION' and r['state']=='E' and r['component'] in ('local96_CLS','central32_CLS')])

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,task in zip(axes[0],('visibility','entity')):
        for arm in arm_summaries:
            fr=sorted([r for r in frames if r['arm']==arm and r['split']=='SOURCE_VALIDATION'],key=lambda r:r['frame_index'])
            ax.plot([r['frame_index'] for r in fr],[r[f'{task}_errors_available'] for r in fr],label=arm,marker='.',lw=1)
        ax.set(xlabel='Original held frame index (all 45)',ylabel='Saved errors / available known rows',title=f'Held weak {task} errors, threshold 0');ax.legend(ncol=4,fontsize=8)
    for ax,comp in zip(axes[1],('target_context_CLS','local96_CLS')):
        source=frame_context if comp=='target_context_CLS' else component_arrays[comp]
        key='nearest_distance' if comp=='target_context_CLS' else 'nearest_train_distance'
        vals=[[r[key] for r in source if r['split']==sp] for sp in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION')]
        ax.boxplot(vals,labels=['TRAIN leave frame out','held → TRAIN'],showfliers=False)
        ax.set(ylabel='Cosine distance',title=comp+(' (frame balanced)' if comp=='target_context_CLS' else ' (query rows)'))
    fig.suptitle('Existing source-only diagnosis: frozen descriptors and saved outputs\nWeak assistant labels; descriptive associations, no causal or adoption claim',fontsize=12)
    fig.savefig(O/'SOURCE_COVERAGE_SAVED_ERRORS.png',dpi=180);plt.close(fig)
except Exception as e: writej('FIGURE_UNAVAILABLE.json',dict(error=str(e)))
writej('INPUT_BINDINGS_AND_CODE_HASHES.json',dict(source_inputs=BINDINGS,script=dict(path=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    rules_sha256=hashlib.sha256((O/'ANALYSIS_RULES_BEFORE_FEATURE_AND_SCORE_BODIES.json').read_bytes()).hexdigest(),
    prohibited_body_access_count=0,model_forward_fit_probe_count=0))
outputs=[]
for path in sorted(O.iterdir()):
    if path.is_file() and path.name!='OUTPUT_HASHES.json': outputs.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
writej('OUTPUT_HASHES.json',outputs)
print(json.dumps(dict(completed=True,rows={k:len(v['rows']) for k,v in datasets.items()},eligible_MR_nearest_rows=len(nearest_rows),aux_nearest_rows=len(aux_nearest),output_directory=str(O)),indent=2))
