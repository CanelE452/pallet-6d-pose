"""V201 source input metadata census. Standard library only; no tensor/pixel I/O."""
import csv, hashlib, json, math, os, stat
from collections import Counter, defaultdict
from pathlib import Path

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_train_native_end_q8_correspondence_20261025_v201'
H=V/'q8_provenance_a'
A=P/'query_visibility_query_jitter_paired_source_training_20261024_v183'
C=P/'query_visibility_local_support_raw_head_20261024_v181'
D=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
S=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
I=P/'query_visibility_actual_initializer_source_20261024_v132'
T=P/'query_visibility_train_only_target_contract_readout_20261024_v197'
ACCESSES=[]

def safe(p):
    p=Path(p)
    if 'v112' in str(p).lower(): raise ValueError('Forbidden lexical path')
    if not p.is_absolute() or '..' in p.parts: raise ValueError('Absolute literal path required')
    for n in [p,*p.parents]:
        if stat.S_ISLNK(os.lstat(n).st_mode): raise ValueError('Symlink refused')
    if p.suffix not in ('.json','.jsonl','.py','.md'): raise ValueError('Metadata/code only')
    return p

def read(p,expected=None):
    p=safe(p);b=p.read_bytes();digest=hashlib.sha256(b).hexdigest()
    if expected: assert digest==expected,(str(p),'SHA mismatch')
    ACCESSES.append(dict(path=str(p),sha256=digest,bytes=len(b),kind='JSON_METADATA'))
    return json.loads(b)

def bound(b):return read(b['path'],b['sha256'])

def lines(b):
    p=safe(b['path']);h=hashlib.sha256();out=[]
    with p.open('rb') as f:
        for l in f:
            h.update(l)
            if l.strip():out.append(json.loads(l))
    assert h.hexdigest()==b['sha256'],(str(p),'SHA mismatch')
    ACCESSES.append(dict(path=str(p),sha256=h.hexdigest(),rows_parsed=len(out),kind='JSONL_METADATA_FULL_CONTAINER'))
    return out

def finiteq(q):return isinstance(q,list) and len(q)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in q)
def normalized(q):return None if q is None or q==[None,None] else q
def eq(a,b):return normalized(a)==normalized(b)
def positiveL(x):return isinstance(x,(int,float)) and math.isfinite(x) and x>0
def write(name,d):
    p=H/name
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def csvwrite(name,rows):
    assert rows
    p=H/name
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def meta(m):return dict(row_id=m['row_id'],original_global_index=m['original_global_index'],native_slot=m['native_slot'],query_xy=m['query_xy'],L=m['L'],original_query_available=m['original_query_available'])
def checkmeta(a,b):
    for k in ('row_id','original_global_index','native_slot','original_query_available'):assert a[k]==b[k],k
    assert eq(a['query_xy'],b['query_xy']) and a['L']==b['L']
def group8(rr):
    rr=sorted(rr,key=lambda r:r['native_slot']);assert len(rr)==8 and [r['native_slot'] for r in rr]==list(range(8))
    assert len({r['L'] for r in rr})==1,'Group shared L'
    return rr
def q8export(rr):
    rr=group8(rr)
    return dict(native_slot=list(range(8)),query_xy8=[r['query_xy'] for r in rr],L8=[r['L'] for r in rr],original_query_available8=[r['original_query_available'] for r in rr],original_global_indices8=[r['original_global_index'] for r in rr],original_row_ids8=[r['row_id'] for r in rr])

def run():
    scope=read(V/'ROOT_TRAIN_ONLY_CORRESPONDENCE_SCOPE.json','4db9e8d14c7d0af2641a903191dbe9ef1779364c38a003d4fd7f61bb7480bc38')
    prev=read(P/'query_visibility_native_query_conditioning_feasibility_20261024_v198/source_contract_review_a/REVIEW.json','cfb91437d67361780e71833c96fafcbae413fdcf6a1ecb0c9d4b1717dfd88da7')
    augc=read(A/'augmentation_training_inputs_v1/COMPLETE.json','beadda20f8e9534e4c8efa2a0b479a3e0dc101188d4a66d7b899da3f4d3d746f')
    pubc=read(C/'source_raw96_cache_v1/COMPLETE.json','b49f641551190aa27fd520acce51c5a1f0a9acc002b91c5b0b1709d4946cd3e4')
    dc=read(D/'direct_control_raw_sam_cache_v1/COMPLETE.json','4bad28053a1ba96d20e64ab4b1039edfbbfb90afd3aac586b8d06d6548bbe197')
    splitc=read(D/'source_component_split_v1/COMPLETE.json','f2dd0929ad73e843f2d0044c296c4c54f618ead69899400e75b2b6bac486d445')
    aug=lines(augc['metadata']);scenes=lines(augc['scene_metadata'])
    public=lines(pubc['domains']['PUBLIC_TRAIN1480']['metadata']);cf=lines(pubc['domains']['CF3816']['metadata'])
    direct=lines(dc['metadata']);dgroups=lines(dc['SAM_groups'])
    sb={k:lines(splitc['outputs'][k+'_SPLIT_ROWS.jsonl']) for k in ('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL')}
    populations={'SELF_SCENE15':[r for r in aug if r['family']=='SELF_SCENE15'],'BEFORE_ANCHOR4':[r for r in aug if r['family']=='BEFORE_ANCHOR4'],'PUBLIC':public,'CF':cf,'DIRECT':direct}
    role={}
    for name,rows in [('SYNTH_AUG65536',aug),('PUBLIC_TRAIN1480',public),('CF3816',cf),('DIRECT_CONTROL',direct)]:
        idx={r['row_id']:r for r in sb[name]};assert len(idx)==len(rows)
        for r in rows:
            z=idx[r['row_id']];checkmeta(r,z);role[r['row_id']]=z['role'];assert z['role'] in ('TRAIN','HOLDOUT')
            if name=='DIRECT_CONTROL':assert r['control_join_key']==z['control_join_key'] and r['source_packet']==z['source_packet']
    orig={};sc={}
    for s in scenes:
        assert s['split']=='TRAIN' and s['scene_index'] not in sc
        sc[s['scene_index']]=s;rr=s['original_input_metadata8'];assert len(rr)==8
        for k,r in enumerate(rr):
            assert r['native_slot']==k and r['original_global_index']==s['scene_index']*8+k
            assert r['split']=='TRAIN' and r['scene_index']==s['scene_index']
            q8=r['original_initial_points8'];assert q8==rr[0]['original_initial_points8']
            assert eq(r['original_query_xy'],None if q8 is None else q8[k])
            orig[r['original_global_index']]=r
    assert len(orig)==4096 and len(sc)==512
    # Original initializer query metadata: read TRAIN prefix only, never the VAL suffix.
    ip=read(I/'frozen_visibility_preparation/PROTOCOL_BEFORE_VISIBILITY_FORWARD.json')
    b=ip['initializer_queries'];p=safe(b['path']);h=hashlib.sha256();matched=0
    with p.open('rb') as f:
        for i in range(4096):
            l=f.readline();assert l;h.update(l);q=json.loads(l);r=orig[i]
            assert q['split']=='TRAIN' and q['scene_index']==i//8 and q['native_slot']==i%8
            assert q['actual_original_initializer'] is True
            assert eq(q['query_xy'],r['original_query_xy']) and q['roi_side']==r['original_L']
            assert q['original_query_continuous_valid']==r['original_input_flags']['native_continuous_valid']
            matched+=1
    ACCESSES.append(dict(path=str(p),kind='TRAIN_PREFIX_ONLY_NO_VAL_READ',rows_parsed=4096,TRAIN_prefix_sha256=h.hexdigest(),full_sha256_inherited_not_recomputed=b['sha256']))
    transforms=Counter();reconstruct=0;originalflag=0;active_mismatch=0
    for r in aug:
        a=orig[r['anchor_original_global_index']];im=orig[r['source_image_original_global_index']]
        assert a['native_slot']==im['native_slot']==r['native_slot']
        assert r['original_query_available']==a['original_input_flags']['query_available']
        assert r['original_flags_ref']==[a['scene_index'],r['native_slot']] and r['image_flags_ref']==[im['scene_index'],r['native_slot']]
        assert r['scene_index']==im['scene_index'] and r['anchor_scene_index']==a['scene_index']
        q=a['original_query_xy'];L=a['original_L'];want=None if q is None else [q[0]+r['dx'],q[1]+r['dy']]
        assert eq(want,r['query_xy']) and r['L']==(None if L is None else L*r['L_scale'])
        if r['family']=='SELF_SCENE15':
            assert a['original_global_index']==im['original_global_index'];transforms[r['state_name']]+=1
            if role[r['row_id']]=='TRAIN' and finiteq(q) and positiveL(L):
                assert (r['query_xy'][0]-q[0],r['query_xy'][1]-q[1])==(r['dx'],r['dy'])
                assert r['L']/L==r['L_scale'] or abs(r['L']/L-r['L_scale'])<1e-15
                reconstruct+=1;active_mismatch+=not eq(q,r['query_xy']) or L!=r['L']
        else:assert a['variant']=='BEFORE' and r['state_index']==0 and r['dx']==r['dy']==0 and r['L_scale']==1
        originalflag+=1
    pg=defaultdict(list);cg=defaultdict(list);dg=defaultdict(list)
    for r in public:
        o=r['original'];assert eq(r['query_xy'],o['query_xy']) and r['L']==o['L'] and r['original_query_available']==o['native_continuous_valid']
        assert o['split']=='SOURCE_DEVELOPMENT';pg[r['sam_group_id']].append(r)
    public_origin={(r['original']['source_id'],r['native_slot']):r for r in public}
    for r in cf:
        o=r['original'];p=public_origin[(o['source_id'],r['native_slot'])]
        assert eq(r['query_xy'],p['query_xy']) and r['L']==p['L'] and r['original_query_available']==p['original_query_available']
        assert o['original_query_id']==p['row_id'] and o['original_frame_index']==p['original']['frame_index']
        cg[r['sam_group_id']].append(r)
    dm={g['group_id']:g for g in dgroups};assert len(dm)==len(dgroups)==393
    direct_origin={'SYNTH_FINITE_END':0,'REAL_SEMANTIC_END_WEAK':0}
    for r in direct:dg[r['group_id']].append(r)
    for key,rr in dg.items():
        g=dm[key];group8(rr);assert len(g['metadata8'])==8
        for r,m in zip(sorted(rr,key=lambda r:r['native_slot']),g['metadata8']):checkmeta(r,m)
        if g['domain']=='SYNTH_FINITE_END':
            for r in rr:
                o=orig[g['source_base_index']*32+r['native_slot']]
                assert eq(r['query_xy'],o['original_query_xy']) and r['L']==o['original_L'] and r['original_query_available']==o['original_input_flags']['query_available']
        else:
            for r in rr:
                o=public_origin[(next(x['original']['source_id'] for x in public if x['original']['frame_index']==g['source_frame_index']),r['native_slot'])]
                assert eq(r['query_xy'],o['query_xy']) and r['L']==o['L'] and r['original_query_available']==o['original_query_available']
        direct_origin[g['domain']]+=1
    for gg in (pg,cg,dg):
        for rr in gg.values():group8(rr);assert len({role[r['row_id']] for r in rr})==1
    out=[];native=[]
    for name,rr in populations.items():
        tr=[r for r in rr if role[r['row_id']]=='TRAIN'];ho=[r for r in rr if role[r['row_id']]=='HOLDOUT']
        if name in ('SELF_SCENE15','BEFORE_ANCHOR4'):
            contexts={r['sam_group_id'] for r in rr};traincontexts={r['sam_group_id'] for r in tr}
        else:contexts={r['sam_group_id' if name!='DIRECT' else 'group_id'] for r in rr};traincontexts={r['sam_group_id' if name!='DIRECT' else 'group_id'] for r in tr}
        row=dict(population=name,physical_rows=len(rr),TRAIN_rows=len(tr),HOLDOUT_role_metadata_rows=len(ho),all_inputU=sum(not r['original_query_available'] for r in rr),TRAIN_inputU=sum(not r['original_query_available'] for r in tr),context_groups_all=len(contexts),context_groups_TRAIN=len(traincontexts),TRAIN_finite_q=sum(finiteq(r['query_xy']) for r in tr),TRAIN_null_q=sum(not finiteq(r['query_xy']) for r in tr),TRAIN_null_L=sum(not positiveL(r['L']) for r in tr))
        out.append(row)
        for k in range(8):
            a=[r for r in tr if r['native_slot']==k];native.append(dict(population=name,native_slot=k,TRAIN_rows=len(a),inputU=sum(not r['original_query_available'] for r in a),finite_q=sum(finiteq(r['query_xy']) for r in a),null_L=sum(not positiveL(r['L']) for r in a)))
    # No new selection: exact previously frozen V197 fixed12.
    tc=read(T/'fixed_train_target_contract_viewer_v1/COMPLETE.json','4a9efcd72b20b13f7814396d5f02535af90f5a009fea5791aa989e124791a316')
    sel=bound(tc['selection']);selected=[]
    for rank,x in enumerate(sel['selected']):
        gg=[g for g in dgroups if g['domain']==x['domain'] and g['attempt_id']==x['attempt_id']]
        assert len(gg)==3
        variants=[]
        for g in sorted(gg,key=lambda z:z['variant_index']):
            assert all(role[r['row_id']]=='TRAIN' for r in dg[g['group_id']])
            variants.append(dict(group_id=g['group_id'],variant=g['variant'],source_variant=g.get('source_variant',g['variant']),source_base_index=g.get('source_base_index'),source_frame_index=g.get('source_frame_index'),**q8export(dg[g['group_id']])))
        assert all(v['query_xy8']==variants[0]['query_xy8'] and v['L8']==variants[0]['L8'] and v['original_query_available8']==variants[0]['original_query_available8'] for v in variants)
        selected.append(dict(selection_index=rank,domain=x['domain'],attempt_id=x['attempt_id'],selected_native_slot=x['native'],original_context_group=variants[0]['group_id'],variants=variants))
    assert len(selected)==12
    exports={}
    exports['populations']=csvwrite('POPULATION_TRAIN_Q8_CENSUS.csv',out)
    exports['native']=csvwrite('TRAIN_NATIVE_Q8_CENSUS.csv',native)
    exports['selected12']=write('SELECTED12_ORIGINAL_Q8_METADATA.json',dict(completed=True,PASS=True,selection=tc['selection'],rows=selected,contains_only_original_input_scalars=True,labels_targets_poses_end_enclosures_or_predictions_included=False,new_selection=False))
    facts=dict(completed=True,PASS=True,kind='SOURCE_TRAIN_INPUT_METADATA_CENSUS_NOT_MODEL_EVALUATION',populations=out,native=native,checks=dict(all_metadata_join_exact=True,original4096_q_L_match_actual_initializer_TRAIN_prefix=matched,original512_q8_slot_membership_exact=True,all65536_anchor_transform_and_flags_exact=originalflag,SELF_state_counts=dict(transforms),TRAIN_SELF_finite_anchor_transform_recoverable_rows=reconstruct,TRAIN_SELF_active_q_or_L_differs_from_original_anchor_rows=active_mismatch,all_PUBLIC185_native8_and_CF477_native8_exact=True,CF_groups_inherit_same_original_public_q8_despite_edited_RGB=True,DIRECT393_all8_groups_exact=True,direct_group_origins=direct_origin,no_historical_V190_direct_rows_opened=True),exports=exports,accesses=ACCESSES,limits=['Full metadata containers include HOLDOUT role rows; TRAIN aggregates filter immutable role sidecars. No HOLDOUT pixel/target/score body was read.','Inherited pixel/array/checkpoint bindings are strings only. No referenced binary was statted, hashed, opened or decoded.','SELF original q8 is original-scene context; individual jitter rows do not certify a newly inferred simultaneous jitter8.','No model input tensor, normalization, fitting, forward, threshold or data selection was produced.'])
    write('CENSUS_RESULT.json',facts)
    print(json.dumps(dict(completed=True,PASS=True,populations=out,selected12=exports['selected12'],actual_initializer_TRAIN_prefix_matches=matched),ensure_ascii=False))

if __name__=='__main__':run()
