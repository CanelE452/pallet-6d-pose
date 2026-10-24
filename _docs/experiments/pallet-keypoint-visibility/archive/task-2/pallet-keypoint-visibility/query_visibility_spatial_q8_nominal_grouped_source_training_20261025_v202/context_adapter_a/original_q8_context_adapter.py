"""Target-free original-initializer context from immutable input metadata only.

No source binary, image, target, score, checkpoint, model or old module import.
Importing this module performs no filesystem I/O. Runtime arrays are derived
only from the whitelisted original input scalars; no NPZ is read or written.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, json, os, stat
import numpy as np

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202'
H=V/'context_adapter_a'
A=P/'query_visibility_query_jitter_paired_source_training_20261024_v183'
C=P/'query_visibility_local_support_raw_head_20261024_v181'
D=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
SCOPE={'path':str(V/'ROOT_PROSPECTIVE_SPATIAL_Q8_NOMINAL_SOURCE_SCOPE.json'),'sha256':'6e758b87f0e99180a1f4b41921d01b3b0a6ec4f5e7e80aee507a10bfdb86ddd6'}
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL')

def b(p,sha):return {'path':str(p),'sha256':sha}
INPUT_BINDINGS={
 'augmentation_complete':b(A/'augmentation_training_inputs_v1/COMPLETE.json','beadda20f8e9534e4c8efa2a0b479a3e0dc101188d4a66d7b899da3f4d3d746f'),
 'public_CF_complete':b(C/'source_raw96_cache_v1/COMPLETE.json','b49f641551190aa27fd520acce51c5a1f0a9acc002b91c5b0b1709d4946cd3e4'),
 'direct_complete':b(D/'direct_control_raw_sam_cache_v1/COMPLETE.json','4bad28053a1ba96d20e64ab4b1039edfbbfb90afd3aac586b8d06d6548bbe197'),
 'split_complete':b(D/'source_component_split_v1/COMPLETE.json','f2dd0929ad73e843f2d0044c296c4c54f618ead69899400e75b2b6bac486d445'),
 'SYNTH_AUG65536':b(A/'augmentation_source_train_v1/AUG65536_ROWS.jsonl','33863d44b0648d498e1e913a0a8e708becb0edeb8230f8522e10ab84b350959d'),
 'original_scenes':b(A/'augmentation_preparation_a/SCENE512_ORIGINALS_INPUT_ONLY.jsonl','f5942772e7b0072c398d2abb1c009c43276110d5945e3f4853095fe336c99285'),
 'PUBLIC_TRAIN1480':b(C/'source_raw96_cache_v1/PUBLIC_TRAIN1480_ROWS.jsonl','08754bf88a98fcdc1a56cd0d84d6dd40804ff9de2f1ad7693a9e5bdd1fa31ff0'),
 'CF3816':b(C/'source_raw96_cache_v1/CF3816_ROWS.jsonl','c0499c1940a3f51d74e98d61f0cb45c6d68fd974bc4eb875a4166b264763530c'),
 'DIRECT_CONTROL':b(D/'direct_control_raw_sam_cache_v1/ALL_CONSTRUCTED_ROWS.jsonl','5746f0962f3a5cd55b7212ec1114612d05467364d9fda313e0a104305328424d'),
 'direct_groups':b(D/'direct_control_raw_sam_cache_v1/ALL_SAM_GROUPS.jsonl','aaf4539ddbe03e32ac60b019d84a8700e08cb1a7b227227fce036fce7d833627'),
}
for _key,_sha in zip(DOMAINS,('7d695977ecab1f9f5659f787c3cc7626b95f34d4654186fc8dc23204d679fde9','b04d5d7b1779b3df346a3fe6db26074b7376733317212471eef838439d72e0cc','c69ea16911554e6ce1afd87f29bce4e1530bc4e76b3767f1b9dfa197fa38b0a9','9ba6db7cfddad297d1fa060c55483d46a6e61eef1acb26cd0dc98fd32e51e046')):
    INPUT_BINDINGS[_key+'_roles']=b(D/'source_component_split_v1'/(_key+'_SPLIT_ROWS.jsonl'),_sha)

def require(ok,message):
    if not ok:raise ValueError(message)

def safe_path(p):
    p=Path(p)
    require('v112' not in str(p).casefold(),'Forbidden lexical path before filesystem')
    require(p.is_absolute() and '..' not in p.parts,'Literal absolute path required')
    for q in [p,*p.parents]:require(not stat.S_ISLNK(os.lstat(q).st_mode),'Symlink refused')
    require(p.suffix in ('.json','.jsonl','.py','.md'),'Metadata/code suffix only')
    return p

def read_bound(binding,accesses):
    p=safe_path(binding['path']);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    require(sha==binding['sha256'],'Bound metadata SHA mismatch')
    if p.suffix=='.jsonl':data=[json.loads(x) for x in raw.splitlines() if x.strip()]
    else:data=json.loads(raw)
    accesses.append(dict(path=str(p),sha256=sha,bytes=len(raw),records=len(data) if isinstance(data,list) else None))
    return data

def same(a,b):
    def normalized(x):return None if x is None or x==[None,None] else x
    return normalized(a)==normalized(b)

def native_index(k):
    require(isinstance(k,(int,np.integer)) and not isinstance(k,(bool,np.bool_)) and 0<=int(k)<8,'native_slot strict integer0..7')
    return int(k)

def bool_flag(x):
    require(isinstance(x,(bool,np.bool_)),'Original availability must be bool')
    return bool(x)

def q64(q):
    a=np.asarray(q,np.float64)
    require(a.shape==(2,) and np.isfinite(a).all(),'Available original q must be finite xy')
    return a

def build_context_vectors(active_q,active_L,native_slot,active_available,original_q8,original_L8,original_available8):
    """F64 subtraction/division -> clip[-8,8] -> FP32; interleaved slot xy.

    Native onehot is unchanged even for inputU. InputU rows get zero spatial
    context/mask and remain U outside this function. Missing other slots never
    acquire evidence merely because their saved coordinates are finite.
    """
    k=native_index(native_slot);active_available=bool_flag(active_available)
    require(len(original_q8)==len(original_L8)==len(original_available8)==8,'Exactly same original8 required')
    av=[bool_flag(x) for x in original_available8]
    require(av[k]==active_available,'Active original availability exact')
    native=np.eye(8,dtype=np.float32)[k]
    xy=np.zeros((8,2),np.float32);mask=np.zeros(8,bool)
    if active_available:
        aq=q64(active_q);L=float(active_L)
        require(np.isfinite(L) and L>0,'Available active L positive finite')
        for j in range(8):
            if j==k or not av[j]:continue
            oq=q64(original_q8[j]);ol=original_L8[j]
            require(ol is not None and np.isfinite(float(ol)) and float(ol)>0,'Available original L positive finite')
            # No FP32 cast before either F64 operation.
            with np.errstate(over='ignore'):
                relative=(oq-aq)/np.float64(L)
            xy[j]=np.clip(relative,-8.,8.).astype(np.float32)
            mask[j]=True
    require(not mask[k] and np.array_equal(xy[k],np.zeros(2,np.float32)),'Self alwayszero')
    return {'native_onehot8':native,'other_mask8':mask.astype(np.float32),'other_mask8_bool':mask,'other_dxdy16':xy.reshape(16)}

def fit_nominal_eligible(domain,row):
    require(domain in DOMAINS,'Known four cache views only')
    if domain!='SYNTH_AUG65536':return True
    require(row['family'] in ('SELF_SCENE15','BEFORE_ANCHOR4'),'Known synthetic family')
    if row['family']=='BEFORE_ANCHOR4':
        require(row['state_name']=='NOMINAL' and row['state_index']==0,'BEFORE nominal contract')
        return True
    require(isinstance(row['state_index'],int) and 0<=row['state_index']<15,'SELF original15 state index')
    require((row['state_index']==0)==(row['state_name']=='NOMINAL'),'SELF state name/index exact')
    return row['state_index']==0

def meta6(r):
    return {k:r[k] for k in ('row_id','original_global_index','native_slot','query_xy','L','original_query_available')}

def assert_meta6(a,b):
    for k in ('row_id','original_global_index','native_slot','original_query_available'):require(a[k]==b[k],'Exact META6 '+k)
    require(same(a['query_xy'],b['query_xy']) and a['L']==b['L'],'Exact META6 q/L')

def ordered8(rr):
    rr=sorted(rr,key=lambda x:native_index(x['native_slot']))
    require(len(rr)==8 and [r['native_slot'] for r in rr]==list(range(8)),'Exact unique native8 membership')
    require(len({r['L'] for r in rr})==1,'Original group shared L8')
    return rr

def load_bound_context_metadata():
    """Explicit metadata-only loader; caller later matches META6 to input arrays.

    Source role masks and eligibility are administrative masks, never head
    features. All rows retain their cache order, including source HOLDOUT role
    metadata and inputU. No target fields or nested metadata escape this loader.
    """
    accesses=[];scope=read_bound(SCOPE,accesses)
    require(scope['phase']=='V202','V202 scope binding')
    completed={k:read_bound(INPUT_BINDINGS[k],accesses) for k in ('augmentation_complete','public_CF_complete','direct_complete','split_complete')}
    require(all(x['completed'] is True and x['PASS'] is True for x in completed.values()),'Original cache/split complete')
    require(completed['augmentation_complete']['metadata']==INPUT_BINDINGS['SYNTH_AUG65536'],'Aug metadata authority')
    require(completed['augmentation_complete']['scene_metadata']==INPUT_BINDINGS['original_scenes'],'Scene metadata authority')
    for name in ('PUBLIC_TRAIN1480','CF3816'):require(completed['public_CF_complete']['domains'][name]['metadata']==INPUT_BINDINGS[name],'Public/CF metadata authority')
    require(completed['direct_complete']['metadata']==INPUT_BINDINGS['DIRECT_CONTROL'] and completed['direct_complete']['SAM_groups']==INPUT_BINDINGS['direct_groups'],'Latest direct3144 authority; no oldV190 rows')
    for name in DOMAINS:require(completed['split_complete']['outputs'][name+'_SPLIT_ROWS.jsonl']==INPUT_BINDINGS[name+'_roles'],'Immutable source role authority')
    rows={name:read_bound(INPUT_BINDINGS[name],accesses) for name in DOMAINS}
    scenes=read_bound(INPUT_BINDINGS['original_scenes'],accesses)
    groups=read_bound(INPUT_BINDINGS['direct_groups'],accesses)
    roles={name:read_bound(INPUT_BINDINGS[name+'_roles'],accesses) for name in DOMAINS}
    original={};scene8={}
    for s in scenes:
        rr=s['original_input_metadata8'];require(len(rr)==8,'Original scene8')
        mm=[]
        for k,r in enumerate(rr):
            require(r['native_slot']==k and r['original_global_index']==s['scene_index']*8+k and r['split']=='TRAIN','Original source scene/native identity')
            q8=r['original_initial_points8'];require(q8==rr[0]['original_initial_points8'],'Original scene q8 repeated exact')
            require(same(r['original_query_xy'],None if q8 is None else q8[k]),'Original q8 membership')
            original[r['original_global_index']]=r
            mm.append(dict(row_id=r['original_occurrence_id'],original_global_index=r['original_global_index'],native_slot=k,query_xy=r['original_query_xy'],L=r['original_L'],original_query_available=r['original_input_flags']['query_available']))
        scene8[s['scene_index']]=ordered8(mm)
    require(len(original)==4096 and len(scene8)==512,'All original source scenes')
    public8=defaultdict(list);public_by_origin={}
    for r in rows['PUBLIC_TRAIN1480']:
        o=r['original'];require(o['split']=='SOURCE_DEVELOPMENT' and same(r['query_xy'],o['query_xy']) and r['L']==o['L'] and r['original_query_available']==o['native_continuous_valid'],'Public original input fields')
        public8[o['frame_index']].append(meta6(r));public_by_origin[(o['source_id'],r['native_slot'])]=r
    public8={k:ordered8(rr) for k,rr in public8.items()}
    cf8=defaultdict(list)
    for r in rows['CF3816']:
        o=r['original'];p=public_by_origin[(o['source_id'],r['native_slot'])]
        require(same(r['query_xy'],p['query_xy']) and r['L']==p['L'] and r['original_query_available']==p['original_query_available'],'CF original public q/L/U')
        require(o['original_query_id']==p['row_id'] and o['original_frame_index']==p['original']['frame_index'],'CF original identity')
        cf8[r['sam_group_id']].append(meta6(r))
    cf8={k:ordered8(rr) for k,rr in cf8.items()}
    direct8={g['group_id']:ordered8(g['metadata8']) for g in groups}
    groupinfo={g['group_id']:g for g in groups}
    require(len(public8)==185 and len(cf8)==477 and len(direct8)==393,'Full original group census')
    views={}
    for name,rr in rows.items():
        idx={r['row_id']:r for r in roles[name]};require(len(idx)==len(rr),'One role row per occurrence')
        n=len(rr);view={key:np.zeros((n,width),np.float32) for key,width in [('native_onehot8',8),('other_mask8',8),('other_dxdy16',16)]}
        administrative_meta=[dict(meta6(r),**({key:r[key] for key in ('family','state_name','state_index')} if name=='SYNTH_AUG65536' else {})) for r in rr]
        view.update(other_mask8_bool=np.zeros((n,8),bool),fit_nominal_eligible=np.zeros(n,bool),immutable_TRAIN_role=np.zeros(n,bool),native_slot=np.array([native_index(r['native_slot']) for r in rr],np.int64),row_id=np.array([r['row_id'] for r in rr]),original_query_available=np.array([bool_flag(r['original_query_available']) for r in rr],bool),meta=administrative_meta,context_origin_group_id=[])
        for i,r in enumerate(rr):
            z=idx[r['row_id']];assert_meta6(r,z);require(z['role'] in ('TRAIN','HOLDOUT'),'Original immutable role')
            view['immutable_TRAIN_role'][i]=z['role']=='TRAIN';view['fit_nominal_eligible'][i]=fit_nominal_eligible(name,r)
            k=native_index(r['native_slot'])
            if name=='SYNTH_AUG65536':
                a=original[r['anchor_original_global_index']];im=original[r['source_image_original_global_index']]
                require(a['native_slot']==im['native_slot']==k and r['scene_index']==im['scene_index'],'Aug exact source/native')
                require(r['original_flags_ref']==[a['scene_index'],k] and r['image_flags_ref']==[im['scene_index'],k],'Aug original flags references')
                require(r['original_query_available']==a['original_input_flags']['query_available'],'Aug source availability')
                q=a['original_query_xy'];L=a['original_L'];want=None if q is None else [q[0]+r['dx'],q[1]+r['dy']]
                require(same(r['query_xy'],want) and r['L']==(None if L is None else L*r['L_scale']),'Aug exact active transform')
                if r['family']=='SELF_SCENE15':require(a['original_global_index']==im['original_global_index'],'SELF own original anchor')
                else:require(a['variant']=='BEFORE' and r['anchor_scene_index']==a['scene_index'] and r['dx']==r['dy']==0 and r['L_scale']==1,'BEFORE original anchor fixed')
                origin=scene8[a['scene_index']];origin_id='SYNTH_ORIGINAL_SCENE_%04d'%a['scene_index']
            elif name=='PUBLIC_TRAIN1480':
                f=r['original']['frame_index'];origin=public8[f];origin_id='PUBLIC_FRAME_%04d'%f;assert_meta6(r,origin[k])
            elif name=='CF3816':
                origin=cf8[r['sam_group_id']];origin_id='PUBLIC_FRAME_%04d'%r['original']['original_frame_index'];assert_meta6(r,origin[k])
            else:
                require(r['control_join_key']==z['control_join_key'] and r['source_packet']==z['source_packet'],'Latest direct occurrence join')
                origin=direct8[r['group_id']];assert_meta6(r,origin[k]);g=groupinfo[r['group_id']]
                if g['domain']=='SYNTH_FINITE_END':
                    base=g['source_base_index'];source=scene8[base*4];origin_id='SYNTH_ORIGINAL_SCENE_%04d'%(base*4)
                else:
                    require(g['domain']=='REAL_SEMANTIC_END_WEAK','Declared source domain only');f=g['source_frame_index'];source=public8[f];origin_id='PUBLIC_FRAME_%04d'%f
                for m,s in zip(origin,source):
                    require(m['native_slot']==s['native_slot'] and same(m['query_xy'],s['query_xy']) and m['L']==s['L'] and m['original_query_available']==s['original_query_available'],'Direct all8 original source q/L/U')
            require(origin[k]['original_query_available']==r['original_query_available'],'Active origin availability')
            context=build_context_vectors(r['query_xy'],r['L'],k,r['original_query_available'],[m['query_xy'] for m in origin],[m['L'] for m in origin],[m['original_query_available'] for m in origin])
            for key in context:view[key][i]=context[key]
            view['context_origin_group_id'].append(origin_id)
        require(np.isfinite(view['other_dxdy16']).all() and np.max(np.abs(view['other_dxdy16']))<=8,'Finite clipped FP32 context')
        views[name]=view
    return {'views':views,'input_bindings':INPUT_BINDINGS,'scope':SCOPE,'accesses':accesses}

def metadata_census():
    """Scope-authorized metadata-only materialization; never writes NPZ/arrays."""
    loaded=load_bound_context_metadata();out=H/'metadata_context_census_v1';out.mkdir(exist_ok=False)
    bindings={};summary=[]
    for domain,vv in loaded['views'].items():
        n=len(vv['row_id']);eligible=vv['fit_nominal_eligible'];train=vv['immutable_TRAIN_role'];av=vv['original_query_available']
        p=out/(domain+'_CONTEXT_ROW_MAPPING.jsonl')
        with p.open('x') as f:
            for i in range(n):
                sha=hashlib.sha256(vv['native_onehot8'][i].tobytes()+vv['other_mask8'][i].tobytes()+vv['other_dxdy16'][i].tobytes()).hexdigest()
                f.write(json.dumps(dict(cache_index=i,row_id=str(vv['row_id'][i]),native_slot=int(vv['native_slot'][i]),context_origin_group_id=vv['context_origin_group_id'][i],original_query_available=bool(av[i]),fit_nominal_eligible=bool(eligible[i]),immutable_TRAIN_role=bool(train[i]),context_FP32_native_mask_xy_sha256=sha),ensure_ascii=False)+'\n')
        raw=p.read_bytes();bindings[domain]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}
        summary.append(dict(domain=domain,physical_rows=n,original_inputU=int((~av).sum()),nominal_eligible_all=int(eligible.sum()),TRAIN_rows=int(train.sum()),TRAIN_nominal_eligible=int((train&eligible).sum()),TRAIN_nominal_excluded=int((train&~eligible).sum()),TRAIN_nominal_eligible_inputU=int((train&eligible&~av).sum()),TRAIN_nominal_eligible_available=int((train&eligible&av).sum()),native6_7_rows_preserved=int(np.isin(vv['native_slot'],[6,7]).sum()),nonzero_other_masks=int(vv['other_mask8_bool'].sum()),all_inputU_context_zero=bool(not vv['other_dxdy16'][~av].any() and not vv['other_mask8'][~av].any())))
    cp=safe_path(Path(__file__).absolute())
    data={'completed':True,'PASS':True,'kind':'TARGETFREE_ORIGINAL_Q8_METADATA_CONTEXT_CENSUS_NOT_FIT','scope':SCOPE,'code':{'path':str(cp),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()},'input_bindings':INPUT_BINDINGS,'summary':summary,'row_mappings':bindings,'physical_metadata_accesses':loaded['accesses'],'binary_header_pixel_weight_model_target_array_or_score_body_reads':0,'unused_label_provenance_fields_physically_present_in_original_metadata_never_features':True,'context_NPZ_written':False,'original_saved_rows_or_split_changed':False,'fitting_or_model_forward':False,'eligibility_applies_both_arms':True,'caller_must_intersect_immutable_TRAIN_later':True}
    p=out/'COMPLETE.json'
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'completed':True,'PASS':True,'summary':summary,'complete':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--metadata-census',action='store_true',required=True);parser.parse_args();metadata_census()
