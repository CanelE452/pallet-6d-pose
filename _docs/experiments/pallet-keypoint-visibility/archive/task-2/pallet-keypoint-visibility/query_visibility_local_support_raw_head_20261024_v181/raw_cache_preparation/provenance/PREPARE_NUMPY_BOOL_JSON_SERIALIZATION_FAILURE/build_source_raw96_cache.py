"""V181 root-entry-only raw crop/SAM cache. Preparation reads metadata, not payloads."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
SCOPE_SHA='8cd21d5eb4caf9663927859ead0cbcbd823e701fd4a2164a66aa765e046bfcdc'
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
V153=P/'query_visibility_mr6d_full_public_source_20261024_v153'
V161=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
V173=P/'query_visibility_full_source_query_causal_census_20261024_v173'
V175=P/'query_visibility_direct_query_owner_control_20261024_v175'
V176=P/'query_visibility_direct_owner_public_ycb_control_20261024_v176'
PROTOCOL=H/'PROTOCOL_BEFORE_RAW_SAM_AND_TARGET_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'; PLAN=H/'ALL9392_INPUT_ROWS_BEFORE_BODIES.jsonl'
OUT=V/'source_raw96_cache_v1'
DOMAINS={'SYNTH_TRAIN':4096,'PUBLIC_TRAIN':1480,'YCB_CF':3816}
DOMAIN_KEYS={'SYNTH_TRAIN':'SYNTH_TRAIN4096','PUBLIC_TRAIN':'PUBLIC_TRAIN1480','YCB_CF':'CF3816'}
OWNER6=['INCIDENT_TARGET_DECLARED_CF','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT']

def require(ok,message):
    if not bool(ok):raise AssertionError(message)
def permitted(path):
    p=Path(path).resolve();require(p.is_relative_to(P),'Outside source project')
    require(not any(s in str(p).lower() for s in ('_v112','_v177','_v178','_v169','current223','/private_evaluation/')),'Forbidden path')
    return p
def sha(path):
    h=hashlib.sha256()
    with permitted(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def checked(b):p=permitted(b['path']);require(sha(p)==b['sha256'],'SHA '+str(p));return p
def read(path):return json.loads(permitted(path).read_text())
def lines(path):return [json.loads(s) for s in permitted(path).read_text().splitlines() if s]
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writelines(path,rows):
    with Path(path).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def qarray(q):return np.array([np.nan,np.nan] if q is None else q,np.float64)
def validq(q):return qarray(q).shape==(2,) and np.isfinite(qarray(q)).all() and not same(qarray(q),[-1.,-1.])

def grid96(q,L):
    require(validq(q) and L is not None and np.isfinite(L) and L>0,'Finite original q/L')
    q=qarray(q);u=(np.arange(96,dtype=np.float64)+.5)/96-.5
    x,y=np.meshgrid(q[0]+u*L,q[1]+u*L);return np.column_stack((x.ravel(),y.ravel()))

def sample_sam(logits,q,L,available,hw):
    """Same original 96-grid, zero-padded bilinear FP64 arithmetic, strict >0."""
    mask=np.zeros((96,96),bool);avail=np.zeros_like(mask)
    if not available or not validq(q) or L is None or not np.isfinite(L) or L<=0:return mask,avail
    require(logits is not None and logits.dtype==np.float32 and logits.shape==tuple(hw) and np.isfinite(logits).all(),'Saved full RGB SAM logits')
    points=grid96(q,L);x,y=points.T;h,w=hw
    ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
    values=np.zeros(len(points),np.float64)
    for ax,ay,wt in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=ix+ax,iy+ay;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        values[inside]+=logits[yy[inside],xx[inside]].astype(np.float64)*wt[inside]
    domain=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    return ((values>0)&domain).reshape(96,96),domain.reshape(96,96)

def collapse_owner(owner6):
    require(owner6.dtype==np.float64 and owner6.ndim==2 and owner6.shape[1]==6,'Original central4 owner6 schema')
    missing=np.isnan(owner6).all(1);require((np.isfinite(owner6).all(1)|missing).all(),'No partial NaN owner rows')
    known=owner6[~missing];require((known>=0).all() and np.allclose(known.sum(1),1,rtol=0,atol=2e-13),'Original owner6 mass')
    # Do not renormalize or drop boundary rows. Class-conditional draws use each raw class mass.
    soft=np.stack((owner6[:,0]+owner6[:,1],owner6[:,2],owner6[:,3]),1)
    return soft,owner6[:,4].copy(),owner6[:,5].copy()

def input_arrays(rows):
    n=len(rows)
    return dict(rgb96=np.zeros((n,96,96,3),np.uint8),valid96=np.zeros((n,96,96),bool),
        sam_mask96=np.zeros((n,96,96),bool),sam_available96=np.zeros((n,96,96),bool),
        original_query_available=np.array([r['original_query_available'] for r in rows],bool),
        row_id=np.array([r['row_id'] for r in rows]),original_global_index=np.array([r['original_global_index'] for r in rows],np.int64),
        native_slot=np.array([r['native_slot'] for r in rows],np.int8),query_xy=np.array([qarray(r['query_xy']) for r in rows],np.float64),
        L=np.array([np.nan if r['L'] is None else r['L'] for r in rows],np.float64))

def target_arrays(rows):
    n=len(rows)
    return dict(support_target=np.full((n,2),-1,np.int8),owner_soft=np.zeros((n,3),np.float64),owner_target=np.full(n,-1,np.int8),
        owner_ambiguous_mass=np.full(n,np.nan,np.float64),owner_invalid_mass=np.full(n,np.nan,np.float64))

def inventories():
    """Metadata-only. Held MR rows are excluded before any future payload loading."""
    paths=dict(synth_manifest=V180/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl',
        public_facts=V153/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl',
        public_crop_manifest=V153/'full230_native_DINO_v1/CROP_PACKETS230_INPUT_ONLY.json',
        CF_manifest=V161/'next_feature_input_metadata/ALL3816_COUNTERFACTUAL_INPUT_OCCURRENCES_NO_FEATURES.jsonl',
        V175_protocol=V175/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_TRAIN_SOFT_BODIES.json',
        V176_protocol=V176/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_OR_CF_FEATURE_AND_TRAIN_SOFT_BODIES.json',
        end_complete=V180/'source_end_supervision_train4096_v1/COMPLETE.json',
        census_complete=V173/'literal_raster_census_v1/COMPLETE.json',
        synth_SAM_complete=V180/'frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json')
    s=lines(paths['synth_manifest']);facts=lines(paths['public_facts']);cf=lines(paths['CF_manifest'])
    p175=read(paths['V175_protocol']);p176=read(paths['V176_protocol']);end=read(paths['end_complete']);census=read(paths['census_complete'])
    crops={r['frame_index']:r['crop_packet'] for r in read(paths['public_crop_manifest'])['sheets']}
    rows=[]
    for i,r in enumerate(s):
        require(i==r['original_global_index'] and r['split']=='TRAIN' and r['native_slot']==i%8,'SYNTH original order')
        rows.append(dict(domain='SYNTH_TRAIN',domain_index=i,row_id=r['original_occurrence_id'],original_global_index=i,native_slot=r['native_slot'],
            query_xy=r['original_query_xy'],L=r['original_L'],original_query_available=r['original_input_flags']['query_available'],
            raw_packet=r['saved_crop_chunk'],raw_row=r['saved_crop_row'],rgb96_sha256=r['saved_rgb96_sha256_inherited'],valid96_sha256=r['saved_valid96_sha256_inherited'],
            sam_group_id='SYNTH_SCENE_%06d'%r['scene_index'],raw_hw=r['raw_hw'],original=r))
    pub=[r for r in facts if r['split']=='SOURCE_DEVELOPMENT']
    for j,r in enumerate(pub):
        rows.append(dict(domain='PUBLIC_TRAIN',domain_index=j,row_id=r['query_id'],original_global_index=r['global_index'],native_slot=r['native_slot'],
            query_xy=r['query_xy'],L=r['L'],original_query_available=r['native_continuous_valid'],raw_packet=crops[r['frame_index']],raw_row=r['native_slot'],
            rgb96_sha256=r['RGB96_HWC_sha256'],valid96_sha256=r['valid96_sha256'],sam_group_id='PUBLIC_FRAME_%04d'%r['frame_index'],raw_hw=[720,1280],original=r))
    for j,r in enumerate(cf):
        f=facts[r['original_global_index']]
        require(r['split']==f['split']=='SOURCE_DEVELOPMENT' and r['original_query_id']==f['query_id'] and r['native_slot']==f['native_slot'] and same(qarray(r['query_xy']),qarray(f['query_xy'])) and r['L']==f['L'],'CF original query join')
        require(r['query_available']==f['native_continuous_valid'],'CF availability source-specific original')
        rows.append(dict(domain='YCB_CF',domain_index=j,row_id=r['input_occurrence_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],
            query_xy=r['query_xy'],L=r['L'],original_query_available=r['query_available'],raw_packet=r['variant_payload'],raw_row=r['native_slot'],
            rgb96_sha256=None,valid96_sha256=None,sam_group_id=r['pair_id']+'/'+r['variant'],raw_hw=[720,1280],original=r))
    require(Counter(r['domain'] for r in rows)==DOMAINS,'All three original domains')
    require(sum(not r['original_query_available'] for r in rows if r['domain']=='SYNTH_TRAIN')==80,'SYNTH U80')
    require(sum(not r['original_query_available'] for r in rows if r['domain']=='PUBLIC_TRAIN')==194,'PUBLIC U194')
    require(sum(not r['original_query_available'] for r in rows if r['domain']=='YCB_CF')==360,'CF U360')
    require(len({r['raw_packet']['path'] for r in rows if r['domain']=='YCB_CF'})==477,'All477 payloads retained')
    require(len(set(p175['public_known_indices']))==253 and len(set(p176['CF_known_indices']))==314,'Frozen owner selection cardinalities')
    for r in rows:
        r['split']=r['original']['split'];r['new_native_visibility_target']=-1;r['new_point_quality_target']=-1
    bindings={k:bind(v) for k,v in paths.items()}
    bindings.update(end_targets=end['outputs']['sidecar'],TRAIN_owner6=census['sidecars']['TRAIN'],public_labels=p175['inputs']['owner_labels'],CF_labels=p176['CF_inputs']['owner_labels'])
    return rows,bindings,p175['public_known_indices'],p176['CF_known_indices']

def load_targets(rows,protocol):
    b=protocol['inputs'];out={d:target_arrays([r for r in rows if r['domain']==d]) for d in DOMAINS}
    ss=[r for r in rows if r['domain']=='SYNTH_TRAIN']
    meta=['original_global_index','native_slot','query_xy','L','split','original_occurrence_id','query_available']
    with np.load(checked(b['end_targets']),allow_pickle=False) as z:
        require(not bool(z['training_admission']) and (z['new_native_visibility_target']==-1).all() and (z['new_point_quality_target']==-1).all(),'Historical sidecar authority')
        for k in meta:
            dst={'original_occurrence_id':'row_id','query_available':'original_query_available'}.get(k,k)
            vals=np.array([qarray(r[dst]) if k=='query_xy' else (np.nan if r[dst] is None else r[dst]) for r in ss],np.float64 if k in ('query_xy','L') else None)
            require(same(z[k],vals),'END exact row join '+k)
        for k in ss[0]['original']['original_input_flags']:
            require(same(z[k],np.array([r['original']['original_input_flags'][k] for r in ss],bool)),'END original flag '+k)
        st=np.column_stack((z['visible_support_target'],z['foreign_front_support_target']))
        require(st.dtype==np.int8 and st.shape==(4096,2) and np.isin(st,[-1,0,1]).all(),'END targets shape/dtype')
        out['SYNTH_TRAIN']['support_target']=st.copy()
    with np.load(checked(b['TRAIN_owner6']),allow_pickle=False) as z:
        require(z['owner_class_order'].tolist()==OWNER6 and not bool(z['training_admission']),'Original owner6 authority')
        for k in meta:
            dst={'original_occurrence_id':'row_id','query_available':'original_query_available'}.get(k,k)
            vals=np.array([qarray(r[dst]) if k=='query_xy' else (np.nan if r[dst] is None else r[dst]) for r in ss],np.float64 if k in ('query_xy','L') else None)
            require(same(z[k],vals),'Owner6 exact row join '+k)
        for k in ss[0]['original']['original_input_flags']:
            require(same(z[k],np.array([r['original']['original_input_flags'][k] for r in ss],bool)),'Owner6 original flag '+k)
        y,amb,invalid=collapse_owner(z['scene_owner_central4'])
    out['SYNTH_TRAIN'].update(owner_soft=y,owner_ambiguous_mass=amb,owner_invalid_mass=invalid)
    # Only SOURCE_DEVELOPMENT rows are deserialized from the historic weak sidecar.
    pp=[r for r in rows if r['domain']=='PUBLIC_TRAIN'];byid={r['original_global_index']:r for r in pp}
    labels={}
    with checked(b['public_labels']).open() as f:
        for i,line in enumerate(f):
            if i not in byid:continue
            z=json.loads(line);r=byid[i];orig=r['original']
            require(z['global_index']==i and z['query_id']==r['row_id'] and z['frame_index']==orig['frame_index'] and z['native_slot']==r['native_slot'] and z['split']==r['split'],'PUBLIC label identity')
            require(same(qarray(z['query_xy']),qarray(r['query_xy'])) and z['L']==r['L'] and z['source_id']==orig['source_id'],'PUBLIC label original qL/source')
            require(not z['human_gold'] and not z['physical_native_gold'] and not z['pixel_coordinate_gold'] and not z['native_visibility_target_defined'] and not z['point_quality_target_defined'],'PUBLIC owner-only authority')
            target=z['owner_aux_target'];require(target in (-1,0,1,2),'PUBLIC owner class')
            if target>=0:require(r['original_query_available'] and z['center4_valid'] and z['prior_owner_observation_present'],'Original PUBLIC weak support')
            labels[i]=target
    require(set(labels)==set(byid),'All PUBLIC TRAIN rows retained')
    ptarget=np.array([labels[r['original_global_index']] for r in pp],np.int8)
    require([r['original_global_index'] for r,y in zip(pp,ptarget) if y>=0]==protocol['public_known_original_indices'],'Frozen public253 exact selection')
    require(np.bincount(ptarget[ptarget>=0],minlength=3).tolist()==[53,52,148],'Public hard known classes')
    cc=[r for r in rows if r['domain']=='YCB_CF'];cl=lines(checked(b['CF_labels']));require(len(cl)==3816,'All CF label rows')
    ctarget=[]
    for j,(r,z) in enumerate(zip(cc,cl)):
        m=r['original'];require(z['cf_global_index']==j,'CF exact original occurrence order')
        for key in ('input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','L','query_available','feature_constructed','central4_available','pair_pixel_proof_PASS','selected_anchor_native_slot'):
            require(z[key]==m[key],'CF weak original metadata '+key)
        require(same(qarray(z['query_xy']),qarray(r['query_xy'])),'CF label q')
        require(z['native_visibility_target']==z['point_quality_target']==-1 and not z['human_gold'] and not z['physical_native_gold'],'CF owner-only authority')
        eligible=bool(m['pair_pixel_proof_PASS'] and m['variant'] in ('COVER','ADJACENT_SHAM') and m['native_slot']==m['selected_anchor_native_slot'] and m['query_available'] and m['feature_constructed'] and m['central4_available'])
        y=z['owner_aux_target'];require((y>=0)==eligible,'Exact CF known eligibility')
        require(y==(1 if m['variant']=='COVER' else 0) if eligible else y==-1,'CF fixed class authority')
        ctarget.append(y)
    ctarget=np.array(ctarget,np.int8)
    require(np.flatnonzero(ctarget>=0).tolist()==protocol['CF_known_occurrence_indices'] and np.bincount(ctarget[ctarget>=0],minlength=3).tolist()==[157,157,0],'Frozen CF314')
    for domain,target in [('PUBLIC_TRAIN',ptarget),('YCB_CF',ctarget)]:
        t=out[domain];t['owner_target']=target;known=target>=0;t['owner_soft'][known]=np.eye(3,dtype=np.float64)[target[known]];t['owner_ambiguous_mass'][known]=0;t['owner_invalid_mass'][known]=0
    return out

def sam_groups(source_seal,new_seal,rows):
    require(source_seal['completed'] and source_seal['PASS'] and source_seal['model_state_byte_unchanged'],'Frozen source SAM completion')
    require(new_seal['completed'] and new_seal['PASS'],'Completed public/CF SAM dependency, never silent none')
    require(source_seal['checkpoint']==new_seal['checkpoint'] and source_seal['source_commit']==new_seal['source_commit'],'Exact same frozen official SAM provenance')
    groups={}
    for s in source_seal['scenes']:
        key='SYNTH_SCENE_%06d'%s['scene_index'];require(key not in groups,'Unique source SAM group');groups[key]=s
    for s in new_seal['groups']:
        key=s['group_id'];require(key not in groups,'Unique new SAM group');groups[key]=s
    require(len(source_seal['scenes'])==512 and len(new_seal['groups'])==662,'All512+185+477 groups')
    expected={r['sam_group_id'] for r in rows};require(set(groups)==expected,'No dropped or additional SAM image group')
    return groups

def load_sam_group(s,rr,used):
    require(len(rr)==8 and [r['native_slot'] for r in rr]==list(range(8)),'Original native8 group')
    mm=s['original_input_metadata8'];require(len(mm)==8,'SAM metadata8')
    for r,m in zip(rr,mm):
        require(m['native_slot']==r['native_slot'] and m['original_global_index']==r['original_global_index'],'SAM original native/ID')
        require(same(qarray(m['original_query_xy']),qarray(r['query_xy'])) and m['original_L']==r['L'],'SAM original q/L')
        av=m.get('original_query_available',m.get('original_input_flags',{}).get('query_available'))
        require(av==r['original_query_available'],'SAM original source-specific availability')
        mid=m['original_occurrence_id'];expected=r['original'].get('original_query_id',r['row_id'])
        require(mid==expected,'SAM original query string')
    runtime=bool(s['runtime_available']);prompt=bool(s['prompt_available'])
    require(runtime or not prompt,'Retain runtime failure and fail cache; no silent mask replacement')
    if s['outputs'] is None:
        require(not runtime and not prompt,'Null SAM only explicit missing original prompt')
        return None,False
    require(runtime and prompt,'Saved SAM output must be actual available result')
    binding=s['outputs'];path=checked(binding);used[str(path)]=binding
    keys=['image_logits','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available']
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in keys}
    logits=a['image_logits'];require(logits.dtype==np.float32 and logits.shape==tuple(rr[0]['raw_hw']) and np.isfinite(logits).all(),'SAM fulloriginal image schema')
    require(a['mask_logits_gt0'].dtype==bool and same(a['mask_logits_gt0'],logits>0),'Frozen SAM strict threshold0')
    values=dict(original_query_xy=np.array([qarray(r['query_xy']) for r in rr],np.float64),original_L=np.array([np.nan if r['L'] is None else r['L'] for r in rr],np.float64),
        native_slot=np.arange(8,dtype=np.int8),original_global_index=np.array([r['original_global_index'] for r in rr],np.int64),original_query_available=np.array([r['original_query_available'] for r in rr],bool))
    for k,val in values.items():require(a[k].dtype==val.dtype and same(a[k],val),'SAM exact saved '+k)
    require(same(a['original_occurrence_id'],np.array([m['original_occurrence_id'] for m in mm])),'SAM exact query strings')
    require(same(a['prediction_available'],values['original_query_available']),'SAM inputU first')
    require(a['box_xyxy'].dtype==np.float64 and same(a['box_xyxy'],s['box_xyxy']),'SAM original q8 box')
    return logits,True

def save_npz(path,arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)==set(arrays),'Cache serialized keys')
        for k,a in arrays.items():require(z[k].dtype==a.dtype and z[k].shape==a.shape and z[k].tobytes()==a.tobytes(),'Cache roundtrip '+k)
    return bind(path)

def run(entry):
    e=read(entry);require(e.get('approved_source_raw_cache') is True and e.get('source_TRAIN_only') is True and e.get('automatic_retry') is False,'Separate approved root cache entry required')
    for key,path in [('scope',SCOPE),('code',Path(__file__)),('protocol',PROTOCOL)]:require(e[key]==bind(path),'Entry '+key)
    p=read(PROTOCOL);require(p['code']==bind(__file__) and p['checks']==bind(CHECKS),'Frozen preparation')
    rows=lines(checked(p['plan']));require(Counter(r['domain'] for r in rows)==DOMAINS,'Frozen all9392 plan')
    require(not OUT.exists(),'Never overwrite or automatic retry');OUT.mkdir()
    write(OUT/'STARTED.json',dict(completed=False,entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),automatic_retry=False))
    start=time.monotonic();used={}
    try:
        source=read(checked(p['inputs']['synth_SAM_complete']));new=read(checked(e['public_CF_SAM_complete']));groups=sam_groups(source,new,rows)
        targets=load_targets(rows,p);result={};raw_path=None;raw=None
        for domain in DOMAINS:
            rr=[r for r in rows if r['domain']==domain];data=input_arrays(rr);localgroups={}
            for j,r in enumerate(rr):localgroups.setdefault(r['sam_group_id'],[]).append((j,r))
            for key,jr in localgroups.items():
                original8=[r for j,r in jr];logits,sceneok=load_sam_group(groups[key],original8,used)
                for j,r in jr:
                    path=Path(r['raw_packet']['path'])
                    if raw_path!=path:
                        checked(r['raw_packet']);used[str(path)]=r['raw_packet']
                        keys=['rgb96','valid96','crop_constructed']+(['query_xy','L','native_slot','query_id'] if domain=='YCB_CF' else ['global_index'])
                        with np.load(path,allow_pickle=False) as z:raw={k:z[k].copy() for k in keys}
                        require(raw['rgb96'].dtype==np.uint8 and raw['rgb96'].ndim==4 and raw['rgb96'].shape[1:]==(96,96,3),'Original RGB96 HWC uint8')
                        require(raw['valid96'].dtype==bool and raw['valid96'].shape==raw['rgb96'].shape[:3],'Original valid96 bool')
                        require(raw['crop_constructed'].dtype==bool and raw['crop_constructed'].shape==(len(raw['rgb96']),),'Original constructed flag')
                        raw_path=path
                    k=r['raw_row'];im=raw['rgb96'][k];valid=raw['valid96'][k];orig=r['original']
                    if domain=='YCB_CF':
                        require(int(raw['native_slot'][k])==r['native_slot'] and str(raw['query_id'][k])==orig['original_query_id'],'CF raw exact original identity')
                        require(raw['query_xy'].dtype==np.float64 and same(raw['query_xy'][k],qarray(r['query_xy'])) and raw['L'].dtype==np.float64 and raw['L'][k]==r['L'],'CF raw original q/L')
                        construct=orig['feature_constructed']
                    else:
                        require(int(raw['global_index'][k])==r['original_global_index'],'Raw original global ID')
                        construct=orig['original_input_flags']['crop_constructed'] if domain=='SYNTH_TRAIN' else orig['feature_constructed']
                        require(ash(im)==r['rgb96_sha256'] and ash(valid)==r['valid96_sha256'],'Original RGB96/valid96 inherited fact hashes')
                    require(bool(raw['crop_constructed'][k])==construct,'Original construction diagnostic preserved')
                    data['rgb96'][j]=im;data['valid96'][j]=valid
                    mask,available=sample_sam(logits,r['query_xy'],r['L'],sceneok and r['original_query_available'],r['raw_hw'])
                    if sceneok and r['original_query_available']:require(same(available,valid),'SAM and RGB exact identical96 samplingdomain')
                    require(not (mask&~available).any(),'SAM unavailable never implies foreground')
                    data['sam_mask96'][j]=mask;data['sam_available96'][j]=available
                    r['actual_raw_RGB96_sha256']=ash(im);r['actual_raw_valid96_sha256']=ash(valid)
                    r['sam_prediction_available']=bool(sceneok and r['original_query_available']);r['sam_prompt_available']=groups[key]['prompt_available'];r['sam_runtime_available']=groups[key]['runtime_available'];r['SAM_output']=groups[key]['outputs']
            name=DOMAIN_KEYS[domain];ip=OUT/(name+'_INPUTS.npz');tp=OUT/(name+'_TARGETS.npz');mp=OUT/(name+'_ROWS.jsonl')
            ib=save_npz(ip,data);tb=save_npz(tp,targets[domain]);writelines(mp,rr)
            t=targets[domain];av=data['original_query_available'];soft=t['owner_soft'];finite=np.isfinite(soft).all(1)
            result[name]=dict(rows=len(rr),inputs=ib,targets=tb,metadata=bind(mp),original_inputU=int((~av).sum()),
                SAM_unavailable_rows=int((~data['sam_available96'].any((1,2))).sum()),support_available_counts=[{str(x):int((av&(t['support_target'][:,c]==x)).sum()) for x in (-1,0,1)} for c in range(2)],
                owner_hard_counts={str(x):int((t['owner_target']==x).sum()) for x in (-1,0,1,2)},owner_available_class_mass=np.sum(soft[av&finite],0,dtype=np.float64).tolist(),
                no_GT_model_input=True,original_qL_native_flags_preserved=True)
            print(json.dumps(dict(domain=name,rows=len(rr),cache_saved=True,seconds=time.monotonic()-start)),flush=True)
        for b in used.values():checked(b)
        for key,b in p['inputs'].items():
            if key in ('end_targets','TRAIN_owner6','public_labels','CF_labels'):checked(b)
        checked(e['public_CF_SAM_complete']);require(p['plan']==bind(PLAN),'Plan unchanged')
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,cache_contract_schema='V181_LOCAL96_PIXEL_CACHE_WITH_SEPARATE_TARGETS_v1',
            entry=bind(entry),scope=p['scope'],code=p['code'],protocol=bind(PROTOCOL),plan=p['plan'],public_CF_SAM_complete=e['public_CF_SAM_complete'],synth_SAM_complete=p['inputs']['synth_SAM_complete'],
            domains=result,rows9392=True,original_all_domains_and_unknown_U_rows_retained=True,all_used_raw_SAM_and_label_sources_SHA_unchanged=True,
            all_original_qL_native_split_flags_unchanged=True,GT_is_supervision_only=True,new_native_visibility_or_quality_targets=False,actual_new_model_forward_fit=0,
            original_V180_training_admission_false_provenance_retained=True,experimental_support_owner_admission_only_under_V181_scope=True,
            selected_model=None,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-start))
    except BaseException as ex:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,PASS=False,error=repr(ex),traceback=traceback.format_exc(),automatic_retry=False));raise

def invented():
    checks={}
    q=[1.25,2.75];L=3.;g=grid96(q,L).reshape(96,96,2)
    checks['q_at_halfpixel47_5_and_exact_center4_symmetric']=same(g[47,47]+g[48,48],np.array(q)*2)
    a=np.arange(30,dtype=np.float32).reshape(5,6)-12
    mask,av=sample_sam(a,q,L,True,a.shape)
    oracle=np.zeros((96,96),bool);oracleav=np.zeros_like(oracle)
    for iy in range(96):
        for ix in range(96):
            x,y=g[iy,ix];xx=int(np.floor(x));yy=int(np.floor(y));fx=x-xx;fy=y-yy
            v=0.
            for tx,ty,w in [(xx,yy,(1-fx)*(1-fy)),(xx+1,yy,fx*(1-fy)),(xx,yy+1,(1-fx)*fy),(xx+1,yy+1,fx*fy)]:
                if 0<=tx<6 and 0<=ty<5:v+=float(a[ty,tx])*w
            oracleav[iy,ix]=-.5<=x<5.5 and -.5<=y<4.5;oracle[iy,ix]=v>0 and oracleav[iy,ix]
    checks['actual_SAM_sampler_independent_scalar_4tap_and_domain']=same(mask,oracle) and same(av,oracleav)
    z,zv=sample_sam(np.zeros_like(a),q,L,True,a.shape);checks['strict_zero_is_nonmask_available']=not z.any() and zv.any()
    for aq in [None,[None,None],[-1.,-1.]]:
        z,zv=sample_sam(None,aq,None,False,a.shape);require(not z.any() and not zv.any(),'Missing never fake available mask')
    checks['missing_q_and_unavailable_SAM_explicit_zerochannels']=True
    mass=np.array([[.2,.1,.3,.1,.1,.2],[0,0,0,0,1,0],[np.nan]*6],np.float64)
    y,amb,inv=collapse_owner(mass)
    checks['partial_known_mass_preserved_no_row_renorm_or_drop']=same(y[0],[.2+.1,.3,.1]) and y[0].sum()<1 and amb[0]==.1 and inv[0]==.2 and y[1].sum()==0 and np.isnan(y[2]).all()
    rr=[dict(row_id='INVENTED0',original_global_index=0,native_slot=0,query_xy=None,L=None,original_query_available=False),dict(row_id='INVENTED1',original_global_index=1,native_slot=1,query_xy=[1.,2.],L=3.,original_query_available=True)]
    inp=input_arrays(rr);t=target_arrays(rr)
    checks['input_whitelist_no_GT_or_native_teacher_fields']=set(inp)=={'rgb96','valid96','sam_mask96','sam_available96','original_query_available','row_id','original_global_index','native_slot','query_xy','L'}
    checks['null_rows_retained_numeric_NaN_and_original_U']=np.isnan(inp['query_xy'][0]).all() and np.isnan(inp['L'][0]) and inp['original_query_available'].tolist()==[False,True]
    checks['targets_separate_unknown_no_native_quality_targets']=set(t)=={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'} and (t['support_target']==-1).all() and (t['owner_target']==-1).all() and not t['owner_soft'].any()
    fp=H/'INVENTED_ONLY_INPUT_AND_TARGET_ROUNDTRIP.npz';np.savez_compressed(fp,**inp,**{'target_'+k:v for k,v in t.items()})
    with np.load(fp,allow_pickle=False) as p:
        checks['actual_npz_roundtrip_input_bytes_and_dtypes']=all(p[k].dtype==v.dtype and p[k].shape==v.shape and p[k].tobytes()==v.tobytes() for k,v in inp.items())
    checks['CF477_payloads_is3816rows_not477_independent_targets']=477*8==3816 and 157*2==314
    require(all(checks.values()),'Invented checks')
    return dict(completed=True,PASS=True,invented_only=True,number_checks=len(checks),checks=checks,code=bind(__file__),source_payload_body_header_hash_reads=0)

def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Frozen scope');rows,b,pub,cf=inventories()
    writelines(PLAN,rows);write(CHECKS,invented())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),plan=bind(PLAN),inputs=b,checks=bind(CHECKS),domain_rows=DOMAINS,
        public_known_original_indices=pub,CF_known_occurrence_indices=cf,
        input_schema='Separate domain INPUTS.npz: rgb96u8[N,96,96,3],valid96bool,sam_mask96bool,sam_available96bool[N,96,96],original_query_availablebool[N],row_idU[N],original_global_indexi64[N],native_sloti8[N],query_xyf64[N,2],Lf64[N]. GT never input.',
        target_schema='Separate TARGETS.npz exact5keys: support_targeti8[N,2](-1/0/1),owner_softf64[N,3],owner_targeti8[N](-1..2),owner_ambiguous_mass/owner_invalid_massf64[N]. Native/quality -1 provenance in ROWS metadata only; target roworder exact INPUTS/ROWS.',
        owner_rule='SYNTH original scene_owner_central4: incident+other=T,FOREIGN=F,BG=BG,rawmass sum<=1 without renormalization/drop; AMBIG/INVALID separate,missing allNaN. Public/CF hard-known onehot+amb/invalid0; unknownsoftzero/hard-1 with amb/invalidNaN (weak unknown, not measured ambiguity). Support targets only fixed V180 TRAIN end sidecar.',
        SAM_rule='Official frozen V180 same model. Exact original q/L float64 96-grid; bilinear full image_logits then strict>0; availability originalinput&prompt&runtime&frozen samplingdomain, never GT, never thresholdfit. Partial image footprint follows original crop domain. Original raw/valid unchanged.',
        public_CF_SAM_dependency='Must bind completed C public185+CF477 SAM seal at actual root entry. All662 groups retained; no silent missing masks. Failure statuses explicit, no retry or substitution.',
        output=str(OUT),new_raw_label_feature_GT_or_SAM_body_header_hash_reads=0,actual_model_forward_fit=0,selected_model=None,fully_solved=False)
    write(PROTOCOL,p)
    write(H/'ROOT_ACTUAL_CACHE_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_source_raw_cache=False,source_TRAIN_only=True,scope=p['scope'],code=p['code'],protocol=bind(PROTOCOL),public_CF_SAM_complete=None,automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],scope=p['scope'],protocol=bind(PROTOCOL),checks=bind(CHECKS),plan=p['plan'],pending_public_CF_SAM_completed_binding=True,actual_source_payload_reads=0))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry');args=ap.parse_args()
    if args.prepare:prepare()
    elif args.entry:run(Path(args.entry).resolve())
    else:ap.error('Use --prepare or separately approved root --entry')
