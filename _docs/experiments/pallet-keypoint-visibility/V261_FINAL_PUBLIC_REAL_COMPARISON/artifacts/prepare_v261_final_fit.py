"""Root feature checks and exact entry assembly for the ONE fixed V261 pair."""
import hashlib, json, os, stat
from collections import Counter
from pathlib import Path
import numpy as np

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
F = R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON'
def guard(value):
    p=Path(value)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112','cache','.pyc'))
    for a in [*reversed(p.parents),p]:
        try: m=os.lstat(a).st_mode
        except FileNotFoundError: continue
        assert not stat.S_ISLNK(m), str(a)
    return p
def bind(p):
    p=guard(p);h=hashlib.sha256();n=0
    with p.open('rb') as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b);n+=len(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=n)
def load(p):return json.loads(guard(p).read_bytes())
def write(p,j):
    p=guard(p)
    with p.open('x',encoding='utf-8') as s:json.dump(j,s,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);s.write('\n')
    return bind(p)
def checked(b):
    assert bind(b['path'])=={k:b[k] for k in ('path','sha256','size_bytes')},b['path']
    return guard(b['path'])
def run():
    cp=F/'public_aux_frozen_features_v1/COMPLETE.json';c=load(cp)
    assert c['completed'] is True and c['PASS'] is True
    assert c['classifier_forwards']==c['optimizer_updates']==0
    for key in ('features','samples','records','frame_manifest'):checked(c[key])
    qa=load(checked(c['outputs']['FEATURE_QA.json']))
    assert qa['PASS'] and qa['encoder_state_sha256_before']==qa['encoder_state_sha256_after']=='bb4c7a80aa873bf3d34c3c1d44a28cee331503d7ccf52aba85f2d23edb65c3c3'
    x=np.load(c['features']['path'],mmap_mode='r',allow_pickle=False)
    with np.load(c['samples']['path'],allow_pickle=False) as s:sample={k:s[k] for k in s.files}
    rows=[json.loads(line) for line in guard(c['records']['path']).read_text().splitlines()]
    frames=load(c['frame_manifest']['path'])['frames'];n=len(rows)
    assert x.shape==(n,1186) and x.dtype==np.float32 and np.isfinite(x).all()
    assert set(sample)=={'target','family','fit','available','row_id','frame_id','coarse_group_id'}
    assert all(len(a)==n for a in sample.values()) and (sample['family']==1).all()
    assert sample['fit'].dtype==sample['available'].dtype==np.bool_
    assert not x[:,1154:].any() and not x[~sample['available']].any()
    assert ((x[:,1152:1154]>=0)&(x[:,1152:1154]<=1)).all()
    for start in (0,384,768):
        norm=np.linalg.norm(x[sample['available'],start:start+384],axis=1)
        assert np.max(np.abs(norm-1))<2e-5
    assert len(frames)==284 and Counter(f['source_role'] for f in frames)=={'TRAIN':168,'DEV':116}
    assert sum(sum(f['label_counts'].values()) for f in frames)==284*4800
    salt='V261_PUBLIC_POINT_STRATIFIED64_V1|'
    for f in frames:
        assert all(f['selected_counts'][k]==min(64,f['label_counts'][k]) for k in ('D','E'))
        assert len(f['selected_reference_grid_indices'])==sum(f['selected_counts'].values())
    by_frame={f['publisher_frame_path']:f for f in frames}
    selected=Counter()
    for i,row in enumerate(rows):
        assert row['sample_index']==i and row['q_xy']==row['r_xy'] and row['q_offset_index']==0 and row['L_pixels']==96
        assert row['native_slot'] is None and row['SELF_state']=='UNKNOWN' and row['family']==1
        assert row['target'] in (0,1) and row['fit']==(row['source_role']=='TRAIN')
        assert row['row_id']==str(sample['row_id'][i]) and row['target']==int(sample['target'][i]) and row['fit']==bool(sample['fit'][i]) and row['available']==bool(sample['available'][i])
        assert row['reference_grid_index'] in by_frame[row['publisher_frame_path']]['selected_reference_grid_indices']
        selected[(row['source_role'],row['target'])]+=1
    verification=write(F/'ROOT_PUBLIC_FEATURE_VERIFICATION.json',dict(completed=True,PASS=True,feature_complete=bind(cp),selected_rows=n,all_reference_rows=284*4800,frames=284,frame_roles={'TRAIN':168,'DEV':116},selected_counts={str(k):v for k,v in sorted(selected.items())},available=int(sample['available'].sum()),unsupported=int((~sample['available']).sum()),checks=['Four exact declared artifact SHA/size','284 frames retained including empty','Fixed class cap64 census','All selected q=r offset0 L96 and AUX1/nativeNone/SELFunknown','Samples/records indices labels roles availability match','Finite1186FP32; zero32; unsupported zeros; three unit blocks; fractions in[0,1]','Official encoder state unchanged; producer first-batch repeat recorded'],no_RGB_encoder_or_classifier_rerun=True))
    e=load(R/'V261_FINAL_PUBLIC_AUX_TRAINER_B/ENTRY_FALSE_TEMPLATE.json')
    originals={
      'original_feature_complete':('V239_ROOT_CANONICAL_CONTEXT/canonical_source_features_v1/COMPLETE.json','e2e796ab5c0580b95204292f1f4ecb90d30842060d9b048eab6c41f942cfc392',11988),
      'original_features':('V239_ROOT_CANONICAL_CONTEXT/canonical_source_features_v1/ORIGINAL_ALL_FEATURES.npy','607acc267dbd6f35e7a3fc6da4bcd1fb5e225f3f7045815702e67357f50463e1',36135176),
      'original_samples':('V233_ROOT_CROP_ADAPTATION/crop_samples_v1/samples.npz','837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99',2325681)}
    for k,(path,sha,size) in originals.items():e[k]=dict(path=str(R/path),sha256=sha,size_bytes=size)
    e.update(public_feature_complete=bind(cp),public_feature_verification=verification)
    for dst,src in [('public_features','features'),('public_samples','samples'),('public_records','records'),('public_frame_manifest','frame_manifest')]:e[dst]={k:c[src][k] for k in ('path','sha256','size_bytes')}
    keys=('original_feature_complete','original_features','original_samples','public_feature_complete','public_features','public_samples','public_records','public_frame_manifest','public_feature_verification')
    e['data_whitelist']=[e[k] for k in keys];e['approved_actual_final_public_aux_fit']=True
    e['root_scope']=write(F/'ACTUAL_FINAL_FIT_SCOPE.json',dict(approved_actual_final_public_aux_fit=True,entry_contract={k:v for k,v in e.items() if k!='root_scope'}))
    eb=write(F/'ACTUAL_FINAL_PUBLIC_AUX_FIT_ENTRY.json',e)
    print(json.dumps(dict(completed=True,PASS=True,verification=verification,entry=eb,counts={str(k):v for k,v in sorted(selected.items())})))
if __name__=='__main__':run()
