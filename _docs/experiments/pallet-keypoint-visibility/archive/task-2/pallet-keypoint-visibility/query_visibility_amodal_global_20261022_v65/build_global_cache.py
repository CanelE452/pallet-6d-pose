"""Join input-only whole-target features with independently read source GT.

The global grid is fixed by the frozen inference prompt box. Targets are
sampled from native-resolution masks; dilation is never done in crop pixels.
"""
from pathlib import Path
import argparse,csv,hashlib,json,time
import cv2
import numpy as np
from PIL import Image
from global_recipe import grid,native_valid,local_mapping

ROOT=Path(__file__).resolve().parent
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
V64=ROOT.parent/'query_visibility_multitarget_20261022_v64'
V61=ROOT.parent.parent/'target_presence_classifier_20261022_v61'
JOINT=np.asarray([[0,-1,-1],[1,2,-1],[3,4,5]],np.int8)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def make(source,split):
    start=time.monotonic();output=ROOT/'source_cache'/f'{source}_{split}_GLOBAL.npz'
    assert not output.exists()
    source_path=V64/'source_cache'/f'{source}_{split}_WITH_SAM.npz'
    with np.load(source_path,allow_pickle=False) as z:old={k:z[k]for k in z.files}
    with np.load(ROOT/'sam_source/GLOBAL_INPUTS.npz',allow_pickle=False) as z:
        good=(z['source']==source)&(np.char.upper(z['split'])==split)
        assert good.any(),(source,split)
        glob={k:z[k][good]for k in z.files if k.startswith('global_')}
    rows={r['id']:r for r in [json.loads(s)for s in(V63/'sam_source/INPUTS.jsonl').read_text().splitlines()]}
    if source=='SYNTH':
        with(V61/'MASK_CONTRACT_AUDIT.csv').open(newline='')as f:raw=list(csv.DictReader(f))
        masks={r['id']:(r['visible_path'],r['amodal_path'])for r in raw};threshold=127
    else:
        path=V63/'ycb'/('train_instances.jsonl'if split=='TRAIN'else'SOURCE_VAL_LOCKED_200.jsonl')
        raw=[json.loads(s)for s in path.read_text().splitlines()]
        masks={f"YCB:{r['scene_id']:06d}:{r['frame_id']:06d}:{r['instance_id']:06d}":(r['mask_vis'],r['mask_full'])for r in raw};threshold=0
    t=len(glob['global_id']);target={s:i for i,s in enumerate(glob['global_id'].astype(str))}
    n=len(old['x']);target_index=np.array([target[s]for s in old['id'].astype(str)],np.int64)
    global_joint=np.empty((t,64,64),np.int8)
    map64=np.empty((n,9,9),np.int64);map8=np.empty_like(map64);queryvalid=np.empty((n,9,9),bool)
    denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    assert not(set(glob['global_image_sha256'].astype(str))&denied)
    assert np.array_equal(glob['global_image_sha256'][target_index],old['image_sha256'])
    label_counts=np.zeros(6,np.int64);centers_covered=0
    for ti,ident in enumerate(glob['global_id'].astype(str)):
        r=rows[ident];h,w=r['raw_hw']
        assert np.array_equal(glob['global_box'][ti],r['box'])
        vis=np.array(Image.open(masks[ident][0]))>threshold
        full=np.array(Image.open(masks[ident][1]))>threshold
        assert vis.shape==full.shape==(h,w)and not(vis&~full).any()
        v2=cv2.dilate(vis.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        f2=cv2.dilate(full.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        xy,p,_,_=grid(r['box']);valid=native_valid(xy,p,[h,w])
        assert np.array_equal(valid,glob['global_valid'][ti])
        px=np.clip(p[...,0],0,w-1);py=np.clip(p[...,1],0,h-1)
        exact=np.where(vis[py,px],0,np.where(full[py,px],1,2))
        near=np.where(v2[py,px],0,np.where(f2[py,px],1,2))
        label=JOINT[exact,near];assert(label>=0).all()
        global_joint[ti]=label;label_counts+=np.bincount(label[valid],minlength=6)
        i64,i8,qv=local_mapping(r['query'],r['side'],r['box'],[h,w])
        ix=np.flatnonzero(target_index==ti)
        for j in ix:
            k=int(old['native_corner'][j]);map64[j]=i64[k];map8[j]=i8[k];queryvalid[j]=qv[k]
            centers_covered+=int(qv[k,4,4])
        if(ti+1)%300==0:print('GLOBAL_GT',source,split,ti+1,t,round(time.monotonic()-start,1),flush=True)
    assert(map64>=0).all()and(map64<4096).all()and(map8>=0).all()and(map8<64).all()
    expected=((map64//64)//8)*8+(map64%64)//8
    assert np.array_equal(map8[queryvalid],expected[queryvalid])
    assert not(queryvalid&~old['dense_valid']).any()
    arrays={**old,**glob,'global_joint':global_joint,'target_index':target_index,
            'global_index64':map64,'global_index8':map8,'global_query_valid':queryvalid}
    np.savez_compressed(output,**arrays)
    write(output.with_suffix('.json'),dict(source=source,split=split,queries=n,targets=t,center_global_mapping_coverage=centers_covered/n,
          valid_global_cells=int(glob['global_valid'].sum()),valid_global_joint_counts=label_counts.tolist(),
          source_cache_sha256=sha(source_path),global_inputs_sha256=sha(ROOT/'sam_source/GLOBAL_INPUTS.npz'),
          cache_sha256=sha(output),recipe_sha256=sha(ROOT/'global_recipe.py'),code_sha256=sha(__file__),
          local_arrays_unchanged=True,GT_features_used=False,current_eval_accessed=False,updates=0,PnP_calls=0,seconds=time.monotonic()-start))
    print(json.dumps(json.loads(output.with_suffix('.json').read_text())),flush=True)

def merge(split):
    output=ROOT/'source_cache'/f'MIXED_{split}_GLOBAL.npz';assert not output.exists()
    inputs=[ROOT/'source_cache'/f'{s}_{split}_GLOBAL.npz'for s in ['SYNTH','YCB']]
    arrays=[]
    for p in inputs:
        with np.load(p,allow_pickle=False)as z:arrays.append({k:z[k]for k in z.files})
    assert arrays[0].keys()==arrays[1].keys()
    arrays[1]['target_index']=arrays[1]['target_index']+len(arrays[0]['global_id'])
    merged={k:np.concatenate([a[k]for a in arrays],axis=0)for k in arrays[0]}
    assert np.array_equal(merged['global_id'][merged['target_index']],merged['id'])
    assert np.array_equal(merged['global_image_sha256'][merged['target_index']],merged['image_sha256'])
    np.savez_compressed(output,**merged)
    write(output.with_suffix('.json'),dict(split=split,queries=len(merged['x']),targets=len(merged['global_id']),input_sha256=[sha(p)for p in inputs],cache_sha256=sha(output),code_sha256=sha(__file__),current_eval_accessed=False,updates=0,PnP_calls=0))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['SYNTH','YCB','MIXED'],required=True);ap.add_argument('--split',choices=['TRAIN','VALIDATION'],required=True);a=ap.parse_args();cv2.setNumThreads(1)
    if a.source=='MIXED':merge(a.split)
    else:make(a.source,a.split)
