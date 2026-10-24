"""Join immutable source-target caches and frozen SAM overlays; no inference."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'source_cache'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
overlay=ROOT/'sam_source/OVERLAY.npz';seal=json.loads((ROOT/'sam_source/SEAL.json').read_text());assert seal['overlay_sha256']==sha(overlay)
with np.load(overlay)as z:ids=z['id'].astype(str);sam=z['sam'];hashes=z['image_sha256'].astype(str)
lookup={s:i for i,s in enumerate(ids)};assert len(lookup)==len(ids)
records={};splits={}
for split in ['TRAIN','VALIDATION']:
    pieces=[];rows=[]
    for source in ['SYNTH','YCB']:
        original=OUT/f'{source}_{split}.npz'
        with np.load(original,allow_pickle=False)as z:a={k:z[k]for k in z.files}
        ii=np.array([lookup[x]for x in a['id'].astype(str)]);assert np.array_equal(hashes[ii],a['image_sha256'].astype(str))
        a['sam']=sam[ii,a['native_corner'].astype(int)];assert a['sam'].shape==(len(ii),2,9,9)
        dest=OUT/f'{source}_{split}_WITH_SAM.npz';assert not dest.exists();np.savez_compressed(dest,**a)
        rows.append(dict(path=str(dest),sha256=sha(dest),source_input_sha256=sha(original),queries=len(ii)))
        pieces.append(a)
    merged={k:np.concatenate([a[k]for a in pieces])for k in pieces[0]};dest=OUT/f'MIXED_{split}_WITH_SAM.npz';assert not dest.exists()
    np.savez_compressed(dest,**merged);rows.append(dict(path=str(dest),sha256=sha(dest),queries=len(merged['x'])))
    splits[split]={k:set(merged[k].astype(str))for k in ['sample_id','group_id','image_sha256']}
    records[split]=rows
assert all(not(splits['TRAIN'][k]&splits['VALIDATION'][k])for k in splits['TRAIN'])
denied=set(json.loads((ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
assert all(not(s['image_sha256']&denied)for s in splits.values())
manifest=dict(schema='MULTITARGET_SOURCE_SAM_MERGE_V1',outputs=records,overlay_sha256=sha(overlay),code_sha256=sha(__file__),all_input_values_preserved=True,
              train_validation_sample_group_image_overlap=0,current_eval_image_overlap=0,current_eval_data_accessed=False,new_training_updates=0,new_PnP_calls=0)
write(OUT/'MERGE_PROVENANCE.json',manifest);print(json.dumps(manifest),flush=True)
