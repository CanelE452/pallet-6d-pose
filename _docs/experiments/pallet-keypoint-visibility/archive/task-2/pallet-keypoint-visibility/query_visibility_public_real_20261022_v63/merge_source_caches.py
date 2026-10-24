"""Merge only source training/validation caches, preserving image exclusion."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'source_cache/v2'
deny=set(json.load(open(ROOT/'contract/EVALUATION_SHA256_DENYLIST.json'))['image_sha256'])
reports={}
for split in ['TRAIN','VALIDATION']:
    paths=[CACHE/f'SYNTH_{split}.npz',CACHE/f'YCB_{split}.npz']
    arrays=[]
    for path in paths:
        with np.load(path)as z:arrays.append({key:z[key]for key in z.files})
    keys=set(arrays[0])&set(arrays[1])
    combined={key:np.concatenate([a[key]for a in arrays])for key in sorted(keys)}
    assert not(set(combined['image_sha256'])&deny)
    assert len(set(combined['sample_id']))==len(combined['y'])
    assert combined['x'].shape==(len(combined['y']),65,9,9)
    assert np.array_equal(combined['y'],combined['dense_y'][:,4,4])
    path=CACHE/f'MIXED_{split}.npz'
    assert not path.exists()
    np.savez_compressed(path,**combined)
    reports[split]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'queries':len(combined['y']),'positive':int(combined['y'].sum()),'sources':{source:int((combined['source']==source).sum())for source in np.unique(combined['source'])},'input_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}}
    if split=='TRAIN':train_ids=set(combined['group_id']);train_hashes=set(combined['image_sha256'])
    else:assert not(train_ids&set(combined['group_id']))and not(train_hashes&set(combined['image_sha256']))
report={'data_sources':['synthetic','YCB-V official train_real'],'current_eval_overlap':0,'TRAIN_VAL_group_overlap':0,'TRAIN_VAL_image_hash_overlap':0,'context':'input-only8ROIvalidweighted64featuremean','output':reports,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(CACHE/'MIXED_CACHE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
