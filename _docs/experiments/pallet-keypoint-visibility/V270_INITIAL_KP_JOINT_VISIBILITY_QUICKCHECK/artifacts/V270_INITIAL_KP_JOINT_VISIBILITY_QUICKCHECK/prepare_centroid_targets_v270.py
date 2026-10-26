"""Add stored renderer centroid to sealed native8 SYN targets; no new labels."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np

HERE=Path(__file__).absolute().parent


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def read_bound(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path']).read_bytes()


def savej(path,value):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    source=safe(HERE/'syn_joint_targets_v1');complete=json.loads((source/'COMPLETE.json').read_text())
    assert complete['PASS'] and complete['completed']
    rows=[json.loads(x) for x in read_bound(complete['outputs']['FRAME_TARGETS1003.jsonl']).decode().splitlines()]
    z=np.load(io.BytesIO(read_bound(complete['outputs']['TARGETS.npz'])),allow_pickle=False)
    arrays={key:z[key] for key in z.files}
    out=safe(HERE/'syn_joint_targets9_v1');assert not out.exists(),out;out.mkdir()
    savej(out/'PROTOCOL_BEFORE_CENTROID_READS.json',{'code':binding(Path(__file__).absolute()),
        'native8_complete':binding(source/'COMPLETE.json'),'centroid_xy':'Stored annotation.projected_cuboid_centroid',
        'centroid_depth':'Mean of source renderer camera-space8corner depths',
        'centroid_visibility_teacher':'NONE. External native loss uses only8native targets.',
        'scope':'Only extends coordinate branch to9; native8VIS/LOC labels preserved byte values.',
        'current223_reads':0,'RGB_reads':0,'M4_reads':0,'model_forwards':0})
    xy9=np.zeros((1003,9,2),np.float32);xy9[:,:8]=arrays['xy_px']
    xy019=np.zeros_like(xy9);xy019[:,:8]=arrays['xy01']
    valid9=np.zeros((1003,9),bool);valid9[:,:8]=arrays['loc_valid']
    centroid=[];counts={'train':0,'validation':0}
    for i,r in enumerate(rows):
        ann=json.loads(read_bound(r['annotation']));o=ann['objects'][0]
        p=np.asarray(o['projected_cuboid_centroid'],np.float64)
        assert p.shape==(2,) and np.isfinite(p).all()
        h,w=r['raw_hw'];d=float(arrays['gt_depth'][i].astype(np.float64).mean())
        inside=bool(d>0 and 0<=p[0]<w and 0<=p[1]<h)
        xy9[i,8]=p.astype(np.float32);xy019[i,8]=((p+.5)/np.asarray([w,h],np.float64)).astype(np.float32)
        valid9[i,8]=inside;counts[r['split']]+=int(inside)
        centroid.append({'frame_index':i,'id':r['id'],'split':r['split'],'centroid_gt_xy_px':p.tolist(),
            'centroid_loc_valid':inside,'centroid_depth':d,'annotation':r['annotation'],'centroid_visibility_y':None})
    assert np.array_equal(xy9[:,:8],arrays['xy_px']) and np.array_equal(valid9[:,:8],arrays['loc_valid'])
    arrays['xy_px']=xy9;arrays['xy01']=xy019;arrays['loc_valid']=valid9
    arrays['gt_depth9']=np.concatenate([arrays['gt_depth'],arrays['gt_depth'].mean(1,keepdims=True)],axis=1)
    np.savez_compressed(safe(out/'TARGETS9.npz'),**arrays)
    safe(out/'CENTROID_TARGETS1003.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in centroid))
    savej(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'native8_complete':binding(source/'COMPLETE.json'),
        'outputs':{n:binding(out/n) for n in ['PROTOCOL_BEFORE_CENTROID_READS.json','TARGETS9.npz','CENTROID_TARGETS1003.jsonl']},
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
        'centroid_loc_valid':counts,'centroid_visibility_teacher':False,'native8_targets_unchanged':True,
        'current223_reads':0,'RGB_reads':0,'M4_reads':0,'model_forwards':0})
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json'),'centroid_loc_valid':counts}),flush=True)


if __name__=='__main__':main()
