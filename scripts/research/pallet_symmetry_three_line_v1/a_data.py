"""Locked same-order RGB loader; metadata only defines allowed supervision orbits."""
import functools
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import env as E

@functools.lru_cache(None)
def validate_permutations(perms):
    """Only previously audited proper rotation groups, never reflections or point swaps."""
    groups=E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    allowed={tuple(tuple(p) for p in g['permutations']) for g in groups}
    allowed.add((tuple(range(9)),))
    if perms not in allowed:raise ValueError('Group differs from audited proper C1/C2/C4 rotations')
    return True

@functools.lru_cache(None)
def records(cohort,split):
    if cohort=='RECT':
        side={r['id']:r for r in E.read(E.RAW/'A/rect_target_sidecar.json')['records']}
        rows=[]
        for source in E.read(E.C.LINE/'SOURCE_MANIFEST.json')['records']:
            if source['source_split']!=split:continue
            r=dict(source);r.update(side[r['id']]);r['raw_hw']=r['raw_shape_hw'];rows.append(r)
        return rows
    side={r['id']:r for r in E.read(E.RAW/'A/square_annotation_target_view.json')['records']}
    perms=E.read(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations']
    return [dict(r,permutations=perms,group_order=4,yolo_target=side[r['id']]['yolo_target'])
            for r in E.read(E.RAW/'A/square_membership.json')[split]['records']]

def target(r):
    return np.array(r['yolo_target'],np.float32) if 'yolo_target' in r else np.array((E.ROOT/r['label']).read_text().split(),np.float32).reshape(-1,32)

def sample(r):
    im=cv2.imread(str(E.ROOT/r['image']));assert im is not None
    h,w=im.shape[:2];gain=min(640/h,640/w);nw,nh=round(w*gain),round(h*gain)
    left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
    im=cv2.resize(im,(nw,nh));im=cv2.copyMakeBorder(im,top,640-nh-top,left,640-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    a=target(r).copy();a[:,1:3]=(a[:,1:3]*[w,h]*gain+[left,top])/640;a[:,3:5]*=np.array([w,h])*gain/640
    kp=a[:,5:].reshape(-1,9,3);kp[...,:2]=(kp[...,:2]*[w,h]*gain+[left,top])/640
    perms=np.tile(np.arange(9),(4,1));p=np.array(r['permutations']);perms[:len(p)]=p
    validate_permutations(tuple(tuple(x) for x in r['permutations']))
    assert np.isfinite(a).all() and all(sorted(x)==list(range(9)) and x[8]==8 for x in p)
    return dict(img=torch.from_numpy(im[:,:,::-1].transpose(2,0,1).copy()),
      cls=torch.from_numpy(a[:,:1].copy()),bboxes=torch.from_numpy(a[:,1:5].copy()),keypoints=torch.from_numpy(kp.copy()),
      permutations=torch.from_numpy(np.tile(perms[None],(len(a),1,1))),
      group_valid=torch.from_numpy(np.tile((np.arange(4)<len(p))[None],(len(a),1))),id=r['id'])

def collate(rows):
    out={k:torch.cat([r[k] for r in rows]) for k in ['cls','bboxes','keypoints','permutations','group_valid']}
    out.update(img=torch.stack([r['img'] for r in rows]),
      batch_idx=torch.cat([torch.full((len(r['cls']),),i,dtype=torch.float32) for i,r in enumerate(rows)]),ids=[r['id'] for r in rows])
    return out

def cuda(batch):
    out={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
    out['img']=out['img'].float()/255
    return out

def order_indices(n,seed,steps=2000):
    rng=np.random.default_rng(seed);out=[]
    while len(out)<steps*16:out.extend(rng.permutation(n).tolist())
    return np.array(out[:steps*16],np.int64)

class FixedDataset(Dataset):
    def __init__(self,cohort,split='train',indices=None):
        self.rows=records(cohort,split);self.indices=np.arange(len(self.rows)) if indices is None else indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return sample(self.rows[self.indices[i]])

def worker_init(_):
    torch.set_num_threads(1);cv2.setNumThreads(1)
