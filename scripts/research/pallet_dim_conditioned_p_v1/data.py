"""Read-only P cache with inference metadata and supervision in separate fields."""
import numpy as np
import torch
import dcp_env as E
from refiner import context,specification

class PaperData:
    def __init__(self):
        self.base=E.old('train').FeatureDataset(E.LINE,E.LINE/'cache')
        self.arrays=self.base.arrays;self.indices=self.base.indices;self.source=self.base.source
        self.train_rows=self.base.train_rows;self.validation_rows=self.base.validation_rows;self.partitions=self.base.partitions
        self.side=np.load(E.RAW/'DIMENSION_SIDECAR.npz');self.norm=E.read(E.DOC/'DIM_NORMALIZATION_LOCK.json')
        assert np.array_equal(self.side['record_index'],self.indices)
    def batch(self,rows,arm,device='cuda',supervision=True):
        keys=('p3','p4','points','boxes','point_valid','input_shape')
        if supervision:keys+=('gt_points','gt_valid')
        batch={k:torch.from_numpy(np.array(self.arrays[k][rows],copy=True)).to(device) for k in keys}
        dim,_=specification(arm)
        if dim:batch['context']=torch.from_numpy(context(self.side['dimensions'][rows],self.side['order'][rows],self.norm,dim==8)).to(device)
        if supervision:
            for k in ['permutations','group_valid']:batch[k]=torch.from_numpy(self.side[k][rows].copy()).to(device)
        return batch

def validate_group(permutations,order):
    c=E.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    expected=[list(range(9))] if order==1 else next(o['permutations'] for o in c['objects'] if o['group_order']==order)
    if np.asarray(permutations).tolist()!=expected:raise ValueError('Not the approved proper rotation group/order; reflections and C2 quarter-turns rejected')
    edges={tuple(sorted(e)) for e in c['edges']}
    for p in permutations:
        assert p[8]==8 and sorted(p)==list(range(9))
        assert {tuple(sorted([p[a],p[b]])) for a,b in edges}==edges
    return True
