"""Freeze GT-independent donor identities; audit TRAIN residual locality only."""
from collections import Counter
import numpy as np
import torch
import dcp_env as E
from data import PaperData
from refiner import local_phase

def donors(ids):
    rng=np.random.default_rng(20260917);p=rng.permutation(len(ids));mapping=np.empty(len(ids),int);mapping[p]=np.roll(p,1)
    assert (mapping!=np.arange(len(ids))).all()
    return [dict(recipient=a,donor=ids[j]) for a,j in zip(ids,mapping)]

def main():
    data=PaperData();held=[r['id'] for r in data.source['records'] if r['partition']=='heldout']
    pe=E.old('paper_evaluation');dev=sorted(r['frame_id'] for r in E.read(pe.POS)['items'])
    E.freeze(E.DOC/'METADATA_DIAGNOSTIC_LOCK.json',dict(seed=20260917,heldout=donors(held),DEV=donors(dev),GT_used=False,
      train_mean_dimensions_WDH=data.side['dimensions'][data.partitions=='train'].mean(0).tolist(),
      wrong_order=dict(C1=2,C2=4,C4=1),D4='standardized dimensions=0, all three group entries=0; explicit diagnostic override, never missing-value fallback'))
    stats={str(k):dict(before=[],after=[],branches=[]) for k in [1,2,4]}
    for start in range(0,len(data.train_rows),4096):
        rows=data.train_rows[start:start+4096];a=data.arrays
        tensors={k:torch.from_numpy(np.array(a[k][rows])) for k in ['points','point_valid','gt_points','gt_valid','boxes']}
        diag=(tensors['boxes'][:,2:]-tensors['boxes'][:,:2]).norm(dim=-1).clamp_min(1)
        gt,v,g,_=local_phase(tensors['points'],tensors['point_valid'],tensors['gt_points'],tensors['gt_valid'],torch.from_numpy(data.side['permutations'][rows]),torch.from_numpy(data.side['group_valid'][rows]),diag)
        before=torch.linalg.vector_norm(tensors['gt_points']-tensors['points'],dim=-1)/diag[:,None];after=torch.linalg.vector_norm(gt-tensors['points'],dim=-1)/diag[:,None]
        for j,row in enumerate(rows):
            key=str(data.side['order'][row]);st=stats[key]
            oldmask=tensors['gt_valid'][j,:8]&tensors['point_valid'][j,:8];newmask=v[j,:8]&tensors['point_valid'][j,:8]
            st['before'].extend(before[j,:8][oldmask].tolist());st['after'].extend(after[j,:8][newmask].tolist());st['branches'].append(int(g[j]))
    out={}
    for k,s in stats.items():
        out['C'+k]=dict(frames=len(s['branches']),branches=dict(Counter(s['branches'])))
        for phase in ['before','after']:
            a=np.array(s[phase]);out['C'+k][phase]=dict(corners=len(a),median=float(np.median(a)) if len(a) else None,P90=float(np.quantile(a,.9)) if len(a) else None,max=float(a.max()) if len(a) else None,
              fraction_in_radius=float((a<=.08).mean()) if len(a) else None,fraction_outside_radius=float((a>.08).mean()) if len(a) else None)
    E.write(E.DOC/'LOCAL_TARGET_RANGE_AUDIT.json',dict(complete=True,paper_train=out,matched_train_rows=len(data.train_rows),GT_for_training_diagnostic_only=True,no_radius_change=True))
    print('DONORS_FROZEN_AND_TRAIN_RANGE_AUDITED',flush=True)
if __name__=='__main__':main()
