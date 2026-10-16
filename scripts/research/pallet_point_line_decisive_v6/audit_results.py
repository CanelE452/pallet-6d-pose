"""Replay metrics with NumPy from stored R/t and independently expanded corners."""
from pathlib import Path
import json
import math
import sys
import numpy as np
import torch
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]


def main():
    run=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
    export=ROOT/'data/pallet/results/pallet_point_line_v5_local/export'
    metadata=json.loads((export/'synth_val.json').read_text())['records']
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1]],float)
    prepared={}
    for rec in metadata:
        O=torch.load(export/rec['observation'],weights_only=True,map_location='cpu')
        S=torch.load(export/rec['supervision'],weights_only=True,map_location='cpu')
        dims=O['dims'][0].numpy().astype(float)
        prepared[rec['id']]=(signs*dims[[0,2,1]]/2,O['K'][0].numpy().astype(float),
            S['points'][0,:8].numpy().astype(float),math.hypot(*O['image_hw'][0].tolist()),rec['symmetry']['order'])
    summary={};maxnorm=maxpx=0.
    for arm in ('point','direct','hough'):
        for seed in (1,2,3):
            name=f'{arm}_seed{seed}'
            pred=json.loads((run/'predictions'/f'{name}_synth_val.json').read_text())
            ev=json.loads((run/'evaluations'/f'{name}_synth_val.json').read_text())
            assert len(pred['rows'])==505 and len(ev['per_frame'])==512
            assert pred['supervision_files_opened']==0 and pred['observation_loads']==505
            saved={r['id']:r for r in ev['per_frame']};allnorm=[];raw=[]
            for p in pred['rows']:
                X,K,gt,diag,order=prepared[p['id']]
                candidates=[]
                for g in range(order):
                    Q=np.diag([-1.,1.,-1.]) if g else np.eye(3)
                    R=np.array(p['R'])@Q;t=np.array(p['t'])
                    cam=X@R.T+t
                    q=cam@K.T;uv=q[:,:2]/np.maximum(cam[:,2,None],1e-6)
                    err=np.linalg.norm(uv-gt,axis=1)
                    candidates.append((float(np.minimum(err/diag,1.).mean()),float(err.mean())))
                n,px=min(candidates,key=lambda x:x[0])
                if not p['pose_valid']:n=1.
                else:raw.append(px)
                allnorm.append(n)
                maxnorm=max(maxnorm,abs(n-saved[p['id']]['normalized']))
                if p['pose_valid']:maxpx=max(maxpx,abs(px-saved[p['id']]['raw_px']))
            full=np.array(allnorm+[1.]*7)
            assert abs(full.mean()-ev['primary_normalized_mean'])<1e-6
            summary[name]=dict(frames=512,raw_frames=len(raw),replayed_primary=float(full.mean()),
                saved_primary=ev['primary_normalized_mean'],raw_mean=float(np.mean(raw)))
    assert maxnorm<2e-6 and maxpx<.002
    gate=json.loads((run/'H_VS_P.json').read_text())
    catastrophic=[]
    for seed in (1,2,3):
        p=json.loads((run/'evaluations'/f'point_seed{seed}_synth_val.json').read_text())['per_frame']
        h=json.loads((run/'evaluations'/f'hough_seed{seed}_synth_val.json').read_text())['per_frame']
        assert [x['id'] for x in p]==[x['id'] for x in h]
        count=sum(a['pose_valid'] and a['raw_px']<=10 and (not b['pose_valid'] or b['raw_px']>50) for a,b in zip(p,h))
        assert count==gate['checks'][seed-1]['new_catastrophic_frames'];catastrophic.append(count)
    output=dict(PASS=True,scope='NumPy recomputation of stored pose predictions, not a second neural inference.',
        frames_per_seed=512,cells=9,max_normalized_delta=maxnorm,max_raw_frame_delta_px=maxpx,
        catastrophic_replayed=catastrophic,metrics=summary)
    (run/'NUMPY_REPLAY_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='metrics'},indent=2))


if __name__=='__main__':main()
