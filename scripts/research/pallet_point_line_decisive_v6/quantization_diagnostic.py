"""Measure target-bin geometry; a peaky-target diagnostic, not a hard bound.

The neural decoder can interpolate locally, so this must not be called a
mathematical precision ceiling. No learned weights or selection are changed.
"""
from pathlib import Path
import json
import sys
import numpy as np
import torch
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT))
from cli_train import Split
from plpose_local.heads import lattice_lines,roi_lines_to_raw
from plpose_local.losses import line_targets
from plpose_local.geometry import EDGES


def main():
    torch.set_num_threads(2)
    run=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
    out=run/'QUANTIZATION_DIAGNOSTIC.json'
    if out.exists():raise SystemExit('diagnostic already exists')
    data=Split(ROOT/'data/pallet/results/pallet_point_line_v5_local/export/calibration.json')
    results={}
    with torch.inference_mode():
        for T,R in ((36,65),(180,257)):
            rows=[];lattice=lattice_lines(T,R)
            for first in range(0,len(data.records),16):
                idx=list(range(first,min(first+16,len(data.records))))
                O,S,_=data.batch(idx,'cpu',targets=True)
                target,ok=line_targets(S['points'],O['box'],T,R)
                line=roi_lines_to_raw(lattice[target][:,:,None],O['box'])[:,:,0]
                u,v=torch.tensor(EDGES).T;n=line[...,:2];c=line[...,2]
                norm=torch.linalg.vector_norm(n,dim=-1).clamp_min(1e-8)
                d=(((S['points'][:,u]*n).sum(-1)+c).abs()+((S['points'][:,v]*n).sum(-1)+c).abs())/(2*norm)
                for j,i in enumerate(idx):
                    for e in range(12):
                        rows.append(dict(id=data.records[i]['id'],edge=e,valid=bool(ok[j,e]),
                                         normal_error_px=float(d[j,e]),kind='height' if e>=8 else 'horizontal'))
            good=np.array([r['normal_error_px'] for r in rows if r['valid']])
            results[f'{T}x{R}']=dict(valid_edges=len(good),median_px=float(np.median(good)),
                mean_px=float(good.mean()),p90_px=float(np.quantile(good,.9)),per_edge=rows)
    out.write_text(json.dumps(dict(complete=True,split='calibration',frames=len(data.records),results=results,
        scope='Exact GT line snapped to its supervised theta/rho bin. Fine-grid result is a '
              'nonlearned reference, not an architecture search or a selected new setting. '
              'Local neural interpolation can beat these errors, so this is not a hard bound.'),indent=2)+'\n')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_edge'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
