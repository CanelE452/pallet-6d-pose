"""Post-v7 diagnostic: can constant line downweighting preserve strong points?

Oracle stock correspondence remains privileged. No weight is selected/deployed.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from run import RUN, V6, MANIFEST, Split, sha, write, pnp, load_model, G, run_solver, metrics, choose_oracle_symmetry, summarize, set_error


def main():
    torch.set_num_threads(2)
    protocol=dict(hypothesis='A constant smaller line weight alone might recover useful fusion of strong stock points. '
        'If positive weights consistently worsen results, simple scale tuning is insufficient in this fixed diagnostic.',
        timing='Adaptive follow-up after all 18 intervention cases finished; reused calibration, not holdout.',
        weights=[0.,.001,.01,.1,1.],iters=5,extra_case=dict(weight=1.,iters=25),
        scope='All cells use GT-assisted stock corner correspondence. No selected architecture or real-performance claim.',
        source_sha256=sha(Path(__file__)),parent_protocol_sha256=sha(RUN/'PROTOCOL.json'))
    out=RUN/'WEIGHT_PROBE.json'
    if out.exists():raise SystemExit('already exists')
    write(RUN/'WEIGHT_PROTOCOL.json',protocol)
    data=Split(MANIFEST);stocks=__import__('json').loads((RUN/'STOCK_PREDICTIONS.json').read_text())['rows']
    allcells={};started=time.monotonic()
    with torch.inference_mode():
        for seed in (1,2,3):
            model,_=load_model(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt','cpu')
            cases=[(w,5) for w in protocol['weights']]+[(1.,25)]
            cells={f'{w}/{it}':[] for w,it in cases}
            for first in range(0,len(data.records),8):
                ids=list(range(first,min(first+8,len(data.records))))
                O,_,orders=data.batch(ids,'cpu',targets=False);o=model(O)
                _,sup,_=data.batch(ids,'cpu',targets=True);gt,_,_=choose_oracle_symmetry(O,sup,orders,o['points'])
                stock=torch.zeros_like(gt)
                for j,i in enumerate(ids):
                    pp=np.array(stocks[i]['points'])
                    u,v=linear_sum_assignment(np.linalg.norm(pp[:,None]-gt[j,:8].numpy()[None],axis=-1))
                    stock[j,v]=torch.as_tensor(pp[u],dtype=gt.dtype)
                stock[:,8]=stock[:,:8].mean(1)
                rr,tt,ok=pnp(O,stock,o);assert all(ok)
                for w,it in cases:
                    obs=dict(O,line_mask=O['line_mask'].float()*w)
                    pred=run_solver(obs,dict(o,points=stock),o['lines'] if w else None,
                                    o['mass'] if w else None,R0=rr,t0=tt,iters=it)
                    met=metrics(O,sup,orders,pred)
                    uv=G.project(O['K'],pred['R'],pred['t'],G.corners(O['dims']))[0]
                    for j,i in enumerate(ids):
                        valid=bool(met['valid'][j]);ge,_=set_error(uv[j,:8].numpy(),sup['points'][j,:8].numpy())
                        cells[f'{w}/{it}'].append(dict(id=data.records[i]['id'],valid=valid,
                            normalized=float(met['normalized'][j]),raw_px=float(met['raw_px'][j]) if valid else None,
                            unordered_px=float(ge.mean()) if valid else None,projected_points=uv[j,:8].tolist()))
            allcells[str(seed)]=dict(summary={k:summarize(v) for k,v in cells.items()},cells=cells)
            print('weight seed',seed,{k:round(v['raw_mean'],5) for k,v in allcells[str(seed)]['summary'].items()},flush=True)
    write(out,dict(complete=True,cells=allcells,protocol_sha256=sha(RUN/'WEIGHT_PROTOCOL.json'),elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
