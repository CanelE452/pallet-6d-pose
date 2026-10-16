"""Reproduce the privileged, pre-training line-headroom diagnostic."""
from pathlib import Path
import sys
import json
import torch
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT))
from diagnostics import load_model,Split,choose_oracle_symmetry,lines_from_points,run_solver,metrics,stats,G,sha


def main():
    torch.set_num_threads(2)
    run=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
    output=run/'ORACLE_PREFLIGHT.json'
    if output.exists():raise SystemExit('oracle preflight already exists; preserve the registered receipt')
    data=Split(ROOT/'data/pallet/results/pallet_point_line_v5_local/export/calibration.json')
    checkpoint=ROOT/'data/pallet/results/pallet_point_line_v5_local/cache_heads/point_seed1/checkpoint_final.pt'
    model,_=load_model(checkpoint,'cpu');rows={}
    with torch.inference_mode():
        for first in range(0,len(data.records),8):
            ii=list(range(first,min(first+8,len(data.records))))
            O,_,orders=data.batch(ii,'cpu',targets=False);o=model(O)
            _,sup,_=data.batch(ii,'cpu',targets=True)
            uv,Rgt,_=choose_oracle_symmetry(O,sup,orders,o['points'])
            lines,_=lines_from_points(uv)
            nearR=(G.exp_so3(torch.tensor([[.03,-.04,.02]])).expand(len(ii),3,3)@Rgt)[:,None]
            nearT=(sup['t']+torch.tensor([[.01,-.01,.02]]))[:,None]
            for init in ('actual','near_gt'):
                for source in ('none','oracle'):
                    pred=run_solver(O,o,lines if source=='oracle' else None,
                        torch.ones_like(lines[...,0]) if source=='oracle' else None,
                        R0=nearR if init=='near_gt' else None,t0=nearT if init=='near_gt' else None)
                    met=metrics(O,sup,orders,pred)
                    for j,i in enumerate(ii):
                        rows.setdefault(f'{init}/{source}',[]).append(dict(id=data.records[i]['id'],
                            normalized=float(met['normalized'][j]),
                            raw_px=float(met['raw_px'][j]) if bool(met['valid'][j]) else None,
                            valid=bool(met['valid'][j]),oracle_best_start=float(met['oracle_best_start'][j])))
    summary={k:stats(v) for k,v in rows.items()}
    output.write_text(json.dumps(dict(complete=True,checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),summary=summary,per_frame=rows,
        scope='Privileged diagnostic, synthetic calibration, frozen existing point seed1. '
              'Not a deployable prediction.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
