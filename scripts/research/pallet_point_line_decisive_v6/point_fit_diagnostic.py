"""Separate train-fit errors in fixed-role pose from unordered geometry."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import torch
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT))
from diagnostics import load_model,Split,run_solver,metrics,G
from stock_geometry_baseline import set_error


def main():
    torch.set_num_threads(2)
    run=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
    out=run/'POINT_GEOMETRY_FIT_DIAGNOSTIC.json'
    if out.exists():raise SystemExit('fit diagnostic already exists')
    cells={}
    with torch.inference_mode():
        for seed in (1,2,3):
            model,_=load_model(run/'cache_heads'/f'point_seed{seed}/checkpoint_final.pt','cpu')
            cells[str(seed)]={}
            for split in ('train','calibration'):
                data=Split(ROOT/f'data/pallet/results/pallet_point_line_v5_local/export/{split}.json')
                ids=sorted(range(len(data.records)),key=lambda i:hashlib.sha256(('fit-v6:'+data.records[i]['id']).encode()).digest())[:128]
                rows=[]
                for start in range(0,len(ids),8):
                    ii=ids[start:start+8];O,_,orders=data.batch(ii,'cpu',targets=False);o=model(O)
                    _,sup,_=data.batch(ii,'cpu');pred=run_solver(O,o,None,None);met=metrics(O,sup,orders,pred)
                    uv=G.project(O['K'],pred['R'],pred['t'],G.corners(O['dims']))[0]
                    for j,i in enumerate(ii):
                        pose_error,_=set_error(uv[j,:8].numpy(),sup['points'][j,:8].numpy())
                        obs_error,_=set_error(o['points'][j,:8].numpy(),sup['points'][j,:8].numpy())
                        rows.append(dict(id=data.records[i]['id'],pose_valid=bool(pred['valid'][j]),
                            role_aware_raw_px=float(met['raw_px'][j]),
                            unordered_pose_raw_px=float(pose_error.mean()),
                            unordered_point_observation_raw_px=float(obs_error.mean())))
                cells[str(seed)][split]=dict(n=len(rows),rows=rows,means={
                    k:float(np.mean([r[k] for r in rows])) for k in
                    ('role_aware_raw_px','unordered_pose_raw_px','unordered_point_observation_raw_px')})
    out.write_text(json.dumps(dict(complete=True,cells=cells,
        selection='First128 by SHA256(fit-v6:+id) separately in train and calibration; no new updates.',
        scope='Same fixed point head; role-aware pose metric versus optimal scoring-only corner '
              'assignment. Large train errors may reflect underfitting, representation, or role '
              'ambiguity; this experiment does not identify a unique cause.'),indent=2)+'\n')
    print(json.dumps({s:{k:v['means'] for k,v in c.items()} for s,c in cells.items()},indent=2))


if __name__=='__main__':main()
