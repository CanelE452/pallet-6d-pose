"""V29 independent experts, original-disjoint router training, immutable parents."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_direction_utility_v27'))
import context27 as OLD27
from context27 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_adaptive_experts_v29'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_adaptive_experts_v29.md'
V6=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
STEPS=1200
ROUTER_STEPS=800
BATCH=16
EXPERTS=('point','guided','independent')
ROUTERS=('adaptive','no_state','point_router')
EVAL_SPLITS=('prospective','synth_val','real')

def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:
        f.write('\n\n### V29 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')

def save_prediction(split,c,name,m,points,**extra):
    folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True)
    detected=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=detected)
    for k,v in dict(points=points,**extra).items():
        z=v.detach().cpu().numpy() if torch.is_tensor(v) else v
        a=np.full((len(detected),)+z.shape[1:],np.nan,np.float32);a[detected]=z;data[k]=a
    np.savez_compressed(folder/(name+'.npz'),**data)
