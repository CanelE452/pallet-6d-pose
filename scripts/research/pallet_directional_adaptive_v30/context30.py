"""State-free directional gain learning with preserved strong point parents."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_adaptive_experts_v29'))
import context29 as V29
from context29 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_directional_adaptive_v30'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_directional_adaptive_v30.md'
ARMS=('point_gain','whole_line','normal_no_evidence','normal_gain')
STEPS=2000
BATCH=16

def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V30 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')

def save(split,c,name,m,O,out):
    folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(original_points=O['points'],**out).items():
        z=v.detach().cpu().numpy();a=np.full((len(det),)+z.shape[1:],np.nan,np.float32);a[det]=z;data[k]=a
    np.savez_compressed(folder/(name+'.npz'),**data)
