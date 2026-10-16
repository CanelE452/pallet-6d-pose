from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_quality_v23'))
import context23 as PREV
from context23 import *
from train23 import training
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_identifiability_v24'
ARMS=('unit_sign','unit_sign_local','unit_sign_parent')
TINY_ARMS=('unit_tiny','unit_tiny_local')
def save24(split,c,name,m,O,out):
    d=RUN/'predictions'/split/c;d.mkdir(parents=True,exist_ok=True);det=np.array([x['detected'] for x in m['rows']]);data=dict(ids=np.array([x['id'] for x in m['rows']]),detected=det)
    for k,v in dict(original_points=O['points'],**{k:out[k] for k in ('points','base','proposal_delta','alpha','coverage')}).items():
        z=v.detach().cpu().numpy();a=np.full((len(det),)+z.shape[1:],np.nan,np.float32);a[det]=z;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)
