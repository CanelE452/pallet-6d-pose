import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.research.pallet_verified_anchor_v1.common import contract

ROOT=Path(__file__).resolve().parents[3]
NAME='pallet_011067_corner_contract_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
FIG=DOC/'figures'
FRAME='plastic_day_01:011067'
ANN=ROOT/'data/evaluation/pallet_eval_v1/final/positive/annotations/plastic_day_01/011067.json'
RGB=ROOT/'data/evaluation/pallet_eval_v1/final/positive/sessions/plastic_day_01/rgb/011067.png'
ANCHOR=ROOT/'_docs/experiments/pallet_verified_anchor_v1'
LR=[(0,1),(3,2),(4,5),(7,6)]
TB=[(0,3),(1,2),(4,7),(5,6)]
FR=[(0,4),(1,5),(2,6),(3,7)]

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bind(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==text,f'Immutable output changed: {path}'
    else:
        with path.open('x') as f:f.write(text)
def object_data():return read(ANN)['objects'][0]
def xyz(w,h,d):
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[0,0,0]],float)
    return signs*np.array([w,h,d])/2
def points():return np.asarray([p['xy'] for p in object_data()['keypoint_annotations']],float)
def direct_mask():return np.array([p['source']=='manual_click' and p['visibility']==2 and p['in_frame'] for p in object_data()['keypoint_annotations']])
def area(q):return float(abs(np.dot(q[:,0],np.roll(q[:,1],-1))-np.dot(q[:,1],np.roll(q[:,0],-1)))/2)
def figure_table(path,title,columns,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,max(2.8,1+len(rows)*.55)))
    ax.axis('off');ax.set_title(title,pad=20)
    t=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center')
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.7)
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)
