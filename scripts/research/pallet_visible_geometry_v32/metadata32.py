"""Restore stock point reference for auxiliary catastrophe counts; predictions fixed."""
from context32 import *
from eval32 import SPLITS,conditions,inputs,render_data

def main():
    if (RUN/'METADATA_REFERENCE_FIX.json').exists():return
    rows=[]
    for split in SPLITS:
        for c in conditions(split):
            m,O,S=inputs(split,c,1);original=O.get('points')
            if original is None:original=render_data(1,'check',c)[1]['points']
            det=np.array([r['detected'] for r in m['rows']]);full=np.full((len(det),8,2),np.nan,np.float32);full[det]=original.numpy()
            for p in (RUN/'predictions'/split/c).glob('*.npz'):
                z=dict(np.load(p));before=__import__('hashlib').sha256(z['points'].tobytes()).hexdigest();z['original_points']=full;np.savez_compressed(p,**z);after=__import__('hashlib').sha256(np.load(p)['points'].tobytes()).hexdigest();assert before==after;rows.append(dict(path=str(p.relative_to(ROOT)),points_sha256=before))
    write(RUN/'METADATA_REFERENCE_FIX.json',dict(complete=True,reason='original_points metadata referenced V12 base rather than stock YOLO. Corrected from frozen inputs for consistent auxiliary new_corner_catastrophes; no output points/GT/checkpoints changed. Recompute summary and hashes afterward.',rows=rows))
if __name__=='__main__':main()
