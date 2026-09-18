"""Reuse original prediction-only PnP and group metrics; new roots only."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch,cv2
import cv_env as E
from pose import infer,metric,metadata,pose_auc

def main():
    torch.set_num_threads(1);cv2.setNumThreads(1);out={}
    for pop,old in [('SYNTH','SYNTH_HELDOUT'),('DEV','REAL_DEV'),('SQUARE','SQUARE_DEV')]:
        meta,truth=metadata(old);out[pop]={}
        arms=E.ARMS['B'] if pop=='SQUARE' else E.ARMS['A']+E.ARMS['B']
        for arm in arms:
            for seed in [1,2,3]:
                name=f'{arm}_seed{seed}';dst=E.RAW/f'pose_predictions/{pop}/{name}.json';mp=E.RAW/f'pose_metrics/{pop}/{name}.json'
                if not dst.exists():
                    rows=E.read(E.RAW/f'predictions/{pop}/{name}_PRIMARY.json')['records'];preds={}
                    for r in rows:
                        idx=r['selected_index'];points=None if idx is None else r['candidates'][idx]['keypoints_xy'];preds[r['id']]=infer(points,*meta[r['id']])
                    E.write(dst,dict(records=preds,GT_input=False,complete=True))
                preds=E.read(dst)['records'];assert set(preds)==set(truth)
                if not mp.exists():
                    with ProcessPoolExecutor(max_workers=4) as pool:metrics=list(pool.map(metric,[(fid,p,truth[fid]) for fid,p in preds.items()],chunksize=64))
                    E.write(mp,metrics)
                metrics=E.read(mp);valid=[r for r in metrics if r['available']]
                s=dict(frames=len(metrics),available=len(valid),coverage=len(valid)/len(metrics),ADDsym_AUC_full=pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in metrics],1.))
                for k in ['translation_cm','rotation_deg','yaw_deg','IoU3D']:s[k]=dict(median=float(np.median([r[k] for r in valid])) if valid else None,P90=float(np.quantile([r[k] for r in valid],.9)) if valid else None)
                out[pop][name]=s;print('CODE_POSE',pop,name,flush=True)
    E.write(E.DOC/'POSE_SECONDARY_RESULTS.json',dict(complete=True,summary=out,GT_input=False,solver='Unchanged DCP prediction-only W/D selector + SQPnP/RefineLM',
      group_metric='Same approved proper C1/C2/C4 rotations, corresponding-corner ADD',reference='Real/square reconstructed geometry, not independent physical pose ground truth',primary_decision_use=False))
if __name__=='__main__':main()
