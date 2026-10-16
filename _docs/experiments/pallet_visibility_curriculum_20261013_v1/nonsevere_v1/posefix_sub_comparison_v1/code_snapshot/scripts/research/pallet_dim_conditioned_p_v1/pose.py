"""Unchanged prediction-only selector/SQPnP/LM; declared proper-group metrics."""
import argparse,sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch,cv2
import dcp_env as E
sys.path.append(str(E.ROOT/'scripts/paper/pose_metric_closure_v1'))
from run_pose_evaluation import cuboid,solve
from symmetry_aware_pose_metrics import pose_auc
from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d

def rotations(n):
    assert n in [1,2,4]
    return [np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]]) for t in np.arange(n)*2*np.pi/n]

def infer(points,K,xyz,source=False):
    if points is None:return dict(available=False)
    p=np.array(points,float);valid=np.isfinite(p[:8]).all(-1)
    if valid.sum()<6:return dict(available=False)
    try:
        if abs(xyz[0]-xyz[2])<1e-9:
            dims=np.array(xyz);name='SQUARE_IDENTICAL_WD'
        else:
            result=select_pnp_hypotheses(p,K,dict(x=max(xyz[0],xyz[2]),y=xyz[1],z=min(xyz[0],xyz[2])),None)
            h=next((h for h in result.hypotheses if h.name==result.selected_hypothesis and h.success),None)
            if h is None:return dict(available=False)
            d=h.camera_facing_dimensions.as_dict();dims=np.array([d['width'],d['height'],d['depth']]);name=h.name
        solved=solve(cuboid(*dims),p[:8],K,valid)
        if solved is None:return dict(available=False)
        R,t,residual=solved
        if t[2]<=0 or not np.isfinite(R).all() or not np.isfinite(t).all():return dict(available=False)
        Q=np.eye(3) if abs(dims[0]-xyz[0])<1e-6 else rotations(4)[1]
        physical=R@Q
        if source:physical=physical@np.diag([1.,-1.,-1.])
        return dict(available=True,R_cf=R.tolist(),R_physical=physical.tolist(),centroid=t.tolist(),cf_extents=dims.tolist(),selected_hypothesis=name,reprojection_px=float(residual))
    except (cv2.error,ValueError):return dict(available=False)

def metric(task):
    fid,p,g=task
    if not p['available']:return dict(id=fid,available=False)
    R=np.array(p['R_physical']);G=np.array(g['R']);t=np.array(p['centroid']);gt=np.array(g['t']);xyz=np.array(g['xyz']);x=cuboid(*xyz)
    rotation=[];yaw=[];add=[]
    for Q in rotations(g['order']):
        target=G@Q;rel=target.T@R
        rotation.append(float(np.degrees(np.arccos(np.clip((np.trace(rel)-1)/2,-1,1)))))
        yaw.append(abs(float((np.degrees(np.arctan2(rel[0,2],rel[2,2]))+180)%360-180)))
        add.append(float(np.linalg.norm((R@x.T).T+t-((target@x.T).T+gt),axis=1).mean()))
    diameter=np.linalg.norm(xyz);a=min(add)
    iou=oriented_iou_3d(np.array(p['R_cf']),t,p['cf_extents'],np.array(g['body_R']),gt,g['body_xyz'])
    return dict(id=fid,available=True,translation_cm=float(np.linalg.norm(t-gt)*100),rotation_deg=min(rotation),yaw_deg=min(yaw),IoU3D=float(iou),ADDsym_m=a,ADDsym_normalized=a/diameter)

def metadata(split):
    meta={};truth={}
    if split=='SYNTH_HELDOUT':
        g=dict(np.load(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'));ix={str(s):i for i,s in enumerate(g['stems'])}
        side={r['frame_id']:r for r in E.read(E.RAW/'DIMENSION_SIDECAR.json')['records']}
        for r in E.read(E.LINE/'SOURCE_MANIFEST.json')['records']:
            if r['partition']!='heldout':continue
            fid=r['id'];i=ix[fid];fx,fy,cx,cy=g['K'][i];K=np.array([[fx,0,cx-g['pad'][i]],[0,fy,cy-g['pad'][i]],[0,0,1]])
            meta[fid]=(K,g['dims'][i],True);truth[fid]=dict(R=g['R'][i],t=g['t'][i],xyz=g['dims'][i],body_R=g['R'][i],body_xyz=g['dims'][i],order=side[fid]['symmetry_order'])
    elif split=='REAL_DEV':
        pe=E.old('paper_evaluation');ids={pe.canonical_key(item.image):item.frame_id for item in pe.population().positive.items};oldgt=E.read(E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
        from inference import registry_input
        for r in E.read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']:
            fid=ids[pe.canonical_key(r['image'])];raw=E.read(E.ROOT/r['annotation'])['camera_data']['intrinsics'];K=np.array([[raw['fx'],0,raw['cx']],[0,raw['fy'],raw['cy']],[0,0,1]])
            dims,order=registry_input(r['object_type']);xyz=dims[[0,2,1]];meta[fid]=(K,xyz,False)
            g=oldgt[r['frame_id']];d=g['physical_dimensions_m'];cf=np.array([d['across'],d['height'],d['along']]);Rcf=np.array(g['R_gt_representative'])
            # Deterministic physical registry-X correspondence, no per-prediction GT branch.
            Q=np.eye(3) if abs(cf[0]-xyz[0])<1e-6 else rotations(4)[1]
            truth[fid]=dict(R=Rcf@Q,t=np.array(g['t_gt']),xyz=xyz,body_R=Rcf,body_xyz=cf,order=order)
    else:
        from square_data import square_metadata
        meta,truth=square_metadata()
    return meta,truth

def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['paper','square','mixed'],default='paper');args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(1)
    arms=['OLD_P',*E.ARMS] if args.track=='paper' else E.SQUARE_ARMS if args.track=='square' else E.MIXED_ARMS
    splits=['SYNTH_HELDOUT','REAL_DEV'] if args.track=='paper' else ['SQUARE_DEV'] if args.track=='square' else ['SYNTH_HELDOUT','REAL_DEV','SQUARE_DEV'];allrows={};summaries={}
    for split in splits:
        meta,gt=metadata(split);allrows[split]={};summaries[split]={}
        for arm in arms:
            for seed in [1,2,3]:
                name=f'{arm}_seed{seed}';dst=E.RAW/f'pose_predictions/{split}/{name}.json'
                if dst.exists():preds=E.read(dst)['records']
                else:
                    rows=E.read(E.RAW/f'predictions/{split}/{name}.json')['records'];preds={}
                    for r in rows:
                        idx=r['selected_index'];points=None if idx is None else r['candidates'][idx]['keypoints_xy'];preds[r['id']]=infer(points,*meta[r['id']])
                    E.write(dst,dict(records=preds,GT_input=False))
                assert set(preds)==set(gt)
                with ProcessPoolExecutor(max_workers=4) as pool:metrics=list(pool.map(metric,[(fid,p,gt[fid]) for fid,p in preds.items()],chunksize=64))
                available=[r for r in metrics if r['available']];a=[r['ADDsym_normalized'] if r['available'] else float('inf') for r in metrics]
                s=dict(frames=len(metrics),available=len(available),coverage=len(available)/len(metrics),ADDsym_AUC_full=pose_auc(a,1.0),
                  ADDsym_AUC_conditional=pose_auc([r['ADDsym_normalized'] for r in available],1.0) if available else None)
                for k in ['translation_cm','rotation_deg','yaw_deg','IoU3D']:s[k]=dict(median=float(np.median([r[k] for r in available])) if available else None,P90=float(np.quantile([r[k] for r in available],.9)) if available else None)
                allrows[split][name]=metrics;summaries[split][name]=s;print('POSE',split,name,s['coverage'],flush=True)
    E.write(E.RAW/f'{args.track.upper()}_POSE_METRICS.json',allrows);E.write(E.DOC/f'{args.track.upper()}_POSE_RESULTS.json',dict(complete=True,summary=summaries,
      solver='unchanged prediction-only W/D selector + SQPnP/RefineLM; identical square W/D hypotheses merged as prior audited contract',
      source_basis='fixed Rx(pi) Y-up transform from previous correction, applied before any current results',
      ADD='minimum whole-object proper-group corresponding 8-corner ADD; AUC 1001 points, max0.1*object diameter, no unrestricted nearest neighbour',
      reference='real DEV/square geometry reconstructed, NOT independent physical GT; source C1 physical front ambiguity retained',
      no_GT_match_gate_for_PnP=True))
if __name__=='__main__':main()
