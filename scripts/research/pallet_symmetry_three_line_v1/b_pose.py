"""Unchanged prediction-only PnP; centroid and explicit physical-frame symmetry metrics."""
import importlib,math
import numpy as np
import torch,cv2
import env as E
from preflight import rotations
from b_model import MODES

def infer(points,camera,xyz):
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    module=importlib.import_module('run_pose_evaluation')
    if points is None:return dict(available=False)
    points=np.asarray(points,float);valid=np.isfinite(points[:8]).all(-1)
    if valid.sum()<6:return dict(available=False)
    try:
        result=select_pnp_hypotheses(points,camera,dict(x=max(xyz[0],xyz[2]),y=xyz[1],z=min(xyz[0],xyz[2])),None)
        h=next((h for h in result.hypotheses if h.name==result.selected_hypothesis and h.success),None)
        if h is None:return dict(available=False)
        d=h.camera_facing_dimensions.as_dict();dims=np.array([d['width'],d['height'],d['depth']])
        model=module.cuboid(*dims);solved=module.solve(model,points[:8],camera,valid)
        if solved is None:return dict(available=False)
        R,t,residual=solved
        # Physical fixed X aligned to CF X if the selected across size is physical X.
        # Otherwise rotate the physical frame by +90 about height. No GT phase chooses this.
        Q=np.eye(3) if abs(dims[0]-xyz[0])<1e-6 else rotations(4)[1]
        return dict(available=True,R_cf=R.tolist(),R_physical=(R@Q).tolist(),
          centroid=(R@model.mean(0)+t).tolist(),cf_extents=dims.tolist(),reprojection_px=residual,
          selected_hypothesis=result.selected_hypothesis,physical_phase_rule='identity if CF across equals fixed physical X, else +Y90; no GT orbit selection')
    except (cv2.error,ValueError):return dict(available=False)

def error(p,truth):
    if not p['available']:return dict(available=False)
    R=np.array(p['R_physical']);gt=np.array(truth['R_physical']);errors=[]
    for Q in rotations(truth['order']):
        angle=np.arccos(np.clip((np.trace(R.T@gt@Q)-1)/2,-1,1));errors.append(float(np.degrees(angle)))
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    iou=oriented_iou_3d(np.array(p['R_cf']),np.array(p['centroid']),p['cf_extents'],np.array(truth['R_cf']),np.array(truth['centroid']),truth['cf_extents'])
    return dict(available=True,centroid_translation_cm=float(np.linalg.norm(np.array(p['centroid'])-truth['centroid'])*100),
      equivalent_rotation_deg=min(errors),body_extent_IoU=float(iou),symmetry_order=truth['order'])

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);pe=E.C.old('paper_evaluation')
    geometry=dict(np.load(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'));index={str(v):i for i,v in enumerate(geometry['stems'])}
    manifests=E.read(E.EXPORT/'synth_val.json')['records'];metadata={};truth={}
    for row in manifests:
        fid=row['frame_id'];i=index[fid];fx,fy,cx,cy=geometry['K'][i];K=np.array([[fx,0,cx-geometry['pad'][i]],[0,fy,cy-geometry['pad'][i]],[0,0,1]])
        metadata[fid]=(K,geometry['dims'][i])
        X=geometry['Xcf'][i];Q=np.stack([X[1]-X[0],X[3]-X[0],X[4]-X[0]],axis=1);dims=np.linalg.norm(Q,axis=0);Q/=dims
        assert abs(np.linalg.det(Q)-1)<1e-8
        truth[fid]=dict(R_physical=geometry['R'][i],R_cf=geometry['R'][i]@Q,centroid=geometry['t'][i],cf_extents=dims,order=row['symmetry_order'])
    # DEV metadata has registered dimensions/intrinsics; GT pose is used below only in error().
    import pose_evaluation_paths as paths
    contract=paths.load_pose_object_contract(str(E.C.POSE/'POSE_EVAL_OBJECT_CONTRACT.json'))
    devlist=E.read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list'];oldtruth=E.read(E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    devmeta={};devtruth={}
    for frame in devlist:
        k=pe.canonical_key(frame['image']);a=E.read(E.ROOT/frame['annotation'])['camera_data']['intrinsics'];s=paths.object_spec(contract,frame['object_type'])
        devmeta[k]=(np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float),np.array([s['long_m'],s['height_m'],s['short_m']]))
        t=oldtruth[frame['frame_id']];Q=np.eye(3) if t['physical_long_axis']=='CF_WIDTH' else rotations(4)[1];d=t['physical_dimensions_m']
        devtruth[k]=dict(R_physical=np.array(t['R_gt_representative'])@Q,R_cf=np.array(t['R_gt_representative']),centroid=np.array(t['t_gt']),cf_extents=[d['across'],d['height'],d['along']],order=2)
    summary={};allrows={}
    for split in ['synth_val','DEV']:
        summary[split]={};allrows[split]={}
        for arm in ['B0_P',*MODES,'B4_SHUFFLE_DIAG']:
            for seed in [1,2,3]:
                name=f'{arm}_seed{seed}';dst=E.RAW/f'B/pose_predictions/{split}/{name}.json'
                if dst.exists():preds=E.read(dst)['records']
                else:
                    if split=='synth_val':
                        p=torch.load(E.RAW/f'B/synth_val_predictions/{name}.pt',map_location='cpu',weights_only=False)
                        preds={fid:infer(points,*metadata[fid]) for fid,points in zip(p['ids'],p['points'])}
                    else:
                        frames=E.read(E.RAW/f'B/DEV_predictions/{name}.json')['frames'];preds={}
                        for key in devmeta:
                            c=max(frames[key],key=lambda c:c['score']) if frames[key] else None
                            preds[key]=infer(None if c is None else c['keypoints_xy'],*devmeta[key])
                    E.write(dst,dict(records=preds,GT_pose_input=False,solver='unchanged prediction-only axis selector + SQPnP + RefineLM'))
                gt=truth if split=='synth_val' else devtruth
                rows=[dict(id=k,**error(p,gt[k])) for k,p in preds.items()];available=[r for r in rows if r['available']]
                summary[split][name]=dict(denominator=len(rows),available=len(available),coverage=len(available)/len(rows),
                  **{f:dict(median=float(np.median([r[f] for r in available])),P90=float(np.quantile([r[f] for r in available],.9))) for f in ['centroid_translation_cm','equivalent_rotation_deg','body_extent_IoU']})
                allrows[split][name]=rows;print('POSE',split,name,len(available),flush=True)
    E.write(E.DOC/'B/POSE_SECONDARY.json',dict(complete=True,summary=summary,solver_changed=False,
      coordinate_origin='Existing canonical cuboid and source geometry use centroid origin: R*X_centroid+t equals t.',
      physical_frame='Prediction-only dimension correspondence maps CF rotation to fixed physical X/Y/Z before global C1/C2 SO3 distance. DEV physical X is registered long axis; source physical XYZ is fixed renderer metadata.',
      phase_limit='Camera-facing R0 labels do not guarantee recovery of a non-symmetric asset physical front. For source C1, the deterministic physical phase can have large error; no GT phase is supplied to inference.',
      body_extent='Predicted selected CF extents used for predicted cuboid, GT CF extents for reference; does not reuse GT extent as predicted extent.',
      GT_quality='DEV geometry-reconstructed pose reference, not independent physical metrology.',
      legacy_metric_warning='New explicit physical-frame orientation/body-extent metrics are not retroactive replacements for old paper pose tables.'))
    E.write(E.RAW/'B/POSE_FULL_METRICS.json',allrows)
if __name__=='__main__':main()
