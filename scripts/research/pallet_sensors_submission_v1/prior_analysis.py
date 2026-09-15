"""Full-precision same-population prior contrasts; missing support is explicit."""
import numpy as np
from env import *
def run():
    analysis=import_old('analysis_panel');analysis.RAW=OLD_RAW
    stores,poses,summary,sources=analysis.load(True)
    assert summary==read(OLD_DOC/'UNIFIED_DEV_RESULTS.json')['methods'],'Historical numbers changed'
    pe=old('paper_evaluation');ag=old('aggregate_results');pop=pe.population();meta={r['frame_id']:r for r in read(pe.POS)['items']}
    baseline=read(LINE/'baseline/FULL_CANDIDATES.json')['frames']
    imageids={pe.canonical_key(i.image):i.frame_id for i in pop.positive.items}
    poseids={r['frame_id']:imageids[pe.canonical_key(r['image'])] for r in read(C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']}
    assert len(poseids)==len(set(poseids.values()))==319
    for seed in (1,2,3):
        for suffix in ('','_raw'):
            name=f'PRIOR{seed}{suffix}';p=RAW/f'evaluation/{name}';frames=read(p/'PREDICTIONS.json')['frames'];rows=[];den=0;nonfinite=0
            for item in pop.positive.items:
                target=pe.E._legacy_forbidden_target(item);cand=frames[pe.canonical_key(item.image)];ref=baseline[pe.canonical_key(item.image)]
                assert len(cand)==len(ref)
                for c,r in zip(cand,ref):
                    assert c['score']==r['score'] and c['box_xyxy']==r['box_xyxy']
                    if c['keypoints_xy'] is not None:assert c['keypoints_xy'][8]==r['keypoints_xy'][8]
                top=max(cand,key=lambda x:x['score']) if cand else None;mask=target.keypoint_supervision_mask;den+=int(mask.sum())
                matched=top is not None and top['keypoints_xy'] is not None and pe.E._box_iou(np.asarray(top['box_xyxy']),target.box_xyxy)>=.5
                err=np.linalg.norm(np.asarray(top['keypoints_xy'])-target.keypoints_xy,axis=-1) if matched else np.full(9,np.inf)
                nonfinite+=int((~np.isfinite(err[mask])).sum()) if matched else 0
                e=err[mask] if matched else np.empty(0);e8=err[:8][mask[:8]] if matched else np.empty(0)
                rows.append(dict(frame_id=item.frame_id,session_id=meta[item.frame_id]['session_id'],errors=e,errors8=e8,gt=int(mask.sum())))
            stores[name]=rows;full=np.concatenate([r['errors'] for r in rows]);finite=full[np.isfinite(full)];e8=np.concatenate([r['errors8'] for r in rows]);e8=e8[np.isfinite(e8)]
            pose=read(p/'POSE_PER_FRAME_BY_ARM.json')['per_frame'][name];mapped=[dict(r,frame_id=poseids[r['frame_id']],session_id=meta[poseids[r['frame_id']]]['session_id']) for r in pose];poses[name]={r['frame_id']:r for r in mapped}
            ps=ag.pose_summary(mapped);ps.update(yaw_median_deg=float(np.median([r['yaw_error_deg'] for r in mapped])),coverage=len(mapped)/319,excluded_frame_ids=sorted(set(meta)-set(poses[name])))
            summary[name]=dict(median_px=float(np.median(finite)) if len(finite) else None,p90_px=float(np.quantile(finite,.9)) if len(finite) else None,frame_mean_px=float(np.mean([r['errors'].mean() for r in rows if len(r['errors'])])) if not nonfinite else None,gross20=float((full>20).mean()) if len(full) else None,corner8_median_px=float(np.median(e8)) if len(e8) else None,matched_frames=sum(bool(len(r['errors'])) for r in rows),supervised_points=len(full),finite_supervised_points=len(finite),nonfinite_points=nonfinite,gt_denominator=den,ALL_GT_PCK={str(t):float((full<=t).sum()/den) for t in (5,10,20)},pose=ps)
            sources[name]=[bound(p/f) for f in ('PREDICTIONS.json','PAPER_2D.json','POSE_PER_FRAME_BY_ARM.json')]
    support=[[len(r['errors']) for r in stores[name]] for name in ['R0','P1','P2','P3','PRIOR1','PRIOR2','PRIOR3']]
    same=all(s==support[0] for s in support) and all(summary[f'PRIOR{s}']['nonfinite_points']==0 for s in (1,2,3))
    paired={level:analysis.contrast(stores,'P','PRIOR',level) for level in ('session','frame')} if same else dict(status='SUPPORT_MISMATCH_NO_PRIMARY_PAIRED_PRECISION',policy='Report full-GT PCK and conditional error separately; do not use a favorable intersection')
    paired.update(role='DEV_EXPLORATORY',multiplicity='No correction; secondary contrasts, not confirmatory superiority tests',common_support_exact=same)
    write(DOC/'UNIFIED_DEV_RESULTS.json',dict(complete=True,methods=summary,sources=sources,role='REUSED_DEV',prior_common_support=same,historical_result_recomputed_exact=True))
    write(DOC/'P_VS_PRIOR_PAIRED.json',paired)
    write(RAW/'DEV_FULL_PRECISION_ERRORS.json',{name:[dict(r,errors=r['errors'].tolist(),errors8=r['errors8'].tolist()) for r in rows] for name,rows in stores.items()})
    print('PRIOR_PAIRED',paired,flush=True)
