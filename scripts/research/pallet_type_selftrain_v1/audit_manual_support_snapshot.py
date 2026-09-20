"""Snapshot newly saved manual supervision; no training or evaluation edits."""
import json
from pathlib import Path
import numpy as np
from . import common as C
from .pseudo import top

DOC=C.DOC/'large_corner_recovery_v1/manual_support_snapshot_v1'
WORKSPACE=C.OUT/'large_corner_recovery_v1/real_support_annotations/WORKSPACE.json'


def main():
    workspace=C.read(WORKSPACE)
    saved=[r for r in workspace['rows'] if (C.ROOT/r['annotation']).exists()]
    rows=[]
    for row in saved:
        path=C.ROOT/row['annotation'];annotation_binding=C.bound(path);doc=C.read(path)
        obj=doc['objects'][0];entries=obj['keypoint_annotations'];h=doc['camera_data']['height'];w=doc['camera_data']['width']
        assert obj['split']==row['split'] and row['role']=='proposed_support'
        C.verify(row['image'])
        valid=np.array([i<8 and p.get('source')=='manual_click' and p.get('visibility')==2
            and p.get('in_frame') is True for i,p in enumerate(entries)])
        gt=np.array([p['xy'] if p.get('xy') is not None else [0,0] for p in entries],float)
        assert np.isfinite(gt[valid]).all() and ((gt[valid]>=0)&(gt[valid]<[w,h])).all()
        pred_path=C.RAW/'pseudo_frames'/(row['id']+'.json');frame=C.read(pred_path)
        assert frame['image']==row['image']
        results={}
        # C2 is used only for this annotation-based diagnostic, never for inference.
        perms=[list(range(9)),[5,4,7,6,1,0,3,2,8]]
        for name in ['raw','refined']:
            if frame.get(name) is None or top(frame[name]) is None:
                results[name]=dict(available=False);continue
            q=np.array(top(frame[name])['keypoints_xy'],float)
            e=[np.linalg.norm(q[p][valid]-gt[valid],axis=1) for p in perms]
            branch=int(np.argmin([x.mean() for x in e]))
            results[name]=dict(available=True,diagnostic_C2_branch=branch,
                native_errors_px=np.linalg.norm(q[valid]-gt[valid],axis=1).tolist(),
                errors_px=e[branch].tolist(),mean_px=float(e[branch].mean()),
                over20=int((e[branch]>20).sum()),within10=int((e[branch]<=10).sum()))
        a=np.array(results['raw']['errors_px']);b=np.array(results['refined']['errors_px'])
        rows.append(dict(id=row['id'],annotation=annotation_binding,image=row['image'],
            prediction=C.bound(pred_path),manual_corner_indices=np.flatnonzero(valid).tolist(),
            unconfirmed_signed_axis=obj.get('pose_status'),old_prediction_results=results,
            hard_recovered_to10=int(((a>20)&(b<=10)).sum()),hard_total=int((a>20).sum()),
            warning='Single manually annotated support image, not heldout evaluation; stored frozen Replay prediction predates this annotation. No retraining and no population-level performance claim.'))
        C.verify(annotation_binding)
    assert len(rows)==1,'This snapshot is specifically the first manually saved image; use a new snapshot for additional labels'
    result=dict(complete=True,saved_images=1,manual_visible_corners=sum(len(r['manual_corner_indices']) for r in rows),
        validation_images=sum((C.ROOT/r['annotation']).exists() for r in workspace['rows'] if r['role']=='proposed_validation'),
        training_performed=False,evaluation_exclusions_applied=False,goal_complete=False,
        sources=[C.bound(__file__),C.bound(WORKSPACE)],records=rows)
    C.freeze(DOC/'AUDIT.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
