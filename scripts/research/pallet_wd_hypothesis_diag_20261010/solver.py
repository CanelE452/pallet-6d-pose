"""Prediction-only fitting/selection. No files, references or human labels."""
import copy
import cv2
import numpy as np

_POSE=None
NAMES=('long-face-front','short-face-front')

def configure(pose):
    global _POSE
    _POSE=pose

def dimensions(xyz, name):
    x,y,z=np.asarray(xyz,float)
    return np.array([max(x,z),y,min(x,z)] if name==NAMES[0] else [min(x,z),y,max(x,z)])

def actual_from_fit(R,t,residual,dims,xyz,name,source=False):
    Q=np.eye(3) if abs(dims[0]-xyz[0])<1e-6 else _POSE.rotations(4)[1]
    physical=R@Q
    if source:physical=physical@np.diag([1.,-1.,-1.])
    available=bool(t[2]>0 and np.isfinite(R).all() and np.isfinite(t).all())
    if not available:return dict(available=False)
    return dict(available=True,R_cf=R.tolist(),R_physical=physical.tolist(),centroid=t.tolist(),
                cf_extents=dims.tolist(),selected_hypothesis=name,reprojection_px=float(residual))

def project_pose(actual,K):
    if not actual.get('available'):return None
    x=np.vstack([_POSE.cuboid(*actual['cf_extents']),np.zeros((1,3))])
    R=np.asarray(actual['R_cf']);t=np.asarray(actual['centroid'])
    cam=x@R.T+t
    projected=cam@np.asarray(K,float).T
    if not np.isfinite(projected).all() or np.any(projected[:,2]==0):return None
    return projected[:,:2]/projected[:,2:]

def fit(q,K,xyz,name,subset_indices,source=False):
    assert _POSE is not None, 'configure() with the frozen pose module first'
    q=np.asarray(q,float);K=np.asarray(K,float);xyz=np.asarray(xyz,float)
    dims=dimensions(xyz,name);mask=np.zeros(8,bool);mask[list(subset_indices)]=True
    finite=np.isfinite(q[:8]).all(-1)
    if np.sum(mask&finite)!=len(subset_indices) or len(subset_indices)<6:
        return dict(success=False,actual_pose=dict(available=False),failure='INVALID_OR_TOO_FEW_SUBSET_POINTS',
                    subset_indices=list(subset_indices),reprojection_rmse_9_px=None,subset_center_rmse_px=None)
    try:
        solved=_POSE.solve(_POSE.cuboid(*dims),q[:8],K,mask)
        if solved is None:raise ValueError('SQPNP_FAILED')
        R,t,residual=solved
        actual=actual_from_fit(R,t,residual,dims,xyz,name,source)
        projected=project_pose(actual,K)
        idx=list(subset_indices)+[8]
        score=None if projected is None or not np.isfinite(q[idx]).all() else float(np.sqrt(np.mean(np.sum((projected[idx]-q[idx])**2,axis=1))))
        rmse=None if projected is None or not np.isfinite(q).all() else float(np.sqrt(np.mean(np.sum((projected-q)**2,axis=1))))
        return dict(success=bool(actual['available']),actual_pose=actual,subset_indices=list(subset_indices),
                    reprojection_rmse_9_px=rmse,subset_center_rmse_px=score,reprojection_mean_8_px=float(residual))
    except (cv2.error,ValueError,np.linalg.LinAlgError) as e:
        return dict(success=False,actual_pose=dict(available=False),failure=type(e).__name__,
                    subset_indices=list(subset_indices),reprojection_rmse_9_px=None,subset_center_rmse_px=None)

def production(q,K,xyz,source=False,support=None):
    """One actual original F call, then alternative final-fit diagnostics."""
    assert _POSE is not None
    q=np.asarray(q,float);K=np.asarray(K,float);xyz=np.asarray(xyz,float)
    trace=[];original=_POSE.select_pnp_hypotheses
    def capture(*a,**kw):
        result=original(*a,**kw);trace.append(result);return result
    _POSE.select_pnp_hypotheses=capture
    try:actual=_POSE.infer(q,K,xyz,source)
    finally:_POSE.select_pnp_hypotheses=original
    selected=actual.get('selected_hypothesis')
    selector={h.name:h for h in trace[0].hypotheses} if trace else {}
    subset=np.flatnonzero(np.isfinite(q[:8]).all(-1)).tolist()
    candidates={}
    for name in NAMES:
        h=selector.get(name)
        if name==selected and actual.get('available'):
            proj=project_pose(actual,K)
            rmse=None if proj is None or not np.isfinite(q).all() else float(np.sqrt(np.mean(np.sum((proj-q)**2,axis=1))))
            candidate=dict(success=True,actual_pose=copy.deepcopy(actual),subset_indices=subset,
                           reprojection_rmse_9_px=rmse,subset_center_rmse_px=rmse,
                           reprojection_mean_8_px=actual['reprojection_px'])
        else:candidate=fit(q,K,xyz,name,subset,source)
        candidate.update(formal_score=None if h is None else h.score,
                         score_components={} if h is None else dict(h.score_components),
                         selector_success=False if h is None else h.success,
                         selector_failure='F_SELECTOR_NOT_REACHED_OR_FAILED_INPUTS' if h is None else h.failure_reason)
        candidates[name]=candidate
    scores={name:c['formal_score'] for name,c in candidates.items()}
    alternative=next((name for name in NAMES if name!=selected),None)
    gap=None if selected not in scores or scores[selected] is None or scores[alternative] is None else float(scores[alternative]-scores[selected])
    simple=[(c['reprojection_rmse_9_px'],i,name) for i,(name,c) in enumerate(candidates.items()) if c['success'] and c['reprojection_rmse_9_px'] is not None]
    simplechoice=min(simple)[2] if simple else None
    return dict(rule='S0',actual_pose=actual,hyp=selected,selected_hypothesis=selected,candidates=candidates,
                candidate_scores=scores,subset_indices=subset,fallback=False,
                status='AVAILABLE' if actual.get('available') else 'NO_POSE',formal_score_gap_px=gap,
                formal_gap_semantics='alternative minus selected weighted official score; nominal px includes penalties',
                simplechoice=simplechoice,selector_status=None if not trace else trace[0].status.value,
                original_F_calls=1,reference_inputs=False)
