"""Fixed local point/line ablation on identical learned observations.

Edges that contribute to any retained corner are removed from line factors.
Each remaining line contributes two endpoint-to-observed-line distances, with
1/sqrt(2) normalization; 3D endpoints are registry geometry, never hidden2D.
The initial pose is an optimizer start, never a residual prior.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from .solver import cuboid,project
from .inference import hidden_mask

EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
POLICY=dict(loss='soft_l1',f_scale_px=8.,max_nfev=50,point_weight=1.,line_weight=1/np.sqrt(2),
    duplicate_policy='remove every line supporting a used corner',initial_pose_prior=False,
    rank_threshold=1e-10,independent_PnP_when_fewer_than4_points=False)

def solve(observations,K,xyz,initial,point_result,hidden):
    """observations contains corners(id,xy,edges) and lines(edge,normal,offset)."""
    usable=[c for c in observations['corners'] if c['id'] not in set(hidden)]
    consumed={e for c in usable for e in c['edges']}
    lines=[l for l in observations['lines'] if l['edge'] not in consumed]
    if not usable and not lines:return dict(available=False,reason='no_independent_factors')
    starts=[]
    if point_result.get('available'):starts.append(point_result)
    if initial.get('available'):starts.append(initial)
    if not starts:return dict(available=False,reason='no_local_initialization')
    answers=[];attempts=[]
    for start in starts:
        dims=np.asarray(start['cf_extents']);X=cuboid(*dims)
        z0=np.r_[cv2.Rodrigues(np.asarray(start['R_cf']))[0].ravel(),start['centroid']]
        def residual(z):
            q=project(X,cv2.Rodrigues(z[:3])[0],z[3:],K)
            rr=[]
            for c in usable:rr.extend(q[c['id']]-np.asarray(c['xy']))
            for line in lines:
                normal=np.asarray(line['normal']);normal=normal/np.linalg.norm(normal)
                rr.extend((q[list(EDGES[line['edge']])]@normal-line['offset'])/np.sqrt(2))
            return np.asarray(rr)
        if len(residual(z0))<6:attempts.append(dict(reason='insufficient_residual_dimensions'));continue
        try:fit=least_squares(residual,z0,loss='soft_l1',f_scale=8.,max_nfev=50)
        except (ValueError,np.linalg.LinAlgError) as error:attempts.append(dict(reason=type(error).__name__));continue
        R=cv2.Rodrigues(fit.x[:3])[0];t=fit.x[3:];J=fit.jac
        J=J/np.maximum(np.linalg.norm(J,axis=0),1e-300);s=np.linalg.svd(J,compute_uv=False);rank=int((s>s[0]*1e-10).sum())
        good=np.isfinite(fit.x).all() and ((R@X.T).T+t)[:,2].min()>1e-9 and rank==6 and fit.success
        info=dict(available=bool(good),reason='local_point_line_estimated' if good else 'local_rank_or_numerical_failure',
            nfev=fit.nfev,rank=rank,singular_values=s,condition_number=float(s[0]/s[-1]) if s[-1]>0 else None,cost=fit.cost,
            point_ids=[c['id'] for c in usable],line_edges=[l['edge'] for l in lines],consumed_edges=sorted(consumed),
            local_initialization_only=len(usable)<4,initial_pose_prior=False,policy=POLICY,
            same_edge_point_line_double_count=False)
        attempts.append(info)
        if good:
            Q=np.eye(3) if abs(dims[0]-xyz[0])<1e-6 else np.array([[0,0,1],[0,1,0],[-1,0,0]])
            info.update(R_cf=R,R_physical=R@Q,centroid=t,cf_extents=dims,projected=project(X,R,t,K),selected_hypothesis=start.get('selected_hypothesis'))
            answers.append(info)
    if not answers:return dict(available=False,reason='local_point_line_failure',attempts=attempts)
    result=dict(min(answers,key=lambda r:r['cost']));result['attempts']=attempts
    return result

def tests():
    K=np.array([[600.,0,320],[0,600,240],[0,0,1]]);xyz=np.array([1.1,.14,1.3]);X=cuboid(*xyz)
    R=cv2.Rodrigues(np.array([.5,.4,.1]))[0];t=np.array([0.,0.,3.]);q=project(X,R,t,K)
    initial=dict(available=True,R_cf=R,centroid=t,cf_extents=xyz,selected_hypothesis='REGISTRY_WD')
    corners=[dict(id=i,xy=q[i],edges=[i,(i-1)%4]) for i in range(4)]
    lines=[]
    for e,(a,b) in enumerate(EDGES):
        normal=np.array([-(q[b]-q[a])[1],(q[b]-q[a])[0]]);normal/=np.linalg.norm(normal)
        lines.append(dict(edge=e,normal=normal,offset=q[a]@normal))
    answer=solve(dict(corners=corners,lines=lines),K,xyz,initial,initial,[])
    assert answer['available'] and not set(answer['line_edges'])&set(answer['consumed_edges'])
    assert np.max(abs(answer['projected']-q))<1e-6
    absent=solve(dict(corners=[],lines=[]),K,xyz,initial,initial,[])
    assert not absent['available']
    return dict(exact_geometry=True,no_edge_double_count=True,all_no_match_insufficient=True)
