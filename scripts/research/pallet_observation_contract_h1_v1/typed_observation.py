"""Diagnostic observation semantics, never a pose solver or real type supplier."""
from __future__ import annotations
import numpy as np
import re
from .b_query import finite_intersection

TYPES=('KNOWN_PHYSICAL_EDGE','TARGET_MEMBERSHIP_ONLY','UNRESOLVED')
EDGE_EVIDENCE=('renderer_mesh_topology','independent_physical_correspondence','analytic_counterexample')


def convex_hull(points):
    pts=sorted(set(map(tuple,np.asarray(points,float))))
    if len(pts)<3:return np.asarray(pts)
    cross=lambda o,a,b:(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[];hi=[]
    for p in pts:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi)>=2 and cross(hi[-2],hi[-1],p)<=0:hi.pop()
        hi.append(p)
    return np.asarray(lo[:-1]+hi[:-1])


def containment(points,projected):
    points=np.asarray(points,float);hull=convex_hull(projected)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)==0 or len(hull)<3 or not np.isfinite(points).all()or not np.isfinite(hull).all():
        return dict(available=False,all_inside=None,inside_count=None,point_count=len(points),projected_area_px2=None)
    edge=np.roll(hull,-1,axis=0)-hull;delta=points[:,None,:]-hull
    cross=edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0]
    # Analytic inclusive polygon relation, with precision in coordinate^2 units.
    tol=32*np.finfo(float).eps*max(1.,float(np.max(np.abs(hull)))**2)
    inside=(cross>=-tol).all(axis=1)
    area=.5*abs(float(np.sum(hull[:,0]*np.roll(hull[:,1],-1)-hull[:,1]*np.roll(hull[:,0],-1))))
    return dict(available=True,all_inside=bool(inside.all()),inside_count=int(inside.sum()),point_count=len(points),projected_area_px2=area,
        identifies_pose=False,large_projection_degeneracy_possible=True,coordinate_squared_roundoff_tolerance=tol)


def evaluate(observation,projected_cuboid,*,image_hw,depth=None,candidate_edges=None):
    kind=observation['type']
    if kind not in TYPES:raise ValueError('Explicit diagnostic type required')
    result=dict(observation_id=observation['id'],type=kind,precise_edge_constraint=False,
        new_correction_applied=0,automatic_real_type_supplier='UNVERIFIED',candidate_ranking_allowed=False,
        evidence_authenticated_by_evaluator=False,independent_physical_ownership_verified=False,
        evidence_validation='Supplied evidence structure only; renderer/physical ownership must be authenticated separately')
    if observation.get('semantic_owner')not in (None,'target'):
        return dict(result,status='OWNERSHIP_CONFLICT',constraint_applied=False,contract_eligible_input=False)
    if kind=='UNRESOLVED':return dict(result,status='UNRESOLVED_RETAINED',constraint_applied=False,contract_eligible_input=False)
    if kind=='KNOWN_PHYSICAL_EDGE':
        evidence=observation.get('edge_evidence',{})
        valid=(evidence.get('association_confirmed')is True and evidence.get('basis')in EDGE_EVIDENCE and
            bool(evidence.get('edge_id'))and bool(evidence.get('target_instance_id')))
        if not valid:return dict(result,status='UNVERIFIED_EDGE_ASSOCIATION',constraint_applied=False,contract_eligible_input=False)
        edge=(candidate_edges or{}).get(evidence['edge_id'])
        if not edge or edge.get('target_instance_id')!=evidence['target_instance_id']or not re.fullmatch('[0-9a-f]{64}',str(edge.get('fixed_pose_binding_sha256',''))):
            return dict(result,status='UNBOUND_CANDIDATE_EDGE',constraint_applied=False,contract_eligible_input=False)
        if depth is None:return dict(result,status='UNVERIFIED_EDGE_DEPTH',constraint_applied=False,contract_eligible_input=False)
        xy=np.asarray(edge['xy'],float)
        if xy.shape!=(2,2):raise ValueError('Candidate physical edge must contain its two projected endpoints')
        relation=finite_intersection(xy[0],xy[1],observation['center'],observation['normal'],image_hw=image_hw,depth=depth)
        return dict(result,status='DIAGNOSTIC_PHYSICAL_EDGE_RELATION',constraint_applied=True,contract_eligible_input=True,
            precise_edge_constraint=True,relation=relation,
            candidate_edge_binding=dict(edge_id=evidence['edge_id'],target_instance_id=edge['target_instance_id'],
                fixed_pose_binding_sha256=edge['fixed_pose_binding_sha256']),
            certification_scope='analytic fixture only'if evidence['basis']=='analytic_counterexample'else'evidence supplied by caller; independently verified ownership still required')
    evidence=observation.get('membership_evidence',{})
    valid=(evidence.get('target_instance_id')and evidence.get('basis')in ('renderer_target_object_mask','independent_target_segmentation','analytic_counterexample'))
    relation=containment(observation.get('target_points',[]),projected_cuboid)
    front=depth is not None and bool(np.isfinite(depth).all()and np.min(depth)>0)
    return dict(result,status='CONDITIONAL_TARGET_CONTAINMENT'if valid else'UNVERIFIED_MEMBERSHIP',
        constraint_applied=bool(valid and relation['available']and front),contract_eligible_input=bool(valid and relation['available']and front),
        relation=relation,precise_edge_constraint=False,
        conditions=['actual target geometry inside registered cuboid','target mask ownership correct','geometry in front of camera'],
        new_pose_accuracy_not_established=True)


def evaluate_fixed_universe(observations,expected_ids,projected_cuboid,*,image_hw,depth=None,candidate_edges=None):
    ids=[o['id']for o in observations]
    if len(ids)!=len(set(ids))or len(expected_ids)!=len(set(expected_ids)):raise ValueError('Unique fixed observation IDs required')
    if set(ids)!=set(expected_ids):
        return dict(status='INCOMPLETE_OBSERVATION_UNIVERSE',missing_ids=sorted(set(expected_ids)-set(ids)),
            extra_ids=sorted(set(ids)-set(expected_ids)),new_correction_applied=0,candidate_ranking_allowed=False,total_cost=None)
    rows=[evaluate(o,projected_cuboid,image_hw=image_hw,depth=depth,candidate_edges=candidate_edges)for o in observations]
    count=sum(r['contract_eligible_input']for r in rows)
    return dict(status='DIAGNOSTIC_ONLY'if count else'NO_CERTIFIED_OBSERVATION_NO_NEW_CORRECTION',observations=rows,
        fixed_observation_count=len(expected_ids),contract_eligible_observations=count,new_correction_applied=0,
        candidate_ranking_allowed=False,total_cost=None,identity_output_is_not_safety_evidence=True)
