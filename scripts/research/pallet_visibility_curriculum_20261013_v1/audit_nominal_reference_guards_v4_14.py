"""Postseal nominal-reference hard-guard gauges, not a runtime selector.

Every registered W/D family and the two declared C2 gauges is evaluated. No
reference pose is fitted/refined, no image/model forward occurs, and no
reference/gauge can alter predictions. Exact nominal-reference infeasibility
does not prove that every pose within5cm is infeasible.
"""
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

from . import audit_remainder_pipeline_v4_14 as A
from . import score_remainder_derived_reference_v4_14 as REF
from . import statewise_partial_BI_outline_solver_v4_13 as G

OUTPUT=A.OUTPUT/'nominal_reference_guard_audit_v1'


def constraints(prepared,R,t):
    xyz,K,states,m,domains,hw,cfg,boundary,regions,Q,prior,requested=prepared
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    p=G.V._pose_vector(R,np.asarray(t),scale);sides=[None]*len(domains)
    if any(d['state']=='T'for d in domains):raise ValueError('Natural143 T0 contract changed; enumerate sides explicitly')
    values,jac=G.P._joint_constraint_pair(p,xyz,K,m,domains,sides,hw,scale,cfg,boundary,regions)
    empty=[]
    base,_=G.P.BASE._constraints(p,xyz,K,m,empty,empty,hw,scale,cfg)
    labels=[dict(type='depth',native_corner=k,unit='m')for k in range(8)]
    for measurement in m:
        k=measurement['native_corner']
        if cfg['enforce_visible_geometry']:labels.append(dict(type='visible_geometry',native_corner=k,unit='cosine'))
        if cfg['enforce_visible_canvas']:
            labels.extend(dict(type='visible_canvas',native_corner=k,side=s,unit='canvas_fraction')
                          for s in ('left','top','right','bottom'))
    if len(labels)!=len(base):raise ValueError('Frozen physical BI guard layout changed')
    for d in domains:
        local,_=G.P.BASE._constraints(p,xyz,K,empty,[d],[None],hw,scale,cfg)
        labels.extend(dict(type='O_region',native_corner=d['native_corner'],unit='mixed_ellipse_or_halfplane_slack')
                      for _ in local[8:])
    labels.extend(dict(type='U_region',native_corner=r['native_corner'],unit='ellipse_normalized')for r in regions)
    for b in boundary:
        for j,line in enumerate(b['lines']):
            labels.extend(dict(type='partial_tangent_interval',native_corner=b['native_corner'],line=j,endpoint=e,unit='pixels')
                          for e in ('low','high'))
    if len(values)!=len(labels):raise ValueError('Frozen constraint layout categories incomplete')
    violations=[{**label,'slack':float(value)}for label,value in zip(labels,values)if value < -cfg['feasibility_tolerance']]
    dropped={b['native_corner']for b in boundary};complete=[r for r in m if r['native_corner']not in dropped]
    _,vj=G.J.visible_residual_and_jacobian(p,xyz,K,complete,scale)
    _,nj=G.P.N._normal_residual_and_jacobian(p,xyz,K,boundary,scale)
    objective,_,parts=G.P._joint_objective_pair(p,xyz,K,m,scale,prior,cfg,boundary,1.)
    return dict(feasible=not violations,minimum_slack=float(values.min())if len(values)else None,
        violations=violations,violation_counts=dict(Counter(r['type']for r in violations)),
        complete_BI_rank=G.V._rank(vj),BI_plus_normal_rank=G.V._rank(np.vstack([vj,nj])),
        non_outline_objective=float(objective),objective_components=parts,
        normal_offset_is_soft_objective_not_hard_guard=True,
        minimum_slack_mixes_units_and_is_not_a_probability_or_physical_distance=True)


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve nominal reference audit')
    A.verify_source_closure();seal=A.C.read(A.POSE/'PREDICTION_SEAL.json');owned=A.C.read(REF.OUTPUT/'SCORE_SEAL.json')
    if seal['rows']!=143 or owned['prediction_seal_sha256']!=A.C.sha(A.POSE/'PREDICTION_SEAL.json'):
        raise ValueError('Full143/ownedpostseal gate required')
    parent=A.C.read(A.POSE/'PROTOCOL.json');config=dict(parent['solver_configuration']);config['prior_weight']=0.
    refs=A.C.read(REF.OUTPUT/'PLAN.json');sources={str(Path(f).relative_to(A.C.ROOT)):A.C.sha(f)
        for f in (__file__,G.__file__,G.P.__file__,REF.__file__)}
    OUTPUT.mkdir(parents=True)
    A.C.write(OUTPUT/'PROTOCOL.json',dict(schema='NOMINAL_REFERENCE_GUARDS_DIAGNOSTIC_V1',rows=143,
        before_own_constraint_values_computed_UTC=datetime.now(timezone.utc).isoformat(),
        stages=['LEGACY_BEFORE_D','TYPED_AFTER_D'],gauges='both savedWD geometries times both fixed properC2 rotations',
        reference_policy=refs['reference_policy'],source_sha256=sources,
        prediction_seal=A.binding(A.POSE/'PREDICTION_SEAL.json'),owned_score_seal=A.binding(REF.OUTPUT/'SCORE_SEAL.json'),
        no_new_fits_or_forwards_or_runtime_GT_gauge_selection=True,exact_nominal_fail_is_not_a_5cm_neighborhood_exclusion=True))
    result=[]
    for annotation in refs['annotations']:
        fid=annotation['id'];stem=fid.replace(':','__')
        prediction=A.C.read(A.POSE/'frames'/(stem+'.json'));row=A.C.read(A.O.OUTPUT/'frames'/(stem+'.json'))
        path=A.C.ROOT/annotation['annotation']
        if A.C.sha(path)!=annotation['annotation_sha256']:raise ValueError('Fixed reference changed')
        reference=REF.reference_from_document(A.C.read(path))
        if reference['R']is None:raise ValueError('All143 reference rotations were eligible; no silent gauge substitute')
        Rgt=np.asarray(reference['R']);records=[]
        for stage,partial in (('LEGACY_BEFORE_D',row['partial_boundary_observation']),
                              ('TYPED_AFTER_D',prediction['partial_global_outline']['merged_partial_observation'])):
            for family in prediction['full_WD_search']['families']:
                coarse=family['coarse_pose'];Q=np.asarray(family['physical_registration_Q'])
                prepared=G._prepare(family['native_xyz'],row['_original_source']['initial_geometry']['K'],
                    row['states'],row['frozen_BI_fit'],row['_original_source']['ot_domains']['domains'],
                    row['runtime']['initial_geometry']['image_hw'],coarse,config,
                    partial['boundary_observations'],row['u_region_observation']['u_regions'])
                for gauge,S in enumerate((np.eye(3),np.diag([-1.,1.,-1.]))):
                    Rcf=Rgt@S@Q.T
                    audit=constraints(prepared,Rcf,reference['t'])
                    records.append(dict(stage=stage,family=family['name'],C2_gauge=gauge,**audit))
        result.append(dict(id=fid,session=annotation['session'],gauges=records,
            any_nominal_gauge_feasible_by_stage={s:any(r['feasible']for r in records if r['stage']==s)
                                                for s in ('LEGACY_BEFORE_D','TYPED_AFTER_D')}))
    summaries={}
    for stage in ('LEGACY_BEFORE_D','TYPED_AFTER_D'):
        failed=[r for r in result if not r['any_nominal_gauge_feasible_by_stage'][stage]]
        categories=Counter()
        for record in failed:
            present={v['type']for g in record['gauges']if g['stage']==stage for v in g['violations']}
            categories.update(present)
        summaries[stage]=dict(frames=143,any_nominal_reference_gauge_feasible=143-len(failed),
            all_nominal_gauges_infeasible=len(failed),failure_category_frame_presence=dict(categories),
            all_infeasible_ids=[r['id']for r in failed])
    A.E._verify_sources(sources);A.verify_source_closure()
    A.C.write(OUTPUT/'AUDIT.json',dict(summary=summaries,rows=result,new_pose_fits=0,new_image_or_model_forwards=0,
        normal_hard_guard_exclusion_count=0,nominal_reference_gauge_is_diagnostic_not_runtime_oracle=True,
        exact_reference_infeasible_does_not_prove_no_5cm_feasible_pose=True,
        signed_axis_or_physical_reference_quality_flags_unchanged=True))
    A.C.write(OUTPUT/'AUDIT_SEAL.json',dict(audit_sha256=A.C.sha(OUTPUT/'AUDIT.json'),
        protocol_sha256=A.C.sha(OUTPUT/'PROTOCOL.json'),source_sha256=sources,all143=True))
    print({k:{n:v for n,v in s.items()if n!='all_infeasible_ids'}for k,s in summaries.items()},flush=True)


if __name__=='__main__':main()
