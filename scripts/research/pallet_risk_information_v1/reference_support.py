"""Separate synthetic reference-only diagnostic, never a production feature.

Run in its own process/output directory after P0. Registered C1/C2 phases are
fixed poses, not new candidate proposals. No solving, RGB forward or mutation.
Every cost comparison uses one original G13 context/family/objective package.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import resource
import tempfile
import time
import numpy as np

from .common import read, write, rows, save_rows, sha
from . import feature_support as F
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_uncertain_solver_v4_bias as BASE
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_boundary_normal_solver_v4_4 as PARTIAL
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_u_region_solver_v4_5 as U


def plain(x):
    if isinstance(x,np.ndarray): return plain(x.tolist())
    if isinstance(x,np.generic): return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x


def registered_reference_phases(target):
    order=int(target['symmetry_order'])
    if order not in (1,2) or order != target['symmetry_order']:
        raise ValueError('Only the original row C1/C2 is admitted, never 90-degree W/D')
    R=np.asarray(target['R_fixed'],float);t=np.asarray(target['t'],float)
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('Original physical R_fixed/centroid required')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0) or abs(np.linalg.det(R)-1)>1e-6:
        raise ValueError('Original reference rotation is not proper')
    result=[]
    for k in range(order):
        angle=k*2*np.pi/order
        S=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        result.append(dict(phase_index=k,angle_radians=float(angle),reference_pose=dict(
            available=True,R_physical=(R@S).tolist(),centroid=t.tolist())))
    return result


def _constraint_summary(values,tolerance):
    values=np.asarray(values,float)
    return dict(count=len(values),finite_count=int(np.isfinite(values).sum()),
        minimum_slack=float(np.min(values)) if len(values) and np.isfinite(values).all() else None,
        feasible=bool(np.isfinite(values).all() and (not len(values) or np.min(values)>=-tolerance)),
        inequalities=plain(values))


def observation_diagnostics(model,p,fixed_audit=None):
    """Original scalar/constraint evaluators, decomposed only for reporting.

    The original full point residual is soft, never declared a failed hard
    residual budget. Partial tangent and O/T/U constraints retain source scale.
    Overall original simultaneous constraints are also stored without subsets.
    """
    c=model.ctx;q,depth,_,_=F.OLD.J.projection_and_jacobian(p,c.xyz,c.K,c.scale)
    tolerance=c.cfg['feasibility_tolerance'];partial_ids={b['native_corner'] for b in c.boundary}
    full=[]
    for m in c.measurements:
        val=BASE._constraints(p,c.xyz,c.K,[m],[],(),c.image_hw,c.scale,c.cfg)[0]
        residual=q[m['native_corner']]-m['xy']
        full.append(dict(native_corner=m['native_corner'],state=m['state'],source=m.get('source'),
            original_observation=plain(m),projected_xy=plain(q[m['native_corner']]),
            likelihood_active=m['native_corner'] not in partial_ids,
            likelihood_replaced_by_partial=m['native_corner'] in partial_ids,
            residual_px=float(np.linalg.norm(residual)),
            mahalanobis=float(np.linalg.norm(m['whitening']@residual)),
            residual_is_soft_not_a_hard_acceptance_test=True,
            geometry_canvas_constraints=_constraint_summary(val[8:],tolerance)))
    partial=[]
    for b in c.boundary:
        val=PARTIAL._boundary_constraint_pair(p,c.xyz,c.K,[b],c.image_hw,c.scale,c.cfg)[0]
        audit=PARTIAL._boundary_audit(p,c.xyz,c.K,[b],c.image_hw,c.scale,c.cfg,
            b.get('fit_reliability_weight',1.))[0]
        partial.append(dict(native_corner=b['native_corner'],state=b['state'],
            original_observation=plain(b),original_audit=plain(audit),
            constraints=_constraint_summary(val,tolerance)))
    domains=[]
    sides=F.OLD._sides(c.domains,q,c.image_hw)
    for d,side in zip(c.domains,sides):
        val=BASE._constraints(p,c.xyz,c.K,[],[d],[side],c.image_hw,c.scale,c.cfg)[0]
        domains.append(dict(native_corner=d['native_corner'],state=d['state'],outside_canvas_branch=side,
            original_observation=plain(d),projected_xy=plain(q[d['native_corner']]),
            constraints=_constraint_summary(val[8:],tolerance)))
    regions=[]
    for u in c.regions:
        val=U._u_constraint_pair(p,c.xyz,c.K,[u],c.scale)[0]
        regions.append(dict(native_corner=u['native_corner'],state='U',original_observation=plain(u),
            projected_xy=plain(q[u['native_corner']]),constraints=_constraint_summary(val,tolerance)))
    original_all=F.OLD.constraints(model,p,0,replace_B=False)[0]
    depth_values=(depth-c.cfg['positive_depth_m'])/c.scale
    base_energy,_,base_components=c.backend.P._joint_objective_pair(
        p,c.xyz,c.K,c.measurements,c.scale,c.prior,c.cfg,c.boundary,1.)
    return dict(full_points=full,partial=partial,OT_domains=domains,U_regions=regions,
        original_objective_components=plain(base_components),
        original_base_energy_excluding_outline=float(base_energy),
        original_weighted_outline_energy=(fixed_audit or {}).get('parts',{}).get('outline'),
        original_joint_constraints=_constraint_summary(original_all,tolerance),
        depth_constraints=_constraint_summary(depth_values,tolerance),
        outline=dict(original_observation=plain(c.outline),active=c.use_outline,
            is_soft_objective_not_hard_support=True,
            observed_sample_count=len(c.outline.get('observed_boundary_samples_xy',[])),
            zero_if_inactive_is_not_correct_material_evidence=True),
        states_fixed_from_initial_N0=list(c.states),states_not_regenerated_at_reference=True,
        self_occluded_S_not_an_image_measurement=True)


def diagnose_frame(frame,e_run,bundle,config,target):
    if target['id']!=frame['id']: raise ValueError('Reference join ID mismatch')
    if target.get('split') not in ('train','validation'): raise ValueError('Synthetic registered split required')
    if target.get('split')!=frame.get('split'): raise ValueError('Synthetic reference/feature split mismatch')
    models=F.compile_models(frame,e_run,bundle,config)
    selected=frame.get('e0_regenerated',{})
    selected_available=F.AD.available(selected)
    selected_scores=F.OLD.candidate_scores(selected,models) if selected_available else None
    selected_metric=F.AD.pose_error(selected,target)
    phases=[]
    for phase in registered_reference_phases(target):
        pose=phase['reference_pose'];scores=F.OLD.candidate_scores(pose,models);byfamily=[]
        for family in F.FAMILIES:
            model=models[family];c=model.ctx;p=F.OLD._native_pose(pose,c)
            actual=F.OLD._actual_pose(p,c)
            Rn=np.asarray(pose['R_physical'])@c.Q.T
            tn=np.asarray(pose['centroid'])-Rn@c.xyz.mean(0)
            reconstructed_R=Rn@c.Q;reconstructed_center=Rn@c.xyz.mean(0)+tn
            # Preserve the original 1e-10 assertion result as a separate audit.
            # Rounded original renderer R_fixed can be valid under inherited
            # 1e-6 rotation admission yet change in the inherited Rodrigues
            # conversion. Do not change tolerance or silently erase its FAIL.
            mapping=dict(raw_Q_rotation_max_abs=float(np.max(np.abs(reconstructed_R-pose['R_physical']))),
                raw_origin_center_max_abs=float(np.max(np.abs(reconstructed_center-pose['centroid']))),
                inherited_Rodrigues_rotation_max_abs=float(np.max(np.abs(np.asarray(actual['R_physical'])-pose['R_physical']))),
                inherited_Rodrigues_center_max_abs=float(np.max(np.abs(np.asarray(actual['centroid'])-pose['centroid']))),
                original_roundtrip_1e_10_pass=bool(np.allclose(actual['R_physical'],pose['R_physical'],atol=1e-10,rtol=0) and
                    np.allclose(actual['centroid'],pose['centroid'],atol=1e-10,rtol=0)),
                original_roundtrip_tolerance=1e-10,
                inherited_original_rotation_admission_tolerance=1e-6,
                original_R_orthogonality_max_abs=float(np.max(np.abs(np.asarray(pose['R_physical']).T@np.asarray(pose['R_physical'])-np.eye(3)))),
                original_R_determinant=float(np.linalg.det(pose['R_physical'])),
                normalized_conversion_is_original_evaluator_not_a_pose_solve=True)
            q,depth,_,_=F.OLD.J.projection_and_jacobian(p,c.xyz,c.K,c.scale)
            finite=F.FIN.support_audit(bundle,q,depth) if bundle is not None else None
            raw_camera=c.xyz@Rn.T+tn;homogeneous=raw_camera@c.K.T
            z=homogeneous[:,2];den=np.where(np.abs(z)<1e-10,np.where(z<0,-1e-10,1e-10),z)
            raw_projection=homogeneous[:,:2]/den[:,None]
            raw_finite=F.FIN.support_audit(bundle,raw_projection,raw_camera[:,2]) if bundle is not None else None
            mapping['raw_physical_vs_inherited_projected_xy_max_abs']=float(np.max(np.abs(raw_projection-q)))
            comparisons=[]
            for mode in F.MODES:
                gs=scores[family];a=F.OLD._finite_feature(gs['scores'][mode])
                b=F.OLD._finite_feature(selected_scores[family]['scores'][mode]) if selected_scores else None
                comparisons.append(dict(objective='G13_FIXED_POSE_'+mode,family=family,
                    reference_cost=a,selected_E0_cost=b,
                    selected_minus_reference=b-a if a is not None and b is not None else None,
                    selected_E0_strictly_lower_cost=b<a if a is not None and b is not None else None,
                    reference_original_constraints_feasible=gs['feasible'][mode],
                    reference_cost_detail=plain(gs['audits'][mode]),
                    selected_cost_detail=plain(selected_scores[family]['audits'][mode]) if selected_scores else None,
                    reference_h_min_slack=plain(gs['h_min_slack'][mode]),
                    selected_h_min_slack=plain(selected_scores[family]['h_min_slack'][mode]) if selected_scores else None,
                    cost_comparison_valid=a is not None and b is not None,
                    same_stage_family_observations_objective=True))
            byfamily.append(dict(family=family,physical_registration_Q=plain(c.Q),
                native_origin_mean=plain(c.xyz.mean(0)),native_xyz=plain(c.xyz),
                physical_mapping_audit=mapping,
                native_pose_parameter_vector=plain(p),reference_projected_native=plain(q),
                raw_physical_projection_native=plain(raw_projection),raw_physical_finite_support=plain(raw_finite),
                finite_support=plain(finite),observations=observation_diagnostics(model,p,scores[family]['audits']['fixed']),
                original_feasible={m:bool(scores[family]['feasible'][m]) for m in F.MODES},
                same_objective_comparisons=comparisons))
        phases.append(dict(**phase,families=byfamily))
    summary={}
    for family in F.FAMILIES:
        ff=[next(x for x in p['families'] if x['family']==family) for p in phases]
        n=ff[0]['finite_support']['fixed_profile_count'] if ff[0]['finite_support'] else None
        summary[family]=dict(B_observation_available=n is not None and n>0,
            original_phase_finite_support=bool(ff[0]['finite_support']['finite_support']) if n else None,
            any_allowed_phase_finite_support=any(x['finite_support']['finite_support'] for x in ff) if n else None,
            all_allowed_phases_finite_rejected=not any(x['finite_support']['finite_support'] for x in ff) if n else None,
            raw_original_phase_finite_support=bool(ff[0]['raw_physical_finite_support']['finite_support']) if n else None,
            raw_any_allowed_phase_finite_support=any(x['raw_physical_finite_support']['finite_support'] for x in ff) if n else None,
            original_phase_original_feasible=ff[0]['original_feasible'],
            any_allowed_phase_original_feasible={m:any(x['original_feasible'][m] for x in ff) for m in F.MODES},
            any_single_allowed_phase_joint_finite_and_original_feasible={m:any(
                x['original_feasible'][m] and x['finite_support']['finite_support'] for x in ff) if n else None for m in F.MODES})
    material_keys=[k for k in ('material_id','material_ids','mesh','mesh_path','visibility_metadata','material_boundary') if k in target]
    return dict(schema='SYNTHETIC_REFERENCE_FIXED_SUPPORT_FRAME_V1',id=frame['id'],split=frame['split'],
        symmetry_order=int(target['symmetry_order']),registered_phases=phases,summary=summary,
        selected_E0_available=selected_available,selected_E0_metric_diagnostic_only=selected_metric,
        selected_E0_pose=selected if selected_available else None,
        material_metadata_fields_present=material_keys,
        material_owner_status='METADATA_AVAILABLE_NOT_PHYSICALLY_CERTIFIED' if material_keys else 'MATERIAL_OWNER_UNVERIFIED',
        material_metadata={k:plain(target[k]) for k in material_keys},
        reference_used_only_in_this_diagnostic=True,production_feature_or_candidate_insertion=False,
        new_solve_calls=0,new_PnP_calls=0,new_model_forward_calls=0,new_candidate_calls=0)


def aggregate(records):
    groups={}
    for split in ('train','validation','ALL1003'):
        subset=[r for r in records if split=='ALL1003' or r['split']==split]
        result=dict(frames=len(subset),families={})
        for family in F.FAMILIES:
            summaries=[r['summary'][family] for r in subset]
            phases=Counter();reasons=Counter();raw_reasons=Counter();comparisons=Counter();obs=Counter()
            for row in subset:
                for phase in row['registered_phases']:
                    f=next(x for x in phase['families'] if x['family']==family)
                    key='original' if phase['phase_index']==0 else 'additional_allowed'
                    phases[key+'/evaluations']+=1
                    finite=f['finite_support']
                    if finite:
                        phases[key+'/profiles']+=finite['fixed_profile_count'];phases[key+'/supported_profiles']+=finite['supported_profiles']
                        reasons.update(key+'/'+p['geometry_status'] for p in finite['profiles'])
                        raw=f['raw_physical_finite_support']
                        raw_reasons.update(key+'/'+p['geometry_status'] for p in raw['profiles'])
                        phases[key+'/raw_vs_parameterized_profile_status_differences']+=sum(
                            a['geometry_status']!=b['geometry_status'] for a,b in zip(finite['profiles'],raw['profiles']))
                    for kind,key2 in [('full_points','geometry_canvas_constraints'),('partial','constraints'),('OT_domains','constraints'),('U_regions','constraints')]:
                        for a in f['observations'][kind]:
                            obs[key+'/'+kind+'/observations']+=1
                            obs[key+'/'+kind+'/rejected']+=not a[key2]['feasible']
                    for c in f['same_objective_comparisons']:
                        prefix=key+'/'+c['objective']
                        comparisons[prefix+'/valid_comparisons']+=c['cost_comparison_valid']
                        comparisons[prefix+'/reference_feasible_but_failed_E0_lower_cost']+=bool(
                            c['reference_original_constraints_feasible'] and c['selected_E0_strictly_lower_cost'] and
                            not row['selected_E0_metric_diagnostic_only']['success_5cm5deg'])
            result['families'][family]=dict(frames=len(subset),
                original_roundtrip_1e_10_failures=sum(not next(x for x in p['families'] if x['family']==family)['physical_mapping_audit']['original_roundtrip_1e_10_pass']
                    for r in subset for p in r['registered_phases']),
                B_input_frames=sum(s['B_observation_available'] for s in summaries),
                B_absent_frames=sum(not s['B_observation_available'] for s in summaries),
                original_phase_finite_pass=sum(s['original_phase_finite_support'] is True for s in summaries),
                any_allowed_phase_finite_pass=sum(s['any_allowed_phase_finite_support'] is True for s in summaries),
                raw_original_phase_finite_pass=sum(s['raw_original_phase_finite_support'] is True for s in summaries),
                raw_any_allowed_phase_finite_pass=sum(s['raw_any_allowed_phase_finite_support'] is True for s in summaries),
                all_allowed_phases_finite_rejected=sum(s['all_allowed_phases_finite_rejected'] is True for s in summaries),
                representation_dependent_finite=sum(s['original_phase_finite_support'] is False and s['any_allowed_phase_finite_support'] is True for s in summaries),
                any_allowed_phase_original_feasible={m:sum(s['any_allowed_phase_original_feasible'][m] for s in summaries) for m in F.MODES},
                phase_counts=dict(phases),finite_reason_counts=dict(reasons),raw_physical_finite_reason_counts=dict(raw_reasons),observation_conflicts=dict(obs),
                same_G13_objective_comparisons=dict(comparisons))
        groups[split]=result
    return dict(schema='B_REFERENCE_SUPPORT_AUDIT_V1',groups=groups,diagnostic_only=True,
        feature_training_inference_use=False,production_bank_mutations=0,
        independent_physical_ground_truth='UNVERIFIED; registered renderer reference only',
        source_of_constraints_and_costs='original evaluators reused; independent geometry crosscheck required separately',
        phase_policy='row C1/C2 only; phase0 original and any-permitted phase separated; no 90-degree expansion',
        no_profile_policy='absence, not finite confidence; no vacuous pass counted',
        new_solve_calls=0,new_PnP_calls=0,new_model_forward_calls=0,new_candidate_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','feature-seal','cache','config','references','reference-seal','plan-go','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--limit',type=int)
    args=p.parse_args();plan=read(args.plan_go)
    if plan.get('status')!='FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE':raise ValueError('P0 frozen plan required')
    out=Path(args.output)
    if not out.name.startswith('b_reference_'): raise ValueError('Separate b_reference_* diagnostic directory required')
    out.mkdir(parents=True,exist_ok=False)
    # No code in F1/train imports this module or accepts these reference objects.
    import cv2
    cv2.setNumThreads(1);F.preload_backends()
    temp=out/'native_diagnostic_tmp';temp.mkdir();tempfile.tempdir=str(temp)
    start,cpu=time.monotonic(),time.process_time()
    F.AD.bound(args.features,args.feature_seal)
    reference_seal=read(args.reference_seal);refhash=sha(args.references)
    def contains(x):
        return x==refhash if isinstance(x,str) else any(contains(y) for y in x.values()) if isinstance(x,dict) else any(contains(y) for y in x) if isinstance(x,list) else False
    if not contains(reference_seal): raise ValueError('Original reference seal does not bind bytes')
    targets={r['id']:r for r in rows(args.references)}
    features=list(rows(args.features))
    if len(targets)!=1003 or len(features)!=1003:raise ValueError('Exact original synthetic1003 required')
    if set(targets)!=set(f['id'] for f in features):raise ValueError('Original synthetic reference IDs mismatch')
    config=read(args.config);index=F._frame_paths(args.cache);result=[];timings=[]
    chosen=features if args.limit is None else features[:args.limit]
    with F.fixed_pose_only() as tripwire:
        for frame in chosen:
            before,pcpu=time.monotonic(),time.process_time();folder,s,sealp=index[frame['id']]
            for filename,key in [('E_RUN.jsonl.gz','e_run_sha256'),('BOUNDARY_BUNDLE.jsonl.gz','boundary_bundle_sha256')]:
                if sha(folder/filename)!=s[key]:raise ValueError('Original fixed observations changed')
            er=next(rows(folder/'E_RUN.jsonl.gz'));bundle=next(rows(folder/'BOUNDARY_BUNDLE.jsonl.gz'))
            result.append(plain(diagnose_frame(frame,er,bundle,config,targets[frame['id']])))
            timings.append(dict(id=frame['id'],wall_seconds=time.monotonic()-before,CPU_seconds=time.process_time()-pcpu))
        if tripwire['forbidden_invocation_attempts']: raise AssertionError('Forbidden solve attempted')
    save_rows(out/'B_REFERENCE_SUPPORT_ROWS.jsonl.gz',result)
    summary=aggregate(result);summary.update(complete=args.limit is None,actual_rows=len(result),
        subset_technical_only=args.limit is not None,wall_seconds=time.monotonic()-start,
        CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        per_frame_timing=timings)
    write(out/'B_REFERENCE_SUPPORT_AUDIT.json',summary)
    write(out/'DIAGNOSTIC_SEAL.json',dict(schema='B_REFERENCE_SUPPORT_SEAL_V1',complete=args.limit is None,
        features_sha256=sha(args.features),feature_seal_sha256=sha(args.feature_seal),
        original_reference_sha256=refhash,original_reference_seal_sha256=sha(args.reference_seal),
        config_sha256=sha(args.config),plan_sha256=sha(args.plan_go),source_sha256={**F.source_bindings(),str(Path(__file__)):sha(__file__)},
        rows_sha256=sha(out/'B_REFERENCE_SUPPORT_ROWS.jsonl.gz'),summary_sha256=sha(out/'B_REFERENCE_SUPPORT_AUDIT.json'),
        production_feature_or_candidate_insertion=False,new_solve_calls=0,new_PnP_calls=0,
        new_model_forward_calls=0,new_candidate_calls=0,reference_access='separate process diagnostic only'))


if __name__=='__main__':main()
