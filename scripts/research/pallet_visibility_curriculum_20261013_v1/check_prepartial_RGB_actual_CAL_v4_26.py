"""Actual CAL integration, before any v26 primary predictions or metrics."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np

from . import evaluate_prepartial_RGB_preservation_v4_26 as V

C=V.C
OUTPUT=V.OUTPUT.parent/'prepartial_RGB_actual_CAL_checks_v4_26'


def main(*,smoke=True):
    if OUTPUT.exists():raise FileExistsError('Preserve actual CAL integration evidence')
    audit=C.read(V.OUTPUT.parent/'prepartial_RGB_source_audit_v4_26/AUDIT.json')
    eligible=[r for r in audit['partial_records']if r['partition']=='CAL'and r['nonfallback_prepartial_restore_candidate']]
    ids=list(dict.fromkeys(next(r['id']for r in eligible if r['state']==state)for state in('B','I')))
    protocol=C.read(V.PREVIOUS.OUTPUT/'PROTOCOL.json')
    tasks={}
    for row in C.rows(V.PREVIOUS.OUTPUT/'INPUTS.jsonl.gz'):
        if row['partition']!='CAL':break
        if row['id']in ids:tasks[row['id']]=row
    if set(tasks)!=set(ids):raise ValueError('Two actual CAL B/I restoration examples required')
    models=V._verified_models();checks=[];maximum=0.
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'PLAN.json',dict(ids=ids,source_sha256=C.sha(__file__),driver_sha256=C.sha(V.__file__),
        helper_sha256=C.sha(V.RESTORE.__file__),CAL_only=True,new_primary_reference_reads=0,
        model_sha256=C.sha(V.RESTORE_CAL.OUTPUT/'MODEL.json'),finite_difference_step=1e-6))
    for ident in ids:
        task=tasks[ident];regions,oparams,original_S=V.OLD._prior_inputs(task,models)
        priors={mode:V._prepare_pose_prior(task,models,mode)for mode in('FULL','TRANSLATION')}
        restored=V._restored_ids(task)
        for family_name,family in task['families'].items():
            spec=task['spec']
            original=V.OLD.BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],
                spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],
                config={**spec['config'],'prior_weight':0.},boundary_observations=task['boundary_merged'],
                u_regions=task['u_regions'],edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
            for profile,mode,containment,_ in V.PROFILES:
                base,bundle=V._problem(task,family,models)
                rebound=V.OLD.P.S.rebind_bounded_visibility_trial_geometry(original_S,family['native_xyz'])
                bounded=V.OLD.P.BoundedGeometricPriorsProblem(base,regions,oparams,rebound)
                bound=V.PRIOR.rebind_coarse_pose_prior(priors[mode],family['native_xyz'],family['physical_registration_Q'])
                regularized=V.PRIOR.CoarsePosePriorProblem(bounded,bound)
                cfg=V.BASE.OUTLINE._outline_config(task['outline']['configuration'])
                problem=V.P.ContinuousContainmentProblem(regularized,task['outline'],cfg)if containment else regularized
                coarse=family['coarse_pose']
                p=V.BASE.NATIVE.V._pose_vector(np.asarray(coarse['R']),np.asarray(coarse['t']),problem.scale)
                p=p+np.array([.0007,-.0003,.0005,.0008,-.0004,.0011])
                value,gradient,parts=problem.objective(p)
                fd=np.zeros(6)
                for j in range(6):
                    step=np.zeros(6);step[j]=1e-6
                    fd[j]=(problem.objective(p+step)[0]-problem.objective(p-step)[0])/2e-6
                relative=float(np.max(np.abs(fd-gradient)/np.maximum(1.,np.abs(fd))))
                maximum=max(maximum,relative)
                if relative>2e-5:raise ValueError(('Actual objective derivative mismatch',ident,profile,relative))
                for sides in original.sides:
                    a,ja=original.constraints(p,sides);b,jb=problem.constraints(p,sides)
                    if not np.array_equal(a,b)or not np.array_equal(ja,jb):raise ValueError('Original hard guards changed')
                costs=parts['native_costs'];counts=Counter(r['native_corner']for r in costs if r['kind']=='COMPLETE_BI')
                if any(counts[k]!=1 for k in restored):raise ValueError('Restored RGB q must enter exactly once')
                if any(r['native_corner']in restored for r in costs
                        if r['kind']in('UNREPLACED_PARTIAL_NORMAL','UNCERTAIN_ARCHIVED_TANGENT')):
                    raise ValueError('Restored owner old normal/tangent duplicated')
                plan=base.plan
                if any(o['native_corner']in restored for g in plan['active_groups']for o in g['activation_owners']):
                    raise ValueError('Restored owner still activates an edge cost')
                if mode=='FULL'and(parts['CAL_coarse_pose_prior_objective']!=0.
                        or parts['CAL_coarse_pose_prior_audit']['factor_weight']!=0.):raise ValueError('noP control active')
                checks.append(dict(id=ident,family=family_name,profile=profile,restored=restored,
                    FD_max_relative_error=relative,objective=float(value),hard_guards_and_jacobian_exact=True,
                    restored_q_exactly_once=True,restored_own_normal_tangent_edge_owner_removed=True))
        if smoke:
            print('CAL smoke '+ident,flush=True)
            row=V._run(task,models)
            if set(row['arms'])!=set(V.ARMS)or row['input_sha256']!=V.E._digest(task):raise ValueError('Seven-arm smoke/input mismatch')
            if row['arms'][V.ARMS[0]]!=task['baseline']:raise ValueError('Frozen e changed')
            C.write(OUTPUT/(ident.replace(':','__')+'_SMOKE.json'),row)
    for key in('source_sha256','input_sha256','protected_user_sha256'):V.E._verify_sources(protocol[key])
    C.write(OUTPUT/'ACTUAL_CAL_INTEGRATION_CHECKS_V1.json',dict(rows=len(ids),cases=ids,checks=checks,
        check_count=len(checks),failure_count=0,maximum_FD_relative_error=maximum,
        all_original_guards_and_jacobians_exact=True,full7arm_smoke_per_case=smoke,
        new_primary_numeric_reference_reads=0,new_pixel_NN_or_measurement_queries=0,
        smoke_is_CAL_not_primary_metric_selection=True,source_sha256=C.sha(__file__),
        driver_sha256=C.sha(V.__file__),helper_sha256=C.sha(V.RESTORE.__file__)))
    C.write(OUTPUT/'CHECK_SEAL.json',dict(report_sha256=C.sha(OUTPUT/'ACTUAL_CAL_INTEGRATION_CHECKS_V1.json'),
        plan_sha256=C.sha(OUTPUT/'PLAN.json'),assets={p.name:C.sha(p)for p in OUTPUT.glob('*_SMOKE.json')}))
    print(dict(CAL_cases=ids,checks=len(checks),maximum_FD_relative_error=maximum,primary_metrics_read=0),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-smoke',action='store_true')
    main(smoke=not parser.parse_args().no_smoke)
