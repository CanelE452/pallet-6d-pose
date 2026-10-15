"""First completed CAL frame: actual support, independent math and FD only.

Engineering variances exercise API/arithmetic; they are not fitted noise or
runtime parameters. No RGB/reference decode, pose fit, score or frame search.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import numpy as np

from . import common as C
from . import prepare_geometry_guided_native_evidence_v4_29 as P
from . import geometry_guided_edge_problem_v4_29 as G
from . import geometry_guided_edge_authentication_v4_29 as AUTH
from . import audit_geometry_guided_edges_v4_29 as A


def main():
    out=P.B/'geometry_guided_independent_actual_smoke_v4_29'
    if out.exists():raise FileExistsError('Preserve every actual arithmetic smoke')
    task=next(iter(C.rows(P.TASKS/'INPUTS.jsonl.gz')))
    if task['partition']!='CAL':raise ValueError('First original CAL frame only')
    spool=P.OBS/'row_spool'/(hashlib.sha256(task['id'].encode()).hexdigest()+'.jsonl.gz')
    row=next(iter(C.rows(spool)));P.validate_join(task,row)
    noise=dict(normal_variance_component_px2=2.,angular_variance_component_sine2=.001)
    gamma=.2;records=[];failures=[];checks=0
    def check(passed,reason,**fields):
        nonlocal checks
        checks+=1
        if not passed:failures.append(dict(reason=reason,**fields))
    for name,family in task['families'].items():
        base=P.native_baseline(task,family)
        bundle=AUTH.authenticate_geometry_guided_edges(base,row['observation'],
            image_binding=dict(sha256=task['image_sha256'],image_hw=task['spec']['image_hw']),
            initial_geometry=row['initial_geometry'],noise_provenance=dict(fit_partition='CAL',
                observation_family=G.OBSERVATION_FAMILY,mode='ENGINEERING_SMOKE_VALUES_NOT_FITTED_MODEL'),
            calibration_preparation=True,**noise)
        for policy in G.SOURCE_POLICIES:
            problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy=policy,tangent_gamma=gamma)
            for passed,reason in A.observation_checks(task,row['observation'],problem.plan,noise):
                check(passed,reason,family=name,policy=policy)
            p0=G.V._pose_vector(np.asarray(family['coarse_pose']['R']),np.asarray(family['coarse_pose']['t']),base.scale)
            for index,perturb in enumerate(([0.]*6,[.004,-.003,.002,.003,-.002,.007])):
                p=p0+perturb;R,t=G.V._pose(p,base.scale)
                value,gradient,parts=problem.objective(p)
                independent=A.recompute_native(task,family,R,t,problem.plan,gamma,raw=base.guard_base)
                for passed,reason in independent['checks']:check(passed,reason,family=name,policy=policy)
                check(np.isclose(value,independent['objective'],atol=1e-9,rtol=1e-10),'manual actual native cost',family=name,policy=policy)
                check(problem.evaluate(R,t)['accepted']==independent['accepted'],'manual actual guard acceptance',family=name,policy=policy)
                if index==0:
                    numeric=G.V._jacobian(lambda x:np.array([problem.objective(x)[0]]),p)[0]
                    values,jac=problem.constraints(p,problem.sides[0])
                    guard_numeric=G.V._jacobian(lambda x:problem.constraints(x,problem.sides[0])[0],p)
                    check(np.allclose(gradient,numeric,atol=2e-4,rtol=4e-5),'actual objective FD',family=name,policy=policy)
                    check(np.allclose(jac,guard_numeric,atol=3e-4,rtol=4e-5),'actual guard FD',family=name,policy=policy)
                    records.append(dict(family=name,policy=policy,active_edges=len(problem.groups),owners=problem.plan['new_owner_ids'],
                        objective=value,independent_objective=independent['objective'],
                        maximum_objective_FD_difference=float(np.max(np.abs(gradient-numeric))),
                        maximum_guard_FD_difference=float(np.max(np.abs(jac-guard_numeric)))))
    out.mkdir()
    report=dict(id=task['id'],checks=checks,failure_count=len(failures),failures=failures,records=records,
        exact_actual_observation_spool=dict(path=str(spool.relative_to(C.ROOT)),sha256=C.sha(spool)),
        original_task_digest=A._digest(task),engineering_variance_exercise_only=dict(**noise,tangent_gamma=gamma),
        these_are_not_CAL_fitted_or_runtime_parameters=True,new_RGB_or_reference_decodes=0,new_pose_fits_or_scores=0,
        source_sha256={str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__) for m in (P,G,AUTH,A)},
        smoke_source_sha256=C.sha(__file__),default_adopted=False)
    C.write(out/'RESULT.json',report);C.write(out/'SEAL.json',dict(result_sha256=C.sha(out/'RESULT.json')))
    print(dict(id=task['id'],checks=checks,failures=len(failures),records=records),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
