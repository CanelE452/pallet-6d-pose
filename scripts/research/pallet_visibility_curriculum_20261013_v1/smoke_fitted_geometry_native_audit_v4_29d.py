"""Independent native costs at the first CAL smoke's fitted-model endpoints.

Only the fixed first CAL input/observation and the already saved smoke poses
are decoded. The new consumer/planner/objective is not called. This checks
native observation costs/support, not O/S/C costs or full-population accuracy.
"""
from pathlib import Path

import numpy as np

from . import audit_geometry_guided_edges_v4_29 as A

C=A.C
B=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
TASKS=B/'coarse_prior_containment_eval_v4_25'
OBS=B/'geometry_guided_edge_observations_v4_29b'
SMOKE=B/'geometry_guided_actual_runtime_smoke_v4_29'
OUTPUT=B/'geometry_guided_fitted_native_independent_smoke_v4_29d'


def main(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every independent actual-model smoke')
    # The existing smoke binds the exact original task path; avoid guessing a
    # similarly named experiment directory or reading any primary rows.
    before=C.read(SMOKE/'BEFORE_SMOKE_BINDING.json');smoke=C.read(SMOKE/'RESULT.json')
    smoke_seal=C.read(SMOKE/'SEAL.json')
    if (smoke['failure_count']!=0 or C.sha(SMOKE/'RESULT.json')!=smoke_seal['result_sha256']
            or C.sha(SMOKE/'BEFORE_SMOKE_BINDING.json')!=smoke_seal['before_binding_sha256']
            or C.sha(SMOKE/'PREDICTION.json')!=smoke_seal['prediction_sha256']):
        raise ValueError('Successful sealed actual-model runtime smoke required')
    task_paths=[C.ROOT/p for p in before['input_sha256']if p.endswith('/INPUTS.jsonl.gz')]
    if len(task_paths)!=1:raise ValueError('Exactly one frozen original physics task journal required')
    task=next(iter(C.rows(task_paths[0])));row=next(iter(C.rows(OBS/'CAL_OBSERVATIONS.jsonl.gz')))
    if (task['partition']!='CAL' or row['partition']!='CAL'
            or task['id']!=row['id'] or task['id']!=smoke['id'] or task['id']!=before['id']):
        raise ValueError('Only fixed first original CAL frame allowed')
    task['v29_observation']=row['observation'];task['v29_initial_geometry']=row['initial_geometry']
    prediction=C.read(SMOKE/'PREDICTION.json');parameters=before['actual_policy_model_parameters']
    sources={str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__)
             for m in (A,A.KERNEL,A.N)}
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(__file__)
    inputs=dict(before['input_sha256'])
    for p in(SMOKE/'BEFORE_SMOKE_BINDING.json',SMOKE/'RESULT.json',SMOKE/'SEAL.json',SMOKE/'PREDICTION.json'):
        inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
    for mapping in(sources,inputs):
        for p,digest in mapping.items():
            if C.sha(C.ROOT/p)!=digest:raise ValueError('Frozen actual-model smoke source/input changed: '+p)
    output.mkdir(parents=True)
    C.write(output/'BEFORE_BINDING.json',dict(id=task['id'],source_sha256=sources,input_sha256=inputs,
        reference_RGB_primary_rows_and_consumer_objective_calls=0))
    checks=[];records=[];raws={}
    def check(passed,reason,arm):checks.append(dict(passed=bool(passed),reason=reason,arm=arm))
    for arm,result in prediction['arms'].items():
        if arm=='SAME_FROZEN_E':continue
        family=task['families'][result['WD_family']];policy=result['source_policy'];params=parameters[policy]
        check(result['accepted'],'first fixed CAL arm is feasible',arm)
        if not result['accepted']:continue
        check(params['available'],'actual separately fitted policy parameters available',arm)
        if not params['available']:continue
        R=np.asarray(result['actual_pose']['R_cf']);t=np.asarray(result['actual_pose']['centroid'])
        check(np.allclose(np.asarray(family['native_xyz']).mean(0),0.,atol=1e-12,rtol=0),
              'legacy centroid stores native t; current native model is centered',arm)
        if result['WD_family']not in raws:raws[result['WD_family']]=A.prepare_raw(task,family)
        plan=result['solution']['v29_cost_plan'];gamma=params['tangent']['gamma']
        for passed,reason in A.observation_checks(task,row['observation'],plan,params['noise']):check(passed,reason,arm)
        fresh=A.recompute_native(task,family,R,t,plan,gamma,raw=raws[result['WD_family']])
        for passed,reason in fresh['checks']:check(passed,reason,arm)
        expected=result['solution']['native_observation_objective'];gap=abs(float(fresh['objective'])-float(expected))
        check(gap<=1e-8*max(1.,abs(expected)),'independent native scalar equals selected fitted-model scalar',arm)
        check(fresh['accepted'],'independent native original/new support guards pass',arm)
        manual=A.cost_buckets(fresh['native_costs']);saved=A.cost_buckets(result['solution']['native_costs'])
        check(manual.keys()==saved.keys() and all(np.isclose(v,saved[k],atol=1e-8,rtol=1e-10)
              for k,v in manual.items()),'all native factor identities and costs agree',arm)
        records.append(dict(arm=arm,source_policy=policy,native=fresh['objective'],saved_native=expected,
            absolute_difference=gap,owners=fresh['owner_ids'],minimum_support_slack=fresh['minimum_actual_support_slack']))
    check(len(records)==4,'all four new arms independently checked',None)
    report=dict(id=task['id'],checks=len(checks),failure_count=sum(not c['passed']for c in checks),
        failures=[c for c in checks if not c['passed']],records=records,source_sha256=sources,input_sha256=inputs,
        independent_native_scalar_and_support=True,independent_O_S_C_or_population_audit=False,
        new_RGB_reference_primary_or_optimizer_calls=0,full238_independent_audit_still_required=True)
    C.write(output/'RESULT.json',report)
    C.write(output/'SEAL.json',dict(before_binding_sha256=C.sha(output/'BEFORE_BINDING.json'),result_sha256=C.sha(output/'RESULT.json')))
    print(dict(id=task['id'],checks=report['checks'],failures=report['failure_count'],
        max_difference=max(r['absolute_difference']for r in records)),flush=True)
    if report['failure_count']:raise SystemExit(1)


if __name__=='__main__':main()
