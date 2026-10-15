"""First CAL frame with the actual separately fitted v29 model exporters.

This is an API, gradient and five-arm wiring smoke, never a parameter search.
It reads no reference labels, RGB, primary predictions or scores. The fixed
first original CAL input and its complete fresh observation are used once.
"""
from pathlib import Path
import argparse
import time
import numpy as np

from . import evaluate_geometry_guided_edges_v4_29 as M

C,E,G,PREP=M.C,M.E,M.G,M.PREP
OUTPUT=PREP.B/'geometry_guided_actual_runtime_smoke_v4_29'


def main(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every actual-model smoke')
    task=next(iter(C.rows(PREP.TASKS/'INPUTS.jsonl.gz')))
    row=next(iter(C.rows(PREP.OBS/'CAL_OBSERVATIONS.jsonl.gz')))
    PREP.validate_join(task,row)
    if task['partition']!='CAL':raise ValueError('The fixed first original CAL frame is required')
    task['v29_observation']=row['observation'];task['v29_initial_geometry']=row['initial_geometry']
    before=E._digest(task);models=M._models()
    sources={str(Path(module.__file__).resolve().relative_to(C.ROOT)):C.sha(module.__file__)
             for module in(M,G,M.AUTH,PREP,M.CAL,M.CAL.M,M.BOUNDED,M.CONTAIN)}
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(__file__)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in(
        PREP.TASKS/'INPUTS.jsonl.gz',PREP.TASKS/'INPUT_SEAL.json',
        PREP.OBS/'CAL_OBSERVATIONS.jsonl.gz',PREP.OBS/'CAL_OBSERVATION_SEAL.json',
        M.CAL.OUTPUT/'PREPARATION_SEAL.json',M.CAL.OUTPUT/'MODEL_SEAL.json',PREP.DECLARATION)}
    for relative,digest in C.read(M.CAL.OUTPUT/'MODEL_SEAL.json')['assets'].items():
        inputs[str((M.CAL.OUTPUT/relative).relative_to(C.ROOT))]=digest
    output.mkdir(parents=True)
    C.write(output/'BEFORE_SMOKE_BINDING.json',dict(id=task['id'],task_digest=before,
        source_sha256=sources,input_sha256=inputs,actual_policy_model_parameters=models['V29'],
        reference_or_primary_prediction_or_score_decodes=0,new_RGB_queries=0,
        parameter_selection_or_model_refits=0,fixed_first_CAL_only=True))
    checks=[];records=[];started=time.monotonic()
    def check(passed,reason,**fields):checks.append(dict(passed=bool(passed),reason=reason,**fields))
    try:
        regions,oparams,original_S=M.OLD._prior_inputs(task,models)
        cfg=M.BASE.OUTLINE._outline_config(task['outline']['configuration'])
        for policy in G.SOURCE_POLICIES:
            check(models['V29'][policy]['available'],'actual policy-specific fitted models available',policy=policy)
            for name,family in task['families'].items():
                core,bundle=M._problem(task,family,models,policy)
                rebound=M.BOUNDED.S.rebind_bounded_visibility_trial_geometry(original_S,family['native_xyz'])
                bounded=M.BOUNDED.BoundedGeometricPriorsProblem(core,regions,oparams,rebound)
                p=G.V._pose_vector(np.asarray(family['coarse_pose']['R']),np.asarray(family['coarse_pose']['t']),core.scale)
                for contain in(False,True):
                    problem=M.CONTAIN.ContinuousContainmentProblem(bounded,task['outline'],cfg)if contain else bounded
                    value,gradient,parts=problem.objective(p)
                    numeric=G.V._jacobian(lambda x:np.array([problem.objective(x)[0]]),p)[0]
                    values,jac=problem.constraints(p,problem.sides[0])
                    guard_numeric=G.V._jacobian(lambda x:problem.constraints(x,problem.sides[0])[0],p)
                    R,t=G.V._pose(p,core.scale);actual=problem.evaluate(R,t)
                    expected=(actual['native_observation_objective']+actual['regularizer_objective']+
                        (cfg['outline_factor_weight']*actual.get('global_containment_cost',0.)if contain else 0.))
                    common=dict(policy=policy,family=name,containment=contain)
                    check(np.allclose(gradient,numeric,atol=2e-4,rtol=4e-5),'actual exported model objective FD',**common)
                    check(np.allclose(jac,guard_numeric,atol=3e-4,rtol=4e-5),'actual exported model guard FD',**common)
                    check(np.isclose(value,actual['objective'],atol=1e-8,rtol=1e-10),'objective/evaluate equality',**common)
                    check(np.isclose(expected,value,atol=1e-8,rtol=1e-10),'native plus O/S plus C exactly once',**common)
                    records.append(dict(**common,objective=value,accepted=actual['accepted'],
                        active_physical_edges=len(core.groups),new_owner_ids=core.plan['new_owner_ids'],
                        maximum_objective_FD_difference=float(np.max(np.abs(gradient-numeric))),
                        maximum_guard_FD_difference=float(np.max(np.abs(jac-guard_numeric)))))
        run_start=time.monotonic();prediction=M._run(task,models);run_elapsed=time.monotonic()-run_start
        check(list(prediction['arms'])==list(M.ARMS),'exact five declared arms')
        check(prediction['arms'][M.ARMS[0]]==task['baseline'],'frozen e exact comparison alias')
        check(E._digest(task)==before,'input remains unchanged')
        for arm in M.ARMS[1:]:
            result=prediction['arms'][arm]
            check(not result['fallback_audit']['e_numeric_pose_used'],'no numeric e prior or fallback',arm=arm)
            check(result['keypoint_readout']['S_image_or_2D_terms']==0,'S model-only readout',arm=arm)
            check(result['keypoint_readout']['U_range_only'],'U range-only readout',arm=arm)
            if result['accepted']:
                chosen=next(r for r in result['candidate_selection']['candidates']
                            if r['name']==result['candidate_selection']['selected_name'])
                check(np.isclose(chosen['total_energy'],result['solution']['objective'],atol=1e-9,rtol=1e-10),
                      'accepted selected current objective matches selector',arm=arm)
        for mapping in(sources,inputs):E._verify_sources(mapping)
        C.write(output/'PREDICTION.json',prediction)
        report=dict(id=task['id'],checks=len(checks),failure_count=sum(not r['passed']for r in checks),
            failures=[r for r in checks if not r['passed']],records=records,
            five_arm_CAL_smoke_elapsed_seconds=run_elapsed,total_elapsed_seconds=time.monotonic()-started,
            arms={a:dict(accepted=prediction['arms'][a].get('accepted',prediction['arms'][a]['actual_pose'].get('accepted')),
                family=prediction['arms'][a].get('WD_family'),
                objective=prediction['arms'][a].get('solution',{}).get('objective'))for a in M.ARMS},
            before_binding_sha256=C.sha(output/'BEFORE_SMOKE_BINDING.json'),
            source_sha256=sources,input_sha256=inputs,reference_or_primary_score_decodes=0,
            new_RGB_queries=0,parameter_selection_or_model_refits=0,default_adopted=False)
        C.write(output/'RESULT.json',report)
        C.write(output/'SEAL.json',dict(before_binding_sha256=C.sha(output/'BEFORE_SMOKE_BINDING.json'),
            result_sha256=C.sha(output/'RESULT.json'),prediction_sha256=C.sha(output/'PREDICTION.json')))
        print(dict(id=task['id'],checks=report['checks'],failures=report['failure_count'],
                   elapsed_seconds=run_elapsed,arms=report['arms']),flush=True)
        if report['failure_count']:raise SystemExit(1)
    except Exception as error:
        C.write(output/'FAILURE.json',dict(exception=type(error).__name__,reason=str(error),
            elapsed_seconds=time.monotonic()-started,completed_checks=checks,records=records,
            source_or_model_edits_performed=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    main(parser.parse_args().output)
