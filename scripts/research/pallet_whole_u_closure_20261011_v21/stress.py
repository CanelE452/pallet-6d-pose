"""One fixed natural synthetic stress audit; ROOT alone freezes and runs it.

No mock candidate, refit, rank or convergence result is injected. Synthetic
correspondence correctness means construction from known cuboid R,t, not real
visibility or physical ownership. Unavailable poses have no fabricated error.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


PRIVATE = Path(__file__).resolve().parent
CPU = PRIVATE/'run_cpu.py'
CPU_SHA = 'a2b943464a0c9b489099c9ea7fe4563641d99751d39dce3ca72fd7349e5491fe'
SOLVER_SHA = '1603401ed1f7b1cde301b272bfc1f2f6d3c3935f371ba9c7bafd56e2a893d099'
TEST_SHA = 'd97b927ae0c68f06c59bc035a2f797348b2ec150867ecd3abbcf0ad01b1ac003'
INSTRUMENT_SOURCE = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19/run_geometry.py')
INSTRUMENT_SHA = 'd964a806959951b2b987094b3a7d290cc587a65ce5c583281d8a4e5b33c0c302'
EMPTY = dict(corners=[],lines=[],queries=[])
SIGNS = ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
         (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
PLAN = (
    dict(id='BASE_CORRECT_8',H=[],bad_offsets={},control=None),
    dict(id='GOOD_WRONGLY_REMOVED_1',H=[7],bad_offsets={},control=None),
    dict(id='GOOD_WRONGLY_REMOVED_2',H=[6,7],bad_offsets={},control=None),
    dict(id='BAD_RETAINED_1',H=[],bad_offsets={'2':[60.,-60.]},control=None),
    dict(id='BAD_RETAINED_2',H=[],bad_offsets={'2':[60.,-60.],'5':[-50.,70.]},control=None),
    dict(id='BOTH_ERRORS_1',H=[7],bad_offsets={'2':[60.,-60.]},control=None),
    dict(id='BOTH_ERRORS_2',H=[6,7],bad_offsets={'2':[60.,-60.],'5':[-50.,70.]},control=None),
    dict(id='INSUFFICIENT_U3',H=[3,4,5,6,7],bad_offsets={},control='INSUFFICIENT_POINT_OBSERVATIONS'),
    dict(id='IMAGE_COLLINEAR_U4',H=[4,5,6,7],bad_offsets={},control='DEGENERATE_POINT_OBSERVATIONS'))
LIMITS = dict(synthetic_only=True,real_rows_RGB_models_annotations_training_loaded=0,
    masks_are_wrong_correspondence_exclusion_not_physical_visibility_truth=True,
    exact_frozen_fixture_square_True_noise_True=True,new_seed_or_parameter_sweep=False,
    normal_solver_paths_only=True,mocked_refits_candidates_rank_success=0,
    truth_Rt_used_as_solver_seed_or_fallback=False,automatic_retry=False,
    no_accuracy_threshold_or_performance_tuning=True,old_V16_full_scalar_FAIL_remains_FAIL=True,
    diagnostic_PASS_means_execution_input_counter_and_restoration_contract_not_pose_success=True,
    case_quality_is_injected_ID_truth_not_pose_inlier_membership=True,
    ADDsym_sample='eight fixed cuboid vertices; nearest transformed true vertex mean',
    true_rotation_error_is_direct_physical_axis_error_not_symmetry_minimized=True,
    unavailable_pose_errors=None,global_root_completeness_proven=False,
    elapsed_is_verification_work_not_pipeline_latency=True)


def require(value,message):
    if not value: raise ValueError(message)


def cpu_helpers():
    raw=CPU.read_bytes();require(hashlib.sha256(raw).hexdigest()==CPU_SHA,'immutable CPU driver')
    tree=ast.parse(raw);names=('binding','write','git_snapshot','instrument_tree','load_module')
    selected=[]
    for name in names:
        found=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(found)==1,'unique unchanged CPU helper '+name);selected.append(copy.deepcopy(found[0]))
    module=ast.Module(body=selected,type_ignores=[])
    h=dict(Path=Path,hashlib=hashlib,json=json,os=os,subprocess=subprocess,sys=sys,
        ast=ast,copy=copy,importlib=importlib,require=require,
        INSTRUMENT_SOURCE=INSTRUMENT_SOURCE,INSTRUMENT_SHA=INSTRUMENT_SHA)
    exec(compile(ast.fix_missing_locations(module),str(CPU)+':unchanged stress lifecycle helpers','exec'),h)
    return h,dict(source=h['binding'](CPU),functions=list(names),whole_function_AST_changes=0,
        canonical_AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest())


def fixture_tree(path):
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==TEST_SHA,'immutable synthetic fixture source')
    tree=ast.parse(raw);selected=[]
    for name in ('rotation','project','fixture'):
        found=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(found)==1,'unique original fixture function '+name);selected.append(copy.deepcopy(found[0]))
    module=ast.Module(body=selected,type_ignores=[])
    fixture=selected[-1]
    signs=[n.value for n in ast.walk(fixture) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='signs' for t in n.targets)]
    require(len(signs)==1 and isinstance(signs[0],ast.Call)
        and ast.literal_eval(signs[0].args[0])==[list(v) for v in SIGNS],'cuboid ID layout from exact fixture source')
    return module,dict(source_sha256=TEST_SHA,function_names=['rotation','project','fixture'],
        whole_function_AST_changes=0,canonical_AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),
        square=True,noise=True,fixture_model_signs=[list(v) for v in SIGNS])


def scalar_rotation(r):
    # Independent standard-library Rodrigues calculation; no cv2/numpy call.
    theta=math.sqrt(math.fsum(float(v)**2 for v in r));x,y,z=map(float,r)
    W=((0.,-z,y),(z,0.,-x),(-y,x,0.));I=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    if theta==0.:return [list(v) for v in I]
    A=math.sin(theta)/theta;B=(1.-math.cos(theta))/(theta*theta)
    return [[I[i][j]+A*W[i][j]+B*math.fsum(W[i][k]*W[k][j] for k in range(3))
             for j in range(3)] for i in range(3)]


def true_errors(result,data):
    if not result['available']:return None
    R=result['R_physical'];t=result['centroid'];R0=scalar_rotation(data['z'][:3]);t0=data['z'][3:]
    require(all(math.isfinite(float(v)) for row in R for v in row)
        and all(math.isfinite(float(v)) for v in t),'finite approved synthetic pose')
    trace=math.fsum(float(R[i][j])*R0[i][j] for i in range(3) for j in range(3))
    rotation=math.degrees(math.acos(max(-1.,min(1.,(trace-1.)/2.))))
    def transform(matrix,translation):
        return [[math.fsum(float(matrix[i][j])*float(p[j]) for j in range(3))+float(translation[i])
                 for i in range(3)] for p in data['model']]
    observed,truth=transform(R,t),transform(R0,t0)
    return dict(translation_cm=100.*math.dist(t,t0),rotation_deg=rotation,
        ADDsym_cm=100.*math.fsum(min(math.dist(p,q) for q in truth) for p in observed)/len(observed),
        ADDsym_sample_count=8,reference='ANALYTIC_SYNTHETIC_CUBOID_Rt',units=dict(translation='cm',rotation='degree',ADDsym='cm'))


def layout(ids):
    return dict(ids=list(ids),fixed_model_xyz_signs={str(k):list(SIGNS[k]) for k in ids},
        common_fixed_axes=[axis for axis in range(3) if ids and len({SIGNS[k][axis] for k in ids})==1],
        layout_is_combinatorial_fixed_fixture_ID_face_incidence_not_new_rank=True)


def counters_delta(before,after):
    return {key:{name:value-before.get(key,{}).get(name,0) for name,value in values.items()}
            for key,values in after.items()}


def json_value(value,np):
    if isinstance(value,dict):return {str(k):json_value(v,np) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v,np) for v in value]
    if isinstance(value,np.ndarray):return json_value(value.tolist(),np)
    if isinstance(value,np.generic):return json_value(value.item(),np)
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def case_record(plan,data,source,calls):
    np=source.np;points=data['points'].copy();bad=sorted(map(int,plan['bad_offsets']))
    for key,delta in plan['bad_offsets'].items():points[int(key)]+=delta
    if plan['id']=='IMAGE_COLLINEAR_U4':
        for k in range(4):points[k]=[260.+12.*k,230.]
        bad=list(range(4))
    H=list(plan['H']);U=[k for k in range(8) if k not in H]
    correct=[k for k in U if k not in bad];before=copy.deepcopy(calls);original=points.copy()
    record=dict(id=plan['id'],plan=copy.deepcopy(plan),input_points=points.tolist(),K=data['K'].tolist(),xyz=data['xyz'].tolist(),
        analytic_truth=dict(rvec=data['z'][:3].tolist(),centroid=data['z'][3:].tolist(),model=data['model'].tolist()),
        hidden_good_removed=H,bad_coordinates_retained=bad,used=U,remaining_correct=layout(correct),
        wrong_removed_count=len(H),bad_retained_count=len(bad),remaining_correct_count=len(correct),synthetic_correspondence_truth_only=True,
        no_baseline_fallback_supplied=True,exception=None,contract_failures=[])
    try:
        bank=source.P4.PoseBank(points,data['K'],data['xyz'],image_size=(640,480))
        subject=source.PointClosureSolver(bank)
        result=subject.solve(U,hidden=H,observation=copy.deepcopy(EMPTY),
            initial_pose=dict(available=False),point_result=dict(available=False))
        approved=bool(result['available']);fit=list(result.get('fit_input_ids',[])) if approved else []
        final=list(result.get('final_inliers',[])) if approved else []
        best=result.get('best_candidate');issues=record['contract_failures']
        if not np.array_equal(points,original):issues.append('INPUT_MUTATED')
        if result['used']!=U or result.get('fit_line_edges',[]):issues.append('U_OR_POINT_ONLY_CONTRACT')
        for group in ('proposal_solutions','all_candidate_solutions'):
            if any(set(c.get('fit_input_ids',())).intersection(H+[8]) for c in result.get(group,[])):
                issues.append('H_OR_CENTER_IN_FIT:'+group)
        if approved:
            if len(fit)<4 or not set(fit)<=set(final) or not set(final)<=set(U):issues.append('ACTUAL_FIT_OR_FINAL_INLIER_CONTRACT')
            output=np.asarray(result['points_final']);untouched=[k for k in range(9) if k not in H]
            if not np.array_equal(output[untouched],original[untouched]):issues.append('NON_H_OUTPUT_CHANGED')
            if H and not np.array_equal(output[H],np.asarray(result['projected'])[H]):issues.append('H_NOT_FINAL_PROJECTION')
        assignments=result['operation_counts']['hidden_reprojection_assignments']
        if assignments!=(len(H) if approved else 0):issues.append('H_ASSIGNMENT_COUNT')
        if result.get('reprojected_points_reused_as_observations',False):issues.append('H_REPROJECTION_REUSED')
        if plan['control'] and (approved or result['state']!=plan['control']):issues.append('EXPECTED_TYPED_CONTROL')
        record.update(NEW=approved,fallback=False,failed=not approved,state=result['state'],reason=result['reason'],
            true_pose_errors=true_errors(result,data),actual_fit_ids=fit,actual_final_inlier_ids=final,
            correct_actual_fit_ids=[k for k in fit if k not in bad],bad_actual_fit_ids=[k for k in fit if k in bad],
            correct_actual_final_ids=[k for k in final if k not in bad],bad_actual_final_ids=[k for k in final if k in bad],
            actual_fit_layout=layout(fit),actual_final_layout=layout(final),
            stored_best_is_not_approved_pose=not approved and best is not None,
            selected_hypothesis=result.get('selected_hypothesis') if approved else None,
            dimension_index=result.get('dimension_index') if approved else None,
            approved_geometry=result.get('point_line_geometry') if approved else None,
            stable_incidence_guard=result.get('stable_incidence_guard') if approved else None,
            point_bank_operation_counts=dict(bank.ledger),raw_solver=json_value(result,np))
    except BaseException as error:
        record.update(NEW=False,fallback=False,failed=True,
            exception=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),true_pose_errors=None)
    record['actual_numerical_calls']=counters_delta(before,calls)
    return json_value(record,np)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','run'))
    for name in ('repo','source-root','baseline-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--original-repo',type=Path);parser.add_argument('--output',type=Path,default=PRIVATE/'stress')
    a=parser.parse_args();repo=a.repo.resolve();original=(a.original_repo or a.source_root).resolve();out=a.output.resolve()
    require(out==PRIVATE/'stress' and not any(p.is_symlink() for p in (a.output,*a.output.parents)),'exclusive nonsymlink stress output')
    for protected in (repo,original):require(not out.is_relative_to(protected) and not protected.is_relative_to(out),'protected repo disjoint')
    require(a.source_root.is_dir() and a.baseline_root.is_dir(),'explicit source and baseline roots')
    h,helper_proof=cpu_helpers();bind,write=h['binding'],h['write']
    paths=dict(solver=PRIVATE/'solver.py',tests=PRIVATE/'test_solver.py',CPU_driver=CPU,stress=Path(__file__).resolve(),instrumentation_source=INSTRUMENT_SOURCE)
    for p in sorted((repo/'scripts/research').rglob('*.py')):paths['source:'+str(p.relative_to(repo))]=p
    require(any(k.startswith('source:') for k in paths),'public production Python closure')
    inputs={k:bind(p) for k,p in paths.items()};require(inputs['solver']['sha256']==SOLVER_SHA,'frozen solver unchanged')
    fixture,fixture_proof=fixture_tree(paths['tests']);instrument,instrument_proof=h['instrument_tree']()
    environment=dict(PALLET_SOURCE_ROOT=str(a.source_root.resolve()),PALLET_BASELINE_ROOT=str(a.baseline_root.resolve()))
    protection=dict(research=h['git_snapshot'](repo),original=h['git_snapshot'](original))
    authority=dict(inputs=inputs,plan=list(PLAN),limits=LIMITS,environment=environment,protection=protection,
        repo=str(repo),original_repo=str(original),CPU_helpers=helper_proof,fixture_source_proof=fixture_proof,
        instrumentation_source_proof=instrument_proof,source_frozen_before_scientific_import=True,opencv_threads=1)
    proto=out/'STRESS_PROTOCOL.json'
    if a.phase=='freeze':
        require(not out.exists(),'fresh absent stress output; failures preserved');out.mkdir(parents=True)
        write(proto,dict(schema='fixed_synthetic_point_closure_stress_protocol_v21',**authority));print('STRESS_FROZEN');return
    frozen=json.loads(proto.read_bytes())
    for k,v in authority.items():require(frozen[k]==v,'exact frozen stress '+k)
    require(not(out/'STRESS_STARTED.json').exists() and not(out/'STRESS_CHECKS.json').exists(),'one stress run without retry')
    write(out/'STRESS_STARTED.json',dict(protocol=bind(proto),inputs=inputs,plan=list(PLAN),limits=LIMITS,environment=environment))
    old_env={k:os.environ.get(k) for k in environment};old_path=list(sys.path);old_bytecode=sys.dont_write_bytecode
    os.environ.update(environment);sys.path.insert(0,str(repo));sys.dont_write_bytecode=True
    calls,restore,originals,loaded={},[],[],[];error=None;cleanup=[];completed=[];parents_restored=None;started=time.monotonic()
    try:
        import cv2
        import numpy as np
        scope=dict(calls=calls,restore=restore)
        exec(compile(ast.fix_missing_locations(instrument),str(INSTRUMENT_SOURCE)+':unchanged stress hooks','exec'),scope)
        hook=scope['instrument'];threads=cv2.getNumThreads();cv2.setNumThreads(1);restore.append(lambda:cv2.setNumThreads(threads))
        for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix'):
            originals.append((cv2,name,getattr(cv2,name)));hook(cv2,name,'cv2.'+name)
        originals.append((np.linalg,'svd',np.linalg.svd));hook(np.linalg,'svd','numpy.linalg.svd')
        parent=importlib.import_module('scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.solver')
        original_opt=parent.least_squares;scope['original_opt']=original_opt
        calls['least_squares']=dict(entries=0,returns=0,exceptions=0,success_reports=0,residual_entries=0,jacobian_entries=0)
        parent.least_squares=scope['optimizer'];originals.append((parent,'least_squares',original_opt));restore.append(lambda:setattr(parent,'least_squares',original_opt))
        source,previous=h['load_module']('private_v21_stress_solver',paths['solver']);loaded.append(('private_v21_stress_solver',previous))
        originals.append((source.PointClosureSolver,'solve',source.PointClosureSolver.solve));hook(source.PointClosureSolver,'solve','V21.solve')
        parents=(source.S15,source.S13,source.P4,source.G14,source.S19,source.S13.V5,source.S13.V9)
        snapshots={m:dict(vars(m)) for m in parents};policies={m:copy.deepcopy(m.POLICY) for m in parents}
        fixture_scope=dict(np=np,math=math)
        exec(compile(ast.fix_missing_locations(fixture),str(paths['tests'])+':unchanged natural fixture only','exec'),fixture_scope)
        with (out/'STRESS_ROWS.jsonl').open('x') as stream:
            for plan in PLAN:
                data=fixture_scope['fixture'](square=True,noise=True)
                record=case_record(plan,data,source,calls)
                json.dump(record,stream,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
                completed.append({k:v for k,v in record.items() if k not in ('raw_solver','approved_geometry','stable_incidence_guard')})
        parents_restored=all(set(vars(m))==set(snapshots[m]) and all(vars(m)[k] is v for k,v in snapshots[m].items())
            and m.POLICY==policies[m] for m in parents)
    except BaseException as exception:error=dict(type=type(exception).__name__,message=str(exception),traceback=traceback.format_exc())
    finally:
        for callback in reversed(restore):
            try:callback()
            except BaseException as exception:cleanup.append(dict(type=type(exception).__name__,message=str(exception)))
        for name,previous in reversed(loaded):
            if previous is None:sys.modules.pop(name,None)
            else:sys.modules[name]=previous
        for k,v in old_env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        sys.path[:]=old_path;sys.dont_write_bytecode=old_bytecode
    after={k:bind(p) for k,p in paths.items()};protected_after=dict(research=h['git_snapshot'](repo),original=h['git_snapshot'](original))
    restored=dict(primitive_optimizer_solver_identities=all(getattr(obj,name) is value for obj,name,value in originals),
        split_environment=all(os.environ.get(k)==v for k,v in old_env.items()),sys_path=sys.path==old_path,
        bytecode_setting=sys.dont_write_bytecode==old_bytecode,loaded_private_namespaces=all(sys.modules.get(n) is v for n,v in loaded),
        parent_globals_and_policies=parents_restored,opencv_threads=cv2.getNumThreads()==threads if 'threads' in locals() else None)
    closed=all(v['entries']==v['returns']+v['exceptions'] for v in calls.values())
    passed=error is None and len(completed)==len(PLAN) and all(not r['exception'] and not r['contract_failures'] for r in completed)
    passed=passed and after==inputs and protected_after==protection and not cleanup and all(v is True for v in restored.values()) and closed
    checks=dict(schema='fixed_synthetic_point_closure_stress_checks_v21',complete=len(completed)==len(PLAN),passed=passed,
        configured_cases=len(PLAN),completed_cases=len(completed),case_results=completed,exception=error,cleanup_errors=cleanup,
        actual_numerical_calls=calls,counter_dispatch_closed=closed,restoration=restored,source_unchanged=after==inputs,
        protection_before=protection,protection_after=protected_after,inputs=inputs,protocol=bind(proto),code=inputs['stress'],
        limits=LIMITS,CPU_helpers=helper_proof,fixture_source_proof=fixture_proof,instrumentation_source_proof=instrument_proof,
        solver_source_isolation_proof=source.SOURCE_PROOF if 'source' in locals() else None,
        raw_cases=bind(out/'STRESS_ROWS.jsonl') if (out/'STRESS_ROWS.jsonl').is_file() else None,
        elapsed_verification_seconds=time.monotonic()-started)
    write(out/'STRESS_CHECKS.json',checks);print(json.dumps(dict(passed=passed,completed=len(completed),exception=error)))
    if not passed:sys.exit(1)


if __name__=='__main__':main()
