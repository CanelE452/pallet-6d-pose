"""ROOT-only fixed245 point closure; no model/decoder/GT or control replay fit.

V19's source-population reader, protection and numerical instruments are kept
as whole AST functions. Only the one-view private pipeline dispatch and its
source/CPU authority differ. This execution is geometry, never latency data.
"""
import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PRIVATE=Path(__file__).resolve().parent
PARENT='pallet_three_head_observation_20261010_v7'
ORIGINAL_DRIVER=Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19/run_geometry.py')
ORIGINAL_SHA='d964a806959951b2b987094b3a7d290cc587a65ce5c583281d8a4e5b33c0c302'
EXPECTED=dict(solver='1603401ed1f7b1cde301b272bfc1f2f6d3c3935f371ba9c7bafd56e2a893d099',
    pipeline='570ff5feece0a10cb81ba1685141af8f0334751a3b0d903f7514a382f242774f',
    cpu_runner='a2b943464a0c9b489099c9ea7fe4563641d99751d39dce3ca72fd7349e5491fe',
    tests='d97b927ae0c68f06c59bc035a2f797348b2ec150867ecd3abbcf0ad01b1ac003')
METHODS=('HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE',)
LIMITS=dict(training_updates=0,new_RGB=0,new_model_forwards=0,new_corner_selection=0,
    new_endpoint_validation_calls=0,new_line_solve_calls=0,actual_solver_calls_per_frame=1,
    known_DEV=True,unseen_evaluation=False,threshold_search=False,automatic_retry=False,
    old564_scalar_FAIL_retained=True,GT_allowed=False,replay_time_is_latency=False,
    score_only_after_independent_full_geometry_PASS=True,
    coverage_addition_is_not_implementation_bug_or_accuracy_certificate=True)


def require(v,s):
    if not v:raise ValueError(s)

def binding(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return dict(bytes=Path(p).stat().st_size,sha256=d.hexdigest())

def read(p):return json.loads(Path(p).read_bytes())

def write(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def git_snapshot(root):
    def command(*args):return subprocess.check_output(['git','-C',str(root),*args])
    return dict(HEAD=command('rev-parse','HEAD').decode().strip(),
        index_sha256=hashlib.sha256(command('ls-files','--stage','-z')).hexdigest(),
        tracked_diff_sha256=hashlib.sha256(command('diff','--binary')).hexdigest(),
        status_sha256=hashlib.sha256(command('status','--porcelain=v1','-uall')).hexdigest())

def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank source row');yield json.loads(line)

def group(stream,identity,key,names):
    result={}
    for name in names:
        row=next(stream,None);require(isinstance(row,dict) and row['id']==identity and row[key]==name,
            'exact ordered source population required: '+identity+' '+name);result[name]=row
    return result


def source_proof():
    require(binding(ORIGINAL_DRIVER)['sha256']==ORIGINAL_SHA,'immutable original driver source')
    original=ast.parse(ORIGINAL_DRIVER.read_bytes());current=ast.parse(Path(__file__).read_bytes())
    names=('require','binding','read','write','git_snapshot','rows','group','instrument','optimizer')
    for name in names:
        old=[n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name==name]
        new=[n for n in ast.walk(current) if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(old)==len(new)==1 and ast.dump(old[0],include_attributes=False)==ast.dump(new[0],include_attributes=False),
            'whole source function changed: '+name)
    return dict(original_driver=binding(ORIGINAL_DRIVER),whole_functions_unchanged=list(names),
        numerical_instrument_callback_AST_changes=0,scientific_projection_fit_formula_changes=0,
        one_view_dispatch_and_source_CPU_authority_changed=True,
        serialization_only_common_finite_sites=2,parent_algorithm_globals_permanently_mutated=False,
        numerical_hooks_temporary_and_restored=True)


def paths(repo):
    doc=repo/'_docs/experiments';parent=doc/PARENT
    result={name:parent/name for name in ('PROTOCOL.json','GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json',
        'GEOMETRY_SEALED.jsonl.gz','OBSERVATIONS.jsonl.gz','FIXED_GEOMETRY_SEALED.jsonl.gz')}
    for name in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'):
        result['V7:'+name]=parent/name
    result['cohort']=doc/'pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
    result['completion']=doc/'pallet_hybrid_factor_subset_consensus_20261011_v16/PARENT_POPULATION_CHECKS.json'
    result['old_failed_validation']=doc/'pallet_hybrid_factor_subset_consensus_20261011_v16/VALIDATION_READY_CHECKS.json'
    for name in ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json'):
        result['CPU:'+name]=PRIVATE/'cpu'/name
    for key,name in dict(solver='solver.py',pipeline='pipeline.py',cpu_runner='run_cpu.py',tests='test_solver.py').items():
        result[key]=PRIVATE/name
        require(binding(result[key])['sha256']==EXPECTED[key],'fixed private source changed: '+key)
    for path in sorted((repo/'scripts/research').rglob('*.py')):
        result['source:'+str(path.relative_to(repo))]=path
    result['runner']=Path(__file__).resolve();result['original_driver']=ORIGINAL_DRIVER
    return result


def source_authority(p,b):
    seal,receipt,cohort,pop,old,cpu=[read(p[k]) for k in ('GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json',
        'cohort','completion','old_failed_validation','CPU:CPU_CHECKS.json')]
    require(seal['complete'] and seal['GT_read_allowed'] is False and seal['frames']==245 and
        seal['rows']==1960 and seal['fixed_rows']==490,'full sealed V7 geometry')
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),
        ('observations','OBSERVATIONS.jsonl.gz'),('fixed_geometry','FIXED_GEOMETRY_SEALED.jsonl.gz')):
        require(all(seal[key][k]==b[name][k] for k in ('bytes','sha256')),'original seal bytes '+key)
    require(receipt['complete'] and receipt['actual_complete_frames']==245 and receipt['cleanup_error'] is None
        and receipt['inference_error'] is None and receipt['GT_access_during_inference'] is False,'clean original inference')
    done=pop['completion'];require(pop['complete'] and pop['passed'] and pop['frames']==245
        and pop['ordered_ids']==cohort['ids'] and done['complete'],'same full cohort completion')
    require(done['protocol_sha256']==b['PROTOCOL.json']['sha256'] and
        done['geometry_sha256']==b['GEOMETRY_SEALED.jsonl.gz']['sha256'] and
        done['inference_receipt_sha256']==b['INFERENCE_RECEIPT.json']['sha256'],'actual completion identity')
    require(cohort['counts']==dict(clean=153,moderate=92,severe_excluded=74,original=319)
        and len(cohort['ids'])==len(set(cohort['ids']))==245,'unchanged Clean/Moderate245')
    require(old['complete'] and old['passed'] is False and old['checks']==198134007
        and old['failure_count']==564,'preserve original full scalar FAIL')
    v7,vs,vc=[read(p['V7:'+name]) for name in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')]
    require(v7['schema']=='supplemental_v7_point_scalar_validation_protocol_v1'
        and vc['schema']=='supplemental_v7_all_point_candidate_scalar_checks_v1'
        and vc['complete'] is True and vc['passed'] is True and vc['failure_count']==0
        and vc['inputs']==v7['inputs'] and vc['protocol']==vs['protocol']
        and all(vc['protocol'][k]==b['V7:VALIDATION_PROTOCOL.json'][k] for k in ('bytes','sha256')),
        'actual independent V7 coordinate geometry PASS')
    for key,alias in (('GEOMETRY_SEAL.json','GEOMETRY_SEAL.json'),('accuracy_protocol','PROTOCOL.json'),
        ('GEOMETRY_SEALED.jsonl.gz','GEOMETRY_SEALED.jsonl.gz'),('OBSERVATIONS.jsonl.gz','OBSERVATIONS.jsonl.gz'),
        ('FIXED_GEOMETRY_SEALED.jsonl.gz','FIXED_GEOMETRY_SEALED.jsonl.gz'),
        ('checker_code','source:scripts/research/pallet_three_head_observation_20261010_v7/validation_checks.py')):
        require(all(v7['inputs'][key][k]==b[alias][k] for k in ('bytes','sha256')),'V7 validated raw identity '+key)
    cp,cs=[read(p['CPU:'+name]) for name in ('CPU_PROTOCOL.json','CPU_STARTED.json')]
    require(cp['schema']=='whole_U_closure_actual_CPU_protocol_v21' and cpu['schema']=='whole_U_closure_actual_CPU_checks_v21'
        and cpu['complete'] is True and cpu['passed'] is True and cpu['tests_run']==7
        and cpu['failures']==cpu['errors']==cpu['skipped']==cpu['cleanup_errors']==[]
        and cpu['exception'] is None and cpu['source_unchanged'] is True
        and cpu['counter_dispatch_closed'] is True and all(v is True for v in cpu['restoration'].values())
        and cpu['protection_before']==cpu['protection_after']==cp['protection']
        and cpu['inputs']==cp['inputs']==cs['inputs']
        and all(cpu['protocol'][k]==cs['protocol'][k]==b['CPU:CPU_PROTOCOL.json'][k] for k in ('bytes','sha256')),
        'actual unchanged seven-group CPU PASS with restoration')
    for key,expected in cp['inputs'].items():
        current=binding(Path(expected['path']))
        require(current=={k:expected[k] for k in ('bytes','sha256')},'current CPU-tested source '+key)
    return seal,cohort,done


def load_pipeline(path):
    specification=importlib.util.spec_from_file_location('private_v21_geometry_pipeline',path)
    require(specification is not None and specification.loader is not None,'explicit private pipeline loader')
    module=importlib.util.module_from_spec(specification);previous=sys.modules.get(specification.name)
    sys.modules[specification.name]=module
    try:specification.loader.exec_module(module)
    except BaseException:
        if previous is None:sys.modules.pop(specification.name,None)
        else:sys.modules[specification.name]=previous
        raise
    return module,previous


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','run'))
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,default=PRIVATE/'geometry')
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--original-repo',type=Path);a=parser.parse_args()
    repo,out=a.repo.resolve(),a.output.resolve();original=(a.original_repo or a.source_root).resolve()
    require(out==PRIVATE/'geometry','exclusive fixed private21/geometry output required')
    for root in (repo,original):require(not out.is_relative_to(root) and not root.is_relative_to(out),'output disjoint from protected repositories')
    p=paths(repo);b={k:binding(v) for k,v in p.items()};proof=source_proof();seal,cohort,completion=source_authority(p,b)
    environment=dict(PALLET_SOURCE_ROOT=str(a.source_root.resolve()),PALLET_BASELINE_ROOT=str(a.baseline_root.resolve()))
    cpu_protocol=read(p['CPU:CPU_PROTOCOL.json']);require(cpu_protocol['environment']==environment,'same actual CPU configured source/baseline environment')
    protection=dict(research=git_snapshot(repo),original=git_snapshot(original))
    if a.phase=='freeze':
        require(not out.exists(),'fresh geometry output required');out.mkdir(parents=True)
        write(out/'GEOMETRY_PROTOCOL.json',dict(schema='whole_U_closure_geometry_protocol_v21',inputs=b,
            environment=environment,protection=protection,frames=245,rows=245,ledger_rows=245,
            limits=LIMITS,methods=list(METHODS),ordered_ids=cohort['ids'],configured_before_geometry=True,
            source_adapter_proof=proof,python=sys.executable))
        print('V21_GEOMETRY_FROZEN',flush=True);return
    protocol=read(out/'GEOMETRY_PROTOCOL.json');require(protocol['inputs']==b and protocol['environment']==environment
        and protocol['protection']==protection and protocol['limits']==LIMITS and protocol['methods']==list(METHODS)
        and protocol['source_adapter_proof']==proof,'exact frozen geometry inputs')
    require(not(out/'GEOMETRY_STARTED.json').exists(),'one actual geometry run only')
    write(out/'GEOMETRY_STARTED.json',dict(protocol=binding(out/'GEOMETRY_PROTOCOL.json'),inputs=b,
        environment=environment,protection=protection,limits=LIMITS,source_adapter_proof=proof))
    saved_env={k:os.environ.get(k) for k in environment};saved_path=list(sys.path);saved_bytecode=sys.dont_write_bytecode
    os.environ.update(environment);sys.path.insert(0,str(repo));sys.dont_write_bytecode=True
    calls={};restore=[];error=None;states=Counter();execution=Counter();cleanup=[];resources=[];t0=time.monotonic();originals=[]
    pipeline=None;previous=None
    try:
        import cv2
        import numpy as np
        pipeline,previous=load_pipeline(p['pipeline']);source=pipeline.S21
        common=importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.common')
        snap=common.quiet_snapshot();resources.append(snap)
        require(snap['quiet'] and snap['temperature_under_80'],'competing scientific workload or thermal limit')
        threads=cv2.getNumThreads();cv2.setNumThreads(1);restore.append(lambda:cv2.setNumThreads(threads))
        def instrument(obj,name,key):
            original=getattr(obj,name);calls[key]=dict(entries=0,returns=0,exceptions=0)
            def counted(*args,**kwargs):
                calls[key]['entries']+=1
                try:value=original(*args,**kwargs)
                except BaseException:calls[key]['exceptions']+=1;raise
                calls[key]['returns']+=1;return value
            setattr(obj,name,counted);restore.append(lambda:setattr(obj,name,original))
        for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix'):
            originals.append((cv2,name,getattr(cv2,name)));instrument(cv2,name,'cv2.'+name)
        originals.append((np.linalg,'svd',np.linalg.svd));instrument(np.linalg,'svd','numpy.linalg.svd')
        original_opt=source.S13.least_squares
        calls['least_squares']=dict(entries=0,returns=0,exceptions=0,success_reports=0,residual_entries=0,jacobian_entries=0)
        def optimizer(fun,x0,*args,**kwargs):
            counter=calls['least_squares'];counter['entries']+=1
            def residual(*values,**options):
                counter['residual_entries']+=1;return fun(*values,**options)
            jac=kwargs.get('jac')
            if callable(jac):
                def derivative(*values,**options):
                    counter['jacobian_entries']+=1;return jac(*values,**options)
                kwargs['jac']=derivative
            try:value=original_opt(residual,x0,*args,**kwargs)
            except BaseException:counter['exceptions']+=1;raise
            counter['returns']+=1;counter['success_reports']+=int(bool(value.success));return value
        source.S13.least_squares=optimizer;originals.append((source.S13,'least_squares',original_opt));restore.append(lambda:setattr(source.S13,'least_squares',original_opt))
        originals.append((source.PointClosureSolver,'solve',source.PointClosureSolver.solve));instrument(source.PointClosureSolver,'solve','V21.solve')
        geometry,observations,fixed=(iter(rows(p[k])) for k in
            ('GEOMETRY_SEALED.jsonl.gz','OBSERVATIONS.jsonl.gz','FIXED_GEOMETRY_SEALED.jsonl.gz'))
        with gzip.open(out/'GEOMETRY.jsonl.gz','xt',compresslevel=6) as gf,gzip.open(out/'LEDGERS.jsonl.gz','xt',compresslevel=6) as lf:
            for i,identity in enumerate(cohort['ids']):
                g=group(geometry,identity,'method',seal['methods']);o=group(observations,identity,'head_arm',
                    ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'));f=group(fixed,identity,'method',('BASE','N3_SUBPIX'))
                parent,role=g['IMAGE_ROLE_CORNERWISE_HYBRID'],o['IMAGE_ROLE']
                require(all(r['session']==parent['session'] for r in (*g.values(),*o.values(),*f.values())),'same session')
                for r in (*g.values(),*f.values()):
                    require(all(r[k]==parent[k] for k in ('K','xyz','raw_hw','selected_index','fixed_metadata')),'same frozen frame metadata')
                require(f['N3_SUBPIX']['native_points']==role['native_N3_points']
                    and f['BASE']['native_points']==role['original_base_points'],'same original fixed points')
                packet=dict(parent_row=parent,native_N3_points=role['native_N3_points'],parent_completion=completion,role_observation=role)
                execution['frames_started']+=1;live_before=copy.deepcopy(calls)
                output,ledger=pipeline.solve_hybrid_packet(packet)
                require(len(output)==1 and [r['method'] for r in output]==list(METHODS),'one fixed actual output')
                delta={key:{k:value-live_before[key][k] for k,value in counter.items()} for key,counter in calls.items()}
                for field,counter_key in (('primitive_entry_calls','entries'),('primitive_return_calls','returns'),('primitive_exception_calls','exceptions')):
                    ledger[field]={name:delta['cv2.'+name][counter_key] for name in
                        ('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')}
                ledger['svd_dispatch']=delta['numpy.linalg.svd'];ledger['optimizer_dispatch']=delta['least_squares']
                ledger['actual_solver_dispatch']=delta['V21.solve']
                require(delta['V21.solve']==dict(entries=1,returns=1,exceptions=0),'one complete point closure solve per frame')
                for row in output:
                    require(row['id']==identity,'actual identity');states[row['method']+':'+row['output_status']]+=1
                    gf.write(json.dumps(common.finite(row),ensure_ascii=False,allow_nan=False)+'\n');execution['geometry_rows']+=1
                lf.write(json.dumps(common.finite(ledger),ensure_ascii=False,allow_nan=False)+'\n');execution['ledger_rows']+=1
                gf.flush();lf.flush();execution['frames_complete']+=1
                if (i+1)%10==0 or i==244:
                    print(json.dumps(dict(frames=i+1,states=dict(states),optimizer=calls['least_squares'],elapsed_seconds=time.monotonic()-t0)),flush=True)
                    snap=common.quiet_snapshot();resources.append(snap)
                    require(snap['quiet'] and snap['temperature_under_80'],'competing workload during geometry')
                del g,o,f,parent,role,packet,output,ledger
            require(next(geometry,None) is next(observations,None) is next(fixed,None) is None,'full sources exhausted')
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        for restore_call in reversed(restore):
            try:restore_call()
            except BaseException as exc:cleanup.append(dict(type=type(exc).__name__,message=str(exc)))
        if pipeline is not None:
            sys.modules.pop(pipeline._spec.name,None)
            if previous is None:sys.modules.pop('private_v21_geometry_pipeline',None)
            else:sys.modules['private_v21_geometry_pipeline']=previous
        for k,value in saved_env.items():
            if value is None:os.environ.pop(k,None)
            else:os.environ[k]=value
        sys.path[:]=saved_path;sys.dont_write_bytecode=saved_bytecode
    after={k:binding(v) for k,v in p.items()};protected_after=dict(research=git_snapshot(repo),original=git_snapshot(original))
    restored=dict(primitive_and_optimizer_identities=all(getattr(obj,name) is value for obj,name,value in originals),
        environment=all(os.environ.get(k)==v for k,v in saved_env.items()),sys_path=sys.path==saved_path,
        bytecode_setting=sys.dont_write_bytecode==saved_bytecode,
        opencv_threads=cv2.getNumThreads()==threads if 'threads' in locals() else False)
    dispatch_closed=all(value['entries']==value['returns']+value['exceptions'] for value in calls.values())
    complete=error is None and not cleanup and after==b and protected_after==protection
    complete=complete and all(restored.values()) and dispatch_closed and dict(execution)==dict(frames_started=245,geometry_rows=245,ledger_rows=245,frames_complete=245)
    payload=dict(schema='whole_U_closure_geometry_receipt_v21',complete=complete,error=error,cleanup_errors=cleanup,
        actual_execution=dict(execution),actual_numerical_calls=calls,states=dict(states),resource_snapshots=resources,
        inputs_unchanged=after==b,protection_before=protection,protection_after=protected_after,restoration=restored,
        environment_restored=restored['environment'],counter_dispatch_closed=dispatch_closed,limits=LIMITS,
        elapsed_geometry_seconds=time.monotonic()-t0,elapsed_is_not_pipeline_latency=True,source_adapter_proof=proof,
        protocol=binding(out/'GEOMETRY_PROTOCOL.json'),inputs=b,
        geometry=binding(out/'GEOMETRY.jsonl.gz') if (out/'GEOMETRY.jsonl.gz').exists() else None,
        ledgers=binding(out/'LEDGERS.jsonl.gz') if (out/'LEDGERS.jsonl.gz').exists() else None)
    write(out/'GEOMETRY_RECEIPT.json',payload)
    print(json.dumps(dict(complete=complete,error=error,execution=dict(execution),states=dict(states)),ensure_ascii=False),flush=True)
    if not complete:sys.exit(1)

if __name__=='__main__':main()
