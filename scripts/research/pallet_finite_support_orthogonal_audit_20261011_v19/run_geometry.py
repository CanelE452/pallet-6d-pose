"""One frozen bug-repair evaluation; saved observations are not latency data."""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PACKAGE='scripts.research.pallet_finite_support_orthogonal_audit_20261011_v19'
PARENT='pallet_three_head_observation_20261010_v7'
LIMITS=dict(training_updates=0,new_RGB=0,new_model_forwards=0,new_corner_selection=0,
           known_DEV=True,unseen_evaluation=False,threshold_search=False,automatic_retry=False,
           old564_scalar_FAIL_retained=True,GT_allowed=False,replay_time_is_latency=False,
           score_only_after_independent_full_geometry_PASS=True)

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

def paths(repo,private):
    doc=repo/'_docs/experiments';parent=doc/PARENT
    result={name:parent/name for name in ('PROTOCOL.json','GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json',
        'GEOMETRY_SEALED.jsonl.gz','OBSERVATIONS.jsonl.gz','FIXED_GEOMETRY_SEALED.jsonl.gz')}
    result['cohort']=doc/'pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
    result['completion']=doc/'pallet_hybrid_factor_subset_consensus_20261011_v16/PARENT_POPULATION_CHECKS.json'
    result['old_failed_validation']=doc/'pallet_hybrid_factor_subset_consensus_20261011_v16/VALIDATION_READY_CHECKS.json'
    result['point_control']=doc/'pallet_consensus_decision_repair_20261011_v18/GEOMETRY_SEALED.jsonl.gz'
    for n in ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json'):result['CPU:'+n]=private/'cpu'/n
    for p in sorted((repo/'scripts/research').rglob('*.py')):
        result['source:'+str(p.relative_to(repo))]=p
    result['runner']=Path(__file__).resolve()
    return result

def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank source row');yield json.loads(line)

def group(stream,identity,key,names):
    result={}
    for name in names:
        row=next(stream,None);require(isinstance(row,dict) and row['id']==identity and row[key]==name,
            'exact ordered source population required: '+identity+' '+name);result[name]=row
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
    require(cpu['complete'] and cpu['passed'] and cpu['tests_run']==6 and not cpu['failures']
        and not cpu['errors'] and cpu['exception'] is None and cpu['source_unchanged'],'actual CPU6 PASS')
    for name,expected in cpu['source_bindings'].items():
        require(binding(p['source:'+name])==expected,'exact CPU-tested source '+name)
    return seal,cohort,done

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','run'))
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True);a=parser.parse_args()
    repo,private,out=(p.resolve() for p in (a.repo,a.private,a.output));p=paths(repo,private)
    b={k:binding(v) for k,v in p.items()};seal,cohort,completion=source_authority(p,b)
    tree=ast.parse((repo/'scripts/research/pallet_finite_support_orthogonal_audit_20261011_v19/pipeline.py').read_bytes())
    declarations=[n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='METHODS' for t in n.targets)]
    require(len(declarations)==1,'one fixed literal method declaration')
    methods=list(ast.literal_eval(declarations[0]));require(len(methods)==2 and len(set(methods))==2,'paired fixed methods')
    environment=dict(PALLET_SOURCE_ROOT=str(a.source_root.resolve()),PALLET_BASELINE_ROOT=str(a.baseline_root.resolve()))
    protection=dict(research=git_snapshot(repo),original=git_snapshot(a.source_root))
    if a.phase=='freeze':
        require(not out.exists(),'fresh geometry output required');out.mkdir(parents=True)
        write(out/'GEOMETRY_PROTOCOL.json',dict(schema='observed_frame_repair_geometry_protocol_v19',inputs=b,
            environment=environment,protection=protection,frames=245,rows=490,ledger_rows=245,
            limits=LIMITS,methods=methods,ordered_ids=cohort['ids'],configured_before_geometry=True,python=sys.executable))
        print('V19_GEOMETRY_FROZEN',flush=True);return
    protocol=read(out/'GEOMETRY_PROTOCOL.json');require(protocol['inputs']==b and protocol['environment']==environment
        and protocol['protection']==protection and protocol['limits']==LIMITS and protocol['methods']==methods,'exact frozen geometry inputs')
    require(not(out/'GEOMETRY_STARTED.json').exists(),'one actual geometry run only')
    write(out/'GEOMETRY_STARTED.json',dict(protocol=binding(out/'GEOMETRY_PROTOCOL.json'),inputs=b,
        environment=environment,protection=protection,limits=LIMITS))
    saved_env={k:os.environ.get(k) for k in environment};os.environ.update(environment);sys.path.insert(0,str(repo))
    calls={};restore=[];error=None;states=Counter();execution=Counter();cleanup=[];resources=[];t0=time.monotonic()
    try:
        import cv2
        import numpy as np
        source=importlib.import_module(PACKAGE+'.solver');pipeline=importlib.import_module(PACKAGE+'.pipeline')
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
            instrument(cv2,name,'cv2.'+name)
        instrument(np.linalg,'svd','numpy.linalg.svd')
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
        source.S13.least_squares=optimizer;restore.append(lambda:setattr(source.S13,'least_squares',original_opt))
        instrument(source.ObservedFrameFactorSubsetConsensusBank,'solve','V19.solve')
        instrument(pipeline.P16.PoseBank,'solve','PoseBank.solve')
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
                execution['frames_started']+=1
                live_before=copy.deepcopy(calls)
                output,ledger=pipeline.solve_hybrid_packet(packet)
                require(len(output)==2 and [r['method'] for r in output]==list(pipeline.METHODS),'paired actual outputs')
                delta={key:{k:value-live_before[key][k] for k,value in counter.items()} for key,counter in calls.items()}
                for field,counter_key in (('primitive_entry_calls','entries'),('primitive_return_calls','returns'),('primitive_exception_calls','exceptions')):
                    ledger[field]={name:delta['cv2.'+name][counter_key] for name in
                        ('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')}
                ledger['svd_dispatch']=delta['numpy.linalg.svd'];ledger['optimizer_dispatch']=delta['least_squares']
                ledger['actual_solver_dispatch']={prefix+'_'+k:v for prefix,key in
                    (('filtered_solver','V19.solve'),('endpoint_solver','PoseBank.solve')) for k,v in delta[key].items()}
                line_ledger=ledger['point_line_ledger']
                ledger['endpoint_native_bank_operation_counts']={k:v-line_ledger['endpoint_native_bank_before'][k]
                    for k,v in line_ledger['endpoint_native_bank_after'].items()}
                require(ledger['actual_solver_dispatch']['filtered_solver_entries']==2
                    and ledger['actual_solver_dispatch']['endpoint_solver_entries']==line_ledger['endpoint_validation_requests'],'actual per-frame solves')
                for row in output:
                    require(row['id']==identity,'actual identity');states[row['method']+':'+row['output_status']]+=1
                    gf.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');execution['geometry_rows']+=1
                lf.write(json.dumps(ledger,ensure_ascii=False,allow_nan=False)+'\n');execution['ledger_rows']+=1
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
        for k,value in saved_env.items():
            if value is None:os.environ.pop(k,None)
            else:os.environ[k]=value
    after={k:binding(v) for k,v in p.items()};protected_after=dict(research=git_snapshot(repo),original=git_snapshot(a.source_root))
    complete=error is None and not cleanup and after==b and protected_after==protection and execution['frames_complete']==245
    payload=dict(schema='observed_frame_repair_geometry_receipt_v19',complete=complete,error=error,cleanup_errors=cleanup,
        actual_execution=dict(execution),actual_numerical_calls=calls,states=dict(states),resource_snapshots=resources,
        inputs_unchanged=after==b,protection_before=protection,protection_after=protected_after,
        environment_restored=all(os.environ.get(k)==v for k,v in saved_env.items()),limits=LIMITS,
        elapsed_geometry_seconds=time.monotonic()-t0,elapsed_is_not_pipeline_latency=True,
        protocol=binding(out/'GEOMETRY_PROTOCOL.json'),inputs=b,
        geometry=binding(out/'GEOMETRY.jsonl.gz') if (out/'GEOMETRY.jsonl.gz').exists() else None,
        ledgers=binding(out/'LEDGERS.jsonl.gz') if (out/'LEDGERS.jsonl.gz').exists() else None)
    write(out/'GEOMETRY_RECEIPT.json',payload)
    print(json.dumps(dict(complete=complete,error=error,execution=dict(execution),states=dict(states)),ensure_ascii=False),flush=True)
    if not complete:sys.exit(1)

if __name__=='__main__':main()
