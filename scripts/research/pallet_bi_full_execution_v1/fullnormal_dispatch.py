"""Headroom-authorized orchestration; immutable fullnormal row primitives.

Only scheduling changes. A bounded ordered queue can drain at row checkpoints;
the original helper truthfully retains its own SHA in every execution receipt.
"""
from __future__ import annotations
import os
for _variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_variable]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
from pathlib import Path
import time
import traceback

from . import fullnormal_parallel as ROW
from . import fullnormal_batch as K,calibration as CAL,qcov as Q

ROW_HELPER_EXPECTED_SHA256='54f2607ba0528ab96db5a283dbd39a00740131b8d54f852ce248f2efe2b4ddb2'
initialize_worker=ROW.initialize_worker
row_job=ROW.row_job
aggregate=ROW.aggregate
verify_packet_bindings=ROW.verify_packet_bindings


def source_bindings():
    helper=Q.sha(ROW.__file__)
    if helper!=ROW_HELPER_EXPECTED_SHA256:raise ValueError('Immutable reviewed row helper changed')
    return dict(orchestration_sha256=Q.sha(__file__),row_helper_sha256=helper,
        numerical_kernel_source_binding=K.source_binding())


def verify_reuse_root(root,packet_seal,binding):
    """Authenticate staging and origin artifacts; row helper rechecks payload."""
    if root is None:return None
    root=Path(root);manifest=Q.read(root/'REUSE_MANIFEST.json')
    if manifest['packet_seal_sha256']!=Q.sha(packet_seal):raise ValueError('Reuse staging requested packet seal differs')
    rows=manifest['rows'];seen=set()
    for row in rows:
        key=(row['domain'],row['ordinal'],row['arm'])
        if key in seen:raise ValueError('Duplicate exact reuse row')
        seen.add(key);folder=root/row['arm']/row['domain']/f"{row['ordinal']:04d}";seal=Q.read(folder/'FRAME_SEAL.json')
        if Q.sha(folder/'FRAME_SEAL.json')!=row['source_frame_seal_sha256']:raise ValueError('Reuse row seal differs from origin')
        if seal['source_binding']!=binding or seal['cache_payload_sha256']!=row['desired_cache_payload_sha256']:raise ValueError('Reuse row numerical/source closure differs')
        if Q.sha(Path(row['source'])/'FRAME_SEAL.json')!=row['source_frame_seal_sha256']:raise ValueError('Original exact row seal changed')
        for name,digest in seal['artifacts'].items():
            if Q.sha(folder/name)!=digest:raise ValueError('Reuse artifact changed')
    if len(rows)!=manifest['eligible_n']:raise ValueError('Reuse staging row count differs')
    return dict(root=str(root),manifest_sha256=Q.sha(root/'REUSE_MANIFEST.json'),authenticated_rows=len(rows),
        source_payload_and_artifact_equality_verified=True,desired_payload_recomputed_by_unchanged_row_helper=True)


def execute_static(jobs,output,workers,initializer,initargs,callable_row,stop_file=None):
    """Bounded FIFO submission; finish every active row after a stop request."""
    if workers not in (1,2,3):raise ValueError('Registered orchestration supports workers1..3')
    output=Path(output);stop_file=Path(stop_file)if stop_file else None;results=[];failures=[];next_index=0;started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=initializer,initargs=initargs)as pool:
        pending={}
        def enqueue():
            nonlocal next_index
            while next_index<len(jobs)and len(pending)<workers and not(stop_file and stop_file.exists()):
                job=jobs[next_index];next_index+=1;pending[pool.submit(callable_row,job)]=job
        enqueue()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                domain,ordinal,frame,*_=pending.pop(future)
                try:
                    result=future.result();results.append(result);Q.write(output/'row_jobs'/domain/f'{ordinal:04d}.json',result)
                except Exception as error:
                    record=dict(domain=domain,ordinal=ordinal,id=frame['id'],status='INCOMPLETE_TECHNICAL',error=repr(error),traceback=traceback.format_exc())
                    failures.append(record);Q.write(output/'row_jobs'/domain/f'{ordinal:04d}_FAILURE.json',record)
                print(dict(stage='FULL_NORMAL_ORDERED_DISPATCH',completed_jobs=len(results),technical_failures=len(failures),
                    total_jobs=len(jobs),submitted_jobs=next_index,workers=workers,wall_seconds=time.monotonic()-started),flush=True)
            enqueue()
    return results,failures,dict(submitted_jobs=next_index,unsubmitted_jobs=len(jobs)-next_index,
        checkpoint_stop_observed=bool(stop_file and stop_file.exists()),active_rows_drained_before_exit=True)


def run(prior_run,packet_root,packet_seal,environment,output,*,arms=('C0','CP','CB'),workers=3,reuse_root=None,root_GO=False,stop_file=None):
    if root_GO is not True:raise PermissionError('Root measured-headroom production GO required')
    if workers not in (1,2,3):raise ValueError('Registered orchestration supports workers1..3')
    prior_run,packet_root,packet_seal,environment,output=map(Path,(prior_run,packet_root,packet_seal,environment,output))
    output.mkdir(parents=True,exist_ok=True);started=time.monotonic();binding=source_bindings()
    frames={domain:CAL.load_bound_frames(prior_run,domain)[0]for domain in ('real','synthetic')}
    if {d:len(ff)for d,ff in frames.items()}!={'real':238,'synthetic':1003}:raise ValueError('Full original1241 required')
    expected,validation=verify_packet_bindings(packet_root,packet_seal,frames)
    validation.update(orchestration_sha256=binding['orchestration_sha256'],row_helper_sha256=binding['row_helper_sha256'])
    Q.write(output/'PACKET_VALIDATION.json',validation)
    reuse=verify_reuse_root(reuse_root,packet_seal,binding['numerical_kernel_source_binding'])
    p0=Q.read(prior_run/'P0_INPUT_CONTRACT.json')
    jobs=[(domain,i,f,p0[domain]['inventory'][i]['observation_folder'],expected[(domain,i)])for domain,ff in frames.items()for i,f in enumerate(ff)]
    lock=dict(schema='FULL_NORMAL_ORDERED_DISPATCH_V1',workers=workers,arms=list(arms),
        driver_sha256=binding['row_helper_sha256'],row_helper_sha256=binding['row_helper_sha256'],orchestration_sha256=binding['orchestration_sha256'],
        driver_field_scope='immutable row helper implementation; new scheduling separately bound by orchestration_sha256',
        numerical_kernel_source_binding=binding['numerical_kernel_source_binding'],numerical_kernel_unchanged_from_pilot=True,
        packet_root=str(packet_root),packet_seal=str(packet_seal),packet_seal_sha256=Q.sha(packet_seal),
        packet_validation_receipt_sha256=Q.sha(output/'PACKET_VALIDATION.json'),
        environment_path=str(environment),environment_sha256=Q.sha(environment),prior_P0_sha256=Q.sha(prior_run/'P0_INPUT_CONTRACT.json'),
        driver_validation_source_sha256={str(Path(ROW.action_data.__file__).resolve()):Q.sha(ROW.action_data.__file__)},
        exact_reuse_binding=reuse,static_ordered_row_ids={d:[f['id']for f in ff]for d,ff in frames.items()},
        stop_file=str(stop_file)if stop_file else None,dispatch_window=workers,
        resource_policy=dict(CPU_workers=workers,BLAS_threads_per_worker=1,observation_or_backbone_training=False),
        new_driver_only_numerical_and_feature_definitions_unchanged=True)
    if (output/'PRODUCER_LOCK.json').exists()and Q.read(output/'PRODUCER_LOCK.json')!=lock:raise ValueError('Ordered producer execution lock changed')
    Q.write(output/'PRODUCER_LOCK.json',lock)
    results,failures,dispatch=execute_static(jobs,output,workers,initialize_worker,
        (str(prior_run),str(packet_root),str(output),list(arms),str(reuse_root)if reuse_root else None),row_job,stop_file)
    summary=aggregate(output,frames,arms,lock,results,failures,started)
    Q.write(output/'DISPATCH_COMPLETION.json',dict(dispatch,orchestration_sha256=binding['orchestration_sha256'],
        row_helper_sha256=binding['row_helper_sha256'],summary_sha256=Q.sha(output/'SUMMARY.json')))
    return summary


_PROBE_BARRIER=None
def initialize_constructed_probe(barrier):
    """Constructed spawn test only; no model, bank, image or pose access."""
    global _PROBE_BARRIER
    _PROBE_BARRIER=barrier;barrier.wait(timeout=30)


def constructed_probe(job):
    domain,ordinal,frame,*_=job;time.sleep(.03)
    return dict(domain=domain,ordinal=ordinal,id=frame['id'],worker_pid=os.getpid(),
        thread_limits={v:os.environ[v]for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
        actual_new_optimizer_calls_this_job=0,new_H1_fits=0)


def main():
    parser=argparse.ArgumentParser()
    for key in ('prior-run','packet-root','packet-seal','environment','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--arms',nargs='+',default=['C0','CP','CB'])
    parser.add_argument('--reuse-root');parser.add_argument('--stop-file');parser.add_argument('--root-go',action='store_true')
    a=parser.parse_args();summary=run(a.prior_run,a.packet_root,a.packet_seal,a.environment,a.output,arms=a.arms,
        workers=a.workers,reuse_root=a.reuse_root,stop_file=a.stop_file,root_GO=a.root_go)
    print({k:v for k,v in summary.items()if k not in ('missing_rows','technical_failures','full_feature_manifests')})


if __name__=='__main__':main()
