"""New execution driver around the unchanged full-normal numerical primitives.

Two fresh CPU workers process independent rows, keeping all arms of one row
together for exact packet caching. Old pilot receipts remain immutable;
current execution and reuse costs are recorded in separate row envelopes.
"""
from __future__ import annotations
import os
for _thread_variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_thread_variable]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import resource
import shutil
import tempfile
import time
import traceback

import cv2
import numpy as np

from scripts.research.pallet_bi_full_execution_v1 import fullnormal_batch as KERNEL
from scripts.research.pallet_bi_full_execution_v1 import qcov as Q,calibration as CAL
from scripts.research.pallet_boundary_action_repair_v1 import action_data

_WORKER={}


def initialize_worker(prior_run,packet_root,output,arms,reuse_pilot):
    """Fresh worker guard/configuration; no cross-worker mutable model state."""
    prior_run,packet_root,output=map(Path,(prior_run,packet_root,output));cv2.setNumThreads(1)
    directory=output/'runtime_tmp'/str(os.getpid());directory.mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(directory);tempfile.tempdir=str(directory)
    old_run=prior_run.parent/'pallet-statewise-end-to-end-recovery-v1'
    guard=KERNEL.D1.firewall(old_run)
    original_config,_,_=KERNEL.D1._cpu_models();original_config=KERNEL.B.configuration(original_config)
    _WORKER.update(prior_run=prior_run,packet_root=packet_root,output=output,arms=tuple(arms),
        reuse_pilot=Path(reuse_pilot)if reuse_pilot else None,guard=guard,config=original_config,
        feature_config=Q.read(old_run/'CONFIG_FROZEN.json'),kernel_source=KERNEL.source_binding(),driver_sha256=Q.sha(__file__))


def _verify_row(folder,signature,source_binding):
    seal=Q.read(folder/'FRAME_SEAL.json')
    if seal['cache_payload_sha256']!=signature or seal['source_binding']!=source_binding:raise ValueError('Exact pilot/current row numerical closure differs')
    for name,digest in seal['artifacts'].items():
        if Q.sha(folder/name)!=digest:raise ValueError('Completed row artifact changed')
    return seal


def _copy_exact_pilot(source,destination,signature,binding):
    if not (source/'FRAME_SEAL.json').exists():return None
    seal=_verify_row(source,signature,binding);destination.mkdir(parents=True,exist_ok=True)
    for name in (*seal['artifacts'],'FRAME_SEAL.json'):
        target=destination/name
        if target.exists():
            if Q.sha(target)!=Q.sha(source/name):raise ValueError('Preserved pilot copy would overwrite changed bytes')
        else:shutil.copy2(source/name,target)
    return dict(source_folder=str(source),source_frame_seal_sha256=Q.sha(source/'FRAME_SEAL.json'),
        copied_artifact_sha256={name:Q.sha(source/name)for name in seal['artifacts']},
        dependency_payload_equality_verified=True,pilot_source_preserved=True)


def row_job(job):
    """One independent image, all covariance arms, no label/reference reads."""
    state=_WORKER;domain,ordinal,frame,observation_folder,expected_packet_sha256=job;start=time.monotonic();cpu=time.process_time()
    observation_folder=Path(observation_folder);oseal=Q.read(observation_folder/'OBSERVATION_SEAL.json')
    observation_path=observation_folder/'OBSERVATION.jsonl.gz';mask_path=observation_folder/'MASK.npz'
    if oseal['id']!=frame['id']or Q.sha(observation_path)!=oseal['observation_sha256']or Q.sha(mask_path)!=oseal['mask_sha256']:
        raise ValueError('Frozen upstream observation/mask binding differs')
    observation=KERNEL.jsonrow(observation_path)
    with np.load(mask_path)as arrays:mask=arrays['pallet_mask_clean'].copy()
    source=state['packet_root']/'packets'/domain/f'{ordinal:04d}.json.gz'
    if Q.sha(source)!=expected_packet_sha256:raise ValueError('Actual packet bytes differ from preverified sealed manifest')
    packet=Q.read(source)
    if packet['id']!=frame['id']:raise ValueError('Requested packet frame identity differs')
    cache={};records=[];calls_this_job=0
    for arm in state['arms']:
        tick=time.monotonic();cpu_tick=time.process_time();folder=state['output']/arm/domain/f'{ordinal:04d}';folder.mkdir(parents=True,exist_ok=True)
        requested=packet['selections'][arm]
        payload=KERNEL.cache_payload(frame,requested,observation,Q.sha(mask_path),state['config'],state['feature_config']);signature=Q.P.digest(payload)
        receipt_path=folder/'ROW_EXECUTION.json';seal_path=folder/'FRAME_SEAL.json';prior_receipt=None
        if receipt_path.exists():
            prior_receipt=Q.read(receipt_path)
            if prior_receipt['driver_sha256']!=state['driver_sha256']or prior_receipt['cache_payload_sha256']!=signature:raise ValueError('Completed new-driver row dependency changed')
        pilot_reuse=None
        if not seal_path.exists()and state['reuse_pilot']is not None:
            pilot_reuse=_copy_exact_pilot(state['reuse_pilot']/arm/domain/f'{ordinal:04d}',folder,signature,state['kernel_source'])
        if seal_path.exists():
            original_seal=_verify_row(folder,signature,state['kernel_source']);result=Q.read(folder/'RESULT.json.gz');feature=Q.read(folder/'FEATURE.json.gz')
            cache[signature]=(payload,result,feature,arm);actual_calls=0;origin='SEALED_PILOT'if pilot_reuse else'PRIOR_SEALED_CURRENT_ROW';reused=True
        elif signature in cache:
            canonical,result,feature,origin_arm=cache[signature]
            if canonical!=payload:raise AssertionError('Canonical payload hash collision')
            actual_calls=0;origin=origin_arm;reused=True
            Q.write(folder/'RESULT.json.gz',result);Q.write(folder/'FEATURE.json.gz',feature)
            original_seal=dict(id=frame['id'],arm=arm,domain=domain,ordinal=ordinal,complete=True,
                cache_payload_sha256=signature,source_binding=state['kernel_source'],exact_reused=True,cache_origin_arm=origin_arm,
                artifacts={name:Q.sha(folder/name)for name in ('RESULT.json.gz','FEATURE.json.gz')},
                resource=dict(actual_pose_optimizer_calls=0),numeric_reference_inputs=False)
            Q.write(seal_path,original_seal)
        else:
            result=KERNEL.solve_frame(frame,requested,observation,mask,state['config'])
            feature=KERNEL.feature_frame(frame,result,state['feature_config']);actual_calls=result['operations']['pose_optimizer_calls'];origin=arm;reused=False
            cache[signature]=(payload,result,feature,arm)
            Q.write(folder/'RESULT.json.gz',result);Q.write(folder/'FEATURE.json.gz',feature)
            original_seal=dict(id=frame['id'],arm=arm,domain=domain,ordinal=ordinal,complete=True,
                cache_payload_sha256=signature,source_binding=state['kernel_source'],exact_reused=False,cache_origin_arm=arm,
                artifacts={name:Q.sha(folder/name)for name in ('RESULT.json.gz','FEATURE.json.gz')},
                resource=dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu_tick,actual_pose_optimizer_calls=actual_calls),
                numeric_reference_inputs=False)
            Q.write(seal_path,original_seal)
        KERNEL.D1.P.validate_input(frame)
        action_data.validate_frame(feature)
        receipt=dict(id=frame['id'],domain=domain,ordinal=ordinal,arm=arm,complete=True,
            driver_sha256=state['driver_sha256'],numerical_kernel_source_binding=state['kernel_source'],
            cache_payload_sha256=signature,requested_selection_sha256=Q.P.digest(requested),
            requested_packet_file_sha256=Q.sha(source),known_unused_cache_fields=['full_points[*].uncertainty_calibration'],
            exact_reused=reused,reuse_origin=origin,pilot_reuse_proof=pilot_reuse,
            original_frame_seal_sha256=Q.sha(seal_path),
            resource=dict(current_execution_wall_seconds=time.monotonic()-tick,current_execution_CPU_seconds=time.process_time()-cpu_tick,
                actual_new_pose_optimizer_calls=actual_calls,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            firewall_receipt=state['guard'].receipt(),numeric_reference_inputs=False)
        if prior_receipt is not None:
            # Resume preserves the original execution account; a completed
            # optimizer is never counted as another current optimization.
            receipt=prior_receipt
        else:Q.write(receipt_path,receipt)
        records.append(dict(arm=arm,path=str(receipt_path),sha256=Q.sha(receipt_path),receipt=receipt))
        calls_this_job+=actual_calls
    return dict(domain=domain,ordinal=ordinal,id=frame['id'],records=records,
        current_job_wall_seconds=time.monotonic()-start,current_job_CPU_seconds=time.process_time()-cpu,
        actual_new_optimizer_calls_this_job=calls_this_job)


def aggregate(output,expected_frames,arms,producer_lock,job_results,failures,started):
    output=Path(output);manifests={};missing=[]
    for domain,frames in expected_frames.items():
        for arm in arms:
            records=[];receipts=[];complete=True
            for ordinal,frame in enumerate(frames):
                folder=output/arm/domain/f'{ordinal:04d}'
                if not (folder/'ROW_EXECUTION.json').exists():
                    complete=False;missing.append(dict(domain=domain,arm=arm,ordinal=ordinal,id=frame['id']));continue
                receipt=Q.read(folder/'ROW_EXECUTION.json');seal=Q.read(folder/'FRAME_SEAL.json')
                if receipt['id']!=frame['id']or Q.sha(folder/'FRAME_SEAL.json')!=receipt['original_frame_seal_sha256']:raise ValueError('Aggregate ID/seal contract changed')
                for name,digest in seal['artifacts'].items():
                    if Q.sha(folder/name)!=digest:raise ValueError('Aggregate row bytes changed')
                records.append(Q.read(folder/'FEATURE.json.gz'));receipts.append(receipt)
            if not complete:continue
            if [r['id']for r in records]!=[r['id']for r in frames]:raise ValueError('Full original ordered denominator mismatch')
            if domain=='synthetic'and CounterSplit(records)!={'train':768,'validation':235}:raise ValueError('Original synthetic split changed')
            destination=output/arm/domain/'FEATURES.jsonl.gz';Q.P.save_rows(destination,records)
            seal=dict(schema='BI_FULL_NORMAL_FEATURE_SEAL_V1',complete=True,domain=domain,arm=arm,rows=len(records),
                features_sha256=Q.sha(destination),head_features_sha256=Q.sha(destination),producer_lock_sha256=Q.sha(output/'PRODUCER_LOCK.json'),
                all_original_ids_and_algorithm_fallbacks_retained=True,numeric_reference_inputs=False,
                source_binding=producer_lock['numerical_kernel_source_binding'],driver_sha256=producer_lock['driver_sha256'],
                frame_seal_sha256={f'{i:04d}':Q.sha(output/arm/domain/f'{i:04d}'/'FRAME_SEAL.json')for i in range(len(frames))},
                row_execution_sha256={f'{i:04d}':Q.sha(output/arm/domain/f'{i:04d}'/'ROW_EXECUTION.json')for i in range(len(frames))})
            Q.write(output/arm/domain/'FEATURE_SEAL.json',seal)
            manifests[arm+'/'+domain]=dict(path=str(destination),feature_seal=str(output/arm/domain/'FEATURE_SEAL.json'),
                rows=len(records),features_sha256=Q.sha(destination),
                actual_new_pose_optimizer_calls=sum(r['resource']['actual_new_pose_optimizer_calls']for r in receipts),
                exact_reused_rows=sum(r['exact_reused']for r in receipts),
                pilot_exact_reused_rows=sum(r['pilot_reuse_proof']is not None for r in receipts))
    summary=dict(status='COMPLETE_NEW'if not missing and not failures else'INCOMPLETE',
        arms=list(arms),expected_real=238,expected_synthetic=1003,full_feature_manifests=manifests,
        missing_rows=missing,technical_failures=failures,worker_count=producer_lock['workers'],
        new_pose_optimizer_calls=sum(Q.read(path)['resource']['actual_new_pose_optimizer_calls']for path in output.glob('*/*/*/ROW_EXECUTION.json')),
        current_invocation_new_pose_optimizer_calls=sum(j['actual_new_optimizer_calls_this_job']for j in job_results),
        exact_reused_rows=sum(v['exact_reused_rows']for v in manifests.values()),
        wall_seconds=time.monotonic()-started,producer_lock_sha256=Q.sha(output/'PRODUCER_LOCK.json'),
        numerical_kernel_unchanged_from_pilot=True,new_H1_fits=0,new_observation_model_fits=0,
        reference_numeric_inputs=False,all_failure_rows_kept=True)
    Q.write(output/'SUMMARY.json',summary);return summary


def CounterSplit(records):
    from collections import Counter
    return dict(Counter(r['split']for r in records))


def verify_packet_bindings(packet_root,packet_seal,expected_frames):
    """Authenticate every requested packet before any row job is scheduled."""
    packet_root,packet_seal=Path(packet_root),Path(packet_seal);manifest=Q.read(packet_seal);expected={}
    if 'artifacts'in manifest:
        for name,digest in manifest['artifacts'].items():
            parts=Path(name).parts
            if len(parts)==3 and parts[0]=='packets'and parts[1]in expected_frames and parts[2].endswith('.json.gz'):
                ordinal=int(parts[2].split('.')[0]);key=(parts[1],ordinal)
                if key in expected:raise ValueError('Duplicate packet binding')
                expected[key]=digest
        schema='S2_PACKETS_ARTIFACT_SEAL'
    elif 'frames'in manifest:
        for frame in manifest['frames']:
            domain=frame['domain'];ordinal=int(frame['index']);key=(domain,ordinal)
            if domain not in expected_frames or key in expected:raise ValueError('Unknown/duplicate packet domain/index')
            expected[key]=frame['packet_sha256']
        schema='F_FRAME_PACKET_MANIFEST'
    else:raise ValueError('No recognized sealed per-packet SHA256 mapping')
    requested={(domain,index)for domain,frames in expected_frames.items()for index in range(len(frames))}
    if set(expected)!=requested:raise ValueError('Sealed manifest must bind every original real238/synthetic1003 packet exactly')
    for domain,ordinal in sorted(requested):
        path=packet_root/'packets'/domain/f'{ordinal:04d}.json.gz'
        if Q.sha(path)!=expected[(domain,ordinal)]:raise ValueError('Packet SHA256 mismatch before scheduling: '+str(path))
    receipt=dict(schema=schema,manifest_path=str(packet_seal),manifest_sha256=Q.sha(packet_seal),
        bound_packet_n=len(expected),all_actual_packet_bytes_verified_before_scheduling=True,
        per_row_worker_reverification_required=True,
        packet_digest_of_digests=Q.P.digest({domain+'/'+str(ordinal):digest for(domain,ordinal),digest in sorted(expected.items())}),
        driver_source_sha256=Q.sha(__file__))
    return expected,receipt


def run(prior_run,packet_root,output,*,arms=('C0','CP','CB'),workers=2,reuse_pilot=None,root_GO=False,packet_seal=None,environment=None):
    if root_GO is not True:raise PermissionError('Root full-production GO required after fixed pilot resource review')
    if workers not in (1,2):raise ValueError('Registered parallel driver supports1or2CPU workers')
    started=time.monotonic();prior_run,packet_root,output=map(Path,(prior_run,packet_root,output));output.mkdir(parents=True,exist_ok=True)
    inputs={};jobs=[];p0=Q.read(prior_run/'P0_INPUT_CONTRACT.json')
    for domain in ('real','synthetic'):
        inputs[domain],binding=CAL.load_bound_frames(prior_run,domain)
    packet_seal=Path(packet_seal)if packet_seal else packet_root/'PACKET_SEAL.json'
    environment=Path(environment)if environment else packet_root.parent/'ENVIRONMENT_INITIAL.json'
    expected_packets,packet_validation=verify_packet_bindings(packet_root,packet_seal,inputs)
    Q.write(output/'PACKET_VALIDATION.json',packet_validation)
    for domain,frames in inputs.items():
        for ordinal,frame in enumerate(frames):jobs.append((domain,ordinal,frame,p0[domain]['inventory'][ordinal]['observation_folder'],expected_packets[(domain,ordinal)]))
    lock=dict(schema='BI_FULL_NORMAL_PARALLEL_EXECUTION_V1',driver_sha256=Q.sha(__file__),workers=workers,
        driver_validation_source_sha256={str(Path(action_data.__file__).resolve()):Q.sha(action_data.__file__)},
        numerical_kernel_source_binding=KERNEL.source_binding(),numerical_kernel_unchanged_from_pilot=True,
        packet_root=str(packet_root),packet_seal=str(packet_seal),packet_seal_sha256=Q.sha(packet_seal),arms=list(arms),
        environment_path=str(environment),environment_sha256=Q.sha(environment),
        packet_validation_receipt_sha256=Q.sha(output/'PACKET_VALIDATION.json'),
        prior_P0_sha256=Q.sha(prior_run/'P0_INPUT_CONTRACT.json'),
        reuse_pilot=str(reuse_pilot)if reuse_pilot else None,static_ordered_row_ids={d:[f['id']for f in ff]for d,ff in inputs.items()},
        resource_policy=dict(CPU_workers=workers,BLAS_threads_per_worker=1,observation_or_backbone_training=False),
        new_driver_only_numerical_and_feature_definitions_unchanged=True)
    if (output/'PRODUCER_LOCK.json').exists()and Q.read(output/'PRODUCER_LOCK.json')!=lock:raise ValueError('Parallel producer execution lock changed')
    Q.write(output/'PRODUCER_LOCK.json',lock);results=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=initialize_worker,
            initargs=(str(prior_run),str(packet_root),str(output),list(arms),str(reuse_pilot)if reuse_pilot else None))as pool:
        futures={pool.submit(row_job,job):job for job in jobs}
        for future in as_completed(futures):
            domain,ordinal,frame,_,_=futures[future]
            try:
                result=future.result();results.append(result)
                Q.write(output/'row_jobs'/domain/f'{ordinal:04d}.json',result)
            except Exception as error:
                failure=dict(domain=domain,ordinal=ordinal,id=frame['id'],status='INCOMPLETE_TECHNICAL',error=repr(error),traceback=traceback.format_exc())
                failures.append(failure);Q.write(output/'row_jobs'/domain/f'{ordinal:04d}_FAILURE.json',failure)
            print(dict(stage='FULL_NORMAL_PARALLEL',completed_jobs=len(results),technical_failures=len(failures),total_jobs=len(jobs),wall_seconds=time.monotonic()-started),flush=True)
    return aggregate(output,inputs,arms,lock,results,failures,started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prior-run',required=True);parser.add_argument('--packet-root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--arms',nargs='+',default=['C0','CP','CB']);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--reuse-pilot');parser.add_argument('--root-go',action='store_true')
    parser.add_argument('--packet-seal');parser.add_argument('--environment')
    args=parser.parse_args();result=run(args.prior_run,args.packet_root,args.output,arms=args.arms,workers=args.workers,reuse_pilot=args.reuse_pilot,root_GO=args.root_go,
        packet_seal=args.packet_seal,environment=args.environment)
    print({k:v for k,v in result.items()if k not in ('missing_rows','technical_failures','full_feature_manifests')})


if __name__=='__main__':main()
