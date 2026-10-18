"""Conditional six visual fits, only after an actual sealed synthetic gate.

Launch in a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8. This module
imports no Torch, model or training module before the environment/gate checks.
Only filtered synthetic1003 inputs/cache enter the training provider.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from .common import read,rows,sha,write,append_log


def gate_check(run_dir,gate_path,extra_seal,extra_normalizer,cache_root):
    gate=read(gate_path)
    if gate.get('run_visual') is not True:raise PermissionError('Root conditional visual gate is not true')
    expected=dict(plan_sha256=sha(Path(run_dir)/'EXPERIMENT_PLAN.json'),
        input_manifest_sha256=sha(Path(run_dir)/'INPUT_MANIFEST.json'),
        extra_feature_seal_sha256=sha(extra_seal),extra_normalizer_sha256=sha(extra_normalizer),
        synthetic_cache_seal_sha256=sha(Path(cache_root)/'CACHE_SEAL.json'))
    representative=Path(run_dir)/'training/R_ORDER_F1_s20261018/RUN_RECEIPT.json'
    expected['F1_representative_run_receipt_sha256']=sha(representative)
    for key,digest in expected.items():
        if gate.get(key)!=digest:raise ValueError('Conditional V gate binding changed: '+key)
    result=read(representative)
    if result.get('complete') is not True or result.get('arm')!='R_ORDER_F1' or result.get('seed')!=20261018:
        raise ValueError('Predeclared F1 representative fit is not complete')
    if result['best_epoch_diagnostics']['validation']['counts'].get('missed_good_action',0)<1:
        raise ValueError('No residual synthetic validation selection failure; V must not run')
    return gate,expected


def validate_cache(inputs_path,cache_root,features,items):
    """Hash every original cache object and inspect at most eight actions."""
    import numpy as np
    input_rows=list(rows(inputs_path));geometry={r['id']:r for r in input_rows}
    flattened=[item for split in('train','validation')for item in items[split]];by_id={r['id']:r for r in flattened}
    if len(input_rows)!=1003 or len(geometry)!=1003 or set(geometry)!=set(by_id):raise ValueError('Only exact filtered synthetic1003 visual inputs are permitted')
    if any(r.get('domain')!='synthetic' for r in input_rows):raise ValueError('Real visual metadata must not enter training')
    root=Path(cache_root);seal=read(root/'CACHE_SEAL.json')
    if seal.get('status')!='COMPLETE' or seal.get('domain')!='synthetic' or seal.get('reference_inputs') is not False:
        raise ValueError('Only complete reference-free synthetic visual cache permitted')
    if seal.get('input_sha256')!=sha(inputs_path)or seal.get('features_sha256')!=sha(features):raise ValueError('Visual cache original input/bank binding changed')
    records=seal['records'];index={r['id']:r for r in records}
    if len(records)!=1003 or set(index)!=set(by_id):raise ValueError('Cache must cover all original1003 frames')
    bindings={};actions=0;chunks=0
    for fid,item in by_id.items():
        record=index[fid];name=record['array_file']
        if Path(name).name!=name or not name.endswith('.npy'):raise ValueError('Cache path is not an explicit local array basename')
        array_path=root/name;meta_path=array_path.with_suffix('.json')
        if sha(array_path)!=record['array_sha256']:raise ValueError('Cached synthetic RGB query bytes changed')
        meta=read(meta_path);expected_ids=[c['candidate_id']for c in item['entries']]
        if record.get('metadata_sha256')!=sha(meta_path)or meta.get('array_sha256')!=record['array_sha256']:
            raise ValueError('Cache production metadata/array hash chain changed')
        if meta['id']!=fid or meta['candidate_ids']!=expected_ids or meta.get('no_GT_inputs') is not True:raise ValueError('Visual cache candidate identity/GT exclusion changed')
        pose_hash=hashlib.sha256(str([(c['candidate_id'],c['actual_pose'])for c in item['entries']]).encode()).hexdigest()
        if meta['action_pose_sha256']!=pose_hash or meta['image_sha256']!=geometry[fid]['image_sha256']:raise ValueError('Cache physical pose/original image binding changed')
        if geometry[fid]['split']!=item['split']or meta['raw_hw']!=geometry[fid]['raw_hw']:raise ValueError('Synthetic split/raw canvas join changed')
        expected_shape=(len(expected_ids),20,4,16,16)
        array=np.load(array_path,mmap_mode='r',allow_pickle=False)
        if array.shape!=expected_shape or array.dtype!=np.float16 or tuple(record['shape'])!=expected_shape or meta['shape']!=list(expected_shape):raise ValueError('Frozen visual dimensions/FP16 cache changed')
        for lo in range(0,len(array),8):
            a=np.asarray(array[lo:lo+8],np.float32);mask=a[:,:,3]
            if not np.isfinite(a).all()or not np.isin(mask,[0.,1.]).all()or np.any(a[:,:,:3]<0)or np.any(a[:,:,:3]>1):raise ValueError('Cached RGB/mask values outside the fixed contract')
            if np.any(a[:,:,:3]*(1-mask[:,:,None])!=0):raise ValueError('Off-canvas samples contain invented RGB')
            chunks+=1
        actions+=len(expected_ids);bindings[fid]=dict(array_sha256=record['array_sha256'],metadata_sha256=sha(meta_path))
        del array
    return dict(status='PASS',frames=1003,actions=actions,verified_candidate_chunks=chunks,files=bindings,
        input_sha256=sha(inputs_path),cache_seal_sha256=sha(root/'CACHE_SEAL.json'),
        real_metadata_RGB_or_cache_read=False,max_candidate_chunk=8)


def configure_cuda():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Fresh V CLI must set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch/CUDA import')
    import torch
    from .train import threads
    threads();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Registered V GPU execution unavailable; preserve the branch as not run')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    return torch,dict(CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),GPU=torch.cuda.get_device_name(0))


def run(args):
    if not args.training_GO:raise PermissionError('Root explicit conditional V training GO required')
    # Fail before CUDA imports/initialization and before any optimizer creation.
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Required fresh-CLI CUBLAS environment absent')
    run_dir=Path(args.run_dir);started,cpu=time.monotonic(),time.process_time()
    for path in(args.features,args.feature_seal,args.labels_dir):
        if not Path(path).resolve().is_relative_to(Path(args.old_run).resolve()):raise ValueError('Original D1 feature/label inputs must remain under the registered read-only old run')
    gate,gate_bindings=gate_check(run_dir,args.gate,args.extra_seal,args.extra_normalizer,args.synthetic_cache)
    from .train import prepare_items,attach_extra,train_one
    from .visual_patches import PatchProvider
    items,bindings,diagnostics=prepare_items(args.features,args.feature_seal,args.labels_dir,args.targets,args.target_seal,args.split_lock)
    bindings.update(attach_extra(items,args.extra_features,args.extra_seal,args.extra_normalizer))
    cache_receipt=validate_cache(args.visual_inputs,args.synthetic_cache,args.features,items)
    receipt_path=run_dir/'V_TRAINING_CACHE_VALIDATION.json';write(receipt_path,cache_receipt)
    expected={(arm,seed)for arm in('V_ORDER_NULL','V_ORDER_RGB')for seed in(20261018,20261019,20261020)}
    attempts_path=run_dir/'GLOBAL_FIT_ATTEMPTS.jsonl';existing=list(rows(attempts_path))if attempts_path.exists()else[]
    if len(existing)+6>18:raise RuntimeError('Conditional six fits would exceed total18 actual attempts')
    if any((r['arm'],r['seed'])in expected for r in existing):raise RuntimeError('Preserve previous V attempts; explicit reviewed resume is required')
    torch,backend=configure_cuda();provider=PatchProvider(args.visual_inputs,args.synthetic_cache)
    bindings.update(visual_inputs_sha256=sha(args.visual_inputs),synthetic_cache_seal_sha256=sha(Path(args.synthetic_cache)/'CACHE_SEAL.json'),
        conditional_V_gate_sha256=sha(args.gate),visual_cache_validation_sha256=sha(receipt_path),
        visual_launcher_sha256=sha(__file__),visual_provider_sha256=sha(Path(__file__).with_name('visual_patches.py')),backend=backend)
    write(run_dir/'V_PRE_OPTIMIZER_EXECUTION_LOCK.json',dict(schema='CONDITIONAL_VISUAL_FIT_EXECUTION_LOCK_V1',gate_bindings=gate_bindings,
        bindings=bindings,supervision_diagnostics=diagnostics,planned_new_fits=6,prior_actual_attempts=len(existing),real_visual_training_inputs=False))
    append_log(run_dir/'RUN_LOG.jsonl',dict(event='START',operation='CONDITIONAL_SIX_VISUAL_FITS',bindings=bindings))
    completed=[];failed=[]
    for arm in('V_ORDER_NULL','V_ORDER_RGB'):
        for seed in(20261018,20261019,20261020):
            try:
                completed.append(train_one(items,arm,seed,run_dir,bindings,training_GO=True,visual_provider=provider,device='cuda:0'))
            except Exception as error:
                # Preserve the failed optimizer attempt; do not tune, retry or
                # substitute a seed. Other already authorized fits may proceed.
                failed.append(dict(arm=arm,seed=seed,error=repr(error),failure_file=str(run_dir/'training'/f'{arm}_s{seed}'/'FAILED_TECHNICAL.json')))
                append_log(run_dir/'RUN_LOG.jsonl',dict(event='FAILED',operation='VISUAL_SINGLE_FIT',arm=arm,seed=seed,error=repr(error)))
                torch.cuda.empty_cache()
    matched=[]
    for seed in(20261018,20261019,20261020):
        folder=run_dir/'training';first_receipt=folder/f'V_ORDER_NULL_s{seed}/RUN_RECEIPT.json';second_receipt=folder/f'V_ORDER_RGB_s{seed}/RUN_RECEIPT.json'
        if not first_receipt.exists()or not second_receipt.exists():
            matched.append(dict(seed=seed,status='NOT_AVAILABLE_FAILED_FIT',representative_replaced=False));continue
        a=read(first_receipt);b=read(second_receipt)
        if a['initial_state_sha256']!=b['initial_state_sha256']:raise AssertionError('NULL/RGB matched initialization violated')
        first=read(folder/f'V_ORDER_NULL_s{seed}/HISTORY.json');second=read(folder/f'V_ORDER_RGB_s{seed}/HISTORY.json')
        for x,y in zip(first,second):
            if x['training_order_sha256']!=y['training_order_sha256']or x['sampled_pair_sequence_sha256']!=y['sampled_pair_sequence_sha256']:
                raise AssertionError('NULL/RGB shared epoch frame/pair schedule changed')
        matched.append(dict(seed=seed,initial_state_sha256=a['initial_state_sha256'],shared_epochs_checked=min(len(first),len(second)),status='PASS'))
    for fid,binding in cache_receipt['files'].items():
        prefix=Path(args.synthetic_cache)/hashlib.sha256(fid.encode()).hexdigest()
        if sha(str(prefix)+'.npy')!=binding['array_sha256']or sha(str(prefix)+'.json')!=binding['metadata_sha256']:raise ValueError('Training cache changed while fitting')
    torch.cuda.synchronize()
    receipt=dict(schema='CONDITIONAL_SIX_VISUAL_FITS_EXECUTION_V1',status='COMPLETE'if not failed else'COMPLETE_WITH_FAILED_FITS',
        planned_new_fits=6,successful_new_fits=len(completed),failed_fits=failed,new_optimizer_attempts=len(list(rows(attempts_path)))-len(existing),
        matched_controls=matched,gate_sha256=sha(args.gate),bindings=bindings,backend=backend,
        model_forwards_in_original_backbones=0,real_reference_or_RGB_cache_training_reads=0,
        GPU_host_wall_upper_bound_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
        exact_CUDA_activity_seconds=None,GPU_time_scope='Whole CLI host wall conservative upper bound, includes CPU input checks and I/O; no exact CUDA event/activity integral was measured.',
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        existing_original_archive_modified=False)
    write(run_dir/'V_TRAINING_EXECUTION_RECEIPT.json',receipt)
    append_log(run_dir/'RUN_LOG.jsonl',dict(event='COMPLETED',operation='CONDITIONAL_SIX_VISUAL_FITS',receipt_sha256=sha(run_dir/'V_TRAINING_EXECUTION_RECEIPT.json')))
    return receipt


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('run-dir','old-run','gate','features','feature-seal','labels-dir','targets','target-seal','split-lock',
        'extra-features','extra-seal','extra-normalizer','visual-inputs','synthetic-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-GO',action='store_true')
    return p


if __name__=='__main__':print(json.dumps(run(parser().parse_args()),allow_nan=False))
