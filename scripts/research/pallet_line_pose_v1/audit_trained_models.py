"""Independent CPU-only checkpoint, sampler, validation and cache metadata audit.

This script never opens P3/P4 feature arrays, images, or GT label files. It does
not run a model forward pass. --all-complete can reuse the audit after training;
the default checks only the requested completed arm/seed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import PalletLinePoseHead


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def metadata(run_dir):
    protocol_path, source_path = run_dir/'TRAIN_PROTOCOL.json', run_dir/'SOURCE_MANIFEST.json'
    protocol, source = read(protocol_path), read(source_path)
    require(protocol['complete'] and protocol['steps'] == 6000 and protocol['batch'] == 16, 'Wrong main training protocol')
    for path, digest in protocol['source_sha256'].items():
        require(sha(path) == digest, f'Frozen protocol source changed: {path}')
    directory = run_dir/'cache'
    manifest_path, complete_path = directory/'CACHE_MANIFEST.json', directory/'CACHE_COMPLETE.json'
    manifest, complete = read(manifest_path), read(complete_path)
    require(manifest['stage'] == 'main' and complete['complete'] and complete['PASS'], 'Incomplete/non-main feature cache')
    require(complete['manifest_sha256'] == sha(manifest_path)
            and manifest['source_manifest_sha256'] == sha(source_path)
            and manifest['train_protocol_sha256'] == sha(protocol_path), 'Cache source/protocol binding failed')
    require(manifest['n_records'] == complete['n_records'] == complete['completed_rows'] == len(source['records']) == 60000,
            'Source/cache denominator changed')
    for path, digest in manifest['dependency_sha256'].items():
        require(sha(path) == digest, f'Cache dependency changed: {path}')
    require(sha(directory/'done.npy') == complete['bitmap_sha256'], 'Cache bitmap hash changed')
    done = np.load(directory/'done.npy', mmap_mode='r')
    require(done.dtype == np.bool_ and done.shape == (60000,) and done.all(), 'Incomplete cache bitmap')
    shard_files = sorted((directory/'shards').glob('*.json'))
    require({p.name:sha(p) for p in shard_files} == complete['shard_marker_sha256'], 'Committed chunk marker changed')
    committed_indices = []
    for marker in shard_files:
        data = read(marker)
        require(data['complete'] and data['manifest_sha256'] == complete['manifest_sha256'], 'Invalid committed shard identity')
        committed_indices.extend(data['source_record_indices'])
    indices = np.asarray(manifest['record_indices'], np.int64)
    require(np.array_equal(indices, np.arange(60000)) and committed_indices == indices.tolist(), 'Cache/shard order changed')
    # Only these small metadata/loss-mask arrays are opened. P3/P4 are untouched.
    names = ('record_index', 'matched', 'matched_gt_index', 'gt_valid')
    arrays = {name:np.load(directory/manifest['arrays'][name]['file'], mmap_mode='r') for name in names}
    for name, value in arrays.items():
        specification = manifest['arrays'][name]
        require(str(value.dtype) == specification['dtype'] and list(value.shape) == specification['shape'], 'Small cache array contract changed')
    require(np.array_equal(arrays['record_index'], indices), 'Cache record lineage changed')
    records = source['records']
    require([r['index'] for r in records] == indices.tolist(), 'Source index order changed')
    partitions = np.asarray([r['partition'] for r in records])
    train_source = np.asarray([r['source_split'] == 'train' for r in records])
    require(np.array_equal(partitions == 'train', train_source), 'Training partition contains original validation records')
    eligible = np.flatnonzero((partitions == 'train') & arrays['matched'] & arrays['gt_valid'][:, :8].any(-1))
    validation = np.flatnonzero(partitions != 'train')
    require(len(eligible) == 55915 and train_source.sum() == 55980 and len(validation) == 4020, 'Unexpected eligible/source/validation counts')
    require(all(records[i]['source_split'] == 'val' for i in validation), 'Non-validation record in validation population')
    require(np.array_equal(arrays['matched_gt_index'] >= 0, arrays['matched']), 'Matched GT index sentinel mismatch')
    require(all(int(arrays['matched_gt_index'][i]) < len(records[i]['targets']) for i in np.flatnonzero(arrays['matched'])), 'Matched GT index out of range')
    require(not arrays['gt_valid'][~arrays['matched']].any(), 'Unmatched frame acquired supervised corners')
    source_counts = dict(Counter(r['source'] for r in records))
    split_counts = dict(Counter(f"{r['source']}:{r['source_split']}" for r in records))
    partition_counts = dict(Counter(partitions.tolist()))
    require(source_counts == complete['source_counts'] and partition_counts == complete['partition_counts'], 'Cache completion denominator summary differs')
    require(split_counts == read(run_dir/'SOURCE_DATA_AUDIT.json')['source_split_counts'], 'Source split audit counts differ')
    identity = dict(train_protocol_sha256=sha(protocol_path), source_manifest_sha256=sha(source_path),
        cache_manifest_sha256=sha(manifest_path), cache_completion_sha256=sha(complete_path),
        model_source_sha256=sha(HERE/'model.py'), train_source_sha256=sha(HERE/'train.py'))
    return protocol, records, eligible, validation, identity, dict(
        records=60000, bitmap_all_complete=True, committed_shards=len(shard_files),
        committed_marker_hashes_verified=True, committed_data_slice_rehash_performed=False,
        large_feature_arrays_opened=False, source_counts=source_counts, source_split_counts=split_counts,
        partition_counts=partition_counts, source_train_frames=55980, eligible_train_frames=55915,
        source_train_rows_excluded_from_loss=65, validation_frames=4020,
        small_arrays_opened=['done', *names], matched_gt_index_lineage_valid=True)


def sampler_audit(saved, eligible, seed, steps, batch):
    """Reconstruct PCG64 permutations independently of train.ShuffledRows."""
    state = saved['sampler_state']
    exposures = steps*batch
    rng = np.random.default_rng(seed)
    order = rng.permutation(eligible)
    sequence, remaining = [], exposures
    while remaining >= len(order):
        sequence.append(order.copy())
        remaining -= len(order)
        order = rng.permutation(eligible)
    sequence.append(order[:remaining].copy())
    sequence = np.concatenate(sequence)
    require(np.array_equal(state['order'], order), 'Sampler final permutation differs from the same-seed independent reconstruction')
    require(state['position'] == remaining and state['epochs_started'] == exposures//len(eligible)+1, 'Sampler state does not account for the fixed exposure budget')
    require(state['rng'] == rng.bit_generator.state, 'Sampler RNG state differs')
    require(np.array_equal(np.sort(state['order']), eligible) and np.isin(sequence, eligible).all(), 'Sampler contains ineligible rows')
    require((state['epochs_started']-1)*len(eligible)+state['position'] == exposures == 96000, 'Exposure accounting failed')
    return dict(eligible_rows=len(eligible), batch=batch, optimizer_steps=steps, exposures=exposures,
        epochs_started=int(state['epochs_started']), position=int(state['position']),
        full_same_seed_sequence_reconstructed=True, final_permutation_and_rng_exact=True,
        every_sample_from_eligible_training_only=True,
        exposure_sequence_int64_sha256=hashlib.sha256(sequence.astype('<i8').tobytes()).hexdigest())


def audit_one(run_dir, arm, seed, protocol, eligible, validation, identity):
    folder = run_dir/'runs'/f'{arm}_seed{seed}'
    checkpoint_path, completion_path = folder/'last.pt', folder/'COMPLETION.json'
    completion = read(completion_path)
    completion_hash = sha(completion_path)
    checkpoint_hash = sha(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require(saved.get('schema') == 'pallet_line_pose_checkpoint_v1' and saved.get('complete') is True
            and saved.get('smoke') is False and saved['step'] == saved['expected_steps'] == 6000
            and (saved['arm'], saved['seed']) == (arm, seed), 'Incomplete or mismatched checkpoint')
    require(completion['complete'] and completion['PASS'] and not completion['smoke'] and completion['step'] == 6000
            and completion['checkpoint_sha256'] == checkpoint_hash, 'Training completion proof mismatch')
    for payload in (saved, completion):
        for name, digest in identity.items():
            require(payload[name] == digest, f'{name} binding failed')
    baseline = Path(saved['baseline_checkpoint'])
    require(sha(baseline) == saved['baseline_checkpoint_sha256'] == protocol['baseline']['sha256'], 'Frozen baseline checkpoint binding failed')
    require(saved['model_config'] == protocol['model_config'], 'Architecture changed')
    torch.manual_seed(seed)
    model = PalletLinePoseHead(**protocol['model_config'])
    initial = {key:value.detach().clone() for key,value in model.state_dict().items()}
    names = {name for name,_ in model.named_parameters()}
    parameters = sum(p.numel() for p in model.parameters())
    require(parameters == completion['head_parameters'] == 19810, 'Parameter count changed')
    require(set(initial) == set(saved['model_state_dict']), 'Checkpoint state keys differ from initial architecture')
    require(all(value.device.type == 'cpu' and torch.isfinite(value).all() for value in saved['model_state_dict'].values()), 'Nonfinite or non-CPU checkpoint tensor')
    changes = {}
    for name in names:
        value = saved['model_state_dict'][name]
        require(value.shape == initial[name].shape and value.dtype == initial[name].dtype, 'State tensor shape/dtype changed')
        delta = value.double()-initial[name].double()
        changes[name] = dict(elements=value.numel(), changed_elements=int((value != initial[name]).sum()),
                            max_abs_change=float(delta.abs().max()), l2_change=float(delta.norm()))
    changed = sum(v['changed_elements'] for v in changes.values())
    require(changed > 0, 'Completed checkpoint remains identical to same-seed initialization')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    optimizer = saved['optimizer_state_dict']
    require(all(int(state['step']) == 6000 for state in optimizer['state'].values()) and len(optimizer['state']) > 0,
            'Optimizer state does not prove 6000 updates')
    require(all(torch.isfinite(v).all() for state in optimizer['state'].values() for v in state.values() if isinstance(v,torch.Tensor)), 'Nonfinite optimizer state')
    sampler = sampler_audit(saved, eligible, seed, saved['step'], protocol['batch'])
    history = [json.loads(line) for line in (folder/'HISTORY.jsonl').read_text().splitlines() if line.strip()]
    require(history[-1]['step'] == 6000 and history[-1]['exposed_instances'] == 96000, 'Training history does not end at fixed budget')
    require(all(r['observed_training_frames'] == len(eligible) and r['step'] <= 6000 for r in history), 'History training denominator differs')
    require(all(np.isfinite(r[k]) for r in history for k in ('loss','line_loss','corner_loss','gradient_norm','lr')), 'Nonfinite recorded training history')
    require(set([1, *range(protocol['log_every'],6001,protocol['log_every'])]).issubset({r['step'] for r in history}), 'Missing expected training progress records')
    index_path = run_dir/'validation/validation_indices.npy'
    logits_path = run_dir/'validation'/f'{arm}_seed{seed}_logits.npy'
    receipt_path = logits_path.with_suffix('.json')
    receipt = read(receipt_path)
    require(receipt['complete'] and (receipt['arm'],receipt['seed']) == (arm,seed) and receipt['n_frames'] == 4020,
            'Validation receipt incomplete')
    require(receipt['checkpoint_sha256'] == checkpoint_hash and receipt['validation_indices_sha256'] == sha(index_path)
            and receipt['logits_sha256'] == sha(logits_path), 'Validation file/checkpoint hash mismatch')
    require(np.array_equal(np.load(index_path), validation), 'Validation does not preserve all original validation indices and order')
    require(all(receipt[name] == digest for name,digest in identity.items()), 'Validation source binding failed')
    logits = np.load(logits_path,mmap_mode='r')
    require(logits.shape == (4020,8,222) and logits.dtype == np.float32 and np.isfinite(logits).all(), 'Invalid validation logits')
    require(sha(checkpoint_path) == checkpoint_hash and sha(completion_path) == completion_hash, 'Checkpoint/completion changed during CPU audit')
    return dict(arm=arm,seed=seed,PASS=True,checkpoint_sha256=checkpoint_hash,completion_sha256=sha(completion_path),
        complete=True,smoke=False,step=6000,head_parameters=parameters,parameter_elements_changed=changed,
        parameter_tensors_changed=sum(v['changed_elements']>0 for v in changes.values()),parameter_tensor_count=len(names),
        changes_by_parameter=changes, same_seed_cpu_initialization_reconstructed=True,
        optimizer_states=len(optimizer['state']),optimizer_all_recorded_steps=6000,sampler=sampler,
        history_rows=len(history),history_last_step=history[-1]['step'],initial_probe=completion['initial_probe'],final_probe=completion['final_probe'],
        validation=dict(frames=4020,shape=list(logits.shape),dtype=str(logits.dtype),all_finite=True,
            logits_sha256=receipt['logits_sha256'],indices_sha256=receipt['validation_indices_sha256'],
            checkpoint_bound=True,original_validation_only=True,training_overlap=0),
        artifact_sha256={str(p):sha(p) for p in (checkpoint_path,completion_path,folder/'HISTORY.jsonl',receipt_path,index_path)},
        source_binding=identity)


def run(run_dir, arm, seed, all_complete, output):
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU audit must not initialize CUDA')
    protocol, records, eligible, validation, identity, cache = metadata(run_dir)
    if all_complete:
        requested = [(a,s) for s in protocol['seeds'] for a in protocol['arms']
                     if (run_dir/'runs'/f'{a}_seed{s}'/'COMPLETION.json').exists()
                     and (run_dir/'validation'/f'{a}_seed{s}_logits.json').exists()]
    else:
        requested = [(arm,seed)]
    require(requested, 'No completed models selected for audit')
    results = [audit_one(run_dir,a,s,protocol,eligible,validation,identity) for a,s in requested]
    require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly initialized CUDA')
    report = dict(schema='pallet_line_pose_trained_model_audit_v1',complete=True,PASS=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),n_audited_models=len(results),runs=results,cache=cache,
        audit_source_sha256=sha(Path(__file__)),source_sha256={str(HERE/name):sha(HERE/name) for name in ('model.py','train.py')},
        gpu_used=False,model_forward_executed=False,large_feature_array_read_or_rehash_performed=False,
        source_image_or_gt_label_opened=False,new_real_inference_executed=False,
        scope='Independent CPU audit of actual completed optimization and synthetic validation artifacts. Metadata/bitmap/chunk-marker integrity is checked without rereading74GB features. This does not establish real accuracy, convergence, or superiority.')
    output = output or run_dir/('TRAINED_MODELS_AUDIT.json' if all_complete else 'FIRST_TRAINED_MODEL_AUDIT.json')
    write(output,report)
    print(json.dumps(dict(PASS=True,output=str(output),runs=[dict(arm=r['arm'],seed=r['seed'],parameters=r['head_parameters'],
        changed=r['parameter_elements_changed'],exposures=r['sampler']['exposures']) for r in results]),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--arm',default='image_joint')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--all-complete',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.run_dir.resolve(),args.arm,args.seed,args.all_complete,args.output)
