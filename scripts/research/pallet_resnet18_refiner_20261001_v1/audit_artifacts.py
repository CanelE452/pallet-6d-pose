"""CPU-only post-training artifact audit; no forwards, fitting or selection.

Run ``python -B audit_artifacts.py train`` after all six final head fits.
The audit is deliberately outside the sealed training implementation. It writes
one immutable TRAINING_ARTIFACT_AUDIT.json and never repairs an artifact.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent

def local(name, filename):
    spec = importlib.util.spec_from_file_location(HERE.name + '_audit_' + name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

C = local('common', 'common.py')


def finite_number(value, *, minimum=None, positive=False):
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    assert math.isfinite(value), value
    if minimum is not None:
        assert value >= minimum, value
    if positive:
        assert value > 0, value
    return float(value)


def validate_trace(rows, protocol):
    assert len(rows) == 6000
    zero_support = {'P': [], 'D': []}
    for step, row in enumerate(rows, 1):
        assert row['step'] == step
        expected_lr = C.learning_rate(step, protocol)
        assert math.isclose(finite_number(row['lr'], positive=True), expected_lr, rel_tol=0., abs_tol=1e-15)
        assert set(row['arms']) == {'P', 'D'}
        for arm, values in row['arms'].items():
            finite_number(values['loss'], minimum=0.)
            finite_number(values['preclip_grad_norm'], minimum=0.)
            finite_number(values['parameter_update_norm'], positive=True)
            frames, corners = values['supported_frames'], values['supported_corners']
            assert isinstance(frames, int) and isinstance(corners, int)
            assert 0 <= frames <= 16 and frames <= corners <= frames * 8
            if frames == 0:
                zero_support[arm].append(step)
    return dict(updates=6000, arms=['P','D'],
        zero_joint_supervision_steps=zero_support,
        all_recorded_values_finite=True, all_recorded_parameter_updates_positive=True)


def validate_state(state, expected):
    assert set(state) == set(expected), 'Model tensor keys differ'
    parameters = 0
    for key, value in state.items():
        reference = expected[key]
        assert isinstance(value, torch.Tensor) and value.device.type == 'cpu' and not value.is_meta, key
        assert value.shape == reference.shape and value.dtype == reference.dtype, key
        if value.is_floating_point():
            assert torch.isfinite(value).all(), key
        parameters += value.numel()
    return parameters


def validate_resumes(checkpoint, receipt):
    """Allow only the producer's explicit step6000 receipt-only finalization."""
    assert isinstance(checkpoint,list) and isinstance(receipt,list)
    for row in checkpoint + receipt:
        assert set(row)=={'time','from_step'} and isinstance(row['time'],str)
        assert isinstance(row['from_step'],int) and 0 <= row['from_step'] <= 6000
    if checkpoint == receipt:
        return False
    assert len(receipt)==len(checkpoint)+1 and receipt[:-1]==checkpoint
    assert receipt[-1]['from_step']==6000
    return True


def validate_optimizer(state, model, optimizer_spec, final_lr):
    groups = state['param_groups']
    assert len(groups) == 1
    group = groups[0]
    parameters = list(model.parameters())
    ids = group['params']
    assert len(ids) == len(parameters) and len(set(ids)) == len(ids)
    assert set(state['state']) == set(ids), 'All trained parameters require optimizer state'
    assert tuple(group['betas']) == tuple(optimizer_spec['betas'])
    assert group['weight_decay'] == optimizer_spec['weight_decay']
    assert math.isclose(group['lr'], final_lr, rel_tol=0., abs_tol=1e-15)
    for index, parameter in zip(ids, parameters):
        item = state['state'][index]
        assert set(item) == {'step', 'exp_avg', 'exp_avg_sq'}
        assert isinstance(item['step'], torch.Tensor) and item['step'].numel() == 1
        assert float(item['step']) == 6000
        for field in ('exp_avg','exp_avg_sq'):
            value = item[field]
            assert value.shape == parameter.shape and value.dtype == parameter.dtype
            assert value.device.type == 'cpu' and torch.isfinite(value).all()
        assert (item['exp_avg_sq'] >= 0).all()
    return dict(parameter_tensors=len(parameters), optimizer_step=6000,
        state_finite=True, optimizer_hyperparameters_match=True)


def train_audit():
    protocol = C.verify_lock()
    assert protocol['steps'] == 6000 and protocol['batch'] == 16
    assert protocol['seeds'] == [1,2,3] and protocol['fits'] == {'P':3,'D':3}
    assert protocol['model_config'] == C.CONFIG and protocol['baseline_trainable'] is False
    assert protocol['training']['real_training'] == 0
    protocol_sha = C.sha(C.DOC / 'PROTOCOL.json')
    seen = {}
    def remember(path):
        path = Path(path).resolve()
        value = C.bound(path)
        value['bytes'] = path.stat().st_size
        seen[str(path)] = value
        return value
    def verify(binding):
        path = (C.ROOT / binding['path']).resolve()
        assert path.is_relative_to(C.ROOT.resolve())
        value = remember(path)
        assert value['sha256'] == binding['sha256'], path
        if 'bytes' in binding:
            assert value['bytes'] == binding['bytes'], path
        return path
    def read(path):
        remember(path)
        return C.read(path)
    for entry in protocol['bindings']:
        verify(entry)
    baseline = verify(protocol['baseline'])
    assert baseline == C.WEIGHTS.resolve()
    baseline_sha = protocol['baseline']['sha256']
    for key in ('baseline_protocol','baseline_training_complete'):
        if key in protocol:
            verify(protocol[key])
    complete = read(C.DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] is True and complete['fits'] == 6
    assert complete['steps_per_fit'] == 6000 and complete['total_head_updates'] == 36000
    assert complete['total_exposures'] == 576000 and complete['baseline_sha256'] == baseline_sha
    assert len(complete['seeds']) == 3
    for key in ('baseline_protocol','baseline_training_complete'):
        if key in protocol:
            assert complete[key] == protocol[key]
    smoke = read(C.DOC / 'GPU_SMOKE.json')
    assert smoke['PASS'] and smoke['heads_discarded'] and smoke['contract_only_no_performance_selection']
    assert smoke['optimizer_updates'] == {'P':2,'D':2} and smoke['seed'] == 20261001
    for entry in smoke['bindings']:
        verify(entry)
    cache = read(C.DOC / 'SOURCE_CACHE_COMPLETE.json')
    assert cache['complete'] and cache['PASS'] and cache['rows'] == 60000
    assert cache['protocol_sha256'] == protocol_sha
    assert verify(cache['source']) == C.SOURCE.resolve()
    assert cache['all_image_and_label_hashes_verified'] is True
    for entry in cache['arrays']:
        verify(entry)
    manifest = read(verify(cache['cache_manifest']))
    assert manifest['n'] == 60000 and manifest['protocol_sha256'] == protocol_sha
    assert manifest['source'] == cache['source']
    records = read(C.SOURCE)['records']
    assert len(records) == 60000
    partitions = np.array([r['partition'] for r in records])
    arrays = {}
    for name in ('done','matched','point_valid','gt_valid'):
        path = C.RAW / 'cache' / (name + '.npy')
        assert any(verify(b) == path.resolve() for b in cache['arrays']), name
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        assert arrays[name].dtype == np.bool_
        assert arrays[name].shape == ((60000,9) if name in ('point_valid','gt_valid') else (60000,))
    assert arrays['done'].all()
    usable = arrays['matched'] & (arrays['point_valid'][:,:8] & arrays['gt_valid'][:,:8]).any(-1)
    for name,count in dict(train=55980,calibration=1004,selection=1031,heldout=1985).items():
        mask=partitions==name
        assert int(mask.sum())==count==cache['partition_counts'][name]['total']
        assert int((mask & arrays['matched']).sum())==cache['partition_counts'][name]['matched']
        assert int((mask & usable).sum())==cache['partition_counts'][name]['usable']
    training_rows = np.flatnonzero((partitions == 'train') & usable)
    assert len(training_rows) > 0 and len(training_rows) == cache['partition_counts']['train']['usable']
    batch_order = read(C.DOC / 'BATCH_ORDER.json')
    assert batch_order['D_P_same_order'] is True and batch_order['per_seed_exposures'] == 96000
    assert batch_order['train_rows'] == len(training_rows)
    row_sha = hashlib.sha256(training_rows.astype('<i8').tobytes()).hexdigest()
    assert batch_order['train_rows_sha256'] == row_sha and set(batch_order['seeds']) == {'1','2','3'}
    R = local('refiner', 'refiner.py')
    classes = {'P':R.GenericPointRefiner,'D':R.DirectResidualControl}
    results = {}
    for offset, seed in enumerate((1,2,3)):
        receipt = read(C.DOC / ('TRAIN_SEED%d.json' % seed))
        assert complete['seeds'][offset] == receipt
        assert receipt['complete'] is True and receipt['seed'] == seed
        assert receipt['updates_per_arm'] == 6000 and receipt['exposures_per_arm'] == 96000
        assert receipt['arms'] == ['P','D'] and receipt['usable_training_rows'] == len(training_rows)
        assert receipt['real_training'] == 0 and receipt['backbone_retrained'] is False
        assert receipt['final_checkpoint_only'] and receipt['finite_all_updates'] and receipt['paired_shared_frozen_features']
        order_path = verify(receipt['order'])
        assert order_path == (C.RAW / ('order_seed%d.npy' % seed)).resolve()
        assert receipt['order'] == batch_order['seeds'][str(seed)]
        order = np.load(order_path, allow_pickle=False)
        assert order.shape == (6000,16) and order.dtype == np.int64
        assert np.isin(order, training_rows).all(), 'Nonusable/nonTRAIN row in order'
        rng = np.random.default_rng(seed); pieces=[]; remaining=96000
        while remaining:
            part=rng.permutation(training_rows)[:remaining]; pieces.append(part); remaining-=len(part)
        assert np.array_equal(order, np.concatenate(pieces).reshape(6000,16))
        ck_path = verify(receipt['checkpoint'])
        assert ck_path == (C.RAW / 'runs' / ('seed%d' % seed) / 'paired_last.pt').resolve()
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        assert ck['complete'] is True and ck['seed'] == seed and ck['step'] == 6000
        assert ck['config'] == C.CONFIG and ck['protocol_sha256'] == protocol_sha
        assert ck['baseline_sha256'] == baseline_sha and ck['order_sha256'] == receipt['order']['sha256']
        assert set(ck['models']) == set(ck['optimizers']) == {'P','D'}
        finalization_only_resume=validate_resumes(ck['resumes'],receipt['resumes'])
        assert ck['elapsed_seconds'] == receipt['elapsed_seconds']
        finite_number(ck['elapsed_seconds'], positive=True)
        assert ck['torch_rng_state'].dtype == torch.uint8 and ck['torch_rng_state'].ndim == 1
        assert len(ck['cuda_rng_state']) >= 1
        assert all(value.dtype == torch.uint8 and value.ndim == 1 for value in ck['cuda_rng_state'])
        metrics = read(ck_path.parent / 'STEP_METRICS.json')
        assert ck['metrics'] == metrics
        trace = validate_trace(metrics, protocol)
        arms = {}
        for arm, cls in classes.items():
            torch.manual_seed(0)
            model = cls(**C.CONFIG)  # CPU shape schema only; no forward or fit.
            validate_state(ck['models'][arm], model.state_dict())
            for name, buffer in model.named_buffers():
                assert torch.equal(ck['models'][arm][name], buffer), ('Frozen sampling buffer changed', name)
            parameter_count = sum(p.numel() for p in model.parameters())
            assert parameter_count == receipt['parameters'][arm]
            optimizer = validate_optimizer(ck['optimizers'][arm], model, protocol['optimizer'], metrics[-1]['lr'])
            gradients = read(C.DOC / ('GRADIENT_%s%d.json' % (arm,seed)))
            assert gradients['frozen_backbone_grads_none'] is True
            assert set(gradients['gradients']) == dict(model.named_parameters()).keys()
            for value in gradients['gradients'].values():
                finite_number(value, minimum=0.)
            assert gradients['gradients']['adapt3.0.weight'] > 0 and gradients['gradients']['adapt4.0.weight'] > 0
            arms[arm] = dict(parameters=parameter_count, state_keys=len(model.state_dict()),
                state_shapes_dtypes_finite=True, first_step_adapter_gradients_positive=True, optimizer=optimizer)
            del model
        results[str(seed)] = dict(checkpoint=remember(ck_path), order=remember(order_path),
            seed=seed, step=6000, arms=arms, trace=trace, resumes=receipt['resumes'],
            checkpoint_resumes=ck['resumes'],finalization_only_resume=finalization_only_resume)
        del ck
    remember(C.DOC / 'PROTOCOL.json')
    remember(HERE / 'common.py'); remember(HERE / 'refiner.py'); remember(__file__)
    result = dict(schema='paired_refiner_training_artifact_audit_v1', complete=True, PASS=True,
        namespace=HERE.name, protocol=remember(C.DOC / 'PROTOCOL.json'),
        training_complete=remember(C.DOC / 'TRAINING_COMPLETE.json'),
        code=remember(__file__), baseline=remember(C.WEIGHTS), seeds=results,
        fits=6, steps_per_fit=6000, total_head_updates=36000, total_exposures=576000,
        source_rows=60000, usable_train_rows=len(training_rows), training_rows_sha256=row_sha,
        exact_deterministic_orders=True, paired_D_P_same_order=True,
        recorded_gradients_and_update_metadata_verified=True,
        gradients_independently_recomputed=False, training_replayed=False,
        new_forwards=0, new_fits=0, GPU_calls=0, new_policy_evaluations=0,
        real_reference_reads=0, image_reads=0,
        quality_claim='Artifact consistency only; no performance or generalization conclusion.',
        reviewed_artifacts=sorted(seen.values(),key=lambda b:b['path']))
    C.write(C.DOC / 'TRAINING_ARTIFACT_AUDIT.json', result)
    print(json.dumps(dict(PASS=True, receipt=C.bound(C.DOC / 'TRAINING_ARTIFACT_AUDIT.json'),
        fits=6,total_head_updates=36000,new_forwards=0,GPU_calls=0)),flush=True)


def selfcheck():
    # Invented metadata only: no repository datasets/checkpoints are opened.
    protocol=dict(steps=6000, optimizer=dict(warmup_steps=10,lr=.001,cosine_final_lr_fraction=.1))
    rows=[dict(step=s,lr=C.learning_rate(s,protocol),arms={a:dict(loss=.2,
        preclip_grad_norm=.1,parameter_update_norm=.01,supported_frames=1,supported_corners=2)
        for a in ('P','D')}) for s in range(1,6001)]
    assert validate_trace(rows,protocol)['updates']==6000
    rows[-1]['step']=5999
    try:validate_trace(rows,protocol)
    except AssertionError:pass
    else:raise AssertionError('Wrong terminal step accepted')
    assert not validate_resumes([],[])
    assert validate_resumes([],[dict(time='synthetic',from_step=6000)])
    try:validate_resumes([],[dict(time='synthetic',from_step=5999)])
    except AssertionError:pass
    else:raise AssertionError('Nonfinal resume mismatch accepted')
    expected={'weight':torch.zeros(2,3)}
    assert validate_state({'weight':torch.ones(2,3)},expected)==6
    try:validate_state({'weight':torch.full((2,3),float('nan'))},expected)
    except AssertionError:pass
    else:raise AssertionError('Nonfinite state accepted')
    model=torch.nn.Linear(2,3)
    parameters=list(model.parameters())
    spec=dict(betas=[.9,.999],weight_decay=.01)
    optimizer=dict(param_groups=[dict(params=list(range(len(parameters))),
        betas=(.9,.999),weight_decay=.01,lr=.0001)],
        state={i:dict(step=torch.tensor(6000.),exp_avg=torch.zeros_like(p),
            exp_avg_sq=torch.ones_like(p)) for i,p in enumerate(parameters)})
    assert validate_optimizer(optimizer,model,spec,.0001)['optimizer_step']==6000
    optimizer['state'][0]['step']=torch.tensor(5999.)
    try:validate_optimizer(optimizer,model,spec,.0001)
    except AssertionError:pass
    else:raise AssertionError('Wrong optimizer step accepted')
    print(json.dumps(dict(PASS=True,synthetic_only=True,actual_checkpoint_reads=0,
        dataset_reads=0,new_forwards=0,new_fits=0,GPU_calls=0)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('train','selfcheck'))
    args=parser.parse_args()
    selfcheck() if args.phase=='selfcheck' else train_audit()

if __name__=='__main__':main()
