"""Two bounded SYN-only fits; frozen V264 is the third, saved-output arm.

Protocol source_bindings must bind tokens.npy, support.npy, dense_labels.npz,
training_packet.npz and the unchanged V264 1000x32 FIT draws.  The packet has
fit,y,available,auxiliary,uv,base_E_logit,base_ids.  Dense labels have soft_target
(N,3,16,16) and known_mass (N,16,16).  Both checkpoints freeze before SYN DEV.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import time
import types
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).absolute().parent


def bootstrap_safe(path):
    p = Path(path)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError('Symlink: ' + str(a))
    return p


def main():
    started = time.time()
    protocol_path = bootstrap_safe(HERE / 'PROTOCOL_BEFORE_FIT.json')
    policy = json.loads(protocol_path.read_text())
    code_path = bootstrap_safe(HERE / 'v266_model.py')
    model_bound = policy['model_code']
    assert str(code_path) == model_bound['path']
    code = code_path.read_bytes()
    assert hashlib.sha256(code).hexdigest() == model_bound['sha256'] and len(code) == model_bound['size_bytes']
    m = types.ModuleType('v266_model')
    m.__file__ = str(code_path)
    sys.modules[m.__name__] = m
    exec(compile(code, str(code_path), 'exec'), m.__dict__)
    assert m.binding(Path(__file__).absolute()) == policy['training_code']
    assert policy['optimizer_updates_each'] == 1000 and policy['seed'] == 17
    assert policy['threshold_E_logit'] == 0
    assert tuple(policy['train_arms']) == m.ARMS
    assert policy['segmentation_loss_weight'] == 1.0 and policy['contrastive_loss_weight'] == .1
    assert policy['contrastive_temperature'] == .1
    assert policy['optimizer'] == {'name': 'AdamW', 'lr': .001, 'weight_decay': .0001}
    assert tuple(policy['owner_classes']) == m.OWNER_CLASSES
    class_weights = np.asarray(policy['segmentation_class_weights'], dtype=np.float32)
    assert class_weights.shape == (3,) and np.isfinite(class_weights).all() and (class_weights > 0).all()
    assert abs(float(class_weights.mean()) - 1) < 1e-5
    sources = policy['source_bindings']
    for bound in sources.values():
        m.checked(bound)
    tokens = np.load(m.checked(sources['tokens']), mmap_mode='r', allow_pickle=False)
    support = np.load(m.checked(sources['support']), allow_pickle=False).astype(bool)
    with np.load(io.BytesIO(m.read_bound(sources['training_packet'])), allow_pickle=False) as z:
        packet = {key: np.array(z[key], copy=True) for key in z.files}
    with np.load(io.BytesIO(m.read_bound(sources['dense_labels'])), allow_pickle=False) as z:
        dense = {key: np.array(z[key], copy=True) for key in z.files}
    draws = np.load(m.checked(sources['draws']), allow_pickle=False)
    n = len(tokens)
    assert tokens.shape == (n, 256, 384) and support.shape == (n, 16, 16)
    assert draws.shape == (1000, 32) and draws.dtype.kind in 'iu'
    required = {'fit', 'y', 'available', 'auxiliary', 'uv', 'base_E_logit', 'base_ids'}
    assert required <= set(packet) and all(len(packet[key]) == n for key in required)
    assert packet['auxiliary'].shape == (n, 12) and packet['uv'].shape == (n, 2)
    assert packet['base_ids'].dtype.kind in 'iu'
    assert dense['soft_target'].shape == (n, 3, 16, 16) and dense['known_mass'].shape == (n, 16, 16)
    for key in ('auxiliary', 'uv', 'base_E_logit'):
        assert np.isfinite(packet[key]).all()
    soft = dense['soft_target'].astype(np.float32)
    mass = dense['known_mass'].astype(np.float32)
    assert np.isfinite(soft).all() and np.isfinite(mass).all()
    assert soft.min() >= 0 and soft.max() <= 1.001 and mass.min() >= 0 and mass.max() <= 1.001
    assert np.max(np.abs(soft.sum(1)[mass > 0] - 1)) < .002
    fit = packet['fit'].astype(bool)
    y = packet['y']
    available = packet['available'].astype(bool) & support.reshape(n, -1).any(1)
    pools = [np.flatnonzero(fit & available & (y == c)) for c in (0, 1)]
    assert [len(pool) for pool in pools] == policy['usable_VIS_FIT_D_E']
    assert fit[draws].all() and available[draws].all()
    assert np.all((y[draws] == 0).sum(1) == 16) and np.all((y[draws] == 1).sum(1) == 16)
    assert not set(packet['base_ids'][fit]) & set(packet['base_ids'][~fit])
    out = m.safe(HERE / 'syn_training_v1')
    assert not out.exists(), out
    out.mkdir()
    device = m.runtime()
    class_weight_tensor = torch.as_tensor(class_weights, device=device)
    rows_sha = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    # Eligibility is built from FIT pixels only.  It never consults DEV or real data.
    candidates = m.contrast_candidates(soft, mass, support, fit)
    proofs, trained = {}, {}
    for arm in m.ARMS:
        torch.manual_seed(17)
        model = m.OwnerVisibilityResidual().to(device)
        initial = m.state_sha(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.Generator(np.random.PCG64(17))
        sample_hash = hashlib.sha256()
        log_path = m.safe(out / (arm + '_TRAIN1000.jsonl'))
        model.train()
        with log_path.open('x') as log:
            for step, batch in enumerate(draws, 1):
                optimizer.zero_grad(set_to_none=True)
                arguments = m.as_batch(tokens, support, packet, batch, device)
                result = model(**arguments)
                yy = torch.as_tensor(np.array(y[batch], copy=True), dtype=torch.float32, device=device)
                soft_batch = torch.as_tensor(soft[batch], device=device)
                mass_batch = torch.as_tensor(mass[batch], device=device)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(result['E_logit'], yy)
                seg = m.segmentation_loss(result['owner_logits'], soft_batch, mass_batch,
                                          arguments['support'], class_weight_tensor)
                contrast = result['embedding'].sum() * 0
                contrast_stats = {'sampled_pixels': 0, 'active_anchors': 0, 'positive_pairs': 0}
                if arm == 'SEGMENTATION_CONTRAST':
                    sample = m.sample_contrast_pixels(batch, candidates, rng)
                    for part in sample:
                        sample_hash.update(part.astype('<i8').tobytes())
                    base_ids = torch.as_tensor(packet['base_ids'][batch], dtype=torch.int64, device=device)
                    contrast, contrast_stats = m.pixel_contrast_loss(result['embedding'], base_ids, sample, .1)
                loss = bce + seg + .1 * contrast
                assert bool(torch.isfinite(loss)), (arm, step)
                loss.backward()
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                optimizer.step()
                record = {'step': step, 'loss': float(loss.detach()), 'BCE_balanced_batch': float(bce.detach()),
                          'segmentation_CE': float(seg.detach()), 'pixel_contrast': float(contrast.detach()),
                          **contrast_stats}
                log.write(json.dumps(record, sort_keys=True) + '\n')
                if step % 100 == 0:
                    log.flush()
                    print(json.dumps({'arm': arm, **record}), flush=True)
        checkpoint_path = m.safe(out / (arm + '_FINAL1000.pt'))
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({'state_dict': state, 'arm': arm, 'steps': 1000, 'threshold_E_logit': 0,
                    'input_dimension': m.INPUT_DIMENSION, 'owner_classes': list(m.OWNER_CLASSES),
                    'model_code_sha256': model_bound['sha256'], 'frozen_baseline_external_input': True}, checkpoint_path)
        proofs[arm] = {'initial_state_sha256': initial, 'final_state_sha256': m.state_sha(state),
                       'parameters': sum(p.numel() for p in model.parameters()), 'steps': 1000,
                       'draws_sha256': rows_sha, 'contrast_sample_sha256': sample_hash.hexdigest(),
                       'final': m.binding(checkpoint_path), 'training_log': m.binding(log_path)}
        trained[arm] = model.eval()
        print(json.dumps({'arm_fit_complete': arm, 'proof': proofs[arm]}), flush=True)
    assert len({p['initial_state_sha256'] for p in proofs.values()}) == 1
    assert len({p['parameters'] for p in proofs.values()}) == 1
    m.write_json(out / 'BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json', {
        'completed': True, 'arms': proofs, 'current_reads': 0, 'SYN_DEV_model_forwards': 0,
        'new_backbone_or_spatial_or_baseline_updates': 0, 'fixed_final1000': True})
    indices = np.flatnonzero(~fit)
    reports = {}
    for arm in m.ARMS:
        result = m.infer(trained[arm], tokens, support, packet, indices, device)
        np.savez_compressed(out / (arm + '_SYN_DEV_PREDICTIONS.npz'), source_index=indices, y=y[indices], **result)
        reports[arm] = m.summary(y[indices], result['pred'])
        reports[arm]['segmentation'] = m.segmentation_summary(
            result['owner_probability'], soft[indices], mass[indices], support[indices])
    baseline_pred = np.where(available[indices], np.where(packet['base_E_logit'][indices] > 0, 1, 0), -1).astype(np.int8)
    reports['CURRENT_HEATMAP_LOCAL'] = m.summary(y[indices], baseline_pred)
    m.write_json(out / 'SYN_DEV_RESULTS.json', {'completed': True, 'fixed_threshold': 0, 'results': reports,
                  'no_recipe_or_checkpoint_or_threshold_selection': True})
    output_names = ['BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json', 'SYN_DEV_RESULTS.json']
    output_names += [arm + suffix for arm in m.ARMS for suffix in
                     ('_FINAL1000.pt', '_TRAIN1000.jsonl', '_SYN_DEV_PREDICTIONS.npz')]
    m.write_json(out / 'COMPLETE.json', {'completed': True, 'PASS': True,
        'protocol': m.binding(protocol_path), 'outputs': {name: m.binding(out / name) for name in output_names},
        'arms': proofs, 'SYN_DEV_results': reports, 'elapsed_seconds': time.time() - started,
        'segmentation_class_weights': class_weights.tolist(),
        'segmentation_class_weights_source': 'SYN_FIT_ONLY; protocol-bound inverse sqrt class frequency, arithmetic mean normalized to one',
        'current_or_real_fit_reads': 0, 'new_optimizer_updates': 2000,
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'GPU': torch.cuda.get_device_name(0)}})
    m.write_json(HERE / 'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json', {
        'completed': True, 'PASS': True, 'protocol': m.binding(protocol_path),
        'model_code': m.binding(code_path), 'training_complete': m.binding(out / 'COMPLETE.json'),
        'checkpoints': {arm: proofs[arm]['final'] for arm in m.ARMS}, 'threshold_E_logit': 0,
        'U_policy': 'original unavailable or token unsupported only', 'train_arms': list(m.ARMS),
        'crop_geometry_native_query_and_location_heads_unchanged': True,
        'current_fit_or_threshold_or_recipe_selection': 0, 'no_further_automatic_fits': True,
        'SYN_DEV_results': reports})
    print(json.dumps({'completed': True, 'SYN_DEV_results': reports,
                      'freeze': m.binding(HERE / 'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json')}), flush=True)


if __name__ == '__main__':
    main()
