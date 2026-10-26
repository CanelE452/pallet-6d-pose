"""Bounded paired run: only SYN features/labels; final1000, no checkpoint search."""
from pathlib import Path
import hashlib
import io
import json
import sys
import time
import types
import numpy as np
import torch

HERE = Path(__file__).absolute().parent


def main():
    path = HERE / 'v264_model.py'
    for a in reversed((path, *path.parents)):
        assert not a.is_symlink()
    m = types.ModuleType('v264_model'); m.__file__ = str(path); sys.modules[m.__name__] = m
    exec(compile(path.read_bytes(), str(path), 'exec'), m.__dict__)
    policy = json.loads(m.safe(HERE / 'PROTOCOL_BEFORE_FIT.json').read_text())
    assert m.binding(Path(__file__).absolute()) == policy['training_code']
    assert m.binding(path) == policy['model_code']
    out = m.safe(HERE / 'syn_training_v1'); assert not out.exists(); out.mkdir()
    started = time.time(); device = m.runtime()
    for b in policy['source_bindings'].values():
        m.checked(b['path'], b['sha256'], b['size_bytes'])
    b = policy['source_bindings']
    tokens = np.load(m.safe(b['tokens']['path']), mmap_mode='r', allow_pickle=False)
    support = np.load(m.safe(b['support']['path']), allow_pickle=False)
    with np.load(io.BytesIO(m.read_bound(b['targets'])), allow_pickle=False) as z:
        target = {k: np.array(z[k], copy=True) for k in z.files}
    fit = target['fit'].astype(bool); y = target['visibility_target']; native = target['native_slot']
    assert fit.sum() == 5504 and len(y) == 7040
    frozen = m.frozen_heatmap_model(device)
    frozen_sha = m.state_sha(frozen.state_dict())
    packet = m.pool_features(frozen, tokens, support, native, device)
    np.savez_compressed(out / 'SYN_POOLED_FEATURES.npz', **packet, fit=fit, y=y, native=native)
    available = packet['available'] & target['original_available']
    pools = [np.flatnonzero(fit & available & (y == a)) for a in (0, 1)]
    assert [len(z) for z in pools] == [4626, 190]
    rng = np.random.Generator(np.random.PCG64(17))
    draws = np.concatenate([pool[rng.integers(0, len(pool), size=(1000, 16))] for pool in pools], 1)
    for row in draws:
        rng.shuffle(row)
    assert fit[draws].all() and (y[draws] >= 0).all()
    np.save(out / 'FIT_DRAWS1000x32.npy', draws)
    rows_hash = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    proofs = {}; trained = {}
    for arm in m.ARMS:
        torch.manual_seed(17)
        classifier = m.VisibilityReadout().to(device)
        initial = m.state_sha(classifier.state_dict())
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.as_tensor(m.design(packet, arm), device=device)
        yy = torch.as_tensor(y, dtype=torch.float32, device=device)
        logs = []
        classifier.train()
        for step, batch in enumerate(draws, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(classifier(x[batch]), yy[batch])
            assert bool(torch.isfinite(loss))
            loss.backward(); optimizer.step()
            logs.append({'step': step, 'BCE_balanced_batch': float(loss.detach())})
        torch.save({'state_dict': classifier.cpu().state_dict(), 'arm': arm, 'steps': 1000,
                    'threshold_E_logit': 0, 'input_dimension': 780}, out / (arm + '_FINAL1000.pt'))
        (out / (arm + '_TRAIN1000.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in logs))
        proofs[arm] = {'initial_state_sha256': initial, 'final_state_sha256': m.state_sha(classifier.state_dict()),
                       'parameters': sum(z.numel() for z in classifier.parameters()), 'steps': 1000,
                       'draws_sha256': rows_hash, 'final': m.binding(out / (arm + '_FINAL1000.pt'))}
        trained[arm] = classifier.to(device).eval()
        print(json.dumps({'arm_fit_complete': arm, 'proof': proofs[arm]}), flush=True)
    assert len({z['initial_state_sha256'] for z in proofs.values()}) == 1
    assert len({z['parameters'] for z in proofs.values()}) == 1
    assert frozen_sha == m.state_sha(frozen.state_dict())
    m.write_json(out / 'BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json',
                 {'completed': True, 'arms': proofs, 'frozen_spatial_state_sha256': frozen_sha,
                  'current_reads': 0, 'new_spatial_updates': 0, 'SYN_DEV_classifier_forwards': 0})
    reports = {}; indices = np.flatnonzero(~fit)
    for arm in m.ARMS:
        logits, pred = m.infer(trained[arm], m.design(packet, arm)[indices], available[indices], device)
        np.savez(out / (arm + '_SYN_DEV_PREDICTIONS.npz'), source_index=indices, E_logit=logits,
                 pred=pred, y=y[indices], available=available[indices])
        reports[arm] = m.summary(y[indices], logits, pred)
    m.write_json(out / 'SYN_DEV_RESULTS.json', {'completed': True, 'fixed_threshold': 0, 'results': reports,
                                              'no_recipe_or_checkpoint_or_threshold_selection': True})
    outputs = {name: m.binding(out / name) for name in [
        'SYN_POOLED_FEATURES.npz', 'FIT_DRAWS1000x32.npy', 'BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json',
        'SYN_DEV_RESULTS.json', *[arm + suffix for arm in m.ARMS for suffix in
        ['_FINAL1000.pt', '_TRAIN1000.jsonl', '_SYN_DEV_PREDICTIONS.npz']]]}
    m.write_json(out / 'COMPLETE.json', {'completed': True, 'PASS': True, 'protocol': m.binding(HERE/'PROTOCOL_BEFORE_FIT.json'),
                 'outputs': outputs, 'arms': proofs, 'SYN_DEV_results': reports,
                 'elapsed_seconds': time.time()-started, 'current_or_real_fit_reads': 0,
                 'frozen_spatial_state_sha256': frozen_sha, 'new_optimizer_updates': 2000,
                 'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'GPU': torch.cuda.get_device_name(0)}})
    m.write_json(HERE / 'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json', {
        'completed': True, 'PASS': True, 'protocol': m.binding(HERE/'PROTOCOL_BEFORE_FIT.json'),
        'model_code': m.binding(path), 'training_complete': m.binding(out/'COMPLETE.json'),
        'checkpoints': {arm: proofs[arm]['final'] for arm in m.ARMS}, 'threshold_E_logit': 0,
        'U_policy': 'original unavailable or input unsupported only', 'arms': list(m.ARMS),
        'crop_geometry_native_query_and_location_heads_unchanged': True,
        'current_fit_or_threshold_or_recipe_selection': 0, 'no_further_automatic_fits': True,
        'SYN_DEV_results': reports})
    print(json.dumps({'completed': True, 'SYN_DEV_results': reports,
                      'freeze': m.binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json')}), flush=True)


if __name__ == '__main__':
    main()
