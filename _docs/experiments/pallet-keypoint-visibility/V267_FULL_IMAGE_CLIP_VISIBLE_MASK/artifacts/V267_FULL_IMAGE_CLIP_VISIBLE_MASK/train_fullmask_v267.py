"""Two fixed SYN full-frame mask decoder fits. Requires Root's frozen protocol.

Protocol source_bindings: tokensPLAIN/tokensCS (npy1003x1024x512), targets
(npzmask/fit/frame_ids), draws(npy1000x8). No real/current/query/M0 inputs.
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


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def write_json(path, value):
    p = safe(path)
    assert not p.exists()
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    started = time.time()
    policy_path = safe(HERE/'PROTOCOL_BEFORE_FIT.json')
    policy = json.loads(policy_path.read_text())
    mb = policy['model_code']
    model_path = safe(mb['path'])
    raw = model_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == mb['sha256'] and len(raw) == mb['size_bytes']
    m = types.ModuleType('v267_bound_fullmask_head'); m.__file__ = str(model_path); sys.modules[m.__name__] = m
    exec(compile(raw, str(model_path), 'exec'), m.__dict__)
    assert m.binding(Path(__file__).absolute()) == policy['training_code']
    assert tuple(policy['train_arms']) == m.ARMS
    assert policy['seed'] == 17 and policy['optimizer_updates_each'] == 1000
    assert policy['batch_size'] == 8 and policy['threshold'] == .5
    assert policy['optimizer'] == {'name':'AdamW','lr':.001,'weight_decay':.0001}
    b = policy['source_bindings']
    for source in b.values():
        m.checked(source)
    with np.load(io.BytesIO(m.checked(b['targets']).read_bytes()), allow_pickle=False) as z:
        target = np.array(z['mask'], copy=True).astype(np.float32)
        fit = np.array(z['fit'], copy=True).astype(bool)
        ids = np.array(z['frame_ids'], copy=True)
    draws = np.load(m.checked(b['draws']), allow_pickle=False)
    assert target.shape == (1003,1,32,32) and fit.shape == (1003,)
    assert np.array_equal(ids, np.arange(1003)) and fit.sum() == 768 and (~fit).sum() == 235
    assert np.isfinite(target).all() and target.min() >= 0 and target.max() <= 1
    expected_draws = np.random.Generator(np.random.PCG64(17)).choice(np.flatnonzero(fit), size=(1000,8), replace=True)
    assert np.array_equal(draws, expected_draws) and fit[draws].all()
    frequency = float(target[fit].astype(np.float64).mean())
    wpos, wneg = .5/frequency, .5/(1-frequency)
    assert abs(frequency-policy['FIT_foreground_frequency']) < 1e-12
    assert abs(wpos-policy['positive_weight']) < 1e-10 and abs(wneg-policy['negative_weight']) < 1e-10
    tokens = {a: np.load(m.checked(b[m.FEATURE_KEYS[a]]), mmap_mode='r', allow_pickle=False) for a in m.ARMS}
    assert all(t.shape == (1003,1024,512) and t.dtype == np.float16 for t in tokens.values())
    out = m.safe(HERE/'syn_training_v1')
    assert not out.exists(); out.mkdir()
    torch.set_num_threads(1); torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    device = 'cuda:0'
    proofs, trained = {}, {}
    draw_sha = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    for arm in m.ARMS:
        torch.manual_seed(17)
        head = m.FullMaskHead().to(device)
        initial_sha = m.state_sha(head.state_dict())
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        log_path = m.safe(out/(arm+'_TRAIN1000.jsonl'))
        head.train()
        with log_path.open('x') as log:
            for step, indices in enumerate(draws, 1):
                x = torch.as_tensor(np.array(tokens[arm][indices], copy=True), dtype=torch.float32, device=device)
                y = torch.as_tensor(np.array(target[indices], copy=True), dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(x)
                loss, components = m.fullmask_loss(logits, y, wpos, wneg)
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head.parameters())
                optimizer.step()
                row = {'step':step,'loss':float(loss.detach()), **{k:float(v.detach()) for k,v in components.items()}}
                log.write(json.dumps(row, sort_keys=True)+'\n')
                if step % 100 == 0:
                    log.flush(); print(json.dumps({'arm':arm, **row}), flush=True)
        state = {k:v.detach().cpu() for k,v in head.state_dict().items()}
        checkpoint = m.safe(out/(arm+'_FINAL1000.pt'))
        torch.save({'state_dict':state,'arm':arm,'steps':1000,'model_code_sha256':mb['sha256'],
                    'threshold':.5,'target':'Full-frame M4 visible target binary; no native E labels'}, checkpoint)
        proofs[arm] = {'initial_state_sha256':initial_sha,'final_state_sha256':m.state_sha(state),
             'parameters':sum(p.numel() for p in head.parameters()),'steps':1000,'draws_sha256':draw_sha,
             'checkpoint':m.binding(checkpoint),'log':m.binding(log_path)}
        trained[arm] = head.eval()
    assert len({p['initial_state_sha256'] for p in proofs.values()}) == 1
    assert len({p['parameters'] for p in proofs.values()}) == 1
    write_json(out/'BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json', {'completed':True,'arms':proofs,
       'SYN_DEV_head_forwards':0,'current_or_real_reads':0,'frozen_encoder_updates':0,'q_or_native_inputs':0})
    dev = np.flatnonzero(~fit)
    results = {}
    for arm in m.ARMS:
        pred = m.infer_mask(trained[arm], tokens[arm], dev, device=device, batch_size=32)
        np.savez_compressed(m.safe(out/(arm+'_SYN_DEV_MASKS.npz')), frame_ids=ids[dev], **pred)
        results[arm] = m.segmentation_summary(pred['probability'], target[dev])
    write_json(out/'SYN_DEV_RESULTS.json', {'completed':True,'fixed_final1000':True,
            'no_checkpoint_threshold_or_recipe_selection':True,'results':results})
    names = ['BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json','SYN_DEV_RESULTS.json']
    names += [a+s for a in m.ARMS for s in ('_TRAIN1000.jsonl','_FINAL1000.pt','_SYN_DEV_MASKS.npz')]
    write_json(out/'COMPLETE.json', {'completed':True,'PASS':True,'protocol':m.binding(policy_path),
          'code':m.binding(Path(__file__).absolute()),'outputs':{n:m.binding(out/n) for n in names},
          'arms':proofs,'new_optimizer_updates':2000,'target':'M4 only; whole image, no q/native/M0 input',
          'FIT_foreground_frequency':frequency,'positive_weight':wpos,'negative_weight':wneg,
          'current_or_real_fit_reads':0,'SYN_DEV_results':results,'elapsed_seconds':time.time()-started,
          'runtime':{'torch':torch.__version__,'numpy':np.__version__,'GPU':torch.cuda.get_device_name(0)}})
    write_json(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json', {'completed':True,'PASS':True,
          'protocol':m.binding(policy_path),'training_complete':m.binding(out/'COMPLETE.json'),
          'model_code':mb,'checkpoints':{a:proofs[a]['checkpoint'] for a in m.ARMS},
          'threshold':.5,'full_frame_scope':True,'target':'Visible target segmentation, not native semantic E',
          'current_or_real_fit_or_selection':0,'no_extra_automatic_fits':True})
    print(json.dumps({'completed':True,'PASS':True,'results':results,'freeze':m.binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json')}), flush=True)


if __name__ == '__main__':
    main()
