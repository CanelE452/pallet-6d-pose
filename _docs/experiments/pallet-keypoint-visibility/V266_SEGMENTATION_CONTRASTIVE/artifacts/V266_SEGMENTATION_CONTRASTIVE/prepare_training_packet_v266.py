"""Prepare only SYN inputs for V266; no current/evaluation material is read."""
from pathlib import Path
import hashlib
import io
import json
import sys
import types
import numpy as np
import torch

HERE = Path(__file__).absolute().parent
PREV = HERE.parent / 'V264_HEATMAP_LOCAL_VISIBILITY'


def safe(value):
    p = Path(value)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink(), str(a)
    return p


def bind(path):
    p = safe(path)
    raw = p.read_bytes()
    return dict(path=str(p), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(b):
    p = safe(b['path'])
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == b['sha256']
    assert len(raw) == b['size_bytes']
    return raw


def write(path, obj):
    with safe(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')


def main():
    torch.set_num_threads(1)
    out = safe(HERE/'training_inputs_v1')
    assert not out.exists()
    out.mkdir()
    complete = json.loads(safe(PREV/'syn_training_v1/COMPLETE.json').read_bytes())
    protocol = json.loads(read(complete['protocol']))
    code = protocol['model_code']
    assert code['sha256'] == 'd412ed01b844ad8e8374d423c3015f20bdcc42a778855aad6d721492613c8c7b'
    m = types.ModuleType('v266_frozen_v264'); m.__file__ = code['path'];sys.modules[m.__name__] = m
    exec(compile(read(code), code['path'], 'exec'), m.__dict__)
    with np.load(io.BytesIO(read(complete['outputs']['SYN_POOLED_FEATURES.npz'])), allow_pickle=False) as z:
        packet = {k: np.array(z[k], copy=True) for k in z.files}
    with np.load(io.BytesIO(read(protocol['source_bindings']['targets'])), allow_pickle=False) as z:
        targets = {k: np.array(z[k], copy=True) for k in z.files}
    dense_complete = json.loads(safe(HERE/'data_v1/COMPLETE.json').read_bytes())
    # Data builder may return its dense binding under either named or output keys.
    dense = bind(HERE/'data_v1/dense_labels.npz')
    with np.load(io.BytesIO(read(dense)), allow_pickle=False) as z:
        base_ids = np.array(z['base_id'], copy=True)
        dense_fit = np.array(z['fit'], copy=True)
    fit = targets['fit'].astype(bool)
    y = targets['visibility_target']
    assert np.array_equal(packet['fit'],fit) and np.array_equal(packet['y'],y)
    assert np.array_equal(fit, dense_fit)
    available = packet['available'] & targets['original_available']
    model = m.VisibilityReadout().eval()
    cp_binding = complete['arms']['HEATMAP_LOCAL']['final']
    cp = torch.load(io.BytesIO(read(cp_binding)), map_location='cpu')
    model.load_state_dict(cp['state_dict'], strict=True)
    design = m.design(packet, 'HEATMAP_LOCAL')
    logits, pred = m.infer(model, design, available, 'cpu')
    old = complete['outputs']['HEATMAP_LOCAL_SYN_DEV_PREDICTIONS.npz']
    with np.load(io.BytesIO(read(old)), allow_pickle=False) as z:
        indices = z['source_index']; difference = float(np.max(np.abs(logits[indices]-z['E_logit'])))
        assert np.array_equal(pred[indices],z['pred'])
    assert difference < 1e-4
    with safe(out/'training_packet.npz').open('xb') as f:
        np.savez_compressed(f, fit=fit, y=y, available=available,
            auxiliary=packet['auxiliary'], uv=packet['uv'],base_E_logit=logits,base_ids=base_ids)
    draws_binding = complete['outputs']['FIT_DRAWS1000x32.npy']
    draws = np.load(io.BytesIO(read(draws_binding)), allow_pickle=False)
    assert draws.shape == (1000,32) and fit[draws].all() and available[draws].all()
    assert ((y[draws] == 0).sum(1) == 16).all() and ((y[draws] == 1).sum(1) == 16).all()
    safe(out/'FIT_DRAWS1000x32.npy').write_bytes(read(draws_binding))
    result = dict(completed=True, PASS=True, SYN_only=True,current_reads=0,
        base_classifier=cp_binding, previous_syn_complete=bind(PREV/'syn_training_v1/COMPLETE.json'),
        data_complete=bind(HERE/'data_v1/COMPLETE.json'), dense_labels=dense,
        source_bindings=protocol['source_bindings'],
        outputs={n:bind(out/n) for n in ('training_packet.npz','FIT_DRAWS1000x32.npy')},
        fit_bases=len(set(base_ids[fit].tolist())), dev_bases=len(set(base_ids[~fit].tolist())),
        usable_fit_D_E=[int((fit&available&(y==a)).sum()) for a in (0,1)],
        base_CPU_replay_max_abs_logit_difference=difference,
        code=bind(Path(__file__).absolute()))
    write(out/'COMPLETE.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
