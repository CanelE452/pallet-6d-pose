"""공개 package 안의 최종 두 모델·저장 특징으로 SOURCE736 logit/집계를 재현한다."""
import csv, hashlib, importlib.util, io, json, os, stat, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parent
ARMS = ('CONTROL_FIXED_Q', 'TREAT_ATTENTION')

def read(relative, expected_sha):
    p = ROOT / relative
    assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
    assert 'v112' not in str(p).lower(), 'V112 경로 거부: filesystem 접근 전'
    for parent in reversed(p.parents):
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW); os.close(fd)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        with os.fdopen(fd, 'rb', closefd=False) as stream: data = stream.read()
    finally: os.close(fd)
    assert hashlib.sha256(data).hexdigest() == expected_sha, relative + ' SHA 불일치'
    return data

def owned(binding, directory, basename):
    # 원 COMPLETE의 절대경로는 읽지 않는다. 명시한 package leaf에만 매핑한다.
    assert binding['path'].replace('\\', '/').rsplit('/', 1)[-1] == basename
    data = read(directory + '/' + basename, binding['sha256'])
    assert 'size_bytes' not in binding or len(data) == binding['size_bytes']
    return data

def metrics(y, z, np):
    pred = (z >= 0).astype(np.int64)
    cm = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] + [0] for a in (0, 1)]
    out = {'rows': len(y), 'CM_D_E_by_D_E_U': cm, 'forced_binary_U': 0}
    for label, k in [('D', 0), ('E', 1)]:
        truth, guess = sum(cm[k]), cm[0][k] + cm[1][k]
        out[label] = dict(truth_count=truth, prediction_count=guess, precision=cm[k][k]/guess if guess else None, recall=cm[k][k]/truth if truth else None)
    out['balanced_recall'] = (out['D']['recall'] + out['E']['recall']) / 2
    return out

def main():
    complete = json.loads(read('results/TRAINING_COMPLETE.json', '2c658c83fb1ac66134cafdddf991bf09efe4dedd8cd442725aef35ea17fdc536'))
    assert complete['completed'] and complete['PASS'] and complete['steps_each'] == 300 and complete['parameters_each'] == 4561
    head = owned(complete['head'], 'code', 'head.py')
    assert hashlib.sha256(head).hexdigest() == '042ffd9c7bced899e0de2546109cdaafff54db27b47d7cd24113eb056570c5dc'
    feature = owned(complete['feature_packet'], 'features', 'CONTEXT_FEATURES.npz')
    summary = json.loads(owned(complete['summary'], 'results', 'SUMMARY.json'))
    rows = list(csv.DictReader(io.StringIO(owned(complete['row_logits'], 'results', 'ALL736_ARM_QUERY_LOGITS.csv').decode())))
    assert len(rows) == 736 and len({(r['arm'], r['row_id']) for r in rows}) == 736
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location('published_v227_head', ROOT / 'code/head.py'))
    exec(compile(head, str(ROOT / 'code/head.py'), 'exec'), module.__dict__)
    shapes = {'contextCLS': (151,384), 'tokens': (151,256,384), 'localCLS': (368,384), 'fixedQ': (368,384), 'condition32': (368,32), 'normalized_active_q': (368,2)}
    with np.load(io.BytesIO(feature), allow_pickle=False) as packet:
        data = {k: packet[k] for k in ['target','fit_bool','row_id','native','frame_index','context_index','context_frame_index']}
        assert all(data[k].shape == (368,) for k in data if k != 'context_frame_index')
        assert data['context_frame_index'].shape == (151,) and data['fit_bool'].dtype == np.bool_
        assert len(set(data['row_id'].tolist())) == 368 and np.isin(data['target'], [0,1]).all()
        assert ((data['context_index'] >= 0) & (data['context_index'] < 151)).all()
        assert np.array_equal(data['frame_index'], data['context_frame_index'][data['context_index']])
        assert list(np.bincount(data['target'][data['fit_bool']].astype(int), minlength=2)) == [57,8]
        assert list(np.bincount(data['target'][~data['fit_bool']].astype(int), minlength=2)) == [267,36]
        result, max_error, seen = {}, 0.0, set()
        by_id = {(r['arm'], r['row_id']): r for r in rows}
        for arm in ARMS:
            checkpoint = torch.load(io.BytesIO(owned(complete['checkpoints'][arm], 'models', arm+'_FINAL300.pt')), map_location='cpu', weights_only=True)
            assert checkpoint['arm'] == arm and checkpoint['updates'] == 300 and checkpoint['seed'] == 17 and checkpoint['parameters'] == 4561
            assert checkpoint['use_attention'] is (arm == ARMS[1]) and checkpoint['feature_packet'] == complete['feature_packet']
            model = module.ContextReadProbe(arm == ARMS[1]).float().eval().requires_grad_(False)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            result[arm] = {}
            for role, mask in [('FIT', data['fit_bool']), ('DEV', ~data['fit_bool'])]:
                ix = np.flatnonzero(mask); ci = data['context_index'][ix]; features = []
                for key in ['localCLS','fixedQ','contextCLS','tokens','condition32','normalized_active_q']:
                    a = packet[key]; assert a.shape == shapes[key] and a.dtype == np.float32
                    a = np.ascontiguousarray(a[ci if key in ['contextCLS','tokens'] else ix]); assert np.isfinite(a).all()
                    features.append(torch.from_numpy(a))
                with torch.inference_mode(): logits = model(*features).numpy()
                assert np.isfinite(logits).all()
                for j, index in enumerate(ix):
                    r = by_id[(arm, str(data['row_id'][index]))]; seen.add((arm,r['row_id']))
                    assert r['role'] == role and int(r['native']) == int(data['native'][index]) and int(r['frame_index']) == int(data['frame_index'][index])
                    assert int(r['truth_D0_E1']) == int(data['target'][index]) and np.isfinite(float(r['logit']))
                    max_error = max(max_error, abs(float(logits[j])-float(r['logit'])))
                    assert int(r['pred_D0_E1']) == int(float(r['logit']) >= 0) == int(logits[j] >= 0)
                result[arm][role] = metrics(data['target'][ix], logits, np)
                assert result[arm][role] == summary['metrics'][arm][role], 'CM/P/R/count 불일치'
    assert len(seen) == 736 and max_error <= 1e-5, 'logit replay 허용오차 초과'
    print(json.dumps(dict(completed=True, PASS=True, rows=736, max_abs_logit_error=max_error, signs_exact=True, metrics=result, optimizer_steps=0), ensure_ascii=False, allow_nan=False))

if __name__ == '__main__': main()
