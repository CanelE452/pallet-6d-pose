"""Source-only chunked RGB -> exact frozen DINO1928 -> frozen C16.

No GT reader and no model/payload I/O on import. Root binds completed corpora.
"""
import os, sys
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, ast, collections, hashlib, importlib.util, json, time, traceback
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
PARENT = P / 'query_visibility_registered_native_render_20261023_v128/c16_baseline_preparation/FROZEN_C16_PROTOCOL_BEFORE_FORWARD.json'
PROTOCOL = H / 'FROZEN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
OUT = V / 'frozen_C16_source_baseline_v1'
BATCH = 8

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for z in iter(lambda: f.read(1 << 20), b''): h.update(z)
    return h.hexdigest()

def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def check(b):
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])
def read(p): return json.loads(Path(p).read_text())
def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
def ash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def module(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m
def pure(p, names):
    nodes = [n for n in ast.parse(Path(p).read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p) + '::pure', 'exec'), ns)
    return ns

def validate_inputs(rows):
    assert rows and [r['occurrence_index'] for r in rows] == list(range(len(rows)))
    assert len({r['id'] for r in rows}) == len(rows)
    forbidden = {'reference_xy', 'raw_state', 'teacher_label', 'owner', 'part_id', 'witnesses', 'error_px'}
    for r in rows:
        assert not forbidden.intersection(r)
        q8 = np.asarray(r['initial_points'], np.float64)
        q = np.asarray(r['query_xy'], np.float64)
        j = r['carrier_slot']
        assert q8.shape == (8, 2) and q.shape == (2,) and 0 <= j < 8
        assert np.isfinite(q8).all() and np.array_equal(q8[j], q)
        good = np.isfinite(q8).all(1) & ~(q8 == [-1, -1]).all(1)
        points = q8[good]
        assert len(points) >= 2
        box = np.r_[points.min(0), points.max(0)]
        assert np.all(box[2:] > box[:2]) and box.tolist() == r['context_box_xyxy']
        assert float(np.clip(.25 * np.ptp(points, axis=0).max(), 16, max(r['raw_hw']))) == r['L']
        assert r['full_RGB_format'] == 'NPZ_RGB_UINT8_HWC' and r['full_RGB_key'] == 'RGB_uint8'
        assert r['rgb96_row'] >= 0 and set(r['rgb96_chunk']) >= {'path', 'sha256'}

def authorize(entry):
    e = read(entry)
    assert e['approved_source_features_and_frozen_C16'] is True
    assert all(e[k] is False for k in ('fit', 'current223', 'V112', 'PnP', 'initializer', 'automatic_retry'))
    for k in ('code', 'protocol', 'source_policy'): check(e[k])
    p = read(PROTOCOL)
    assert e['code'] == p['code'] == bind(__file__) and e['protocol'] == bind(PROTOCOL)
    assert e['source_policy'] == p['source_policy']
    for k in ('parent_protocol', 'C11_producer', 'C11_protocol', 'C8_preprocess', 'C9_target_preprocess', 'local_crop_code', 'target_crop_code', 'asset_runtime', 'asset_plan', 'weight', 'source_model_freeze', 'checkpoint', 'head_class_code', 'availability_code'): check(p[k])
    for b in p['official_source_bindings']: check(b)
    freeze = read(p['source_model_freeze']['path'])
    assert freeze['completed'] and freeze['selected_update'] == 4500 and freeze['threshold_logit'] == 0
    assert freeze['checkpoint'] == p['checkpoint'] and not freeze['current_used_for_fit_or_selection']
    corpora = []
    assert {d['name']: d['expected_rows'] for d in e['corpora']} == {'NATIVE': 55296, 'PUBLIC_AUX': 918}
    for descriptor in e['corpora']:
        check(descriptor['complete']); check(descriptor['input_rows'])
        done = read(descriptor['complete']['path'])
        assert done['completed'] and done['PASS']
        assert done['input_manifest'] == descriptor['input_rows']
        packet = read(descriptor['input_rows']['path'])
        assert packet['completed'] and len(packet['rows']) == descriptor['expected_rows']
        validate_inputs(packet['rows'])
        corpora.append((descriptor['name'], packet['rows']))
    assert not OUT.exists(), 'Preserve failed/completed attempt; explicit repair/new output only.'
    return e, p, corpora

def run(entry):
    e, p, corpora = authorize(entry)
    import torch
    assert torch.__version__ == p['runtime']['torch'] == '2.1.1+cu118'
    assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve() and torch.cuda.is_available()
    rows = []; origin = []
    for name, rr in corpora:
        rows.extend(rr)
        origin.extend((name, i) for i in range(len(rr)))
    N = len(rows)
    OUT.mkdir(); (OUT / 'feature_chunks').mkdir()
    write(OUT / 'STARTED.json', {'entry': bind(entry), 'rows': N, 'GT_read': False, 'automatic_retry': False})
    started = time.monotonic()
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
        producer = module(p['C11_producer']['path'], 'v129_exact_C11')
        c8 = module(p['C8_preprocess']['path'], 'v129_exact_C8')
        c9 = module(p['C9_target_preprocess']['path'], 'v129_exact_C9')
        tm = module(p['target_crop_code']['path'], 'v129_exact_context')
        crop = producer.pure_crop()
        # Pure geometry helper only; parent I/O/prepare/run functions are never invoked.
        availability = pure(Path(p['availability_code']['path']), ('factual_availability',))['factual_availability']
        image_cache = collections.OrderedDict(); crop_cache = collections.OrderedDict()
        checked = set()
        def load_checked(binding):
            key = (binding['path'], binding['sha256'])
            if key not in checked: check(binding); checked.add(key)
            return key
        def native(r):
            key = load_checked(r['full_RGB'])
            if key not in image_cache:
                with np.load(key[0], allow_pickle=False) as f: image_cache[key] = np.array(f['RGB_uint8'], copy=True)
                image = image_cache[key]
                assert image.dtype == np.uint8 and image.shape == tuple(r['raw_hw']) + (3,) and ash(image) == r['decoded_RGB_sha256']
                if len(image_cache) > 4: image_cache.popitem(last=False)
            image_cache.move_to_end(key)
            return image_cache[key]
        def stored(r):
            key = load_checked(r['rgb96_chunk'])
            if key not in crop_cache:
                with np.load(key[0], allow_pickle=False) as f: crop_cache[key] = (f['rgb96'], f['valid96'], f['occurrence_index'])
                if len(crop_cache) > 2: crop_cache.popitem(last=False)
            crop_cache.move_to_end(key)
            rgb, valid, indices = crop_cache[key]; j = r['rgb96_row']
            assert indices[j] == r['occurrence_index']
            assert ash(rgb[j]) == r['rgb96_sha256'] and ash(valid[j]) == r['valid96_sha256']
            return rgb[j].transpose(2, 0, 1).copy(), valid[j][None].copy()
        sys.path.insert(0, p['official_repo'])
        from dinov2.hub.backbones import dinov2_vits14_reg
        encoder = dinov2_vits14_reg(pretrained=False)
        state = torch.load(p['weight']['path'], map_location='cpu', weights_only=True)
        encoder.load_state_dict(state, strict=True); del state
        encoder = encoder.cuda().float().eval().requires_grad_(False)
        encstate = producer.state_sha(encoder)
        assert encstate == read(p['asset_runtime']['path'])['state_before_SHA256']
        features = np.empty((N, 1928), np.float32)
        factual = np.zeros(N, bool); context = np.zeros(N, bool); qinside = np.zeros(N, bool)
        calls = 0; repeated = False; chunks = []; checkpoint_start = 0
        for start in range(0, N, BATCH):
            batch = rows[start:start + BATCH]; n = len(batch)
            lr = []; lv = []; tr = []; tv = []; fractions = []; central4 = []
            for offset, r in enumerate(batch):
                image = native(r); a, valid = crop(image, np.array(r['query_xy'], np.float64), r['L'], 96)
                saved, savedvalid = stored(r)
                assert np.array_equal(a, saved) and np.array_equal(valid, savedvalid), ('native crop replay', start + offset)
                target, vt = tm.sample_target_context(image, r['context_box_xyxy'])
                fa, ca, aa, c4 = availability(r['query_xy'], r['raw_hw'], valid, vt)
                assert fa == r['factual_query_available'] and ca == r['context_available'] and aa == r['query_available']
                factual[start + offset] = fa; context[start + offset] = ca
                lr.append(a); lv.append(valid); tr.append(target.transpose(2, 0, 1)); tv.append(vt[None])
                fractions.append([np.float32(vt.mean()), np.float32(valid.mean()), np.float32(valid[:, 32:64, 32:64].mean())]); central4.append(c4)
            local, _ = c8.preprocess_views(torch.as_tensor(np.stack(lr), device='cuda'), torch.as_tensor(np.stack(lv), device='cuda'))
            target = c9.preprocess_target(torch.as_tensor(np.stack(tr), device='cuda'), torch.as_tensor(np.stack(tv), device='cuda'))
            views = torch.cat((target, local), 0)
            with torch.inference_mode():
                z = encoder.forward_features(views); calls += 1
                if not repeated:
                    again = encoder.forward_features(views); calls += 1
                    assert all(torch.equal(z[k], again[k]) for k in ('x_norm_clstoken', 'x_norm_patchtokens'))
                    repeated = True
                cl = z['x_norm_clstoken']; pt = z['x_norm_patchtokens']
                x, inside = producer.assemble(cl[:n], pt[:n], cl[n:2*n], pt[n:2*n], cl[2*n:], [r['query_xy'] for r in batch], [r['context_box_xyxy'] for r in batch], fractions, central4)
            end = start + n
            features[start:end] = x.cpu().numpy(); qinside[start:end] = np.asarray(inside, bool)
            if end - checkpoint_start >= 2048 or end == N:
                f = OUT / 'feature_chunks' / f'UNITS_{checkpoint_start:06d}_{end:06d}.npz'
                np.savez_compressed(f, features=features[checkpoint_start:end], global_index=np.arange(checkpoint_start, end, dtype=np.int32))
                chunks.append(bind(f)); checkpoint_start = end
                print(json.dumps({'feature_rows_complete': end, 'rows': N, 'encoder_calls': calls, 'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
        assert np.isfinite(features).all() and np.array_equal(features[:, 1927], qinside.astype(np.float32))
        assert not features[~qinside, 384:768].any() and producer.state_sha(encoder) == encstate
        assert all(not t.requires_grad for t in encoder.parameters()) and all(not t.training for t in encoder.modules())
        del encoder
        available = factual & context
        ff = OUT / 'SOURCE_FEATURES_INPUT_ONLY.npz'
        np.savez_compressed(ff, features=features, query_available=available, factual_query_available=factual, context_available=context, Q_inside=qinside, global_index=np.arange(N, dtype=np.int32), corpus=np.array([x[0] for x in origin]), source_occurrence_index=np.array([x[1] for x in origin], np.int32))
        write(OUT / 'FEATURE_COMPLETE.json', {'completed': True, 'PASS': True, 'entry': bind(entry), 'code': e['code'], 'protocol': e['protocol'], 'corpora': e['corpora'], 'output': bind(ff), 'feature_chunks': chunks, 'rows': N, 'features_shape': list(features.shape), 'encoder_calls': calls, 'first_batch_repeat_exact': True, 'encoder_state_unchanged': True, 'GT_read': False, 'all_saved_crop_replay_exact': True})
        hm = module(p['head_class_code']['path'], 'v129_frozen_C16_head')
        payload = torch.load(p['checkpoint']['path'], map_location='cpu', weights_only=True)
        assert payload['update'] == 4500 and payload['head_class_code'] == p['head_class_code']
        head = hm.SpatialTargetQueryHead().cuda().float().eval().requires_grad_(False)
        head.load_state_dict(payload['state_dict'], strict=True); del payload
        hs = producer.state_sha(head); logits = []
        with torch.inference_mode():
            for lo in range(0, N, 512):
                x = torch.as_tensor(features[lo:lo+512], device='cuda'); z = head(x)
                if lo == 0: assert torch.equal(z, head(x))
                logits.append(z.cpu().numpy())
        raw = np.concatenate(logits).astype(np.float32)
        assert raw.shape == (N,) and np.isfinite(raw).all() and producer.state_sha(head) == hs
        pred = np.where(available, (raw >= 0).astype(np.int8), -1).astype(np.int8)
        pf = OUT / 'FROZEN_C16_PREDICTIONS_INPUT_ONLY.npz'
        np.savez_compressed(pf, query_logit=np.where(available, raw, 0), raw_head_logit_before_U=raw, prediction=pred, query_available=available, global_index=np.arange(N, dtype=np.int32), corpus=np.array([x[0] for x in origin]), source_occurrence_index=np.array([x[1] for x in origin], np.int32))
        write(OUT / 'PREDICTION_SEAL.json', {'completed': True, 'PASS': True, 'entry': bind(entry), 'code': e['code'], 'protocol': e['protocol'], 'features': bind(ff), 'predictions': bind(pf), 'feature_receipt': bind(OUT/'FEATURE_COMPLETE.json'), 'corpora': e['corpora'], 'rows': N, 'available': int(available.sum()), 'decoder': 'D raw<0,E raw>=0; factual/context unavailable U first', 'threshold_logit': 0, 'checkpoint': p['checkpoint'], 'head_state_unchanged': True, 'GT_read': False, 'fit_current_V112_PnP': 0, 'seconds': time.monotonic()-started})
        print(json.dumps({'completed': True, 'seal': bind(OUT/'PREDICTION_SEAL.json'), 'seconds': time.monotonic()-started}), flush=True)
    except BaseException:
        write(OUT / 'FAILED_NO_RESTART.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise

def prepare():
    parent = read(PARENT)
    keys = ('C11_producer', 'C11_protocol', 'C8_preprocess', 'C9_target_preprocess', 'local_crop_code', 'target_crop_code', 'asset_runtime', 'asset_plan', 'weight', 'source_model_freeze', 'checkpoint', 'head_class_code', 'official_repo', 'official_revision', 'official_source_bindings', 'runtime')
    p = {k: parent[k] for k in keys}
    policy = V / 'ROOT_FROZEN_FEATURE_POLICY_BEFORE_FORWARD.json'
    p.update(completed=True, code=bind(__file__), parent_protocol=bind(PARENT), source_policy=bind(policy), availability_code=bind(P/'query_visibility_boundary_query_source_20261023_v125/feature_baseline_preparation/run_frozen_boundary_source.py'), output=str(OUT), input_schema='ChunkedRGB96 HWC/validHW; same exact1928 T/Q/F/C/P operator; input-only rows. No GT sidecar or semantic indices as features.', no_fit_current_V112_PnP=True)
    write(PROTOCOL, p)
    write(H/'ROOT_ENTRY_TEMPLATE.json', {'approved_source_features_and_frozen_C16': False, 'fit': False, 'current223': False, 'V112': False, 'PnP': False, 'initializer': False, 'automatic_retry': False, 'code': p['code'], 'protocol': bind(PROTOCOL), 'source_policy': bind(policy), 'corpora': []})
    print(json.dumps({'code': bind(__file__), 'protocol': bind(PROTOCOL), 'template': bind(H/'ROOT_ENTRY_TEMPLATE.json')}))

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--prepare', action='store_true'); a.add_argument('--entry', type=Path); o = a.parse_args()
    if o.prepare: prepare()
    elif o.entry: run(o.entry)
    else: raise SystemExit('Use --prepare or --entry ROOT_BOUND_ENTRY.json')
