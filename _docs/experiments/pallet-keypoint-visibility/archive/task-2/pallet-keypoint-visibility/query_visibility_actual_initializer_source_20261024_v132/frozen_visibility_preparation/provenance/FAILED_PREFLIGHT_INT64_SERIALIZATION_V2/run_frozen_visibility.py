"""Frozen C16/v61 on all original V132 initializer queries; no GT I/O.

Preparation uses code/metadata and generated arrays only. Actual --entry runs
require a completed initializer seal and separate root entry for each mode.
"""
from pathlib import Path
import argparse, ast, collections, hashlib, importlib.util, json, os, sys, time, traceback
import numpy as np

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V / 'ROOT_INITIALIZER_POLICY_BEFORE_INPUTS.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_VISIBILITY_FORWARD.json'
GENERATED = H / 'GENERATED_BOUNDARY_CHECKS.json'
C16_PARENT = P / 'query_visibility_registered_native_diversity_20261023_v129/feature_preparation/FROZEN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
V61_PARENT = P / 'query_visibility_registered_native_diversity_20261023_v129/v61_baseline_preparation/V61_PROTOCOL_BEFORE_SOURCE_FORWARD.json'
N = 6144


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for z in iter(lambda: f.read(1 << 20), b''):
            h.update(z)
    return h.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']
    return Path(binding['path'])


def same(a, b):
    return a['path'] == b['path'] and a['sha256'] == b['sha256']


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def ash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def pure(path, names, extra=None):
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {'np': np}
    ns.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + '::pure', 'exec'), ns)
    return ns


def finite_query(q):
    if q is None:
        return False
    x = np.asarray(q, np.float64)
    return x.shape == (2,) and bool(np.isfinite(x).all() and not (x == [-1., -1.]).all())


def decode(raw, available, mode):
    raw = np.asarray(raw)
    available = np.asarray(available, bool)
    assert raw.shape == available.shape and np.isfinite(raw[available]).all()
    pred = np.full(raw.shape, -1, np.int8)
    pred[available] = ((raw[available] >= 0) if mode == 'c16' else (raw[available] < 0)).astype(np.int8)
    return pred


def input_geometry(proposals, queries, target, native_grid):
    """Whitelist initializer fields; verify all source slots without GT matching."""
    assert len(proposals) == 768 and len(queries) == N
    rows = []
    for scene_index, s in enumerate(proposals):
        assert s['scene_index'] == scene_index and s['base_index'] == scene_index // 4
        assert s['id'] == 'V129_SCENE_%04d' % scene_index
        assert s['base_id'] == 'V129_BASE_%03d' % (scene_index // 4)
        assert s['variant'] == ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')[scene_index % 4]
        assert s['split'] == ('TRAIN' if scene_index < 512 else 'VALIDATION') and s['raw_hw'] == [512, 512]
        cs = s['candidates']
        selected = None if not cs else int(np.argmax([c['score'] for c in cs]))
        assert selected == s['selected_index']
        if selected is None:
            points = None
            box = None
            side = None
        else:
            all9 = np.asarray(cs[selected]['keypoints_xy'], np.float64)
            assert all9.shape == (9, 2) and np.isfinite(all9).all()
            points = all9[:8].copy()
            _, _, side = native_grid(all9, s['raw_hw'])
            side = float(side)
            box = target['query_extent'](points)
        for slot in range(8):
            i = scene_index * 8 + slot
            q = queries[i]
            for key in ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant', 'selected_index'):
                assert q[key] == s[key], (i, key)
            assert q['query_id'] == s['id'] + ':native' + str(slot) and q['native_slot'] == slot
            assert q['actual_original_initializer'] is True
            expected = None if points is None else points[slot].tolist()
            assert q['query_xy'] == expected and q['roi_side'] == side
            good = finite_query(expected)
            if expected is None:
                cont = nearest = False
                assert q['reason'] == 'NO_DETECTION_U'
            else:
                x, y = expected
                cont = bool(good and 0 <= x < 512 and 0 <= y < 512)
                nearest = bool(good and 0 <= np.floor(x + .5) < 512 and 0 <= np.floor(y + .5) < 512)
            assert q['original_query_continuous_valid'] == cont
            assert q['query_nearest_pixel_valid'] == nearest and q['query_rgb96_centre_domain_valid'] == nearest
            rows.append(dict(occurrence_index=i, occurrence_id=q['query_id'], scene_index=scene_index,
                             base_index=s['base_index'], scene_id=s['id'], base_id=s['base_id'],
                             split=s['split'], variant=s['variant'], native_slot=slot,
                             selected_index=selected, raw_hw=s['raw_hw'], raw_RGB_PNG=s['raw_RGB_PNG'],
                             decoded_RGB_sha256=s['decoded_RGB_sha256'], query_xy=expected,
                             initial_points=None if points is None else points.tolist(), L=side,
                             context_box_xyxy=box, finite_non_sentinel_query=good,
                             native_continuous_valid=cont, native_nearest_valid=nearest,
                             context_geometry_valid=target['valid_box'](box),
                             actual_original_initializer=True, source_emulation=False))
    assert len(rows) == N and len({r['occurrence_id'] for r in rows}) == N
    return rows


def authorize(entrypath):
    e = read(entrypath)
    assert e['approved_frozen_visibility'] is True and e['mode'] in ('c16', 'v61')
    assert all(e[k] is False for k in ('fit', 'current223', 'V112', 'PnP', 'q_correction', 'automatic_retry'))
    assert same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL)) and same(e['source_policy'], bind(POLICY))
    p = read(PROTOCOL)
    assert same(p['code'], e['code']) and same(p['source_policy'], e['source_policy'])
    for key in ('generated', 'c16_parent_protocol', 'v61_parent_protocol'):
        check(p[key])
    assert read(p['generated']['path'])['PASS'] and same(read(p['generated']['path'])['code'], e['code'])
    # Completed initializer seal must be verified before any PNG/model/teacher I/O.
    for key in ('initializer_seal', 'initializer_result', 'initializer_proposals', 'initializer_queries'):
        check(e[key])
        assert same(e[key], p[key]), key
    seal = read(e['initializer_seal']['path'])
    result = read(e['initializer_result']['path'])
    assert seal['completed'] and seal['PASS'] and seal['all768_scenes6144_slots_preserved'] and seal['GT_read'] is False
    assert same(seal['result'], e['initializer_result']) and result['completed'] and result['PASS']
    assert result['frames'] == 768 and result['slots'] == N and result['after_state_unchanged'] and result['checkpoint_unchanged']
    assert result['GT_mixed_buffers_mapping_crop_head_fit_current_V112_PnP_access'] == 0
    assert same(result['proposals'], e['initializer_proposals']) and same(result['queries'], e['initializer_queries'])
    assert same(result['inputs'], read(POLICY)['inputs'])
    assert Path(e['initializer_seal']['path']).resolve() == V / 'actual_initializer_v1/PREDICTION_SEAL.json'
    for key in ('C11_producer', 'C8_preprocess', 'C9_target_preprocess', 'local_crop_code', 'target_crop_code',
                'availability_code', 'asset_runtime', 'asset_plan', 'weight', 'source_model_freeze', 'checkpoint', 'head_class_code'):
        check(p['c16'][key])
    for b in p['c16']['official_source_bindings']:
        check(b)
    for b in p['v61']['assets'].values():
        check(b)
    check(p['native_grid_code'])
    freeze = read(p['c16']['source_model_freeze']['path'])
    assert freeze['completed'] and freeze['selected_update'] == 4500 and freeze['threshold_logit'] == 0
    assert same(freeze['checkpoint'], p['c16']['checkpoint']) and not freeze['current_used_for_fit_or_selection']
    target = pure(p['c16']['target_crop_code']['path'], ('query_extent', 'valid_box'))
    grid = pure(p['native_grid_code']['path'], ('native_grid',))['native_grid']
    rows = input_geometry(lines(e['initializer_proposals']['path']), lines(e['initializer_queries']['path']), target, grid)
    out = V / ('frozen_C16_actual_initializer_v1' if e['mode'] == 'c16' else 'frozen_v61_actual_initializer_v1')
    assert not out.exists(), 'Preserve existing output or failed run; no automatic retry.'
    return e, p, rows, out


class PNGCache:
    def __init__(self):
        self.images = collections.OrderedDict()
        self.checked = set()

    def get(self, row):
        import cv2
        b = row['raw_RGB_PNG']
        key = (b['path'], b['sha256'])
        if key not in self.checked:
            check(b)
            self.checked.add(key)
        if key not in self.images:
            assert Path(b['path']).suffix == '.png'
            bgr = cv2.imread(b['path'], cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.dtype == np.uint8
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            assert list(rgb.shape) == row['raw_hw'] + [3] and ash(rgb) == row['decoded_RGB_sha256']
            self.images[key] = rgb
            if len(self.images) > 4:
                self.images.popitem(last=False)
        self.images.move_to_end(key)
        return self.images[key]


def common_arrays(rows):
    return dict(global_index=np.arange(N, dtype=np.int32), corpus=np.full(N, 'NATIVE_ACTUAL_INITIALIZER'),
                source_occurrence_index=np.arange(N, dtype=np.int32),
                occurrence_id=np.array([r['occurrence_id'] for r in rows]),
                scene_index=np.array([r['scene_index'] for r in rows], np.int32),
                base_index=np.array([r['base_index'] for r in rows], np.int32),
                native_slot=np.array([r['native_slot'] for r in rows], np.int8),
                query_xy=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan, np.nan] for r in rows], np.float64),
                L=np.array([r['L'] if r['L'] is not None else np.nan for r in rows], np.float64),
                native_continuous_valid=np.array([r['native_continuous_valid'] for r in rows], bool),
                native_nearest_valid=np.array([r['native_nearest_valid'] for r in rows], bool),
                no_detection=np.array([r['selected_index'] is None for r in rows], bool))


def configure(torch, cv2, p):
    assert torch.__version__ == '2.1.1+cu118' and np.__version__ == '1.26.4' and cv2.__version__ == '4.9.0'
    assert Path(sys.executable).resolve() == Path(p['c16']['runtime']['executable']).resolve() and torch.cuda.is_available()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_c16(e, p, rows, out):
    os.environ['XFORMERS_DISABLED'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch, cv2
    configure(torch, cv2, p)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    a = p['c16']
    producer = module(a['C11_producer']['path'], 'v132_C11_operator')
    c8 = module(a['C8_preprocess']['path'], 'v132_C8_preprocess')
    c9 = module(a['C9_target_preprocess']['path'], 'v132_C9_preprocess')
    target = module(a['target_crop_code']['path'], 'v132_target_math')
    crop = pure(a['local_crop_code']['path'], ('crop',))['crop']
    availability = pure(a['availability_code']['path'], ('factual_availability',))['factual_availability']
    can = np.array([r['finite_non_sentinel_query'] and r['context_geometry_valid'] and r['L'] is not None for r in rows], bool)
    indices = np.flatnonzero(can)
    features = np.zeros((N, 1928), np.float32)
    factual = np.zeros(N, bool)
    context = np.zeros(N, bool)
    qinside = np.zeros(N, bool)
    c4all = np.zeros(N, bool)
    out.joinpath('crop_chunks').mkdir()
    crop_chunks = []
    cache = PNGCache()
    # All6144 original slots have a stored crop record. Missingq crops are zero
    # placeholders with crop_constructed=False; they are never encoder inputs.
    metadata = [dict(r) for r in rows]
    for lo in range(0, N, 512):
        rr = rows[lo:lo+512]
        rgb = np.zeros((len(rr), 96, 96, 3), np.uint8)
        valid = np.zeros((len(rr), 96, 96), bool)
        built = np.zeros(len(rr), bool)
        for k, r in enumerate(rr):
            if r['finite_non_sentinel_query'] and r['L'] is not None:
                cr, vr = crop(cache.get(r), np.asarray(r['query_xy'], np.float64), r['L'], 96)
                rgb[k] = cr.transpose(1, 2, 0)
                valid[k] = vr[0]
                built[k] = True
                fa, _, _, c4 = availability(r['query_xy'], r['raw_hw'], vr, np.zeros((1, 1), bool))
                factual[lo+k], c4all[lo+k] = fa, c4
            metadata[lo+k].update(rgb96_sha256=ash(rgb[k]), valid96_sha256=ash(valid[k]), crop_constructed=bool(built[k]))
        path = out / 'crop_chunks' / ('CROPS_%06d_%06d.npz' % (lo, lo+len(rr)))
        np.savez_compressed(path, rgb96=rgb, valid96=valid, crop_constructed=built, global_index=np.arange(lo, lo+len(rr), dtype=np.int32))
        binding = bind(path)
        crop_chunks.append(binding)
        for k in range(len(rr)):
            metadata[lo+k].update(rgb96_chunk=binding, rgb96_row=k)
    sys.path.insert(0, a['official_repo'])
    from dinov2.hub.backbones import dinov2_vits14_reg
    encoder = dinov2_vits14_reg(pretrained=False)
    state = torch.load(a['weight']['path'], map_location='cpu', weights_only=True)
    encoder.load_state_dict(state, strict=True)
    del state
    encoder = encoder.cuda().float().eval().requires_grad_(False)
    encstate = producer.state_sha(encoder)
    assert encstate == read(a['asset_runtime']['path'])['state_before_SHA256']
    calls = 0
    repeated = False
    start = time.monotonic()
    for lo in range(0, len(indices), 8):
        ix = indices[lo:lo+8]
        rr = [rows[int(i)] for i in ix]
        lr, lv, tr, tv, fractions, central4 = [], [], [], [], [], []
        for i, r in zip(ix, rr):
            image = cache.get(r)
            cr, vr = crop(image, np.asarray(r['query_xy'], np.float64), r['L'], 96)
            assert ash(cr.transpose(1, 2, 0)) == metadata[int(i)]['rgb96_sha256'] and ash(vr[0]) == metadata[int(i)]['valid96_sha256']
            tc, vt = target.sample_target_context(image, r['context_box_xyxy'])
            fa, ca, av, c4 = availability(r['query_xy'], r['raw_hw'], vr, vt)
            factual[i], context[i], c4all[i] = fa, ca, c4
            assert av == (fa and ca)
            lr.append(cr); lv.append(vr); tr.append(tc.transpose(2, 0, 1)); tv.append(vt[None])
            fractions.append([np.float32(vt.mean()), np.float32(vr.mean()), np.float32(vr[:, 32:64, 32:64].mean())])
            central4.append(c4)
        local, _ = c8.preprocess_views(torch.as_tensor(np.stack(lr), device='cuda'), torch.as_tensor(np.stack(lv), device='cuda'))
        ctx = c9.preprocess_target(torch.as_tensor(np.stack(tr), device='cuda'), torch.as_tensor(np.stack(tv), device='cuda'))
        views = torch.cat((ctx, local), 0)
        n = len(rr)
        with torch.inference_mode():
            z = encoder.forward_features(views)
            calls += 1
            if not repeated:
                again = encoder.forward_features(views)
                calls += 1
                assert all(torch.equal(z[k], again[k]) for k in ('x_norm_clstoken', 'x_norm_patchtokens'))
                repeated = True
            cl, pt = z['x_norm_clstoken'], z['x_norm_patchtokens']
            x, inside = producer.assemble(cl[:n], pt[:n], cl[n:2*n], pt[n:2*n], cl[2*n:],
                                         [r['query_xy'] for r in rr], [r['context_box_xyxy'] for r in rr], fractions, central4)
        features[ix] = x.cpu().numpy()
        qinside[ix] = np.asarray(inside, bool)
        if (lo // 8 + 1) % 64 == 0 or lo+n == len(indices):
            print(json.dumps(dict(feature_rows_constructed=lo+n, total_constructible=len(indices), all_original_slots=N,
                                  encoder_calls=calls, seconds=time.monotonic()-start)), flush=True)
    assert producer.state_sha(encoder) == encstate
    assert all(not v.requires_grad for v in encoder.parameters()) and all(not m.training for m in encoder.modules())
    del encoder
    assert np.isfinite(features).all() and not features[~can].any()
    assert np.array_equal(features[:, 1927], qinside.astype(np.float32)) and not features[~qinside, 384:768].any()
    hm = module(a['head_class_code']['path'], 'v132_C16_head')
    payload = torch.load(a['checkpoint']['path'], map_location='cpu', weights_only=True)
    assert payload['update'] == 4500 and same(payload['head_class_code'], a['head_class_code'])
    head = hm.SpatialTargetQueryHead().cuda().float().eval().requires_grad_(False)
    head.load_state_dict(payload['state_dict'], strict=True)
    del payload
    hs = producer.state_sha(head)
    raw = np.full(N, np.nan, np.float32)
    headcalls = 0
    for lo in range(0, len(indices), 512):
        ix = indices[lo:lo+512]
        with torch.inference_mode():
            x = torch.as_tensor(features[ix], device='cuda')
            z = head(x)
            if lo == 0:
                assert torch.equal(z, head(x))
            raw[ix] = z.cpu().numpy()
        headcalls += 1
    assert producer.state_sha(head) == hs and np.isfinite(raw[can]).all() and np.isnan(raw[~can]).all()
    assert all(not v.requires_grad for v in head.parameters()) and all(not m.training for m in head.modules())
    available = can & factual & context
    common = common_arrays(rows)
    ff = out / 'SOURCE_FEATURES_INPUT_ONLY.npz'
    np.savez_compressed(ff, features=features, feature_constructed=can, query_available=available,
                        factual_query_available=factual, context_available=context, Q_inside=qinside, **common)
    pf = out / 'C16_PREDICTIONS_INPUT_ONLY.npz'
    np.savez_compressed(pf, raw_head_logit_before_U=raw, query_logit=np.where(available, raw, 0).astype(np.float32),
                        prediction=decode(raw, available, 'c16'), query_available=available, head_forwarded=can,
                        factual_query_available=factual, context_available=context, Q_inside=qinside,
                        RGB_central4_available=c4all, **common)
    for i, r in enumerate(metadata):
        r.update(feature_constructed=bool(can[i]), query_available=bool(available[i]), factual_query_available=bool(factual[i]),
                 context_available=bool(context[i]), Q_inside=bool(qinside[i]), RGB_central4_available=bool(c4all[i]))
    mf = out / 'INPUT_ROWS_AND_CROP_FACTS.json'
    write(mf, dict(completed=True, rows=metadata, crop_chunks=crop_chunks, GT_read=False))
    return dict(predictions=bind(pf), features=bind(ff), input_crop_facts=bind(mf), rows=N,
                head_forwarded=int(can.sum()), available=int(available.sum()), input_U=int((~available).sum()),
                encoder_calls=calls, head_batches=headcalls, encoder_first_repeat_exact=repeated if len(indices) else None,
                head_first_repeat_exact=bool(len(indices)), encoder_state_sha256=encstate, head_state_sha256=hs,
                encoder_state_unchanged=True, head_state_unchanged=True, selected_update=4500, threshold_logit=0,
                decoder='D raw<0,E raw>=0; factual continuous&nearest&central4 and context available, otherwise U',
                no_input_raw_score='NaN only when no finite non-sentinel q or positive q8 context; no speculative score',
                crop_placeholder='All6144 retained; zero crop only when crop_constructed=False; never encoder input')


def state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode() + b'\0' + str(a.dtype).encode() + b'\0' + str(a.shape).encode() + b'\0' + a.tobytes())
    return h.hexdigest()


def run_v61(e, p, rows, out):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['YOLO_CONFIG_DIR'] = str(out / 'runtime/ultralytics')
    import torch, cv2
    configure(torch, cv2, p)
    a = p['v61']['assets']
    math = module(a['spatial_math']['path'], 'v132_v90_math')
    bm = module(a['backbone_code']['path'], 'v132_v63_backbone')
    back = bm.FrozenYoloFeatures(a['backbone_checkpoint']['path'], device='cuda:0')
    try:
        loaded = state_sha(back.yolo.model)
        with torch.inference_mode():
            back.predict(np.zeros((512, 512, 3), np.uint8), already_padded=False, cpu_features=True)
        bs = state_sha(back.yolo.model)
        node = next(n for n in ast.parse(Path(a['head_class']['path']).read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ProxyHead')
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == p['v61']['ProxyHead_AST_SHA']
        ns = {'nn': torch.nn}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), a['head_class']['path'], 'exec'), ns)
        head = ns['ProxyHead']()
        assert sum(v.numel() for v in head.parameters()) == 28033
        head.load_state_dict(torch.load(a['head_checkpoint']['path'], map_location='cpu', weights_only=True), strict=True)
        head = head.cuda().float().eval().requires_grad_(False)
        hs = state_sha(head)
        raw = np.full(N, np.nan, np.float32)
        forwarded = np.zeros(N, bool)
        native = np.zeros(N, bool)
        any9 = np.zeros(N, bool)
        c4all = np.zeros(N, bool)
        roi = np.zeros((N, 65, 9, 9), np.float16)
        cache = PNGCache()
        facts = []
        calls = 0
        headcalls = 0
        repeated = False
        first_p3 = None
        first_scene_with_query = None
        for si in range(768):
            ids = np.arange(si*8, si*8+8)
            rr = rows[si*8:si*8+8]
            ix = np.array([int(i) for i in ids if rows[int(i)]['finite_non_sentinel_query'] and rows[int(i)]['L'] is not None], np.int32)
            if not len(ix):
                continue
            rgb = cache.get(rr[0])
            with torch.inference_mode():
                captured = back.predict(np.ascontiguousarray(rgb[:, :, ::-1]), already_padded=False, cpu_features=True)
            calls += 1
            a0 = math.expected_affine(rr[0]['raw_hw'])
            gain, offset = bm.canvas_affine(captured['canvas_shape'], captured['input_shape'])
            assert captured['added_border'] == 100 and list(captured['canvas_shape']) == a0['canvas_shape']
            assert list(captured['input_shape']) == a0['input_shape'] and gain == a0['gain'] and np.array_equal(offset, a0['offset'])
            h3, w3 = a0['p3_actual_hw']
            plane = captured['p3'].numpy()
            assert plane.dtype == np.float16 and plane.shape == (64, 80, 80)
            assert not plane[:, h3:, :].any() and not plane[:, :, w3:].any()
            q = np.array([rows[int(i)]['query_xy'] for i in ix], np.float64)
            side = np.array([rows[int(i)]['L'] for i in ix], np.float64)
            x, av, valid = math.sample_p3(plane[:, :h3, :w3], q, side, rr[0]['raw_hw'], gain, offset, 100)
            nv, cont, near = math.query_validity(q, rr[0]['raw_hw'])
            native[ix], any9[ix], roi[ix] = nv, av, x
            for k, i in enumerate(ix):
                r = rows[int(i)]
                assert bool(cont[k]) == r['native_continuous_valid'] and bool(near[k]) == r['native_nearest_valid']
                _, cv = math.crop(rgb, q[k], side[k], 96)
                c4all[i] = bool(cv[:, 47:49, 47:49].all())
            with torch.inference_mode():
                inp = torch.from_numpy(np.ascontiguousarray(x)).float().cuda()
                zz = head(inp).float().cpu().numpy()
                if not repeated:
                    assert np.array_equal(zz, head(inp).float().cpu().numpy())
                    repeated = True
            raw[ix] = zz
            forwarded[ix] = True
            headcalls += 1
            if first_p3 is None:
                first_p3 = plane.copy()
                first_scene_with_query = si
            facts.append(dict(scene_index=si, global_indices=ix.tolist(), raw_RGB_PNG=rr[0]['raw_RGB_PNG'],
                              raw_hw=rr[0]['raw_hw'], canvas_shape=list(captured['canvas_shape']), input_shape=list(captured['input_shape']),
                              P3_actual_shape=[64, h3, w3], added_border=100, gain=gain, offset=offset.tolist(), full_P3_persisted=False))
            if (si + 1) % 64 == 0:
                print(json.dumps(dict(source_scenes_visited=si+1, backbone_calls=calls, original_slots=N)), flush=True)
        if first_p3 is not None:
            image = cache.get(rows[first_scene_with_query*8])
            with torch.inference_mode():
                repeat = back.predict(np.ascontiguousarray(image[:, :, ::-1]), already_padded=False, cpu_features=True)
            assert np.array_equal(repeat['p3'].numpy(), first_p3)
        assert state_sha(back.yolo.model) == bs and state_sha(head) == hs
        assert all(not v.requires_grad for v in back.yolo.model.parameters()) and all(not m.training for m in back.yolo.model.modules())
        assert all(not v.requires_grad for v in head.parameters()) and all(not m.training for m in head.modules())
        assert np.isfinite(raw[forwarded]).all() and np.isnan(raw[~forwarded]).all()
        available = forwarded & native & any9
        common = common_arrays(rows)
        pf = out / 'V61_PREDICTIONS_INPUT_ONLY.npz'
        np.savez_compressed(pf, raw_visible_logit=raw, raw_hidden_logit=-raw,
                            prediction_D0_E1_Uminus1=decode(raw, available, 'v61'), query_available=available,
                            head_forwarded=forwarded, native_query_valid=native, ROI_any_valid=any9,
                            RGB_central4_available=c4all, **common)
        rois = []
        for lo in range(0, N, 2048):
            path = out / ('ROIS_%06d_%06d.npz' % (lo, min(lo+2048, N)))
            np.savez_compressed(path, p3_roi=roi[lo:lo+2048], head_forwarded=forwarded[lo:lo+2048],
                                global_index=np.arange(lo, min(lo+2048, N), dtype=np.int32))
            rois.append(bind(path))
        factpath = out / 'INPUT_P3_FACTS.json'
        write(factpath, dict(completed=True, rows=facts, source_scenes_without_forward=768-len(facts),
                            reason='No finite non-sentinel original q; all original slots retained U', GT_read=False))
        return dict(predictions=bind(pf), ROI_chunks=rois, input_facts=bind(factpath), rows=N,
                    head_forwarded=int(forwarded.sum()), available=int(available.sum()), input_U=int((~available).sum()),
                    backbone_calls=calls, repeat_backbone_calls=int(first_p3 is not None), blank_warmup_calls=1,
                    head_batches=headcalls, head_first_repeat_exact=repeated if headcalls else None,
                    backbone_first_P3_repeat_exact=first_p3 is not None, loaded_state_sha256=loaded,
                    warmed_state_sha256=bs, standard_warmup_state_change=loaded != bs,
                    head_state_sha256=hs, backbone_state_unchanged=True, head_state_unchanged=True,
                    threshold_visible_logit=0, decoder='D visible_logit>=0,E visible_logit<0; continuous&nearest&ROIany required, else U',
                    no_input_raw_score='NaN for no finite non-sentinel original q; ROI placeholders zero and forwarded=False')
    finally:
        back.close()


def run(entrypath):
    e, p, rows, out = authorize(entrypath)
    out.mkdir()
    write(out / 'STARTED.json', dict(completed=False, entry=bind(entrypath), initializer_seal=e['initializer_seal'],
                                   rows=N, source_scenes=768, GT_read=False, automatic_retry=False))
    started = time.monotonic()
    try:
        result = run_c16(e, p, rows, out) if e['mode'] == 'c16' else run_v61(e, p, rows, out)
        for binding in (p['c16']['checkpoint'], p['c16']['weight'], p['v61']['assets']['backbone_checkpoint'], p['v61']['assets']['head_checkpoint']):
            check(binding)
        seal = dict(completed=True, PASS=True, mode=e['mode'], code=e['code'], protocol=e['protocol'],
                    entry=bind(entrypath), initializer_seal=e['initializer_seal'], initializer_result=e['initializer_result'],
                    initializer_queries=e['initializer_queries'], initializer_proposals=e['initializer_proposals'],
                    all768_scenes6144_slots_preserved=True, GT_read=False, fit_current_V112_PnP_q_correction=0,
                    no_model_or_threshold_choice=True, seconds=time.monotonic()-started, **result)
        write(out / 'PREDICTION_SEAL.json', seal)
        print(json.dumps(dict(completed=True, mode=e['mode'], seal=bind(out / 'PREDICTION_SEAL.json'),
                              available=result['available'], rows=N, seconds=time.monotonic()-started)), flush=True)
    except BaseException:
        write(out / 'FAILED_NO_AUTOMATIC_RETRY.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise


def generated_checks(p):
    checks = {}
    target = pure(p['c16']['target_crop_code']['path'], ('query_extent', 'valid_box'))
    grid = pure(p['native_grid_code']['path'], ('native_grid',))['native_grid']
    pts = np.array([[30.+j*11, 40.+j*7] for j in range(9)], np.float64)
    changed = pts.copy(); changed[8] = [1e6, -1e6]
    assert grid(pts, [512, 512])[2] == grid(changed, [512, 512])[2]
    assert target['query_extent'](pts[:8]) == target['query_extent'](changed[:8])
    checks['centroid_not_in_L_or_context'] = True
    assert not finite_query(None) and not finite_query([-1., -1.]) and finite_query([-100., -100.])
    assert target['query_extent'](np.array([[2., 3.]]*8)) is None
    checks['null_sentinel_OOB_and_degenerate_context_separate'] = True
    z = np.array([-1., -0., 0., 1., np.nan])
    av = np.array([True, True, True, True, False])
    assert decode(z, av, 'c16').tolist() == [0, 1, 1, 1, -1]
    assert decode(z, av, 'v61').tolist() == [1, 0, 0, 0, -1]
    checks['opposite_polarities_signedzero_unavailable_NaN_U'] = True
    # All768 synthetic proposal records: q rows include no-detection and OOB.
    proposals, queries = [], []
    for si in range(768):
        pp = pts.copy()
        pp[0] = [511.75, 10.]
        cs = [] if si % 3 == 0 else [dict(score=.8, keypoints_xy=pp.tolist()), dict(score=.8, keypoints_xy=(pp+1).tolist())]
        selected = None if not cs else 0
        s = dict(scene_index=si, base_index=si//4, id='V129_SCENE_%04d' % si, base_id='V129_BASE_%03d' % (si//4),
                 variant=('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')[si%4], split='TRAIN' if si < 512 else 'VALIDATION',
                 raw_hw=[512, 512], raw_RGB_PNG=dict(path='/generated/%04d.png' % si, sha256='0'*64),
                 decoded_RGB_sha256='1'*64, candidates=cs, selected_index=selected)
        proposals.append(s)
        side = None if selected is None else float(grid(pp, [512, 512])[2])
        for j in range(8):
            xy = None if selected is None else pp[j].tolist()
            good = finite_query(xy)
            cont = bool(good and 0 <= xy[0] < 512 and 0 <= xy[1] < 512)
            near = bool(good and 0 <= np.floor(xy[0]+.5) < 512 and 0 <= np.floor(xy[1]+.5) < 512)
            identity = {k: s[k] for k in ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant', 'selected_index')}
            queries.append(dict(identity, query_id=s['id']+':native'+str(j), native_slot=j, query_xy=xy, roi_side=side,
                                original_query_continuous_valid=cont, query_nearest_pixel_valid=near,
                                query_rgb96_centre_domain_valid=near, actual_original_initializer=True,
                                reason='NO_DETECTION_U' if xy is None else 'VALID_QUERY_NOT_A_VISIBILITY_LABEL'))
    rows = input_geometry(proposals, queries, target, grid)
    assert len(rows) == N and sum(r['query_xy'] is None for r in rows) == 2048
    assert rows[8]['query_xy'] == [511.75, 10.] and rows[8]['native_continuous_valid'] and not rows[8]['native_nearest_valid']
    assert len({r['occurrence_id'] for r in rows}) == N
    checks['all6144_order_null_and_native_boundary_rows_retained'] = True
    broken = [dict(q) for q in queries]; broken[8]['query_xy'] = [511., 10.]
    try:
        input_geometry(proposals, broken, target, grid)
    except AssertionError:
        checks['q_correction_rejected'] = True
    else:
        raise AssertionError('Altered query accepted')
    assert len(checks) == 5 and all(checks.values())
    return dict(completed=True, PASS=True, code=bind(__file__), checks=checks,
                actual_source_RGB_or_GT_or_weight_reads_or_model_calls=0)


def prepare():
    c16 = read(C16_PARENT)
    v61 = read(V61_PARENT)
    policy = read(POLICY)
    assert policy['source_scene_count'] == 768 and policy['original_native_slot_count'] == N
    keys = ('C11_producer', 'C8_preprocess', 'C9_target_preprocess', 'local_crop_code', 'target_crop_code',
            'availability_code', 'asset_runtime', 'asset_plan', 'weight', 'source_model_freeze', 'checkpoint',
            'head_class_code', 'official_repo', 'official_revision', 'official_source_bindings', 'runtime')
    p = dict(completed=True, code=bind(__file__), source_policy=bind(POLICY), c16_parent_protocol=bind(C16_PARENT),
             v61_parent_protocol=bind(V61_PARENT), c16={k:c16[k] for k in keys},
             v61={k:v61[k] for k in ('assets', 'ProxyHead_AST_SHA', 'runtime', 'native_preprocessing', 'availability')},
             native_grid_code=policy['native_grid_code'], source_scene_count=768, rows=N,
             schema='Completed initializer seal→result→proposals768+queries6144, verified before PNG/model reads',
             C16_operator='Exact pinned C11 T/Q/F/C/P1928, C8/C9 preprocessing, DINO vits14reg FP32 mathSDPA batch8; head512; update4500/raw0',
             v61_operator='Exact frozen V63 factory/reflection/640rectFP32 then V90 FP16 ROI65→FP32 ProxyHead/rawvisible0',
             geometry='Unchanged first-argmax native0..7; V60 L excludescentroid; V107 original q8 minmax context; no GT association',
             preservation='All6144slot identity/order retained. Finite OOB unchanged. Null q => no input/rawNaN/U; missingcontext C16U. No snapping.',
             input_RGB='Only original PNG allowed; encodedSHA + cv2decodedRGB SHA verify; no mixedNPZ read.',
             predictions='Inputonly raw score/availability/U, row IDs,q,L; no source GT read/hash until both model seals exist in separate evaluation',
             GT_read=False, training=False, current223=False, V112=False, PnP=False, q_correction=False,
             root_owns_actual_execution=True, automatic_retry=False)
    for k in ('local_crop_code', 'target_crop_code', 'availability_code'):
        check(p['c16'][k])
    check(p['native_grid_code'])
    # Input-only completed producer bindings; no RGB, model or GT access here.
    initializer_seal = V / 'actual_initializer_v1/PREDICTION_SEAL.json'
    init_seal = read(initializer_seal)
    assert init_seal['completed'] and init_seal['PASS'] and init_seal['GT_read'] is False
    initializer_result = check(init_seal['result'])
    init_result = read(initializer_result)
    assert init_result['completed'] and init_result['PASS'] and init_result['frames'] == 768 and init_result['slots'] == N
    p.update(initializer_seal=bind(initializer_seal), initializer_result=bind(initializer_result),
             initializer_proposals=init_result['proposals'], initializer_queries=init_result['queries'])
    check(p['initializer_proposals']); check(p['initializer_queries'])
    actual_rows = input_geometry(lines(p['initializer_proposals']['path']), lines(p['initializer_queries']['path']),
                                 pure(p['c16']['target_crop_code']['path'], ('query_extent', 'valid_box')),
                                 pure(p['native_grid_code']['path'], ('native_grid',))['native_grid'])
    p['input_only_preflight'] = dict(rows=len(actual_rows), no_detection_slots=sum(r['selected_index'] is None for r in actual_rows),
                                   continuous_valid=sum(r['native_continuous_valid'] for r in actual_rows),
                                   nearest_valid=sum(r['native_nearest_valid'] for r in actual_rows),
                                   context_geometry_valid=sum(r['context_geometry_valid'] for r in actual_rows),
                                   q_and_L_exact_completed_initializer=True, all_original_slots_preserved=True)
    g = generated_checks(p)
    write(GENERATED, g)
    p['generated'] = bind(GENERATED)
    write(PROTOCOL, p)
    for mode in ('c16', 'v61'):
        e = dict(approved_frozen_visibility=False, mode=mode, fit=False, current223=False, V112=False,
                 PnP=False, q_correction=False, automatic_retry=False, code=p['code'], protocol=bind(PROTOCOL),
                 source_policy=p['source_policy'], **{k:p[k] for k in ('initializer_seal', 'initializer_result',
                                                                       'initializer_proposals', 'initializer_queries')})
        write(H / ('ROOT_' + mode.upper() + '_ENTRY_TEMPLATE.json'), e)
    receipt = dict(completed=True, PASS=True, code=p['code'], protocol=bind(PROTOCOL), generated=bind(GENERATED),
                   C16_template=bind(H/'ROOT_C16_ENTRY_TEMPLATE.json'), v61_template=bind(H/'ROOT_V61_ENTRY_TEMPLATE.json'),
                   command=c16['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_FILLED_ENTRY.json',
                   actual_model_exec=0, actual_source_RGB_or_GT_or_weight_reads=0, root_owns_actual_execution=True)
    write(H / 'PREPARATION.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    if args.prepare:
        assert args.entry is None
        prepare()
    elif args.entry:
        run(args.entry)
    else:
        raise SystemExit('Use --prepare or root-approved --entry')
