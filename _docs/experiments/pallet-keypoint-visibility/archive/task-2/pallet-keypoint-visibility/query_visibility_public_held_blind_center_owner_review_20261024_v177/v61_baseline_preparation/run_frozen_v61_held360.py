"""Original frozen v61 on MR held360. Preparation never opens model/RGB/NPZ bodies."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V = HERE.parent
SCOPE = V / 'ROOT_V61_PUBLIC_HELD_BASELINE_SCOPE_BEFORE_ASSETS_OR_RGB_BODIES.json'
SCOPE_SHA = '9f34a41537a8288e409b6e41de15e14000b9ffc9a3738ca9eaf38d9c74f73c5b'
PROTOCOL = HERE / 'PROTOCOL_BEFORE_V61_ASSET_OR_RGB_BODIES.json'
CHECKS = HERE / 'GENERATED_PURE_MATH_CHECKS.json'
PREPARATION = HERE / 'PREPARATION_COMPLETE.json'
INPUTS = HERE / 'V61_RUNTIME_INPUTS360_INPUT_ONLY.jsonl'
FRAMES = HERE / 'V61_RUNTIME_FRAMES45_INPUT_ONLY.json'
OUT = V / 'v61_public_held_baseline_v1'
N = 360


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_sha(path)}


def check(binding):
    assert file_sha(binding['path']) == binding['sha256'], 'Binding SHA mismatch'
    return Path(binding['path'])


def metadata(binding):
    p = Path(binding['path'])
    assert p.suffix in ('.json', '.jsonl', '.py'), 'Metadata-only path required'
    return check(binding)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def ash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def finite_input(row):
    if row['query_xy'] is None or row['L'] is None:
        return False
    q = np.asarray(row['query_xy'], np.float64)
    return bool(q.shape == (2,) and np.isfinite(q).all()
                and not (q == [-1., -1.]).all()
                and np.isfinite(row['L']) and row['L'] > 0)


def decode(raw, available):
    raw, available = np.asarray(raw), np.asarray(available, bool)
    assert raw.shape == available.shape and np.isfinite(raw[available]).all()
    pred = np.full(raw.shape, -1, np.int8)
    pred[available] = (raw[available] < 0).astype(np.int8)
    return pred


IDENTITY = ('original_global_index', 'original_feature_index', 'occurrence_unit', 'query_id',
            'frame_index', 'frame_id', 'source_id', 'split', 'native_slot', 'query_xy', 'L',
            'no_detection', 'original_query_available', 'native_continuous_valid',
            'nearest_valid', 'feature_constructed', 'local_central4_available',
            'context_available', 'Q_inside', 'raw_hw', 'canonical_RGB', 'decoded_RGB_sha256')


def assert_identity(row, source):
    assert all(row[k] == source[k] for k in IDENTITY), 'Original input metadata changed'


def input_metadata(scope):
    source = lines(metadata(scope['input_manifest']))
    assert len(source) == N and len({r['query_id'] for r in source}) == N
    rows, frames = [], []
    for j, r in enumerate(source):
        assert r['original_global_index'] == r['original_feature_index'] == r['occurrence_unit'] == 584 + j
        assert r['original_global_index'] == r['frame_index'] * 8 + r['native_slot']
        assert r['split'] == 'SOURCE_VALIDATION' and r['source_id'].split(':')[0] == 'val/000001'
        assert r['query_available'] == r['original_query_available'] == r['native_continuous_valid']
        row = {k: r[k] for k in IDENTITY}
        row.update(evaluation_index=j,
                   source_RGB96_CHW_sha256=r['original_array_hashes_inherited_not_recomputed']['RGB96_CHW_sha256'],
                   source_valid96_sha256=r['original_array_hashes_inherited_not_recomputed']['valid96_sha256'])
        assert_identity(row, r)
        rows.append(row)
    for j in range(45):
        rr = rows[j * 8:j * 8 + 8]
        r = rr[0]
        assert [x['native_slot'] for x in rr] == list(range(8))
        assert all(x['frame_index'] == 73 + j and x['frame_id'] == r['frame_id']
                   and x['raw_hw'] == r['raw_hw'] and x['canonical_RGB'] == r['canonical_RGB']
                   and x['decoded_RGB_sha256'] == r['decoded_RGB_sha256'] for x in rr)
        assert r['raw_hw'] == [720, 1280]
        frames.append(dict(frame_index=r['frame_index'], frame_id=r['frame_id'], source_id=r['source_id'],
                           split=r['split'], raw_hw=r['raw_hw'], canonical_RGB=r['canonical_RGB'],
                           decoded_RGB_sha256=r['decoded_RGB_sha256'], evaluation_indices=list(range(j*8,j*8+8)),
                           original_global_indices=[x['original_global_index'] for x in rr],
                           finite_query_slots=sum(finite_input(x) for x in rr)))
    assert sum(finite_input(x) for x in rows) == 336
    assert sum(f['finite_query_slots'] > 0 for f in frames) == 42
    assert sum(x['native_continuous_valid'] for x in rows) == 252
    assert sum(x['nearest_valid'] for x in rows) == 251
    assert sum(x['native_continuous_valid'] and x['nearest_valid'] for x in rows) == 251
    return rows, frames


def pure_math(path, torch):
    names = {'expected_affine', 'native_grid_explicit', 'sample_p3', 'query_validity', 'crop'}
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = {'np': np, 'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + '::pure', 'exec'), ns)
    return ns


def proxy_node(path, expected):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'ProxyHead')
    assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected
    return node


def generated_checks(scope):
    import torch
    assert torch.__version__ == '2.1.1+cu118' and np.__version__ == '1.26.4'
    a = scope['original_assets_inherited_not_opened_or_hashed']
    op = pure_math(metadata(a['assets']['spatial_math']), torch)
    proxy_node(metadata(a['assets']['head_class']), a['ProxyHead_AST_SHA'])
    passed = ['Original_ProxyHead_AST_and_28033_parameter_formula']
    assert 65*32*9+32 + 32*32*9+32 + 32+1 == 28033
    z = op['expected_affine']([720, 1280])
    assert z['canvas_shape'] == [920, 1480] and z['input_shape'] == [416, 640]
    assert z['p3_actual_hw'] == [52, 80] and z['gain'] == 640/1480
    assert np.array_equal(z['offset'], [0, 9])
    z2 = op['expected_affine']([480, 640])
    assert z2['canvas_shape'] == [680, 840] and z2['input_shape'] == [544, 640]
    assert z2['p3_actual_hw'] == [68, 80] and np.array_equal(z2['offset'], [0, 13])
    passed.append('Two_independent_known_affines_and_actual_P3_shapes')
    q = np.array([[0.,0.], [1279.49,719.49], [1279.5,719.5], [-.25,10.], [1280.,10.], [-1.,-1.]])
    native, cont, near = op['query_validity'](q, [720,1280])
    assert cont.tolist() == [True,True,True,False,False,False]
    assert near.tolist() == [True,True,False,True,False,False]
    assert np.array_equal(native, cont & near)
    passed.append('Continuous_nearest_boundary_and_sentinel_rules')
    q = np.array([[500.,400.], [600.,300.]], np.float64)
    side = np.array([72.,36.], np.float64)
    grid, valid = op['native_grid_explicit'](q, side, [720,1280])
    assert np.array_equal(grid[:,4,4], q) and valid.all()
    assert np.array_equal(grid[:,0,0], q - side[:,None]*4/9)
    passed.append('Explicit_original_per_row_L_and_query_grid_no_recomputed_L')
    yy, xx = np.meshgrid(np.arange(52), np.arange(80), indexing='ij')
    plane = np.stack([xx if k % 2 == 0 else yy for k in range(64)]).astype(np.float16)
    sampled, av, vm = op['sample_p3'](plane, q, side, [720,1280], z['gain'], z['offset'], 100)
    index = ((grid + 100) * z['gain'] + z['offset']) / 8
    full = torch.nn.functional.pad(torch.from_numpy(plane.copy())[None].float(), (0,0,0,28))
    norm = torch.as_tensor(2*(index+.5)/80-1, dtype=torch.float32)
    direct = torch.nn.functional.grid_sample(full.expand(2,-1,-1,-1), norm, align_corners=False,
                                              mode='bilinear', padding_mode='zeros')
    expected = torch.cat((direct, torch.ones((2,1,9,9))), 1).half().numpy()
    assert sampled.dtype == np.float16 and sampled.shape == (2,65,9,9)
    assert np.array_equal(sampled, expected) and av.all() and vm.all()
    ramp_error = float(np.max(np.abs(sampled[:,0].astype(np.float64)-index[...,0])))
    assert ramp_error <= .0625
    passed.append('FP16_actual_plane_to_full80_historical_sampling_byte_parity_and_phase_ramp')
    rejected = 0
    for pp, qq, ss in [(plane.astype(np.float32),q,side), (plane[:,:2,:2],q,side),
                       (plane,q,np.array([0.,36.])), (plane,np.full((2,2),np.nan),side)]:
        try:
            op['sample_p3'](pp, qq, ss, [720,1280], z['gain'], z['offset'], 100)
        except ValueError:
            rejected += 1
    assert rejected == 4
    passed.append('Malformed_dtype_P3_support_L_and_nonfinite_query_fail_closed')
    raw = np.array([-1.,0.,np.nextafter(np.float32(0),np.float32(1)),np.nan], np.float32)
    av = np.array([True,True,True,False])
    assert decode(raw,av).tolist() == [1,0,0,-1]
    forwarded = np.array([True,True,False,True]); native = np.array([True,False,True,True])
    roi = np.array([True,True,True,False]); center = np.zeros(4,bool)
    assert np.array_equal(forwarded & native & roi, [True,False,False,False]) and not center.any()
    assert decode(np.zeros(4,np.float32),forwarded & native & roi).tolist() == [0,-1,-1,-1]
    passed.append('Original_zero_decoder_U_first_and_center4_not_a_gate')
    toy = {k: None for k in IDENTITY}; toy.update(query_xy=[None,None],L=None)
    assert not finite_input(toy)
    changed = dict(toy); changed['native_slot'] = 5
    rejected = False
    try:
        assert_identity(changed,toy)
    except AssertionError:
        rejected = True
    assert rejected
    passed.append('Metadata_identity_drift_rejected_and_null_query_not_forwarded')
    return dict(completed=True, PASS=True, invented_only=True, checks=passed, check_count=len(passed),
                phase_ramp_max_error=ramp_error, actual_model_head_forward_calls=0,
                weight_RGB_P3_score_NPZ_body_or_hash_reads=0,
                runtime=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__))


def prepare():
    assert file_sha(SCOPE) == SCOPE_SHA
    scope = read(SCOPE)
    parent = read(metadata(scope['original_v61_protocol']))
    metadata(scope['original_v61_forward_source'])
    assert parent['v61'] == scope['original_assets_inherited_not_opened_or_hashed']
    for k in ('backbone_code','spatial_math','head_class'):
        metadata(parent['v61']['assets'][k])
    rows, frames = input_metadata(scope)
    common = common_arrays(rows)
    assert common['query_xy'].shape == (N,2) and common['L'].shape == (N,)
    assert np.array_equal(common['global_index'],np.arange(584,944,dtype=np.int32))
    assert int(common['source_original_query_available'].sum()) == 252
    assert int(np.isnan(common['query_xy']).all(1).sum()) == 24
    with INPUTS.open('x') as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False)+'\n')
    write(FRAMES, dict(completed=True, rows=N, frames=frames))
    write(CHECKS, generated_checks(scope))
    p = dict(completed=True, prepared_only=True, code=bind(__file__), scope=bind(SCOPE),
             parent_protocol=scope['original_v61_protocol'], parent_source=scope['original_v61_forward_source'],
             original_input_manifest=scope['input_manifest'], runtime_inputs=bind(INPUTS), runtime_frames=bind(FRAMES),
             checks=bind(CHECKS), v61=parent['v61'], runtime=scope['runtime'], rows=N, frames=45,
             expected_finite_query_rows=336, expected_primary_backbone_calls=42, blank_warmup_calls=1,
             first_P3_repeat_calls=1, primary_head_batches=42, first_head_repeat_calls=1,
             operator='Unchanged V132 run_v61/V63 backbone/V90 explicit qL sampler/ProxyHead28033; MR canonical loader and identity adapter only',
             availability='head_forwarded & native_continuous_valid & native_nearest_valid & ROI_any_valid',
             RGBcentral4='Recorded and checked against original facts only; not an extra v61 gate',
             decoder='Unchanged raw_visible_logit>=0 => D0/Treference; <0=>E1/nonTreference; unavailable=>U-1',
             input_loader='Canonical RGB NPY bound SHA + decoded contiguous uint8 RGB SHA + rawHW; same original decoded RGB, no JPEG decode/re-encoding',
             no_new_proposals_or_qL_changes=True, no_fit=True, no_held_annotation_reads=True,
             output=str(OUT), output_files=['V61_MR_HELD360_PREDICTIONS_INPUT_ONLY.npz',
                                           'V61_MR_HELD360_ROI65_INPUT_ONLY.npz','INPUT_P3_FACTS45_INPUT_ONLY.json','COMPLETE.json'],
             output_score_schema=dict(raw_visible_logit='float32[360]',raw_hidden_logit='float32[360]',
                                      prediction_D0_E1_Uminus1='int8[360]',query_available='own original v61 rule',
                                      identity='global_index/original_global_index/feature_index remain original584..943; evaluation_index0..359; original IDs/q/L/native/split/rawHW/inputflags'),
             full_P3_persisted=False, ROI_cache_shape=[360,65,9,9], ROI_cache_dtype='float16',
             score_blinding='No real raw/prediction/class counts in stdout or JSON receipts. Score NPZ bodies remain sealed from root/annotators until held annotation seals; runner reads no labels.',
             automatic_retry=False, selected_model=None, adopted=False, fully_solved=False)
    write(PROTOCOL,p)
    write(PREPARATION,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),
                           protocol=bind(PROTOCOL),checks=bind(CHECKS),runtime_inputs=bind(INPUTS),runtime_frames=bind(FRAMES),
                           asset_RGB_NPZ_score_body_or_hash_reads=0,actual_forward_or_fit_calls=0))
    write(HERE/'ROOT_ACTUAL_FORWARD_ENTRY_TEMPLATE.json',dict(
        approved_actual_v61_public_held_forward=False,source_only=True,no_held_annotation_reads=True,
        no_private_current_V112_PnP_qL_or_training=True,scope=bind(SCOPE),code=bind(__file__),
        protocol=bind(PROTOCOL),preparation=bind(PREPARATION),output=str(OUT)))
    print(json.dumps(dict(prepared_only=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),
                          checks=bind(CHECKS),preparation=bind(PREPARATION))))


def common_arrays(rows):
    return dict(evaluation_index=np.arange(N,dtype=np.int32),
        global_index=np.array([r['original_global_index'] for r in rows],np.int32),
        original_global_index=np.array([r['original_global_index'] for r in rows],np.int32),
        original_feature_index=np.array([r['original_feature_index'] for r in rows],np.int32),
        occurrence_unit=np.array([r['occurrence_unit'] for r in rows],np.int32),
        query_id=np.array([r['query_id'] for r in rows]),frame_index=np.array([r['frame_index'] for r in rows],np.int32),
        frame_id=np.array([r['frame_id'] for r in rows]),source_id=np.array([r['source_id'] for r in rows]),
        split=np.array([r['split'] for r in rows]),native_slot=np.array([r['native_slot'] for r in rows],np.int8),
        query_xy=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in rows],np.float64),
        L=np.array([r['L'] for r in rows],np.float64),no_detection=np.array([r['no_detection'] for r in rows],bool),
        raw_hw=np.array([r['raw_hw'] for r in rows],np.int32),
        native_continuous_valid=np.array([r['native_continuous_valid'] for r in rows],bool),
        native_nearest_valid=np.array([r['nearest_valid'] for r in rows],bool),
        source_original_query_available=np.array([r['original_query_available'] for r in rows],bool),
        source_original_feature_constructed=np.array([r['feature_constructed'] for r in rows],bool),
        source_original_RGBcentral4_available=np.array([r['local_central4_available'] for r in rows],bool),
        source_original_context_available=np.array([r['context_available'] for r in rows],bool),
        source_original_Q_inside=np.array([r['Q_inside'] for r in rows],bool))


def state_sha(model):
    h = hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        x = v.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def authorize(entry):
    e = read(entry)
    assert e['approved_actual_v61_public_held_forward'] is True and e['source_only'] is True
    assert e['no_held_annotation_reads'] is True and e['no_private_current_V112_PnP_qL_or_training'] is True
    assert file_sha(SCOPE) == SCOPE_SHA and e['scope'] == bind(SCOPE)
    assert e['code'] == bind(__file__) and e['protocol'] == bind(PROTOCOL)
    assert e['preparation'] == bind(PREPARATION) and e['output'] == str(OUT)
    p = read(metadata(e['protocol'])); prep = read(metadata(e['preparation']))
    assert prep['completed'] and prep['PASS'] and p['code'] == e['code'] and p['scope'] == e['scope']
    assert read(metadata(p['checks']))['PASS']
    rows = lines(metadata(p['runtime_inputs'])); frames = read(metadata(p['runtime_frames']))['frames']
    source_rows, source_frames = input_metadata(read(SCOPE))
    assert rows == source_rows and frames == source_frames
    return e,p,rows,frames


class CanonicalCache:
    def __init__(self):
        self.images = {}
        self.bindings = {}

    def get(self,frame):
        i = frame['frame_index']
        if i not in self.images:
            rgb = np.load(check(frame['canonical_RGB']),allow_pickle=False)
            assert rgb.dtype == np.uint8 and list(rgb.shape) == frame['raw_hw']+[3]
            assert ash(rgb) == frame['decoded_RGB_sha256']
            rgb.flags.writeable = False
            self.images[i] = rgb; self.bindings[i] = frame['canonical_RGB']
        return self.images[i]


def run(entry):
    e,p,rows,frames = authorize(entry)  # Before any asset/RGB body hash/load.
    OUT.mkdir(exist_ok=False)
    write(OUT/'STARTED.json',dict(started=True,entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),
                                 score_bodies_blinded_until_held_annotation_seals=True))
    start = time.monotonic()
    back = None
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        os.environ['YOLO_CONFIG_DIR'] = str(OUT/'runtime/ultralytics')
        import torch, cv2, ultralytics
        assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve()
        assert torch.__version__ == '2.1.1+cu118' and np.__version__ == '1.26.4' and cv2.__version__ == '4.9.0'
        assert torch.cuda.is_available() and torch.version.cuda == '11.8'
        torch.set_num_threads(1); cv2.setNumThreads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        assert not torch.is_autocast_enabled()
        a = p['v61']['assets']
        for b in a.values():
            check(b)
        op = pure_math(a['spatial_math']['path'],torch)
        bm = module(a['backbone_code']['path'],'v177_original_v63_backbone')
        back = bm.FrozenYoloFeatures(a['backbone_checkpoint']['path'],device='cuda:0')
        loaded = state_sha(back.yolo.model)
        with torch.inference_mode():
            back.predict(np.zeros((512,512,3),np.uint8),already_padded=False,cpu_features=True)
        warmed = state_sha(back.yolo.model)
        node = proxy_node(a['head_class']['path'],p['v61']['ProxyHead_AST_SHA'])
        ns = {'nn':torch.nn}
        exec(compile(ast.Module(body=[node],type_ignores=[]),a['head_class']['path']+'::ProxyHead_only','exec'),ns)
        head = ns['ProxyHead']()
        assert sum(v.numel() for v in head.parameters()) == 28033
        head.load_state_dict(torch.load(a['head_checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
        head = head.cuda().float().eval().requires_grad_(False)
        hs = state_sha(head)
        raw = np.full(N,np.nan,np.float32); forwarded=np.zeros(N,bool); native=np.zeros(N,bool)
        any9=np.zeros(N,bool); c4=np.zeros(N,bool); roi=np.zeros((N,65,9,9),np.float16)
        roi_valid=np.zeros((N,9,9),bool); cache=CanonicalCache(); facts=[]
        calls=0; headcalls=0; repeated=False; first_p3=None; first_frame=None
        for fr in frames:
            ids=np.array(fr['evaluation_indices'],np.int32)
            ix=np.array([int(i) for i in ids if finite_input(rows[int(i)])],np.int32)
            if not len(ix):
                facts.append(dict(frame_index=fr['frame_index'],original_global_indices=fr['original_global_indices'],
                                  primary_forward=False,skip_reason='No finite original q/positive L',
                                  canonical_RGB_binding_inherited_not_opened=fr['canonical_RGB']))
                continue
            rgb=cache.get(fr)
            with torch.inference_mode():
                captured=back.predict(np.ascontiguousarray(rgb[:,:,::-1]),already_padded=False,cpu_features=True)
            calls+=1
            affine=op['expected_affine'](fr['raw_hw'])
            gain,offset=bm.canvas_affine(captured['canvas_shape'],captured['input_shape'])
            assert captured['added_border']==100 and list(captured['canvas_shape'])==affine['canvas_shape']
            assert list(captured['input_shape'])==affine['input_shape'] and gain==affine['gain'] and np.array_equal(offset,affine['offset'])
            h3,w3=affine['p3_actual_hw']; plane=captured['p3'].numpy()
            assert plane.dtype==np.float16 and plane.shape==(64,80,80)
            assert not plane[:,h3:,:].any() and not plane[:,:,w3:].any()
            q=np.array([rows[int(i)]['query_xy'] for i in ix],np.float64)
            side=np.array([rows[int(i)]['L'] for i in ix],np.float64)
            x,av,valid=op['sample_p3'](plane[:,:h3,:w3],q,side,fr['raw_hw'],gain,offset,100)
            nv,cont,near=op['query_validity'](q,fr['raw_hw'])
            native[ix],any9[ix],roi[ix],roi_valid[ix]=nv,av,x,valid
            for k,i in enumerate(ix):
                r=rows[int(i)]
                assert bool(cont[k])==r['native_continuous_valid'] and bool(near[k])==r['nearest_valid']
                crop,cv=op['crop'](rgb,q[k],side[k],96)
                assert ash(crop)==r['source_RGB96_CHW_sha256'] and ash(cv[0])==r['source_valid96_sha256']
                c4[i]=bool(cv[:,47:49,47:49].all())
                assert bool(c4[i])==r['local_central4_available']
            with torch.inference_mode():
                inp=torch.from_numpy(np.ascontiguousarray(x)).float().cuda()
                zz=head(inp).float().cpu().numpy()
                if not repeated:
                    assert np.array_equal(zz,head(inp).float().cpu().numpy())
                    repeated=True
            assert zz.shape==(len(ix),) and np.isfinite(zz).all()
            raw[ix]=zz; forwarded[ix]=True; headcalls+=1
            if first_p3 is None:
                first_p3=plane.copy(); first_frame=fr
            facts.append(dict(frame_index=fr['frame_index'],original_global_indices=fr['original_global_indices'],
                              primary_forward=True,forwarded_original_global_indices=[rows[int(i)]['original_global_index'] for i in ix],
                              canonical_RGB=fr['canonical_RGB'],decoded_RGB_sha256=fr['decoded_RGB_sha256'],raw_hw=fr['raw_hw'],
                              canvas_shape=list(captured['canvas_shape']),input_shape=list(captured['input_shape']),
                              P3_actual_shape=[64,h3,w3],added_border=100,gain=gain,offset=offset.tolist(),full_P3_persisted=False))
        assert first_p3 is not None and first_frame is not None
        with torch.inference_mode():
            repeat=back.predict(np.ascontiguousarray(cache.get(first_frame)[:,:,::-1]),already_padded=False,cpu_features=True)
        assert np.array_equal(repeat['p3'].numpy(),first_p3)
        assert calls==headcalls==len(cache.images)==42 and len(facts)==45 and repeated
        assert int(forwarded.sum())==336 and int(native.sum())==251 and int(c4.sum())==248
        assert np.isfinite(raw[forwarded]).all() and np.isnan(raw[~forwarded]).all()
        assert state_sha(back.yolo.model)==warmed and state_sha(head)==hs
        assert all(not v.requires_grad for v in back.yolo.model.parameters()) and all(not m.training for m in back.yolo.model.modules())
        assert all(not v.requires_grad for v in head.parameters()) and all(not m.training for m in head.modules())
        for fr in frames:
            if fr['frame_index'] in cache.images:
                check(fr['canonical_RGB']); assert ash(cache.images[fr['frame_index']])==fr['decoded_RGB_sha256']
        for b in a.values():
            check(b)
        available=forwarded & native & any9
        pred=decode(raw,available); common=common_arrays(rows)
        pf=OUT/'V61_MR_HELD360_PREDICTIONS_INPUT_ONLY.npz'
        np.savez_compressed(pf,raw_visible_logit=raw,raw_hidden_logit=-raw,prediction_D0_E1_Uminus1=pred,
                            query_available=available,head_forwarded=forwarded,native_query_valid=native,
                            ROI_any_valid=any9,RGB_central4_available=c4,**common)
        rf=OUT/'V61_MR_HELD360_ROI65_INPUT_ONLY.npz'
        np.savez_compressed(rf,p3_roi=roi,native_grid_valid=roi_valid,head_forwarded=forwarded,
                            global_index=common['global_index'],evaluation_index=common['evaluation_index'])
        ff=OUT/'INPUT_P3_FACTS45_INPUT_ONLY.json'
        write(ff,dict(completed=True,rows=facts,all45_frames_preserved=True,full_P3_persisted=False,
                      held_weak_annotations_read=False,class_counts_or_scores_in_receipt=False))
        result=dict(completed=True,PASS=True,scope=bind(SCOPE),code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),
                    outputs=dict(predictions=bind(pf),ROI65=bind(rf),input_facts=bind(ff)),rows=N,frames=45,
                    original_global_indices=[584,943],head_forwarded=int(forwarded.sum()),available=int(available.sum()),
                    input_U=int((~available).sum()),source_original_available=252,source_original_U=108,
                    native_query_valid=int(native.sum()),RGBcentral4_available=int(c4.sum()),
                    primary_backbone_calls=calls,blank_warmup_calls=1,repeat_backbone_calls=1,
                    primary_head_batches=headcalls,first_head_repeat_calls=1,first_P3_repeat_exact=True,first_head_repeat_exact=True,
                    loaded_backbone_state_sha256=loaded,warmed_backbone_state_sha256=warmed,
                    standard_warmup_state_change=loaded!=warmed,warmed_backbone_state_unchanged=True,
                    head_state_sha256=hs,head_state_unchanged=True,threshold_visible_logit=0,
                    original_q_L_identity_and_RGB96_valid_parity=True,all360_preserved=True,
                    runtime=dict(executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cv2=cv2.__version__,
                                 CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),ultralytics=ultralytics.__version__,
                                 ultralytics_source=str(ultralytics.__file__),TF32=False,autocast=False),
                    score_body_blinded_until_held_annotation_seals=True,no_class_counts_or_real_scores_in_receipt=True,
                    held_annotations_read=False,training=False,private_current_V112_PnP_qL_changes=False,
                    selected_model=None,adopted=False,fully_solved=False,seconds=time.monotonic()-start)
        write(OUT/'COMPLETE.json',result)
        print(json.dumps(dict(completed=True,PASS=True,rows=N,frames=45,head_forwarded=result['head_forwarded'],
                              available=result['available'],input_U=result['input_U'],primary_backbone_calls=calls,
                              score_bodies_blinded_until_held_annotation_seals=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except Exception:
        write(OUT/'FAILED.json',dict(completed=False,PASS=False,code=bind(__file__),protocol=bind(PROTOCOL),
                                    traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        if back is not None:
            back.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--entry',type=Path)
    args=parser.parse_args()
    prepare() if args.prepare else run(args.entry)
