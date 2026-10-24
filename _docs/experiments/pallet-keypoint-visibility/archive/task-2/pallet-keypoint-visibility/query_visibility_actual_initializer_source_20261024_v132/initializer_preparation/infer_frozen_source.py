"""V132: unchanged original initializer on all 768 immutable V129 source RGBs.

The producer reads PNGs only. No GT, mixed NPZ, mapping, crop, or head is read.
"""
from pathlib import Path
import argparse, ast, datetime, hashlib, importlib.util, json, os, time, traceback
import numpy as np

HERE = Path(__file__).resolve().parent
V132 = HERE.parent
BASE = V132.parent
POLICY = V132 / 'ROOT_INITIALIZER_POLICY_BEFORE_INPUTS.json'
POLICY_SHA = '03b12b0cd8f3c3f894fd3e5f2db583812c8f213ca0886a0f5b04b655e31e41ff'
PROTOCOL = HERE / 'PROTOCOL_BEFORE_FORWARD.json'
TESTS = HERE / 'GENERATED_BOUNDARY_AND_V94_PARITY.json'
OUT = V132 / 'actual_initializer_v1'
SOURCE = BASE / 'query_visibility_registered_native_diversity_20261023_v129/actual_corpus_v1'
ROI = BASE.parent / 'external_ray_proxy_head_20261022_v60/head_and_roi.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    assert not path.exists(), str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']


def native_function():
    tree = ast.parse(ROI.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_grid']
    assert len(nodes) == 1
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROI), 'exec'), namespace)
    return namespace['native_grid']


def selected_queries(z, hw):
    # Exact V94 selection/coordinates/L/boundary semantics, tested below by AST replay.
    candidates = z['candidates']
    selected = None if not candidates else int(np.argmax([c['score'] for c in candidates]))
    assert selected == z['selected_index']
    if selected is None:
        return [dict(native_slot=k, query_xy=None, roi_side=None,
                     original_query_continuous_valid=False, query_nearest_pixel_valid=False,
                     query_rgb96_centre_domain_valid=False, reason='NO_DETECTION_U') for k in range(8)]
    q = np.asarray(candidates[selected]['keypoints_xy'], np.float64)
    assert q.shape == (9, 2) and np.isfinite(q).all()
    _, _, L = native_function()(q, hw)
    H, W = hw
    rows = []
    for k in range(8):
        x, y = q[k]
        valid = bool(0 <= x < W and 0 <= y < H)
        nearest = bool(0 <= np.floor(x + .5) < W and 0 <= np.floor(y + .5) < H)
        rgbcentre = bool(-.5 <= x < W - .5 and -.5 <= y < H - .5)
        assert nearest == rgbcentre
        rows.append(dict(native_slot=k, query_xy=q[k].tolist(), roi_side=L,
                         original_query_continuous_valid=valid, query_nearest_pixel_valid=nearest,
                         query_rgb96_centre_domain_valid=rgbcentre,
                         reason='VALID_QUERY_NOT_A_VISIBILITY_LABEL' if valid else 'OOB_OR_INVALID_U'))
    return rows


def serial(z):
    return dict(candidates=[{k: v.tolist() if isinstance(v, np.ndarray) else v
                             for k, v in c.items()} for c in z['candidates']],
                selected_index=z['selected_index'], input_shape=list(z['input_shape']),
                canvas_shape=list(z['canvas_shape']), added_border=z['added_border'])


def state(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def input_rows(policy):
    check(policy['inputs'])
    packet = read(policy['inputs']['path'])
    assert packet['completed'] and len(packet['rows']) == 768
    rows = []
    for i, r in enumerate(packet['rows']):
        assert r['scene_index'] == i and r['base_index'] == i // 4
        assert r['id'] == 'V129_SCENE_%04d' % i and r['base_id'] == 'V129_BASE_%03d' % (i // 4)
        assert r['variant'] == ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')[i % 4]
        assert r['split'] == ('TRAIN' if i < 512 else 'VALIDATION') and r['raw_hw'] == [512, 512]
        png = r['raw_RGB_PNG']
        expected = SOURCE / 'bases' / r['base_id'] / (r['variant'] + '_RGB.png')
        assert Path(png['path']).resolve() == expected.resolve()
        # Deliberately exclude full_RGB NPZ/config/reference/factors from producer rows.
        rows.append({k: r[k] for k in ('scene_index', 'base_index', 'id', 'base_id', 'split',
                                      'variant', 'raw_hw', 'raw_RGB_PNG', 'decoded_RGB_sha256')})
    return rows


def generated(policy):
    ref = Path(policy['reference_producer_code']['path'])
    check(policy['reference_producer_code'])
    nodes = [n for n in ast.parse(ref.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in ('native_function', 'selected_queries', 'serial')]
    assert len(nodes) == 3
    namespace = {'np': np, 'ast': ast, 'ROI': ROI}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ref), 'exec'), namespace)
    q = np.array([[90. + 11*i, 120. + 3*i] for i in range(9)], np.float64)
    candidate = lambda score, xy: dict(candidate_index=0, score=score, box_xyxy=np.array([1., 2., 3., 4.]),
                                      keypoints_xy=xy, keypoints_conf=np.full(9, .75))
    count = 0
    for xy in ([-.5, 0.], [-.25, 10.], [0., 0.], [511.49, 511.49], [511.5, 10.],
               [511.75, 10.], [-100., -100.], [512., 512.]):
        points = q.copy(); points[0] = xy
        z = dict(candidates=[candidate(.8, points), candidate(.8, points + 30)], selected_index=0,
                 input_shape=(640, 640), canvas_shape=(712, 712), added_border=100)
        assert selected_queries(z, [512, 512]) == namespace['selected_queries'](z, [512, 512])
        assert serial(z) == namespace['serial'](z)
        count += 1
    z = dict(candidates=[candidate(.8, q)], selected_index=0)
    baseline = selected_queries(z, [512, 512]); modified = q.copy(); modified[8] = [100000., -100000.]
    assert selected_queries(dict(candidates=[candidate(.8, modified)], selected_index=0), [512, 512]) == baseline
    missing = dict(candidates=[], selected_index=None)
    assert selected_queries(missing, [512, 512]) == namespace['selected_queries'](missing, [512, 512])
    assert len(selected_queries(missing, [512, 512])) == 8
    assert selected_queries(dict(candidates=[candidate(.8, q)], selected_index=0), [512, 512])[0]['query_xy'] == q[0].tolist()
    write(TESTS, dict(completed=True, PASS=True, code=bind(__file__), reference=bind(ref), native_grid=bind(ROI),
                      V94_exact_generated_boundary_and_serial_parity_cases=count,
                      first_argmax_tie_centroid_exclusion_no_detection_all8=True,
                      native_continuous_vs_nearest_RGBcentre_bounds_distinct=True,
                      actual_RGB_weights_GT_model_forward=0))


def prepare():
    assert sha(POLICY) == POLICY_SHA
    policy = read(POLICY)
    for key in ('backbone_code', 'checkpoint', 'reference_producer_code', 'native_grid_code'):
        check(policy[key])
    assert policy['native_grid_code']['path'] == str(ROI) and policy['output'] == str(OUT)
    rows = input_rows(policy)
    generated(policy)
    protocol = dict(completed=True, code=bind(__file__), generated=bind(TESTS), policy=bind(POLICY),
                    inputs=policy['inputs'], backbone_code=policy['backbone_code'], checkpoint=policy['checkpoint'],
                    reference_producer_code=policy['reference_producer_code'], native_grid=policy['native_grid_code'],
                    output=str(OUT), input_scene_count=len(rows), slots=6144, TRAIN_scenes=512, VAL_scenes=256,
                    runtime=dict(python='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python', torch='2.1.1+cu118',
                                 cv2='4.9.0', device='cuda:0', FP32=True, TF32=False, threads=1, deterministic=True),
                    predict='Exact FrozenYoloFeatures defaults, cpu_features=True; first argmax; native0..7 unchanged',
                    decoder='cv2.imread PNG only; per-PNG SHA and decoded contiguous RGB SHA exact before forward',
                    calls=dict(blank_generated_warmup=1, actual_source=768, repeat_first=1),
                    repeated_first='Full candidate serialization and transient FP16 P3 bit-exact; no P3 file saved',
                    state='Loaded vs warmed recorded; warmed model SHA unchanged after all source calls',
                    GT_mixed_buffers_mapping_crop_head_fit_current_V112_PnP_access=0,
                    training_admission=False, automatic_retry=False)
    write(PROTOCOL, protocol)
    template = dict(approved_source_initializer=False, input_only=True, fit=False, current223=False,
                    V112=False, PnP=False, q_correction=False, automatic_retry=False,
                    code=protocol['code'], protocol=bind(PROTOCOL), policy=protocol['policy'])
    write(HERE / 'ROOT_ENTRY_TEMPLATE.json', template)
    print(json.dumps(dict(protocol=bind(PROTOCOL), generated=bind(TESTS), entry_template=bind(HERE / 'ROOT_ENTRY_TEMPLATE.json'),
                          command=protocol['runtime']['python'] + ' ' + str(Path(__file__).resolve()) + ' --entry ROOT_ACTUAL_INITIALIZER_ENTRY.json',
                          actual_forward=0)), flush=True)


def run(entrypath):
    entrypath = Path(entrypath).resolve()
    e = read(entrypath)
    assert e['approved_source_initializer'] is True and e['input_only'] is True
    for key in ('fit', 'current223', 'V112', 'PnP', 'q_correction', 'automatic_retry'):
        assert e[key] is False
    assert e['code'] == bind(__file__) and e['protocol'] == bind(PROTOCOL) and e['policy'] == bind(POLICY)
    assert sha(POLICY) == POLICY_SHA
    p = read(PROTOCOL)
    assert p['code'] == e['code'] and p['policy'] == e['policy'] and p['output'] == str(OUT)
    for key in ('generated', 'inputs', 'backbone_code', 'checkpoint', 'reference_producer_code', 'native_grid'):
        check(p[key])
    assert read(p['generated']['path'])['PASS']
    rows = input_rows(read(POLICY))
    assert not OUT.exists(), 'Preserve existing output/failure; no automatic retry.'
    OUT.mkdir()
    write(OUT / 'INPUT768_LOCK_BEFORE_RGB_MODEL.json', dict(completed=True, entry=bind(entrypath),
          protocol=bind(PROTOCOL), rows=rows, query_slots=6144, actual_RGB_model_GT_reads=0))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'runtime/Ultralytics')
    os.environ['MPLCONFIGDIR'] = str(OUT / 'runtime/matplotlib')
    import cv2, torch
    assert torch.__version__ == p['runtime']['torch'] and cv2.__version__ == p['runtime']['cv2']
    assert Path(os.sys.executable).resolve() == Path(p['runtime']['python']).resolve()
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('v132_original_frozen_initializer', p['backbone_code']['path'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.FrozenYoloFeatures(p['checkpoint']['path'], device='cuda:0')
    hooks = [0]
    handle = model.yolo.model.register_forward_pre_hook(lambda *_: hooks.__setitem__(0, hooks[0] + 1))
    loaded = state(model.yolo.model)
    model.predict(np.zeros((512, 512, 3), np.uint8), already_padded=False, cpu_features=True)
    warmed = state(model.yolo.model); warmhooks = hooks[0]
    write(OUT / 'STARTED.json', dict(entry=bind(entrypath), input_lock=bind(OUT / 'INPUT768_LOCK_BEFORE_RGB_MODEL.json'),
          loaded_state_SHA=loaded, warmed_state_SHA=warmed, standard_inference_initialization_change=loaded != warmed,
          generated_warmup_hooks=warmhooks, source_frames=768, source_slots=6144, GT_access=False, training_admission=False))

    def image(row):
        check(row['raw_RGB_PNG'])
        bgr = cv2.imread(row['raw_RGB_PNG']['path'], cv2.IMREAD_COLOR)
        assert bgr is not None and bgr.dtype == np.uint8 and list(bgr.shape) == [512, 512, 3]
        decoded = hashlib.sha256(np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()).hexdigest()
        assert decoded == row['decoded_RGB_sha256'], row['id']
        return bgr

    started = time.monotonic(); proposal_path = OUT / 'FROZEN_PROPOSALS_ALL768_INPUT_ONLY.jsonl'
    query_path = OUT / 'ORIGINAL_NATIVE_QUERY6144_INPUT_ONLY.jsonl'
    first = first_p3 = None; reasons = {}; no_detection = 0
    with proposal_path.open('x') as proposals, query_path.open('x') as queries:
        for i, row in enumerate(rows):
            z = model.predict(image(row), already_padded=False, cpu_features=True)
            serialized = serial(z)
            assert serialized['canvas_shape'] == [712, 712] and serialized['input_shape'] == [640, 640]
            assert serialized['added_border'] == 100
            gain, offset = module.canvas_affine(z['canvas_shape'], z['input_shape'])
            assert gain == 640 / 712 and np.array_equal(offset, [0., 0.])
            result = dict(row, **serialized, affine_gain=gain, affine_offset=offset.tolist(),
                          decoded_RGB_verified=True, original_factory_unchanged=True)
            proposals.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            no_detection += z['selected_index'] is None
            for q in selected_queries(z, row['raw_hw']):
                reasons[q['reason']] = reasons.get(q['reason'], 0) + 1
                identity = {k: row[k] for k in ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant')}
                qr = dict(identity, query_id=row['id'] + ':native' + str(q['native_slot']),
                          selected_index=z['selected_index'], **q, actual_original_initializer=True,
                          semantic_state='UNKNOWN_NOT_READ_BY_PRODUCER', training_admission=False)
                queries.write(json.dumps(qr, ensure_ascii=False, allow_nan=False) + '\n')
            proposals.flush(); queries.flush()
            if i == 0:
                first = serialized; first_p3 = z['p3'].numpy().copy()
            if (i + 1) % 64 == 0:
                print(json.dumps(dict(frames_done=i + 1, slots_done=8*(i + 1), seconds=round(time.monotonic() - started, 3))), flush=True)
    repeat = model.predict(image(rows[0]), already_padded=False, cpu_features=True)
    assert serial(repeat) == first and np.array_equal(repeat['p3'].numpy(), first_p3)
    assert state(model.yolo.model) == warmed
    check(p['checkpoint'])
    assert all(not x.training for x in model.yolo.model.modules())
    assert all(not x.requires_grad for x in model.yolo.model.parameters())
    assert hooks[0] == warmhooks + 769 and sum(reasons.values()) == 6144
    handle.remove(); model.close()
    receipt = dict(completed=True, PASS=True, entry=bind(entrypath), protocol=bind(PROTOCOL), code=bind(__file__),
                   input_lock=bind(OUT / 'INPUT768_LOCK_BEFORE_RGB_MODEL.json'), inputs=p['inputs'],
                   proposals=bind(proposal_path), queries=bind(query_path),
                   frames=768, slots=6144, TRAIN_frames=512, VAL_frames=256,
                   no_detection_frames=int(no_detection), reason_counts=reasons, all_raw9_candidates_retained=True,
                   first_argmax_no_query_correction=True, all_PNG_and_decoded_RGB_hashes_exact=True,
                   repeat_first_candidates_and_FP16_P3_exact=True, no_fullP3_saved=True,
                   loaded_state_SHA=loaded, warmed_state_SHA=warmed, after_state_unchanged=True,
                   checkpoint_unchanged=True, explicit_predict_calls=770, total_forward_hooks=hooks[0],
                   generated_warmup_hooks=warmhooks, seconds=time.monotonic() - started,
                   GT_mixed_buffers_mapping_crop_head_fit_current_V112_PnP_access=0, training_admission=False)
    write(OUT / 'RESULT.json', receipt)
    write(OUT / 'PREDICTION_SEAL.json', dict(completed=True, PASS=True, result=bind(OUT / 'RESULT.json'),
          entry=bind(entrypath), outputs=[bind(x) for x in sorted(OUT.iterdir()) if x.is_file()],
          all768_scenes6144_slots_preserved=True, GT_read=False, training_admission=False))
    print(json.dumps(dict(completed=True, PASS=True, seal=bind(OUT / 'PREDICTION_SEAL.json'),
                          no_detection_frames=int(no_detection), reason_counts=reasons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    if args.prepare:
        assert args.entry is None
        prepare()
    elif args.entry:
        try:
            run(args.entry)
        except BaseException:
            if OUT.exists() and not (OUT / 'HALTED_FAILURE_NO_AUTOMATIC_RETRY.json').exists():
                write(OUT / 'HALTED_FAILURE_NO_AUTOMATIC_RETRY.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
            raise
    else:
        raise SystemExit('Use --prepare or --entry ROOT_ACTUAL_INITIALIZER_ENTRY.json')
