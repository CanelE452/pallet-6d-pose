"""Actual unchanged original initializer for V136 source PNGs.

Preparation reads code and metadata only. A completed new render seal and an
explicit root entry are required before any PNG, checkpoint or model call.
"""
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
H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
OLD = P / 'query_visibility_actual_initializer_source_20261024_v132'
OLD_CODE = OLD / 'initializer_preparation/infer_frozen_source.py'
OLD_PROTOCOL = OLD / 'initializer_preparation/PROTOCOL_BEFORE_FORWARD.json'
OLD_SEAL = OLD / 'actual_initializer_v1/PREDICTION_SEAL.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_FORWARD.json'
GENERATED = H / 'GENERATED_BOUNDARY_AND_OPERATOR_CHECKS.json'
SOURCE = V / 'actual_corpus_v1'
RENDER = SOURCE / 'RENDER_COMPLETE.json'
INPUTS = SOURCE / 'SCENES_INPUT_ONLY.json'
OUT = V / 'actual_initializer_v1'
POLICY = V / 'ROOT_CONTINUOUS_CORPUS_POLICY_BEFORE_PIXELS.json'
POLICY_SHA = '546f3d28d4454ba8a28a5eecdbfb0d30d91e6513e2ecd3209fb8945be8dd9ec4'
SCENES = 2560
SLOTS = SCENES * 8
VARIANTS = ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')
KEYS = ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant', 'raw_hw', 'raw_RGB_PNG', 'decoded_RGB_sha256')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


def same(a, b):
    return a['path'] == b['path'] and a['sha256'] == b['sha256']


def check(b):
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def canonical_rows(packet):
    assert packet['completed'] and len(packet['rows']) == SCENES
    rows = []
    for i, r in enumerate(packet['rows']):
        assert r['scene_index'] == i and r['base_index'] == i // 4
        assert r['id'] == 'V136_SCENE_%04d' % i and r['base_id'] == 'V136_BASE_%04d' % (i // 4)
        assert r['split'] == ('TRAIN' if i // 4 < 512 else 'VALIDATION')
        assert r['variant'] == VARIANTS[i % 4] and r['raw_hw'] == [512, 512]
        expected = SOURCE / 'bases' / r['base_id'] / (r['variant'] + '_RGB.png')
        assert Path(r['raw_RGB_PNG']['path']).resolve() == expected.resolve()
        # No full_RGB/config/reference/factor/teacher member is dereferenced.
        rows.append({k: r[k] for k in KEYS})
    assert len({r['id'] for r in rows}) == SCENES
    return rows


def generated(old, p):
    # This is operator replay on invented arrays, not old predictions assigned
    # to new V136 images. No source RGB or model asset body is opened.
    ref = check(p['V94_reference_code'])
    nodes = [n for n in ast.parse(ref.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('native_function', 'selected_queries', 'serial')]
    assert len(nodes) == 3
    ns = {'np': np, 'ast': ast, 'ROI': Path(p['native_grid']['path'])}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ref), 'exec'), ns)
    q = np.array([[90. + 11 * k, 120. + 3 * k] for k in range(9)], np.float64)
    def candidate(score, points):
        return {'candidate_index': 0, 'score': score, 'box_xyxy': np.array([1., 2., 3., 4.]), 'keypoints_xy': points, 'keypoints_conf': np.full(9, .75)}
    count = 0
    for xy in ([-.5, 0.], [-.25, 10.], [0., 0.], [511.49, 511.49], [511.5, 10.], [511.75, 10.], [-100., -100.], [512., 512.], [-1., -1.]):
        points = q.copy(); points[0] = xy
        z = {'candidates': [candidate(.8, points), candidate(.8, points + 30)], 'selected_index': 0,
             'input_shape': (640, 640), 'canvas_shape': (712, 712), 'added_border': 100}
        assert old.selected_queries(z, [512, 512]) == ns['selected_queries'](z, [512, 512])
        assert old.serial(z) == ns['serial'](z)
        count += 1
    baseline = old.selected_queries({'candidates': [candidate(.8, q)], 'selected_index': 0}, [512, 512])
    changed = q.copy(); changed[8] = [1e6, -1e6]
    assert old.selected_queries({'candidates': [candidate(.8, changed)], 'selected_index': 0}, [512, 512]) == baseline
    missing = old.selected_queries({'candidates': [], 'selected_index': None}, [512, 512])
    assert len(missing) == 8 and all(r['query_xy'] is None and r['reason'] == 'NO_DETECTION_U' for r in missing)
    dummy = []
    for i in range(SCENES):
        base = i // 4; variant = VARIANTS[i % 4]; bid = 'V136_BASE_%04d' % base
        dummy.append(dict(scene_index=i, base_index=base, id='V136_SCENE_%04d' % i, base_id=bid,
                          split='TRAIN' if base < 512 else 'VALIDATION', variant=variant, raw_hw=[512, 512],
                          raw_RGB_PNG={'path': str(SOURCE / 'bases' / bid / (variant + '_RGB.png')), 'sha256': 'GENERATED_NOT_A_FILE'},
                          decoded_RGB_sha256='GENERATED_NOT_A_FILE', forbidden_GT={'should_not_survive': True}))
    whitelisted = canonical_rows({'completed': True, 'rows': dummy})
    assert all(set(r) == set(KEYS) for r in whitelisted)
    write(GENERATED, dict(completed=True, PASS=True, code=bind(__file__), parent_code=p['parent_code'],
                         generated_V94_query_serial_cases=count, centroid_exclusion_first_argmax_noquery_all8=True,
                         all2560_identity_split_path_whitelist_checks=True, invented_arrays_only=True,
                         actual_RGB_checkpoint_encoder_model_GT_reads=0))


def prepare(policy_path):
    assert Path(policy_path).resolve() == POLICY and sha(POLICY) == POLICY_SHA
    policy_binding = bind(policy_path)
    oldp = read(OLD_PROTOCOL); check(oldp['code'])
    assert same(oldp['code'], bind(OLD_CODE))
    oldseal = read(OLD_SEAL); assert oldseal['completed'] and oldseal['PASS'] and oldseal['GT_read'] is False
    oldresult_path = check(oldseal['result']); oldresult = read(oldresult_path)
    assert oldresult['completed'] and oldresult['PASS'] and oldresult['after_state_unchanged']
    p = dict(completed=True, code=bind(__file__), policy=policy_binding, parent_code=bind(OLD_CODE), parent_protocol=bind(OLD_PROTOCOL),
             backbone_code=oldp['backbone_code'], checkpoint=oldp['checkpoint'], V94_reference_code=oldp['reference_producer_code'],
             native_grid=oldp['native_grid'], runtime=oldp['runtime'], output=str(OUT),
             expected_render_seal=str(RENDER), expected_input_scenes=str(INPUTS), expected_scenes=SCENES, expected_slots=SLOTS,
             expected_bases=640, TRAIN_bases=512, VALIDATION_bases=128, TRAIN_scenes=2048, VALIDATION_scenes=512,
             original_initializer_asset_reference=dict(seal=bind(OLD_SEAL), result=bind(oldresult_path),
                 loaded_state_SHA=oldresult['loaded_state_SHA'], warmed_state_SHA=oldresult['warmed_state_SHA']),
             factory='Exact FrozenYoloFeatures original defaults: reflect100/conf.001/imgsz640/rect=True/FP32/cpu_features=True; first argmax',
             query_operator='Unmodified V132 selected_queries/native_grid/serial; native0..7, centroid8 excluded from L',
             inputs='New V136 completed render metadata and RGB PNG whitelist only; no buffer/config/reference/pose/order/teacher input',
             actual_original_initializer=True, source_emulation=False, old_prediction_replay_on_new_data=False,
             GT_read=False, fit_current_V112_PnP_or_q_correction=0, automatic_retry=False)
    assert p['checkpoint']['sha256'] == '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
    for key in ('backbone_code', 'V94_reference_code', 'native_grid'):
        check(p[key])
    old = module(OLD_CODE, 'v136_inert_original_initializer_operators')
    generated(old, p); p['generated'] = bind(GENERATED)
    write(PROTOCOL, p)
    entry = dict(approved_source_initializer=False, input_only=True, fit=False, current223=False, V112=False,
                 PnP=False, q_correction=False, automatic_retry=False, code=p['code'], protocol=bind(PROTOCOL), policy=p['policy'],
                 render_seal=None, input_scenes=None,
                 fill_after_new_render_complete='Bind completed V136/actual_corpus_v1/RENDER_COMPLETE.json and its input_scenes only.')
    write(H / 'ROOT_ENTRY_TEMPLATE.json', entry)
    print(json.dumps(dict(completed=True, protocol=bind(PROTOCOL), generated=p['generated'], template=bind(H / 'ROOT_ENTRY_TEMPLATE.json'),
                          actual_RGB_checkpoint_model_GT_reads=0)), flush=True)


def authorize(entry_path):
    e = read(entry_path)
    assert e['approved_source_initializer'] is True and e['input_only'] is True
    assert all(e[k] is False for k in ('fit', 'current223', 'V112', 'PnP', 'q_correction', 'automatic_retry'))
    assert same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL))
    p = read(PROTOCOL); assert same(p['code'], e['code']) and same(p['policy'], e['policy'])
    assert Path(p['policy']['path']).resolve() == POLICY and p['policy']['sha256'] == POLICY_SHA
    for k in ('policy', 'parent_code', 'parent_protocol', 'generated', 'backbone_code', 'checkpoint', 'V94_reference_code', 'native_grid'):
        check(p[k])
    assert read(p['generated']['path'])['PASS'] and same(read(p['generated']['path'])['code'], e['code'])
    assert Path(e['render_seal']['path']).resolve() == RENDER.resolve() and Path(e['input_scenes']['path']).resolve() == INPUTS.resolve()
    render = read(check(e['render_seal']))
    assert render['completed'] and render['PASS'] and render['scenes'] == SCENES and render['bases'] == 640
    assert same(render['input_scenes'], e['input_scenes']) and same(render['policy'], p['policy'])
    rows = canonical_rows(read(check(e['input_scenes'])))
    assert not OUT.exists(), 'Preserve completed or failed attempt; no automatic retry.'
    return e, p, rows


def run(entry_path):
    e, p, rows = authorize(entry_path); OUT.mkdir()
    lock = OUT / 'INPUT2560_LOCK_BEFORE_RGB_MODEL.json'
    write(lock, dict(completed=True, entry=bind(entry_path), protocol=bind(PROTOCOL), render_seal=e['render_seal'],
                     input_scenes=e['input_scenes'], rows=rows, query_slots=SLOTS, RGB_checkpoint_model_GT_reads=0))
    input_seal = OUT / 'INPUT_SOURCE_SEAL_BEFORE_RGB_MODEL.json'
    write(input_seal, dict(completed=True, PASS=True, input_lock=bind(lock), render_seal=e['render_seal'],
                          scenes=SCENES, slots=SLOTS, actual_original_initializer=True, source_emulation=False, GT_read=False))
    started = time.monotonic()
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'runtime/Ultralytics')
        os.environ['MPLCONFIGDIR'] = str(OUT / 'runtime/matplotlib')
        import cv2, torch
        assert torch.__version__ == p['runtime']['torch'] and cv2.__version__ == p['runtime']['cv2'] and np.__version__ == '1.26.4'
        assert Path(sys.executable).resolve() == Path(p['runtime']['python']).resolve() and torch.cuda.is_available()
        torch.set_num_threads(1); cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        old = module(p['parent_code']['path'], 'v136_exact_initializer_operators')
        factory = module(p['backbone_code']['path'], 'v136_exact_original_yolo_factory')
        model = factory.FrozenYoloFeatures(p['checkpoint']['path'], device='cuda:0')
        hooks = [0]; handle = model.yolo.model.register_forward_pre_hook(lambda *_: hooks.__setitem__(0, hooks[0] + 1))
        loaded = old.state(model.yolo.model)
        model.predict(np.zeros((512, 512, 3), np.uint8), already_padded=False, cpu_features=True)
        warmed = old.state(model.yolo.model); warmhooks = hooks[0]
        ref = p['original_initializer_asset_reference']
        assert loaded == ref['loaded_state_SHA'] and warmed == ref['warmed_state_SHA']
        write(OUT / 'STARTED.json', dict(entry=bind(entry_path), input_seal=bind(input_seal), loaded_state_SHA=loaded,
              warmed_state_SHA=warmed, same_original_V132_asset_states_exact=True, generated_warmup_hooks=warmhooks,
              scenes=SCENES, slots=SLOTS, GT_read=False))

        def image(row):
            check(row['raw_RGB_PNG'])
            bgr = cv2.imread(row['raw_RGB_PNG']['path'], cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.dtype == np.uint8 and list(bgr.shape) == [512, 512, 3]
            assert hashlib.sha256(np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()).hexdigest() == row['decoded_RGB_sha256']
            return bgr

        proposal_path = OUT / 'FROZEN_PROPOSALS_ALL2560_INPUT_ONLY.jsonl'
        query_path = OUT / 'ORIGINAL_NATIVE_QUERY20480_INPUT_ONLY.jsonl'
        first = first_p3 = None; reasons = {}; no_detection = 0
        with proposal_path.open('x') as proposals, query_path.open('x') as queries:
            for i, row in enumerate(rows):
                z = model.predict(image(row), already_padded=False, cpu_features=True)
                serialized = old.serial(z)
                assert serialized['canvas_shape'] == [712, 712] and serialized['input_shape'] == [640, 640] and serialized['added_border'] == 100
                gain, offset = factory.canvas_affine(z['canvas_shape'], z['input_shape'])
                assert gain == 640 / 712 and np.array_equal(offset, [0., 0.])
                proposals.write(json.dumps(dict(row, **serialized, affine_gain=gain, affine_offset=offset.tolist(),
                                               decoded_RGB_verified=True, original_factory_unchanged=True), ensure_ascii=False, allow_nan=False) + '\n')
                no_detection += z['selected_index'] is None
                for q in old.selected_queries(z, row['raw_hw']):
                    reasons[q['reason']] = reasons.get(q['reason'], 0) + 1
                    identity = {k: row[k] for k in ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant')}
                    qr = dict(identity, query_id=row['id'] + ':native' + str(q['native_slot']), selected_index=z['selected_index'],
                              **q, actual_original_initializer=True, source_emulation=False,
                              semantic_state='UNKNOWN_NOT_READ_BY_PRODUCER', training_admission=False)
                    queries.write(json.dumps(qr, ensure_ascii=False, allow_nan=False) + '\n')
                proposals.flush(); queries.flush()
                if i == 0:
                    first = serialized; first_p3 = z['p3'].numpy().copy()
                if (i + 1) % 64 == 0:
                    print(json.dumps(dict(frames_done=i + 1, slots_done=8 * (i + 1), seconds=time.monotonic() - started)), flush=True)
        repeat = model.predict(image(rows[0]), already_padded=False, cpu_features=True)
        assert old.serial(repeat) == first and np.array_equal(repeat['p3'].numpy(), first_p3)
        assert old.state(model.yolo.model) == warmed and all(not x.training for x in model.yolo.model.modules())
        assert all(not x.requires_grad for x in model.yolo.model.parameters())
        assert hooks[0] == warmhooks + SCENES + 1 and sum(reasons.values()) == SLOTS
        check(p['checkpoint']); handle.remove(); model.close()
        receipt = dict(completed=True, PASS=True, entry=bind(entry_path), protocol=bind(PROTOCOL), code=bind(__file__),
                       policy=p['policy'], render_seal=e['render_seal'], input_lock=bind(lock), input_seal=bind(input_seal),
                       inputs=e['input_scenes'], proposals=bind(proposal_path), queries=bind(query_path), frames=SCENES, slots=SLOTS,
                       TRAIN_frames=2048, VAL_frames=512, no_detection_frames=int(no_detection), reason_counts=reasons,
                       all_raw9_candidates_retained=True, first_argmax_no_query_correction=True,
                       all_PNG_and_decoded_RGB_hashes_exact=True, repeat_first_candidates_and_FP16_P3_exact=True,
                       no_fullP3_saved=True, loaded_state_SHA=loaded, warmed_state_SHA=warmed,
                       same_original_V132_asset_states_exact=True, after_state_unchanged=True, checkpoint_unchanged=True,
                       explicit_predict_calls=SCENES + 2, total_forward_hooks=hooks[0], generated_warmup_hooks=warmhooks,
                       actual_original_initializer=True, source_emulation=False, old_prediction_replay_on_new_data=False,
                       GT_mixed_buffers_mapping_crop_head_fit_current_V112_PnP_access=0, training_admission=False,
                       seconds=time.monotonic() - started)
        write(OUT / 'RESULT.json', receipt)
        seal = dict(completed=True, PASS=True, result=bind(OUT / 'RESULT.json'), entry=bind(entry_path),
                    input_seal=bind(input_seal), outputs=[bind(x) for x in sorted(OUT.iterdir()) if x.is_file()],
                    all2560_scenes20480_slots_preserved=True, actual_original_initializer=True, source_emulation=False,
                    GT_read=False, training_admission=False)
        write(OUT / 'PREDICTION_SEAL.json', seal)
        print(json.dumps(dict(completed=True, PASS=True, seal=bind(OUT / 'PREDICTION_SEAL.json'),
                              no_detection_frames=int(no_detection), reason_counts=reasons)), flush=True)
    except BaseException:
        write(OUT / 'FAILED_NO_AUTOMATIC_RETRY.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--policy', type=Path); parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    if args.prepare:
        assert args.policy is not None; prepare(args.policy)
    elif args.entry:
        run(args.entry)
    else:
        raise SystemExit('Use --prepare --policy V136/ROOT_POLICY.json or root --entry')
