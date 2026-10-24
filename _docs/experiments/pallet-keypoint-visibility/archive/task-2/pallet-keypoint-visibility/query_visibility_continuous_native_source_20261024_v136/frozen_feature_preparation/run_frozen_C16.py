"""V136 GT-free actual-query C16 features and frozen update4500 baseline.

The reviewed V132 run_c16 body is reused without changing its operators,
preprocessing, packing, encoder batch8, head batch512 or availability decoder.
Only native source count, identity validation and corpus metadata are adapted.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
OLD = P / 'query_visibility_actual_initializer_source_20261024_v132'
PARENT_CODE = OLD / 'frozen_visibility_preparation/run_frozen_visibility.py'
PARENT_PROTOCOL = OLD / 'frozen_visibility_preparation/PROTOCOL_BEFORE_VISIBILITY_FORWARD.json'
OLD_C16_SEAL = OLD / 'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json'
INITIALIZER_CODE = V / 'initializer_preparation/infer_frozen_source.py'
INITIALIZER_PROTOCOL = V / 'initializer_preparation/PROTOCOL_BEFORE_FORWARD.json'
PROTOCOL = H / 'FROZEN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
GENERATED = H / 'GENERATED_INPUT_AND_OPERATOR_CHECKS.json'
OUT = V / 'frozen_C16_actual_initializer_v1'
POLICY = V / 'ROOT_CONTINUOUS_CORPUS_POLICY_BEFORE_PIXELS.json'
POLICY_SHA = '546f3d28d4454ba8a28a5eecdbfb0d30d91e6513e2ecd3209fb8945be8dd9ec4'
SCENES = 2560
N = SCENES * 8
VARIANTS = ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')


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


def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s]


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def input_geometry(proposals, queries, operator, target, native_grid):
    """Same V132 query geometry; no GT fields or source-label matching."""
    assert len(proposals) == SCENES and len(queries) == N
    rows = []
    for si, s in enumerate(proposals):
        assert s['scene_index'] == si and s['base_index'] == si // 4
        assert s['id'] == 'V136_SCENE_%04d' % si and s['base_id'] == 'V136_BASE_%04d' % (si // 4)
        assert s['variant'] == VARIANTS[si % 4] and s['split'] == ('TRAIN' if si // 4 < 512 else 'VALIDATION')
        assert s['raw_hw'] == [512, 512]
        cs = s['candidates']; selected = None if not cs else int(np.argmax([c['score'] for c in cs]))
        assert selected == s['selected_index']
        if selected is None:
            points = side = box = None
        else:
            all9 = np.asarray(cs[selected]['keypoints_xy'], np.float64)
            assert all9.shape == (9, 2) and np.isfinite(all9).all()
            points = all9[:8].copy(); side = float(native_grid(all9, s['raw_hw'])[2]); box = target['query_extent'](points)
        for slot in range(8):
            i = si * 8 + slot; q = queries[i]
            for key in ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant', 'selected_index'):
                assert q[key] == s[key], (i, key)
            assert q['query_id'] == s['id'] + ':native' + str(slot) and q['native_slot'] == slot
            assert q['actual_original_initializer'] is True and q['source_emulation'] is False
            expected = None if points is None else points[slot].tolist()
            assert q['query_xy'] == expected and q['roi_side'] == side
            good = operator.finite_query(expected)
            if expected is None:
                cont = near = False; assert q['reason'] == 'NO_DETECTION_U'
            else:
                x, y = expected
                cont = bool(good and 0 <= x < 512 and 0 <= y < 512)
                near = bool(good and 0 <= np.floor(x + .5) < 512 and 0 <= np.floor(y + .5) < 512)
            assert q['original_query_continuous_valid'] == cont
            assert q['query_nearest_pixel_valid'] == near and q['query_rgb96_centre_domain_valid'] == near
            rows.append(dict(occurrence_index=i, occurrence_id=q['query_id'], scene_index=si, base_index=s['base_index'],
                scene_id=s['id'], base_id=s['base_id'], split=s['split'], variant=s['variant'], native_slot=slot,
                selected_index=selected, raw_hw=s['raw_hw'], raw_RGB_PNG=s['raw_RGB_PNG'], decoded_RGB_sha256=s['decoded_RGB_sha256'],
                query_xy=expected, initial_points=None if points is None else points.tolist(), L=side, context_box_xyxy=box,
                finite_non_sentinel_query=good, native_continuous_valid=cont, native_nearest_valid=near,
                context_geometry_valid=bool(target['valid_box'](box)), actual_original_initializer=True, source_emulation=False))
    assert len(rows) == N and len({r['occurrence_id'] for r in rows}) == N
    return rows


def common_arrays(rows):
    return dict(global_index=np.arange(N, dtype=np.int32), corpus=np.full(N, 'NATIVE_V136_ACTUAL_INITIALIZER'),
                source_occurrence_index=np.arange(N, dtype=np.int32), occurrence_id=np.array([r['occurrence_id'] for r in rows]),
                scene_index=np.array([r['scene_index'] for r in rows], np.int32), base_index=np.array([r['base_index'] for r in rows], np.int32),
                native_slot=np.array([r['native_slot'] for r in rows], np.int8),
                query_xy=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan, np.nan] for r in rows], np.float64),
                L=np.array([r['L'] if r['L'] is not None else np.nan for r in rows], np.float64),
                native_continuous_valid=np.array([r['native_continuous_valid'] for r in rows], bool),
                native_nearest_valid=np.array([r['native_nearest_valid'] for r in rows], bool),
                no_detection=np.array([r['selected_index'] is None for r in rows], bool))


def generated(p, op):
    target = op.pure(p['c16']['target_crop_code']['path'], ('query_extent', 'valid_box'))
    grid = op.pure(p['native_grid_code']['path'], ('native_grid',))['native_grid']
    pts = np.array([[30. + j * 11, 40. + j * 7] for j in range(9)], np.float64)
    changed = pts.copy(); changed[8] = [1e6, -1e6]
    assert grid(pts, [512, 512])[2] == grid(changed, [512, 512])[2]
    assert target['query_extent'](pts[:8]) == target['query_extent'](changed[:8])
    assert not op.finite_query(None) and not op.finite_query([-1., -1.]) and op.finite_query([-100., -100.])
    assert target['query_extent'](np.array([[2., 3.]] * 8)) is None
    assert op.decode(np.array([-1., -0., 0., 1., np.nan]), np.array([1, 1, 1, 1, 0], bool), 'c16').tolist() == [0, 1, 1, 1, -1]
    proposals, queries = [], []
    for si in range(SCENES):
        pp = pts.copy(); pp[0] = [511.75, 10.]
        cs = [] if si % 3 == 0 else [dict(score=.8, keypoints_xy=pp.tolist()), dict(score=.8, keypoints_xy=(pp + 1).tolist())]
        selected = None if not cs else 0
        s = dict(scene_index=si, base_index=si // 4, id='V136_SCENE_%04d' % si, base_id='V136_BASE_%04d' % (si // 4),
                 split='TRAIN' if si // 4 < 512 else 'VALIDATION', variant=VARIANTS[si % 4], selected_index=selected,
                 raw_hw=[512, 512], raw_RGB_PNG={'path': '/generated/%04d.png' % si, 'sha256': '0' * 64}, decoded_RGB_sha256='1' * 64, candidates=cs)
        proposals.append(s); side = None if selected is None else float(grid(pp, [512, 512])[2])
        for slot in range(8):
            xy = None if selected is None else pp[slot].tolist(); good = op.finite_query(xy)
            cont = bool(good and 0 <= xy[0] < 512 and 0 <= xy[1] < 512)
            near = bool(good and 0 <= np.floor(xy[0] + .5) < 512 and 0 <= np.floor(xy[1] + .5) < 512)
            identity = {k: s[k] for k in ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant', 'selected_index')}
            queries.append(dict(identity, query_id=s['id'] + ':native' + str(slot), native_slot=slot, query_xy=xy, roi_side=side,
                                original_query_continuous_valid=cont, query_nearest_pixel_valid=near, query_rgb96_centre_domain_valid=near,
                                actual_original_initializer=True, source_emulation=False, reason='NO_DETECTION_U' if xy is None else 'VALID_QUERY_NOT_A_VISIBILITY_LABEL'))
    rows = input_geometry(proposals, queries, op, target, grid)
    assert len(rows) == N and sum(r['query_xy'] is None for r in rows) == 6832
    assert rows[8]['query_xy'] == [511.75, 10.] and rows[8]['native_continuous_valid'] and not rows[8]['native_nearest_valid']
    json.dumps(rows, allow_nan=False)
    # Compare geometry fields to the exact previous builder on invented records
    # with its previous IDs/fold metadata. No previous source RGB/output replay.
    oldp, oldq = [], []
    for si in range(768):
        s = dict(proposals[si]); s.update(id='V129_SCENE_%04d' % si, base_id='V129_BASE_%03d' % (si // 4), split='TRAIN' if si < 512 else 'VALIDATION')
        oldp.append(s)
        for slot in range(8):
            q = dict(queries[si * 8 + slot]); q.update(id=s['id'], base_id=s['base_id'], split=s['split'], query_id=s['id'] + ':native' + str(slot)); oldq.append(q)
    reference_rows = op.input_geometry(oldp, oldq, target, grid)
    keys = ('query_xy', 'initial_points', 'L', 'context_box_xyxy', 'finite_non_sentinel_query', 'native_continuous_valid', 'native_nearest_valid', 'context_geometry_valid')
    assert all(all(a[k] == b[k] for k in keys) for a, b in zip(rows, reference_rows))
    broken = [dict(q) for q in queries]; broken[8]['query_xy'] = [511., 10.]
    try:
        input_geometry(proposals, broken, op, target, grid)
    except AssertionError:
        pass
    else:
        raise AssertionError('Query correction was accepted')
    write(GENERATED, dict(completed=True, PASS=True, code=bind(__file__), parent_code=p['parent_code'],
        all20480_native_order_null_OOB_and_boundary_rows_retained=True, invented_missing_slots=6832,
        geometry_equal_previous_exact_builder_invented6144=True, no_GT_or_corrected_query_accepted=True,
        centroid_excluded_null_sentinel_degenerate_separate=True, hidden_zero_E_noquery_U=True,
        actual_RGB_GT_checkpoint_encoder_model_reads=0))


def prepare(policy_path):
    assert Path(policy_path).resolve() == POLICY and sha(POLICY) == POLICY_SHA
    parent = read(PARENT_PROTOCOL); old = read(OLD_C16_SEAL)
    assert old['completed'] and old['PASS'] and old['GT_read'] is False
    assert same(parent['code'], bind(PARENT_CODE)) and same(old['protocol'], bind(PARENT_PROTOCOL))
    initp = read(INITIALIZER_PROTOCOL); assert same(initp['code'], bind(INITIALIZER_CODE)) and same(initp['policy'], bind(policy_path))
    p = dict(completed=True, code=bind(__file__), source_policy=bind(policy_path), parent_code=bind(PARENT_CODE),
             parent_protocol=bind(PARENT_PROTOCOL), initializer_code=bind(INITIALIZER_CODE), initializer_protocol=bind(INITIALIZER_PROTOCOL),
             c16=parent['c16'], native_grid_code=parent['native_grid_code'], output=str(OUT), rows=N, source_scene_count=SCENES,
             expected_initializer_seal=str(V / 'actual_initializer_v1/PREDICTION_SEAL.json'),
             original_frozen_asset_reference=dict(seal=bind(OLD_C16_SEAL), encoder_state_sha256=old['encoder_state_sha256'],
                                                  head_state_sha256=old['head_state_sha256']),
             exact_function_reuse='V132 run_c16 body unchanged; module N=20480, common_arrays replaced only for count/corpus metadata',
             C16_operator=parent['C16_operator'], geometry='Original actual first-argmax native0..7; centroid excluded; q8 minmax context; no GT association',
             preprocessing='PNG decoded RGB only; exact V90 96 crop/C8 two224/C9 target224/DINO TQFCP1928/head3464→64→1',
             calls='Frozen DINO FP32 mathSDPA batch8; compact constructible-index order; C16 head512; repeat first and state invariance',
             output_schema='SOURCE_FEATURES_INPUT_ONLY.npz, C16_PREDICTIONS_INPUT_ONLY.npz, all20480 cropfacts/raw96 chunks; NaN raw only no feature input',
             actual_original_initializer=True, source_emulation=False, old_prediction_replay_on_new_data=False,
             GT_read=False, training=False, current223=False, V112=False, PnP=False, q_correction=False, automatic_retry=False)
    for key in ('local_crop_code', 'target_crop_code', 'availability_code'):
        check(p['c16'][key])
    check(p['native_grid_code'])
    op = module(PARENT_CODE, 'v136_inert_exact_C16_operator')
    generated(p, op); p['generated'] = bind(GENERATED)
    write(PROTOCOL, p)
    entry = dict(approved_frozen_C16=False, fit=False, current223=False, V112=False, PnP=False, q_correction=False,
                 automatic_retry=False, code=p['code'], protocol=bind(PROTOCOL), source_policy=p['source_policy'],
                 initializer_seal=None, initializer_result=None, initializer_proposals=None, initializer_queries=None,
                 fill_after_new_initializer_complete='Bind actual V136 initializer completed seal→result→proposals/queries; no old source seal substitution.')
    write(H / 'ROOT_C16_ENTRY_TEMPLATE.json', entry)
    print(json.dumps(dict(completed=True, protocol=bind(PROTOCOL), generated=p['generated'], template=bind(H / 'ROOT_C16_ENTRY_TEMPLATE.json'),
                          actual_RGB_GT_checkpoint_encoder_model_reads=0)), flush=True)


def authorize(entry_path):
    e = read(entry_path)
    assert e['approved_frozen_C16'] is True
    assert all(e[k] is False for k in ('fit', 'current223', 'V112', 'PnP', 'q_correction', 'automatic_retry'))
    assert same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL))
    p = read(PROTOCOL); assert same(p['code'], e['code']) and same(p['source_policy'], e['source_policy'])
    assert Path(p['source_policy']['path']).resolve() == POLICY and p['source_policy']['sha256'] == POLICY_SHA
    for key in ('source_policy', 'parent_code', 'parent_protocol', 'initializer_code', 'initializer_protocol', 'generated', 'native_grid_code'):
        check(p[key])
    proof = read(p['generated']['path']); assert proof['PASS'] and same(proof['code'], e['code'])
    for key in ('initializer_seal', 'initializer_result', 'initializer_proposals', 'initializer_queries'):
        check(e[key])
    assert Path(e['initializer_seal']['path']).resolve() == Path(p['expected_initializer_seal']).resolve()
    seal = read(e['initializer_seal']['path']); result = read(e['initializer_result']['path'])
    assert seal['completed'] and seal['PASS'] and seal['GT_read'] is False and seal['all2560_scenes20480_slots_preserved']
    assert seal['actual_original_initializer'] is True and seal['source_emulation'] is False and same(seal['result'], e['initializer_result'])
    assert result['completed'] and result['PASS'] and result['frames'] == SCENES and result['slots'] == N
    assert result['GT_mixed_buffers_mapping_crop_head_fit_current_V112_PnP_access'] == 0
    assert result['after_state_unchanged'] and result['checkpoint_unchanged'] and result['same_original_V132_asset_states_exact']
    assert same(result['code'], p['initializer_code']) and same(result['protocol'], p['initializer_protocol']) and same(result['policy'], p['source_policy'])
    assert same(result['proposals'], e['initializer_proposals']) and same(result['queries'], e['initializer_queries'])
    for key in ('C11_producer', 'C8_preprocess', 'C9_target_preprocess', 'local_crop_code', 'target_crop_code', 'availability_code',
                'asset_runtime', 'asset_plan', 'weight', 'source_model_freeze', 'checkpoint', 'head_class_code'):
        check(p['c16'][key])
    for b in p['c16']['official_source_bindings']:
        check(b)
    freeze = read(p['c16']['source_model_freeze']['path'])
    assert freeze['completed'] and freeze['selected_update'] == 4500 and freeze['threshold_logit'] == 0
    assert same(freeze['checkpoint'], p['c16']['checkpoint']) and not freeze['current_used_for_fit_or_selection']
    op = module(p['parent_code']['path'], 'v136_exact_C16_runtime')
    rows = input_geometry(lines(e['initializer_proposals']['path']), lines(e['initializer_queries']['path']), op,
                          op.pure(p['c16']['target_crop_code']['path'], ('query_extent', 'valid_box')),
                          op.pure(p['native_grid_code']['path'], ('native_grid',))['native_grid'])
    assert not OUT.exists(), 'Preserve completed or failed attempt; no automatic retry.'
    return e, p, rows, op


def run(entry_path):
    e, p, rows, op = authorize(entry_path); OUT.mkdir()
    lock = OUT / 'C16_NATIVE_INPUT_GEOMETRY_BEFORE_RGB.json'
    write(lock, dict(completed=True, rows=rows, initializer_seal=e['initializer_seal'], GT_read=False, actual_original_initializer=True, source_emulation=False))
    input_seal = OUT / 'INPUT_C16_SEAL_BEFORE_RGB_MODEL.json'
    write(input_seal, dict(completed=True, PASS=True, input_lock=bind(lock), entry=bind(entry_path), initializer_seal=e['initializer_seal'],
                          rows=N, source_scenes=SCENES, q8_L_context_from_completed_actual_initializer_exact=True, GT_read=False))
    write(OUT / 'STARTED.json', dict(entry=bind(entry_path), input_seal=bind(input_seal), rows=N, source_scenes=SCENES, GT_read=False))
    started = time.monotonic()
    try:
        # Only these metadata globals change; exact reviewed run_c16 function
        # and every feature/head operator retain their original function body.
        op.N = N
        op.common_arrays = common_arrays
        result = op.run_c16(e, p, rows, OUT)
        reference = p['original_frozen_asset_reference']
        assert result['encoder_state_sha256'] == reference['encoder_state_sha256']
        assert result['head_state_sha256'] == reference['head_state_sha256']
        assert result['rows'] == N and result['encoder_state_unchanged'] and result['head_state_unchanged']
        for binding in (p['c16']['checkpoint'], p['c16']['weight']):
            check(binding)
        result['crop_placeholder'] = 'All20480 original slots retained; zero crop only when crop_constructed=False; never encoder input'
        seal = dict(completed=True, PASS=True, code=e['code'], protocol=e['protocol'], entry=bind(entry_path), source_policy=p['source_policy'],
                    initializer_seal=e['initializer_seal'], initializer_result=e['initializer_result'], initializer_proposals=e['initializer_proposals'],
                    initializer_queries=e['initializer_queries'], input_seal=bind(input_seal), all2560_scenes20480_slots_preserved=True,
                    actual_original_initializer=True, source_emulation=False, same_original_C16_encoder_and_head_states_exact=True,
                    old_prediction_replay_on_new_data=False, GT_read=False, fit_current_V112_PnP_q_correction=0,
                    no_model_or_threshold_choice=True, seconds=time.monotonic() - started, **result)
        write(OUT / 'PREDICTION_SEAL.json', seal)
        print(json.dumps(dict(completed=True, PASS=True, seal=bind(OUT / 'PREDICTION_SEAL.json'), rows=N,
                              available=result['available'], seconds=time.monotonic() - started)), flush=True)
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
