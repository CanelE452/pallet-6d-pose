"""V136 GT-free original-query v61 baseline, root-entry execution only.

The exact reviewed V132 v61 body is reused with its two scene-count constants
changed from768 to2560. All feature, affine, ROI, head and decoder math remains
unchanged. This is no calibration or broadened acceptance rule.
"""
from pathlib import Path
import argparse
import ast
import copy
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
OLD_V61_SEAL = OLD / 'frozen_v61_actual_initializer_v1/PREDICTION_SEAL.json'
FEATURE_CODE = V / 'frozen_feature_preparation/run_frozen_C16.py'
FEATURE_PROTOCOL = V / 'frozen_feature_preparation/FROZEN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
INITIALIZER_CODE = V / 'initializer_preparation/infer_frozen_source.py'
INITIALIZER_PROTOCOL = V / 'initializer_preparation/PROTOCOL_BEFORE_FORWARD.json'
POLICY = V / 'ROOT_CONTINUOUS_CORPUS_POLICY_BEFORE_PIXELS.json'
POLICY_SHA = '546f3d28d4454ba8a28a5eecdbfb0d30d91e6513e2ecd3209fb8945be8dd9ec4'
PROTOCOL = H / 'V61_PROTOCOL_BEFORE_FORWARD.json'
GENERATED = H / 'GENERATED_COUNT_JOIN_AND_BOUNDARY_CHECKS.json'
OUT = V / 'frozen_v61_actual_initializer_v1'
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


def adapted_function():
    original = next(n for n in ast.parse(PARENT_CODE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'run_v61')
    assert sum(isinstance(n, ast.Constant) and type(n.value) is int and n.value == 768 for n in ast.walk(original)) == 2
    loop = next(n for n in ast.walk(original) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Name) and n.iter.func.id == 'range' and len(n.iter.args) == 1
                and isinstance(n.iter.args[0], ast.Constant) and n.iter.args[0].value == 768)
    metadata = next(n for n in ast.walk(original) if isinstance(n, ast.keyword) and n.arg == 'source_scenes_without_forward')
    assert isinstance(metadata.value, ast.BinOp) and isinstance(metadata.value.op, ast.Sub) and metadata.value.left.value == 768
    class Count(ast.NodeTransformer):
        def __init__(self, old, new):
            self.old, self.new, self.changed = old, new, 0
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == self.old:
                self.changed += 1
                return ast.copy_location(ast.Constant(self.new), node)
            return node
    change = Count(768, SCENES); node = change.visit(copy.deepcopy(original)); assert change.changed == 2
    reverse = Count(SCENES, 768); reverted = reverse.visit(copy.deepcopy(node)); assert reverse.changed == 2
    dump = lambda n: ast.dump(n, include_attributes=False)
    assert dump(reverted) == dump(original)
    proof = dict(original_function_AST_sha256=hashlib.sha256(dump(original).encode()).hexdigest(),
                 count_adapted_function_AST_sha256=hashlib.sha256(dump(node).encode()).hexdigest(),
                 old_scene_count=768, new_scene_count=SCENES, replacements=2,
                 changed_fields=['range(source_scene_count)', 'source_scenes_without_forward'], reverse_substitution_exact=True)
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), proof


def generated(p, op, geometry):
    target = op.pure(p['query_math']['target_crop_code']['path'], ('query_extent', 'valid_box'))
    grid = op.pure(p['query_math']['native_grid_code']['path'], ('native_grid',))['native_grid']
    query = module(p['initializer_parent_code']['path'], 'v136_generated_original_query_operator')
    pts = np.array([[30. + j * 11, 40. + j * 7] for j in range(9)], np.float64)
    pts[0] = [511.75, 10.]; pts[1] = [-.25, 10.]; pts[7] = [-1., -1.]
    proposals, queries = [], []
    for si in range(SCENES):
        cs = [] if si % 3 == 0 else [dict(score=.8, keypoints_xy=pts.tolist()), dict(score=.8, keypoints_xy=(pts + 1).tolist())]
        selected = None if not cs else 0
        s = dict(scene_index=si, base_index=si // 4, id='V136_SCENE_%04d' % si, base_id='V136_BASE_%04d' % (si // 4),
                 split='TRAIN' if si // 4 < 512 else 'VALIDATION', variant=VARIANTS[si % 4], selected_index=selected,
                 raw_hw=[512, 512], raw_RGB_PNG={'path': '/generated/%04d.png' % si, 'sha256': '0' * 64}, decoded_RGB_sha256='1' * 64, candidates=cs)
        proposals.append(s)
        for q in query.selected_queries(s, [512, 512]):
            identity = {k: s[k] for k in ('scene_index', 'base_index', 'id', 'base_id', 'split', 'variant', 'selected_index')}
            queries.append(dict(identity, query_id=s['id'] + ':native' + str(q['native_slot']), **q,
                                actual_original_initializer=True, source_emulation=False))
    rows = geometry.input_geometry(proposals, queries, op, target, grid)
    assert len(rows) == N and sum(r['query_xy'] is None for r in rows) == 6832
    assert len({r['occurrence_id'] for r in rows}) == N
    assert rows[8]['query_xy'] == [511.75, 10.] and rows[8]['native_continuous_valid'] and not rows[8]['native_nearest_valid']
    assert rows[9]['query_xy'] == [-.25, 10.] and not rows[9]['native_continuous_valid'] and rows[9]['native_nearest_valid']
    assert not rows[15]['finite_non_sentinel_query'] and rows[15]['query_xy'] == [-1., -1.]
    spatial = op.pure(p['v61']['assets']['spatial_math']['path'], ('query_validity', 'expected_affine', 'native_grid_explicit'))
    q = np.array([[-.25, 10.], [511.75, 10.], [511.49, 10.], [-1., -1.], [np.nan, np.nan]])
    good, cont, near = spatial['query_validity'](q, [512, 512])
    assert good.tolist() == [False, False, True, False, False]
    assert cont.tolist() == [False, True, True, False, False] and near.tolist() == [True, False, True, False, False]
    affine = spatial['expected_affine']([512, 512])
    assert affine['canvas_shape'] == [712, 712] and affine['input_shape'] == [640, 640]
    assert affine['p3_actual_hw'] == [80, 80] and affine['gain'] == 640 / 712 and affine['offset'].tolist() == [0., 0.]
    samplegrid, mask = spatial['native_grid_explicit'](q[:4], 16., [512, 512])
    assert samplegrid.shape == (4, 9, 9, 2) and mask.shape == (4, 9, 9) and not mask[3].any()
    assert op.decode(np.array([-1., -0., 0., 1., np.nan]), np.array([1, 1, 1, 1, 0], bool), 'v61').tolist() == [1, 0, 0, 0, -1]
    broken = [dict(r) for r in queries]; broken[8]['query_xy'] = [511., 10.]
    try:
        geometry.input_geometry(proposals, broken, op, target, grid)
    except AssertionError:
        pass
    else:
        raise AssertionError('Altered query was accepted')
    _, proof = adapted_function(); assert proof == p['count_only_AST_substitution']
    write(GENERATED, dict(completed=True, PASS=True, code=bind(__file__), count_only_AST_substitution=proof,
        all20480_ids_and_missing_OOB_sentinel_rows_retained=True, invented_missing_slots=6832,
        first_argmax_tie_original_queries_and_altered_query_rejection=True, continuous_nearest_separate_factual_U=True,
        original640_rect_reflect100_affine_and_native9grid=True, raw_visible_signedzero_D_negative_E_unavailable_U=True,
        invented_arrays_only=True, actual_RGB_GT_checkpoint_encoder_model_reads=0))


def prepare(policy_path):
    assert Path(policy_path).resolve() == POLICY and sha(POLICY) == POLICY_SHA
    parent = read(PARENT_PROTOCOL); assert same(parent['code'], bind(PARENT_CODE))
    assert Path(sys.executable).resolve() == Path(parent['c16']['runtime']['executable']).resolve() and sys.version == parent['c16']['runtime']['python'], 'AST proofs must be prepared in the pinned model Python runtime.'
    feature = read(FEATURE_PROTOCOL); assert same(feature['code'], bind(FEATURE_CODE)) and same(feature['source_policy'], bind(POLICY))
    initial = read(INITIALIZER_PROTOCOL); assert same(initial['code'], bind(INITIALIZER_CODE)) and same(initial['policy'], bind(POLICY))
    old = read(OLD_V61_SEAL); assert old['completed'] and old['PASS'] and old['GT_read'] is False
    _, proof = adapted_function()
    p = dict(completed=True, code=bind(__file__), source_policy=bind(POLICY), parent_code=bind(PARENT_CODE), parent_protocol=bind(PARENT_PROTOCOL),
             feature_geometry_code=bind(FEATURE_CODE), feature_geometry_protocol=bind(FEATURE_PROTOCOL),
             initializer_code=bind(INITIALIZER_CODE), initializer_protocol=bind(INITIALIZER_PROTOCOL),
             initializer_parent_code=initial['parent_code'], c16={'runtime': parent['c16']['runtime']}, v61=parent['v61'],
             query_math={'target_crop_code': parent['c16']['target_crop_code'], 'native_grid_code': parent['native_grid_code']},
             count_only_AST_substitution=proof, source_scene_count=SCENES, rows=N, output=str(OUT),
             expected_initializer_seal=str(V / 'actual_initializer_v1/PREDICTION_SEAL.json'),
             original_frozen_asset_reference=dict(seal=bind(OLD_V61_SEAL), loaded_state_sha256=old['loaded_state_sha256'],
                warmed_state_sha256=old['warmed_state_sha256'], head_state_sha256=old['head_state_sha256']),
             preprocessing=parent['v61']['native_preprocessing'], availability=parent['v61']['availability'],
             score='raw_visible_logit and its negative raw_hidden_logit. Exact visible0=>D; unavailableU. No threshold/calibration change.',
             preservation='All20480 slots in original identity/order. Native actual q and L unchanged, centroid excluded; no GT/config/order input.',
             output_schema='V61_PREDICTIONS_INPUT_ONLY.npz+ROI65 chunks+INPUT_P3_FACTS+all20480 input whitelist; no full P3 stored',
             actual_original_initializer=True, source_emulation=False, old_prediction_replay_on_new_data=False,
             GT_read=False, training=False, current223=False, V112=False, PnP=False, q_correction=False,
             acceptance_rule_broadened=False, calibration=False, automatic_retry=False)
    for b in (p['v61']['assets']['spatial_math'], p['v61']['assets']['backbone_code'], p['v61']['assets']['head_class'], *p['query_math'].values()):
        check(b)
    op = module(PARENT_CODE, 'v136_inert_exact_v61_operators'); geometry = module(FEATURE_CODE, 'v136_inert_query_geometry')
    generated(p, op, geometry); p['generated'] = bind(GENERATED)
    write(PROTOCOL, p)
    entry = dict(approved_frozen_v61=False, fit=False, current223=False, V112=False, PnP=False, q_correction=False,
                 calibration=False, broaden_acceptance=False, automatic_retry=False, code=p['code'], protocol=bind(PROTOCOL), source_policy=p['source_policy'],
                 initializer_seal=None, initializer_result=None, initializer_proposals=None, initializer_queries=None,
                 fill_after_new_initializer_complete='Bind new actual V136 completed initializer seal→result→proposals/queries, not an old source seal.')
    write(H / 'ROOT_V61_ENTRY_TEMPLATE.json', entry)
    print(json.dumps(dict(completed=True, protocol=bind(PROTOCOL), generated=p['generated'], template=bind(H / 'ROOT_V61_ENTRY_TEMPLATE.json'),
                          actual_RGB_GT_checkpoint_encoder_model_reads=0)), flush=True)


def authorize(entry_path):
    e = read(entry_path); assert e['approved_frozen_v61'] is True
    assert all(e[k] is False for k in ('fit', 'current223', 'V112', 'PnP', 'q_correction', 'calibration', 'broaden_acceptance', 'automatic_retry'))
    assert same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL))
    p = read(PROTOCOL); assert same(p['code'], e['code']) and same(p['source_policy'], e['source_policy'])
    assert Path(sys.executable).resolve() == Path(p['c16']['runtime']['executable']).resolve() and sys.version == p['c16']['runtime']['python'], 'AST proof runtime must match preparation and model runtime.'
    assert Path(p['source_policy']['path']).resolve() == POLICY and p['source_policy']['sha256'] == POLICY_SHA
    for key in ('source_policy', 'parent_code', 'parent_protocol', 'feature_geometry_code', 'feature_geometry_protocol',
                'initializer_code', 'initializer_protocol', 'initializer_parent_code', 'generated'):
        check(p[key])
    proof = read(p['generated']['path']); assert proof['PASS'] and same(proof['code'], e['code'])
    for b in p['v61']['assets'].values():
        check(b)
    for b in p['query_math'].values():
        check(b)
    for key in ('initializer_seal', 'initializer_result', 'initializer_proposals', 'initializer_queries'):
        check(e[key])
    assert Path(e['initializer_seal']['path']).resolve() == Path(p['expected_initializer_seal']).resolve()
    seal = read(e['initializer_seal']['path']); result = read(e['initializer_result']['path'])
    assert seal['completed'] and seal['PASS'] and seal['GT_read'] is False and seal['all2560_scenes20480_slots_preserved']
    assert seal['actual_original_initializer'] is True and seal['source_emulation'] is False and same(seal['result'], e['initializer_result'])
    assert result['completed'] and result['PASS'] and result['frames'] == SCENES and result['slots'] == N
    assert result['GT_mixed_buffers_mapping_crop_head_fit_current_V112_PnP_access'] == 0 and result['after_state_unchanged'] and result['checkpoint_unchanged']
    assert same(result['code'], p['initializer_code']) and same(result['protocol'], p['initializer_protocol']) and same(result['policy'], p['source_policy'])
    assert same(result['proposals'], e['initializer_proposals']) and same(result['queries'], e['initializer_queries'])
    op = module(p['parent_code']['path'], 'v136_exact_v61_runtime'); geometry = module(p['feature_geometry_code']['path'], 'v136_actual_query_whitelist')
    rows = geometry.input_geometry(lines(e['initializer_proposals']['path']), lines(e['initializer_queries']['path']), op,
        op.pure(p['query_math']['target_crop_code']['path'], ('query_extent', 'valid_box')),
        op.pure(p['query_math']['native_grid_code']['path'], ('native_grid',))['native_grid'])
    function, adapted = adapted_function(); assert adapted == p['count_only_AST_substitution']
    exec(compile(function, str(PARENT_CODE) + '::scene_count_only', 'exec'), op.__dict__)
    op.N = N; op.common_arrays = geometry.common_arrays
    assert not OUT.exists(), 'Preserve completed or failed attempt; no automatic retry.'
    return e, p, rows, op


def run(entry_path):
    e, p, rows, op = authorize(entry_path); OUT.mkdir()
    lock = OUT / 'V61_NATIVE_INPUT_GEOMETRY_BEFORE_RGB.json'
    write(lock, dict(completed=True, rows=rows, initializer_seal=e['initializer_seal'], GT_read=False, actual_original_initializer=True, source_emulation=False))
    input_seal = OUT / 'INPUT_V61_SEAL_BEFORE_RGB_MODEL.json'
    write(input_seal, dict(completed=True, PASS=True, input_lock=bind(lock), entry=bind(entry_path), initializer_seal=e['initializer_seal'],
                          rows=N, source_scenes=SCENES, q_L_from_completed_actual_initializer_exact=True, GT_read=False))
    write(OUT / 'STARTED.json', dict(entry=bind(entry_path), input_seal=bind(input_seal), rows=N, source_scenes=SCENES, GT_read=False))
    started = time.monotonic()
    try:
        result = op.run_v61(e, p, rows, OUT)
        reference = p['original_frozen_asset_reference']
        assert result['loaded_state_sha256'] == reference['loaded_state_sha256']
        assert result['warmed_state_sha256'] == reference['warmed_state_sha256']
        assert result['head_state_sha256'] == reference['head_state_sha256']
        assert result['rows'] == N and result['backbone_state_unchanged'] and result['head_state_unchanged']
        check(p['v61']['assets']['backbone_checkpoint']); check(p['v61']['assets']['head_checkpoint'])
        final = dict(completed=True, PASS=True, code=e['code'], protocol=e['protocol'], entry=bind(entry_path), source_policy=p['source_policy'],
                     initializer_seal=e['initializer_seal'], initializer_result=e['initializer_result'], initializer_proposals=e['initializer_proposals'],
                     initializer_queries=e['initializer_queries'], input_seal=bind(input_seal), all2560_scenes20480_slots_preserved=True,
                     actual_original_initializer=True, source_emulation=False, same_original_v61_asset_states_exact=True,
                     old_prediction_replay_on_new_data=False, GT_read=False, fit_current_V112_PnP_q_correction=0,
                     no_model_threshold_or_calibration_choice=True, acceptance_rule_broadened=False,
                     seconds=time.monotonic() - started, **result)
        write(OUT / 'PREDICTION_SEAL.json', final)
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
