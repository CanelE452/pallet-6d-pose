"""V139: exact C29 replay versus original sealed native G42 conditioning.

The pinned V137 run is adapted by explicit, count-checked edits below. The
effective run and its complete diff are saved for review. Its namespace is a
copy: neither the parent module nor its original6144 helper is modified.
Only --entry may read actual feature/GT/G42/checkpoint bodies or execute a fit.
"""
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import importlib.util
import inspect
import json
import numpy as np
import torch

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V / 'ROOT_BROAD_SOURCE_GEOMETRY_POLICY_BEFORE_FIT.json'
POLICY_SHA = 'b1c38b4110f90ada399fe71c15fb29cc931b0adeb852ed920b2b1556dfb00415'
PARENT = P / 'query_visibility_continuous_source_coverage_control_20261024_v137/source_fit_preparation/train_matched_coverage.py'
PARENT_SHA = '9751b158c14cf0e0ba1cc2b8cb41e761f798c6ff8fe6b6dcdd4d69d2b5f9d2b5'
PROTOCOL = H / 'PROTOCOL_BEFORE_SOURCE_FIT.json'
GENERATED = H / 'GENERATED_MEANINGFUL_CHECKS.json'
RUN = V / 'runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000'
ARMS = ('C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL', 'C31_BROAD_SOURCE_NATIVE_G42')
C29 = PARENT.parent.parent / 'runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/C29_CONTINUOUS_NATIVE_DATA_ADDITION'
OLD_GEOMETRY_SEAL = P / 'query_visibility_native_geometry_control_20261024_v133/input_geometry_v1/INPUT_GEOMETRY_SEAL.json'
SCORE_FILES = ('FINAL_OLD51004_SOURCE_SCORES.npz', 'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz', 'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz')
INPUT_KEYS = ('global_index', 'corpus', 'source_occurrence_index', 'occurrence_id', 'scene_index', 'base_index', 'native_slot', 'split',
              'query_xy', 'L', 'G42', 'G42_constructed', 'feature_constructed', 'query_available', 'native_continuous_valid', 'native_nearest_valid')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


assert sha(PARENT) == PARENT_SHA
spec = importlib.util.spec_from_file_location('v139_pinned_V137', PARENT)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
bind, same, check, read, write = h.bind, h.same, h.check, h.read, h.write
q, m = h.q, h.m


def equal_array(a, b):
    # Includes NaN payloads and signed zero, stronger than numerical equality.
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')


def validate_geometry(table, oldpacket, newpacket, old_split, new_split, original):
    """Only G42 is numeric model input; identity/flags below are join guards."""
    assert set(table) == set(INPUT_KEYS)
    n = h.NATIVE + h.NEWN
    assert np.array_equal(table['global_index'], np.arange(n))
    G = table['G42']; constructed = table['G42_constructed']
    assert G.shape == (n, 42) and G.dtype == np.float32 and np.isfinite(G).all()
    assert constructed.dtype == bool and not G[~constructed].any()
    for lo, packet, split, corpus in ((0, oldpacket, old_split, 'OLD_NATIVE'), (h.NATIVE, newpacket, new_split, 'CONTINUOUS_NATIVE')):
        size = len(packet['occurrence_id']); section = slice(lo, lo + size)
        assert np.array_equal(table['corpus'][section], np.repeat(corpus, size))
        assert np.array_equal(table['source_occurrence_index'][section], np.arange(size))
        assert np.array_equal(table['split'][section], split)
        for key in ('occurrence_id', 'scene_index', 'base_index', 'native_slot', 'query_xy', 'L', 'feature_constructed', 'query_available',
                    'native_continuous_valid', 'native_nearest_valid'):
            a, b = table[key][section], packet[key]
            assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b), key
        assert np.array_equal(constructed[section], packet['feature_constructed'])
    assert equal_array(G[:h.NATIVE], original['G42'])
    assert np.array_equal(constructed[:h.NATIVE], original['geometry_constructed'])
    for key in ('query_xy', 'L', 'query_available', 'native_continuous_valid', 'native_nearest_valid', 'scene_index', 'base_index', 'native_slot', 'occurrence_id'):
        a, b = table[key][:h.NATIVE], original[key]
        assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b), key
    return G[:h.NATIVE].copy(), G[h.NATIVE:].copy()


def load_geometry(protocol, data, new):
    # np.load is lazy. No risk/confidence/descriptor/normalized-q arrays are read.
    with np.load(check(protocol['input_table']), allow_pickle=False) as f:
        table = {k: f[k].copy() for k in INPUT_KEYS}
    with np.load(check(protocol['old_geometry_packet']), allow_pickle=False) as f:
        keys = ('G42', 'geometry_constructed', 'query_xy', 'L', 'query_available', 'native_continuous_valid', 'native_nearest_valid',
                'scene_index', 'base_index', 'native_slot', 'occurrence_id')
        original = {k: f[k].copy() for k in keys}
    return validate_geometry(table, data['npack'], new['packet'], data['nmeta'][h.CONVENTIONS[0]]['split'], new['meta'][h.CONVENTIONS[0]]['split'], original)


def validate_extra_bindings(p, entry):
    assert same(entry['effective_run'], p['effective_run']) and same(entry['parent_run_diff'], p['parent_run_diff'])
    for key in ('parent_V137_code', 'parent_V137_protocol', 'parent_V137_policy', 'parent_V137_control', 'input_seal', 'input_table',
                'old_geometry_seal', 'old_geometry_packet', 'historical_C29_fit', 'historical_C29_freeze', 'historical_C29_checkpoint',
                'historical_C29_trace', 'historical_C29_history', 'historical_C29_exposure', 'historical_C29_results', 'effective_run', 'parent_run_diff'):
        check(p[key])
    for b in p['historical_C29_scores'].values(): check(b)
    assert p['arms'] == list(ARMS) and p['run_directory'] == str(RUN)
    assert p['parameters_each'] == 224513 and p['fixed_updates'] == 8000
    assert p['source_selection'] == read(p['parent_V137_protocol']['path'])['source_selection']
    seal = read(p['input_seal']['path']); assert seal['completed'] and seal['PASS'] and seal['rows'] == 26624
    assert all(seal[k] is False for k in ('GT_read', 'C29score_read', 'v61score_read', 'RGB_body_read', 'visual_feature_body_read'))
    assert same(seal['table'], p['input_table'])
    assert same(seal['source_input_bindings']['OLD_NATIVE']['config']['input_crop_seal'], read(p['source_binding_policy']['path'])['native_feature_seal'])
    assert same(seal['source_input_bindings']['CONTINUOUS_NATIVE']['config']['input_crop_seal'], p['new_feature_seal'])
    assert m.NNATIVE == h.NATIVE == 6144


def assert_draw_replay(historical, step, ix, pairs, ni, npi, ei, epi):
    assert historical['update'] == step and historical['extra_source'] == 'V136_NATIVE'
    for value, key in ((ix, 'old_occurrences'), (pairs, 'old_matched_cover_sham_positions'), (ni, 'native_occurrences'),
                       (npi, 'native_pair_pool_positions'), (ei, 'extra_native_occurrences'), (epi, 'extra_pair_pool_positions')):
        assert np.array_equal(value, historical[key]), key


def recursive_equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and equal_array(a.cpu().numpy(), b.cpu().numpy())
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(recursive_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return type(a) == type(b) and len(a) == len(b) and all(recursive_equal(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def final_c29_replay(directory, p):
    """Mandatory control equality, not a choice between fitted outputs."""
    checks = {}
    for filename in SCORE_FILES + ('SAMPLING_EXPOSURE.npz',):
        historical = p['historical_C29_exposure'] if filename == 'SAMPLING_EXPOSURE.npz' else p['historical_C29_scores'][filename]
        with np.load(directory / filename, allow_pickle=False) as a, np.load(check(historical), allow_pickle=False) as b:
            assert a.files == b.files
            checks[filename] = {k: equal_array(a[k], b[k]) for k in a.files}
            assert all(checks[filename].values()), filename
    assert sha(directory / 'TRAIN_DRAWS_AND_LOSSES.jsonl') == p['historical_C29_trace']['sha256']
    assert sha(directory / 'HISTORY.jsonl') == p['historical_C29_history']['sha256']
    current = torch.load(directory / 'FINAL_RESUME_CHECKPOINT.pt', map_location='cpu', weights_only=True)
    previous = torch.load(check(p['historical_C29_checkpoint']), map_location='cpu', weights_only=True)
    state_checks = {k: recursive_equal(current[k], previous[k]) for k in ('state_dict', 'optimizer_state', 'RNG', 'update', 'C16_parent_update', 'threshold_logit')}
    assert all(state_checks.values())
    path = directory / 'C30_EXACT_C29_REPLAY.json'
    write(path, dict(completed=True, PASS=True, all8000_draws_losses_trace_bitexact=True, all81_history_rows_bitexact=True,
                    all_score_arrays_and_exposure_exact=checks, final_state_optimizer_RNG_exact=state_checks,
                    original_C29_fit=p['historical_C29_fit'], original_C29_checkpoint=p['historical_C29_checkpoint'],
                    arm_policy_trainer_metadata_expected_different=True))
    return bind(path)


def adapted_run_source():
    source = inspect.getsource(h.run)
    edits = []
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new); edits.append(dict(before=old, after=new))
    replace("p = read(PROTOCOL)", "p = read(PROTOCOL)\n    validate_extra_bindings(p, e)")
    replace("new = load_new(p); newmeta = new['meta'][CONVENTIONS[0]]", "new = load_new(p); newmeta = new['meta'][CONVENTIONS[0]]\n    native_G, new_G = load_geometry(p, data, new)")
    replace("xn = torch.as_tensor(data['npack']['features'], device='cuda'); xx = torch.as_tensor(new['packet']['features'], device='cuda')", "xn = torch.as_tensor(data['npack']['features'], device='cuda'); xx = torch.as_tensor(new['packet']['features'], device='cuda')\n    ng = torch.as_tensor(native_G, device='cuda'); eg = torch.as_tensor(new_G, device='cuda')")
    replace("actual_geometry_input_used=False))", "actual_geometry_input_used='C30_ZERO_C31_ORIGINAL_SEALED_G42', input_seal=p['input_seal']))")
    replace("for arm in ARMS:\n", "for arm in ARMS:\n            actual_geometry = arm == ARMS[1]\n")
    replace("extra_pool, extra_uniform = (oldpairs, data['native_pools']) if arm == ARMS[0] else (new['pairs'], new['pools'])", "extra_pool, extra_uniform = new['pairs'], new['pools']")
    replace("ex, extarget, exav = (xn, nt, na) if arm == ARMS[0] else (xx, et, ea)", "ex, extarget, exav = xx, et, ea")
    replace("for tensor, constructed, size in ((xn, ni_all, NATIVE), (xx, ei_all, NEWN)):", "for tensor, geometry, constructed, size in ((xn, ng, ni_all, NATIVE), (xx, eg, ei_all, NEWN)):")
    replace("model(tensor[ix], zeros[:len(ix)]).cpu().numpy()", "model(tensor[ix], geometry[ix] if actual_geometry else zeros[:len(ix)]).cpu().numpy()")
    replace("np.array_equal(old0, data['baseline']['query_logit'])", "equal_array(old0, data['baseline']['query_logit'])")
    replace("np.array_equal(native0, data['nbaseline']['raw_head_logit_before_U'], equal_nan=True)", "equal_array(native0, data['nbaseline']['raw_head_logit_before_U'])")
    replace("np.array_equal(new0, new['baseline']['raw_head_logit_before_U'], equal_nan=True)", "equal_array(new0, new['baseline']['raw_head_logit_before_U'])")
    replace("Path(p['historical_C24_trace']['path']).open() as c24:", "Path(p['historical_C24_trace']['path']).open() as c24, Path(p['historical_C29_trace']['path']).open() as c29:")
    replace("prior = json.loads(next(previous)); original = json.loads(next(c24))", "prior = json.loads(next(previous)); original = json.loads(next(c24))\n                    historical = json.loads(next(c29)); assert_draw_replay(historical, step, ix, pairs, ni, npi, ei, epi)")
    replace("extra_meta = oldmeta if arm == ARMS[0] else newmeta", "extra_meta = newmeta")
    replace("extra_available = data['npack']['query_available'] if arm == ARMS[0] else new['packet']['query_available']", "extra_available = new['packet']['query_available']")
    replace("np.add.at(exposure_native if arm == ARMS[0] else exposure_new, ei, 1)", "np.add.at(exposure_new, ei, 1)")
    replace("zo = model(xt[units[oi]], zeros[:192]); zn = model(xn[nn], zeros[:64]); ze = model(ex[ee], zeros[:64])", "zo = model(xt[units[oi]], zeros[:192]); zn = model(xn[nn], ng[nn] if actual_geometry else zeros[:64]); ze = model(ex[ee], eg[ee] if actual_geometry else zeros[:64])")
    replace("assert not model.geometry.weight.any()\n                    trace.write(json.dumps(dict(", "if not actual_geometry: assert not model.geometry.weight.any()\n                    record = dict(")
    replace("extra_source='OLD_NATIVE' if arm == ARMS[0] else 'V136_NATIVE',", "extra_source='V136_NATIVE',")
    replace("gradient_before_clip=float(grad)), allow_nan=False) + '\\n')", "gradient_before_clip=float(grad))\n                    if not actual_geometry: assert record == historical, f'C30 C29 trace mismatch at {step}'\n                    trace.write(json.dumps(record, allow_nan=False) + '\\n')")
    replace("assert next(previous, None) is None and next(c24, None) is None", "assert next(previous, None) is None and next(c24, None) is None and next(c29, None) is None")
    replace("shared = shared_hash.hexdigest(); rng_hash =", "assert shared_hash.hexdigest() == p['historical_C29_draw_hash'] and extra_hash.hexdigest() == p['historical_C29_extra_draw_hash']\n            shared = shared_hash.hexdigest(); rng_hash =")
    replace("write(directory / 'SOURCE_RESULTS.json', dict(", "replay = final_c29_replay(directory, p) if not actual_geometry else None\n            write(directory / 'SOURCE_RESULTS.json', dict(")
    replace("teacher_SU_no_external_gradient=True, actual_geometry_input_used=False,", "teacher_SU_no_external_gradient=True, actual_geometry_input_used=actual_geometry, C30_C29_exact_replay=replay,")
    replace("all_geometry_input_zero=True,", "all_geometry_input_zero=not actual_geometry, native_original_G42_input=actual_geometry, geometry_input_seal=p['input_seal'],")
    replace("source_eligible=eligible, worst_required49_class_recall=worst))\n            arms[arm]", "source_eligible=eligible, worst_required49_class_recall=worst, exact_C29_replay=replay))\n            arms[arm]")
    replace("extra_occurrence_ids_not_claimed_equal=True,", "extra_occurrence_ids_all8000_equal_to_C29_both_arms=True, C30_exact_C29_replay=bind(RUN / ARMS[0] / 'C30_EXACT_C29_REPLAY.json'),\n            only_difference_native_geometry_ZERO_vs_G42=True, native_G42_input_seal=p['input_seal'],")
    return source, edits


EFFECTIVE_SOURCE, EDITS = adapted_run_source()
namespace = dict(vars(h))
namespace.update(__file__=__file__, H=H, V=V, P=P, POLICY=POLICY, POLICY_SHA=POLICY_SHA, PROTOCOL=PROTOCOL, GENERATED=GENERATED, RUN=RUN, ARMS=ARMS,
                 load_geometry=load_geometry, validate_extra_bindings=validate_extra_bindings, assert_draw_replay=assert_draw_replay,
                 final_c29_replay=final_c29_replay, equal_array=equal_array)
exec(compile(EFFECTIVE_SOURCE, str(H / 'EFFECTIVE_V139_RUN.py'), 'exec'), namespace)
run = namespace['run']


def generated_checks():
    torch.set_num_threads(1); torch.manual_seed(1330)
    # Invented only: zero branch preserves complete warm head initially, but
    # nonzero geometry can train its weights while auxiliary G0 cannot.
    a = q.GeometryQueryHead().float(); b = q.GeometryQueryHead().float(); b.load_state_dict(a.state_dict())
    x = torch.randn(64, 1928); G = torch.randn(64, 42); z = torch.zeros_like(G)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters()) == 224513
    assert torch.equal(a(x, z), b(x, G)) and not b.geometry.weight.any()
    y = torch.tensor([1.] * 16 + [0.] * 32 + [1.] * 16); av = torch.ones(64, dtype=torch.bool)
    q.native_bce(a(x, z), y, av).backward(); q.native_bce(b(x, G), y, av).backward()
    assert not a.geometry.weight.grad.any() and b.geometry.weight.grad.abs().sum() > 0
    assert all(torch.equal(p.grad, r.grad) for (k, p), (j, r) in zip(a.named_parameters(), b.named_parameters()) if k != 'geometry.weight')
    # Separate old/native/new mean weights unchanged; unlabelled/missing have no gradient.
    raw = torch.zeros(4, requires_grad=True)
    q.native_bce(raw, torch.tensor([0., 1., -1., 1.]), torch.tensor([1, 1, 1, 0], dtype=torch.bool)).backward()
    assert raw.grad[:2].abs().min() > 0 and not raw.grad[2:].any()
    # Same C29 native sampler, every draw, same RNG consumption; no source bodies.
    pair = np.c_[np.arange(20), np.arange(20, 40)]; uniform = [np.arange(20, 80), np.r_[np.arange(20), np.arange(80, 100)]]
    histories = []
    for arm in range(2):
        pg = torch.Generator().manual_seed(1374); ug = torch.Generator().manual_seed(1372)
        histories.append([q.draw_native64(pair, uniform, pg, ug) for _ in range(100)])
    assert all(np.array_equal(a, b) for aa, bb in zip(*histories) for a, b in zip(aa, bb))
    # Sentinel metadata joins are exact; defined finite OOB geometry stays unchanged.
    def packet(n, offset):
        return dict(occurrence_id=np.array([f'ID{offset+i}' for i in range(n)]), scene_index=np.arange(n)//8, base_index=np.arange(n)//32,
                    native_slot=np.arange(n)%8, query_xy=np.c_[np.arange(n, dtype=float), np.zeros(n)], L=np.full(n, 16.),
                    feature_constructed=np.ones(n, bool), query_available=np.ones(n, bool), native_continuous_valid=np.ones(n, bool), native_nearest_valid=np.ones(n, bool))
    # Exercise helper at actual dimensions using invented metadata, never source data.
    op, npk = packet(h.NATIVE, 0), packet(h.NEWN, h.NATIVE)
    op['feature_constructed'][0] = op['query_available'][0] = False; op['query_xy'][0] = np.nan; op['L'][0] = np.nan
    op['query_xy'][1] = [-2., 15.]; op['query_available'][1] = op['native_continuous_valid'][1] = op['native_nearest_valid'][1] = False
    total = h.NATIVE + h.NEWN
    table = {k: np.concatenate((op[k], npk[k])) for k in op}; table.update(global_index=np.arange(total),
        corpus=np.array(['OLD_NATIVE']*h.NATIVE + ['CONTINUOUS_NATIVE']*h.NEWN), source_occurrence_index=np.r_[np.arange(h.NATIVE), np.arange(h.NEWN)],
        split=np.repeat('TRAIN', total), G42=np.ones((total, 42), np.float32), G42_constructed=np.r_[op['feature_constructed'], npk['feature_constructed']])
    table['G42'][0] = 0
    original={k: table[k][:h.NATIVE].copy() for k in ('G42', 'query_xy', 'L', 'query_available', 'native_continuous_valid', 'native_nearest_valid', 'scene_index', 'base_index', 'native_slot', 'occurrence_id')}
    original['geometry_constructed'] = op['feature_constructed'].copy()
    oldG, newG = validate_geometry(table, op, npk, np.repeat('TRAIN', h.NATIVE), np.repeat('TRAIN', h.NEWN), original)
    assert oldG[1].all() and not oldG[0].any() and newG.shape == (h.NEWN, 42)
    changed = dict(table); changed['query_xy'] = table['query_xy'].copy(); changed['query_xy'][1,0] += .01
    try: validate_geometry(changed, op, npk, np.repeat('TRAIN', h.NATIVE), np.repeat('TRAIN', h.NEWN), original)
    except AssertionError: pass
    else: raise AssertionError('Changed original q accepted')
    # Final replay includes dtype/NaN/state, rejects both value and parameter changes.
    assert equal_array(np.array([np.nan, 1.], np.float32), np.array([np.nan, 1.], np.float32))
    assert not equal_array(np.array([1.], np.float32), np.array([1.], np.float64))
    assert not equal_array(np.array([0.], np.float32), np.array([-0.], np.float32))
    assert not equal_array(np.array([0x7fc00000], np.uint32).view(np.float32), np.array([0x7fc00001], np.uint32).view(np.float32))
    assert recursive_equal({'x': torch.ones(2)}, {'x': torch.ones(2)}) and not recursive_equal({'x': torch.ones(2)}, {'x': torch.zeros(2)})
    # AST validates effective code, and parent globals/legacy native count stay intact.
    ast.parse(EFFECTIVE_SOURCE)
    assert h.RUN != RUN and h.ARMS != ARMS and h.POLICY != POLICY and m.NNATIVE == 6144
    write(GENERATED, dict(completed=True, PASS=True, code=bind(__file__), parent_code=bind(PARENT), invented_arrays_only=True,
        parameters224513_equal_and_initial_nonzero_G42_bitexact=True, old_G0_zero_geometry_gradient=True, native_G42_nonzero_geometry_gradient=True,
        initial_visual_gradients_identical=True, no_SU_or_missing_gradient=True, paired100_draws_RNG_exact=True,
        all26624_invented_identity_q_L_geometry_sentinel_OOB_and_tamper_checks=True, original_old_G42_exact_join_required=True,
        final_replay_dtype_NaN_state_changes_detected=True, explicit_parent_run_edits=len(EDITS), parent_globals_not_mutated=True,
        model_checkpoint_feature_GT_RGB_actual_input_body_reads_or_fit=0))
    print(json.dumps(dict(completed=True, generated=bind(GENERATED), actual_fit=False)), flush=True)


def prepare():
    assert sha(POLICY) == POLICY_SHA
    policy = read(POLICY); assert policy['arms'] == list(ARMS)
    proof = read(GENERATED); assert proof['PASS'] and same(proof['code'], bind(__file__))
    parent = read(check(policy['parent_V137_protocol']))
    assert same(policy['parent_V137_code'], bind(PARENT)) and parent['fixed_updates'] == 8000 and parent['new_TRAIN_support']['pair_count'] == 473
    seal_binding = policy['prior_evidence']['input_only_seal']; seal = read(check(seal_binding))
    assert seal['completed'] and seal['PASS'] and seal['rows'] == 26624
    oldseal = read(OLD_GEOMETRY_SEAL); assert oldseal['completed'] and oldseal['PASS'] and oldseal['GT_read'] is False
    fit = read(C29 / 'FIT_COMPLETE.json'); results = read(C29 / 'SOURCE_RESULTS.json'); freeze = read(C29 / 'MODEL_FREEZE.json')
    assert fit['completed'] and fit['optimizer_updates'] == 8000 and freeze['parameters'] == 224513
    effective = H / 'EFFECTIVE_V139_RUN.py'; effective.write_text(EFFECTIVE_SOURCE)
    diff = H / 'EXACT_PARENT_RUN_DIFF.patch'
    diff.write_text(''.join(difflib.unified_diff(inspect.getsource(h.run).splitlines(True), EFFECTIVE_SOURCE.splitlines(True), fromfile='PINNED_V137_RUN', tofile='EFFECTIVE_V139_RUN')))
    p = dict(parent)
    p.update(code=bind(__file__), policy=bind(POLICY), generated=bind(GENERATED), parent_V137_code=bind(PARENT), parent_V137_protocol=policy['parent_V137_protocol'],
        parent_V137_policy=policy['parent_V137_policy'], parent_V137_control=policy['parent_V137_control'], arms=list(ARMS),
        geometry_all_zero=False, actual_G42_packet_read='ROOT_ENTRY_ONLY', input_seal=seal_binding, input_table=seal['table'],
        input_keys_read=list(INPUT_KEYS), only_model_geometry_input='G42 original42 columns; no recomputation or secondary descriptors',
        old_geometry_seal=bind(OLD_GEOMETRY_SEAL), old_geometry_packet=oldseal['G42'],
        historical_C29_fit=bind(C29 / 'FIT_COMPLETE.json'), historical_C29_freeze=bind(C29 / 'MODEL_FREEZE.json'), historical_C29_checkpoint=freeze['checkpoint'],
        historical_C29_trace=bind(C29 / 'TRAIN_DRAWS_AND_LOSSES.jsonl'), historical_C29_history=bind(C29 / 'HISTORY.jsonl'),
        historical_C29_results=bind(C29 / 'SOURCE_RESULTS.json'), historical_C29_exposure=bind(C29 / 'SAMPLING_EXPOSURE.npz'),
        historical_C29_scores={name: bind(C29 / name) for name in SCORE_FILES},
        historical_C29_draw_hash=results['shared_old192_native64_sha256'], historical_C29_extra_draw_hash=results['extra_native_draw_sha256'],
        effective_run=bind(effective), parent_run_diff=bind(diff), explicit_run_edits=len(EDITS), run_directory=str(RUN),
        exact_C30_C29_every8000_draw_loss_score_state_optimizer_RNG_replay_required=True,
        both_native_sampling_populations='C29 unchanged original6144 and continuous20480, same TRAIN125+473pairs',
        actual_source_input_G42_feature_GT_checkpoint_body_reads_during_preparation=0, actual_fit=False)
    write(PROTOCOL, p)
    template=dict(approved_source_fit=False, current223=False, V112=False, PnP=False, q_correction=False, automatic_retry=False,
        code=p['code'], protocol=bind(PROTOCOL), policy=p['policy'], generated=p['generated'], effective_run=p['effective_run'], parent_run_diff=p['parent_run_diff'])
    write(H / 'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json', template)
    print(json.dumps(dict(completed=True, prepared_only=True, code=bind(__file__), protocol=bind(PROTOCOL), generated=bind(GENERATED),
        effective_run=p['effective_run'], parent_run_diff=p['parent_run_diff'], template=bind(H / 'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'), actual_fit=False)), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--generated', action='store_true'); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--entry', type=Path)
    args=parser.parse_args()
    if args.generated and not args.prepare and args.entry is None: generated_checks()
    elif args.prepare and not args.generated and args.entry is None: prepare()
    elif args.entry and not args.generated and not args.prepare: run(args.entry)
    else: raise SystemExit('Use --generated, --prepare or explicit root --entry')
