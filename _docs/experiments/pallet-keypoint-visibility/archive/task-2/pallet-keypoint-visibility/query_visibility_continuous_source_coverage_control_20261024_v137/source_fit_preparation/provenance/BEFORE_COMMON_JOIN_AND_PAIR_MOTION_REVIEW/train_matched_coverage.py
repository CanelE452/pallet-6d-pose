"""Prospective V137 source-only matched data-coverage control.

Generated checks use invented arrays only. Preparation requires all new source
producer/GT seals. Only an explicit root entry can run either fixed8000 fit.
"""
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import traceback
import numpy as np
import torch

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V / 'ROOT_CONTINUOUS_COVERAGE_POLICY_BEFORE_FIT.json'
POLICY_SHA = '0bfd6a5a30f2705a49636f3cb6b60df735e863c4a2f74e973012a48e3d236c60'
PARENT = P / 'query_visibility_native_pair_supervision_control_20261024_v135/source_fit_preparation/train_native_pairs.py'
PARENT_SHA = '8d0f5e24f7080b5de459c446677b408a1cadd83eb9de499a6a9a0b5027b46857'
V136 = P / 'query_visibility_continuous_native_source_20261024_v136'
DIAGNOSIS = V136 / 'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json'
NEW_FEATURE_SEAL = V136 / 'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json'
NEW_INIT_SEAL = V136 / 'actual_initializer_v1/PREDICTION_SEAL.json'
NEW_RENDER = V136 / 'actual_corpus_v1/RENDER_COMPLETE.json'
SOURCE_POLICY = P / 'query_visibility_native_geometry_control_20261024_v133/ROOT_MATCHED_GEOMETRY_POLICY_BEFORE_INPUTS.json'
C26 = P / 'query_visibility_native_pair_supervision_control_20261024_v135/runs/C26_C27_NATIVE_PAIR_FIXED8000/C26_NATIVE_PAIR_SAMPLER'
PROTOCOL = H / 'PROTOCOL_BEFORE_SOURCE_FIT.json'
GENERATED = H / 'GENERATED_MEANINGFUL_CHECKS.json'
RUN = V / 'runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000'
ARMS = ('C28_OLD_NATIVE_EXTRA_REPLAY_CONTROL', 'C29_CONTINUOUS_NATIVE_DATA_ADDITION')
NOLD, NATIVE, NEWN = 51004, 6144, 20480
CONVENTIONS = ('CF_PRIMARY_DECLARED', 'V4_SENSITIVITY_ONLY')
FOREIGN = ('SAME_TARGET_MATERIAL', 'CARDBOARD_TAN', 'BACKGROUND_LIKE', 'HIGH_CONTRAST')
MATERIALS = ('PROCEDURAL_WOOD', 'DARK_POLYMER')
ASPECTS = ('RECTANGULAR', 'SQUARE')
GEOMETRIES = ('BOARD_BLOCK_RUNNER', 'RIB_LEG_RUNNER')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
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
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line]


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


assert sha(PARENT) == PARENT_SHA
spec = importlib.util.spec_from_file_location('v137_pinned_V135_source_helpers', PARENT)
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
# The original V133/V135 globals remain6144. Only new-source functions below
# use NEWN20480. No original mask, gate, source-label or sampling helper changes.
m = q.m


def parse_new_GT(rr):
    assert len(rr) == 2 * NEWN
    result = {}
    for c, convention in enumerate(CONVENTIONS):
        rs = rr[c::2]
        for i, r in enumerate(rs):
            assert r['convention'] == convention and r['scene_index'] == i // 8 and r['base_index'] == i // 32 and r['native_slot'] == i % 8
            assert r['query_id'] == f'V136_SCENE_{i // 8:04d}:native{i % 8}'
            assert r['id'] == f'V136_SCENE_{i // 8:04d}' and r['base_id'] == f'V136_BASE_{i // 32:04d}' and 0 <= r['canonical_index'] < 8
            assert r['split'] == ('TRAIN' if i // 32 < 512 else 'VALIDATION')
            assert r['variant'] == ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')[(i // 8) % 4]
        state = np.array([r['teacher_state'] for r in rs]); assert np.isin(state, list('DESU')).all()
        meta = dict(state=state, target=np.where(state == 'D', 0, np.where(state == 'E', 1, -1)).astype(np.int8),
                    split=np.array([r['split'] for r in rs]), boundary=np.array([r['correspondence_boundary'] for r in rs], bool),
                    bands=np.array([r['position_band'] for r in rs]), variant=np.array([r['variant'] for r in rs]), rows=rs)
        for key in ('material', 'aspect', 'geometry_family', 'foreign_family'):
            meta[key] = np.array([r['factors'][key] for r in rs])
        result[convention] = meta
    return result


def new_groups(meta):
    out = {}
    for fold in ('ALL', 'TRAIN', 'VALIDATION'):
        f = np.ones(NEWN, bool) if fold == 'ALL' else meta['split'] == fold
        out[fold] = f
        out[fold + '::historical_area_agreement'] = f & ~meta['boundary']
        out[fold + '::historical_area_disagreement'] = f & meta['boundary']
        for band in ('LE3_PX', 'GT3_LE8_PX', 'GT8_LE20_PX', 'GT20_PX', 'NO_QUERY'):
            out[fold + '::' + band] = f & (meta['bands'] == band)
        for key in ('variant', 'material', 'aspect', 'geometry_family', 'foreign_family'):
            for value in np.unique(meta[key]):
                out[fold + '::' + key + '::' + str(value)] = f & (meta[key] == value)
        # Explicit full cross cohorts, including empty support if it occurs.
        for material in MATERIALS:
            for aspect in ASPECTS:
                out[fold + '::material_aspect::' + material + '::' + aspect] = f & (meta['material'] == material) & (meta['aspect'] == aspect)
    return out


def new_report(raw, available, meta):
    pred = np.where(available, (raw >= 0).astype(np.int8), -1)
    return {name: m.external_metric(meta['target'], pred, mask, meta['state']) for name, mask in new_groups(meta).items()}


def new_primary_groups():
    return (['VALIDATION'] + ['VALIDATION::foreign_family::' + x for x in FOREIGN]
            + ['VALIDATION::material_aspect::' + material + '::' + aspect for material in MATERIALS for aspect in ASPECTS]
            + ['VALIDATION::geometry_family::' + x for x in GEOMETRIES])


def new_gates(report, baseline):
    gates = {}
    for group in new_primary_groups():
        rr, bb = report.get(group), baseline.get(group)
        for state in ('D', 'E'):
            row = 0 if state == 'D' else 1
            support = 0 if rr is None else sum(rr['CM_D_E_rows_D_E_U_columns'][row])
            base_support = 0 if bb is None else sum(bb['CM_D_E_rows_D_E_U_columns'][row])
            value = None if rr is None else rr[state + '_recall']
            baseD = None if bb is None else bb['D_recall']
            threshold = (None if baseD is None else baseD - .01) if state == 'D' else .8
            passed = bool(support > 0 and base_support > 0 and value is not None and threshold is not None and value >= threshold)
            gates['NEW_NATIVE::' + group + '::' + state] = dict(value=value, limit=threshold, N=support,
                baseline_N=base_support, PASS=passed, missing_support=bool(support == 0 or base_support == 0))
    assert len(gates) == 22
    return gates


def make_new_pairpool(meta, available):
    pool = []
    for base in range(512):
        for slot in range(8):
            e, d = (base * 4 + 1) * 8 + slot, (base * 4 + 2) * 8 + slot
            if meta['state'][e] == 'E' and meta['state'][d] == 'D' and available[e] and available[d]:
                a, b = meta['rows'][e], meta['rows'][d]
                assert a['canonical_index'] == b['canonical_index']
                assert a['base_index'] == b['base_index'] == base and a['native_slot'] == b['native_slot'] == slot
                assert meta['split'][e] == meta['split'][d] == 'TRAIN'
                assert meta['variant'][e] == 'COVER' and meta['variant'][d] == 'NEAR_SHAM'
                # Historical projected-area disagreement is not a filter.
                pool.append((e, d))
    result = np.asarray(pool, np.int64).reshape(-1, 2)
    assert len(np.unique(result)) == 2 * len(result)
    return result


def load_new(protocol):
    seal = read(check(protocol['new_feature_seal']))
    assert seal['completed'] and seal['PASS'] and seal['GT_read'] is False and seal['all2560_scenes20480_slots_preserved']
    assert seal['actual_original_initializer'] and not seal['source_emulation'] and same(seal['initializer_seal'], protocol['new_initializer_seal'])
    with np.load(check(seal['features']), allow_pickle=False) as f:
        packet = {k: f[k].copy() for k in f.files}
    with np.load(check(seal['predictions']), allow_pickle=False) as f:
        baseline = {k: f[k].copy() for k in f.files}
    assert packet['features'].shape == (NEWN, 1928) and packet['features'].dtype == np.float32 and np.isfinite(packet['features']).all()
    assert np.array_equal(packet['global_index'], np.arange(NEWN))
    assert np.array_equal(packet['feature_constructed'], baseline['head_forwarded']) and np.array_equal(packet['query_available'], baseline['query_available'])
    assert not packet['features'][~packet['feature_constructed']].any()
    gt = parse_new_GT(lines(check(protocol['new_GT_ledger'])))
    meta = gt[CONVENTIONS[0]]
    for i, r in enumerate(meta['rows']):
        assert packet['occurrence_id'][i] == r['query_id'] and packet['scene_index'][i] == i // 8 and packet['native_slot'][i] == i % 8
        expected = r['query_xy'] if r['query_xy'] is not None else [np.nan, np.nan]
        assert np.array_equal(packet['query_xy'][i], expected, equal_nan=True)
        side = np.nan if r['roi_side'] is None else r['roi_side']
        assert (np.isnan(side) and np.isnan(packet['L'][i])) or side == packet['L'][i]
    train = (meta['split'] == 'TRAIN') & packet['query_available']
    uniform = [np.flatnonzero(train & (meta['target'] == t)) for t in (0, 1)]
    pairs = make_new_pairpool(meta, packet['query_available'])
    assert all(len(pool) > 0 for pool in uniform) and len(pairs) > 0, 'New TRAIN class/pair support missing; cannot run fixed sampler.'
    assert all((meta['split'][pool] == 'TRAIN').all() for pool in uniform)
    return dict(packet=packet, baseline=baseline, meta=gt, pools=uniform, pairs=pairs)


def saved_native(path, raw, packet, meta, size):
    np.savez_compressed(path, raw_head_logit_before_U=raw,
        query_logit=np.where(packet['query_available'], raw, 0).astype(np.float32),
        prediction=np.where(packet['query_available'], (raw >= 0).astype(np.int8), -1), query_available=packet['query_available'],
        head_forwarded=packet['feature_constructed'], global_index=np.arange(size), occurrence_id=packet['occurrence_id'],
        query_xy=packet['query_xy'], L=packet['L'], split=meta['split'], raw_teacher_state_CF=meta['state'],
        external_target_CF=meta['target'], convention_boundary=meta['boundary'])


def generated_checks():
    torch.set_num_threads(1); torch.manual_seed(1330)
    model = q.GeometryQueryHead().float(); x = torch.randn(64, 1928); zero = torch.zeros(64, 42)
    assert sum(p.numel() for p in model.parameters()) == 224513 and not model.geometry.weight.any()
    a = torch.ones(64, dtype=torch.bool); target = torch.tensor([1.] * 16 + [0.] * 32 + [1.] * 16)
    z1 = torch.zeros(64, requires_grad=True); z2 = torch.full((64,), .25, requires_grad=True)
    loss = q.native_bce(z1, target, a) + q.native_bce(z2, target, a)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z1, target) + torch.nn.functional.binary_cross_entropy_with_logits(z2, target)
    assert torch.equal(loss, expected); loss.backward(); assert z1.grad[0] == -1 / 128 and z1.grad[16] == 1 / 128
    masked = torch.zeros(4, requires_grad=True)
    q.native_bce(masked, torch.tensor([0., 1., -1., 1.]), torch.tensor([1, 1, 1, 0], dtype=torch.bool)).backward()
    assert masked.grad[:2].abs().min() > 0 and not masked.grad[2:].any()
    q.native_bce(model(x, zero), target, a).backward()
    assert not model.geometry.weight.grad.any() and model.head[0].weight.grad.abs().sum() > 0
    # Equal algorithm/RNG calls, distinct populations and occurrence IDs.
    pools = [np.c_[np.arange(125), 125 + np.arange(125)], np.c_[np.arange(193), 193 + np.arange(193)]]
    uniform = [[np.arange(125, 500), np.r_[np.arange(125), np.arange(500, 510)]],
               [np.arange(193, 700), np.r_[np.arange(193), np.arange(700, 750)]]]
    histories = []; shared = []
    for arm in range(2):
        pg = torch.Generator().manual_seed(1374); ug = torch.Generator().manual_seed(1372)
        spg = torch.Generator().manual_seed(1334); sug = torch.Generator().manual_seed(1332)
        extra = []; ss = []
        for _ in range(100):
            ni, pi = q.draw_native64(pools[arm], uniform[arm], pg, ug)
            assert np.array_equal(ni[:16], pools[arm][pi, 0]) and np.array_equal(ni[16:32], pools[arm][pi, 1])
            assert np.isin(ni[32:48], uniform[arm][0]).all() and np.isin(ni[48:], uniform[arm][1]).all()
            extra.append(ni); ss.append(q.draw_native64(pools[0], uniform[0], spg, sug)[0])
        histories.append((extra, pg.get_state(), ug.get_state())); shared.append(ss)
    assert all(np.array_equal(a, b) for a, b in zip(*shared))
    assert torch.equal(histories[0][1], histories[1][1]) and torch.equal(histories[0][2], histories[1][2])
    assert any(not np.array_equal(a, b) for a, b in zip(histories[0][0], histories[1][0]))
    # All512 source-defined TRAIN pairs are admitted despite historical area flags.
    states = np.full(NEWN, 'S', dtype='<U1'); avail = np.ones(NEWN, bool); invented_GT = []
    for i in range(NEWN):
        if i // 32 < 512 and i % 32 == 8: states[i] = 'E'
        if i // 32 < 512 and i % 32 == 16: states[i] = 'D'
        for convention in CONVENTIONS:
            invented_GT.append(dict(convention=convention, scene_index=i // 8, base_index=i // 32, native_slot=i % 8,
                id=f'V136_SCENE_{i // 8:04d}', base_id=f'V136_BASE_{i // 32:04d}', query_id=f'V136_SCENE_{i // 8:04d}:native{i % 8}',
                split='TRAIN' if i // 32 < 512 else 'VALIDATION', variant=('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')[(i // 8) % 4],
                canonical_index=i % 8, teacher_state=str(states[i]), correspondence_boundary=True, position_band='LE3_PX',
                factors=dict(material=MATERIALS[(i // 32) % 2], aspect=ASPECTS[(i // 64) % 2],
                             geometry_family=GEOMETRIES[(i // 128) % 2], foreign_family=FOREIGN[(i // 256) % 4])))
    parsed = parse_new_GT(invented_GT); meta = parsed[CONVENTIONS[0]]
    assert len(meta['rows']) == NEWN and np.array_equal(meta['state'], states) and new_groups(meta)['ALL'].sum() == NEWN
    malformed = [dict(r) for r in invented_GT]; malformed[16]['query_id'] = 'V129_WRONG_SOURCE:native0'
    try:
        parse_new_GT(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('Malformed new source identity was accepted')
    pairpool = make_new_pairpool(meta, avail); assert pairpool.shape == (512, 2)
    assert all((meta['split'][pairpool[:, k]] == 'TRAIN').all() for k in range(2))
    good = {g: dict(CM_D_E_rows_D_E_U_columns=[[99, 1, 0], [2, 8, 0]], D_recall=.99, E_recall=.8) for g in new_primary_groups()}
    gates = new_gates(good, good); assert len(gates) == 22 and all(v['PASS'] for v in gates.values())
    broken = dict(good); broken.pop(new_primary_groups()[1]); bad = new_gates(broken, good)
    assert len(bad) == 22 and sum(v['missing_support'] for v in bad.values()) == 2 and not all(v['PASS'] for v in bad.values())
    metric = m.external_metric(np.array([0, 0, 1, 1, -1, -1]), np.array([0, -1, 1, -1, 1, 0]), np.ones(6, bool), np.array(list('DDEESU')))
    assert metric['D_recall'] == metric['E_recall'] == .5 and metric['teacher_S_outside_external'] == metric['teacher_U_outside_external'] == 1
    write(GENERATED, dict(completed=True, PASS=True, code=bind(__file__), parent_code=bind(PARENT), invented_data_only=True,
        parameters224513_zero_geometry_visual_gradient=True, native_BCE64_means_separate_equal_weights_no_hinge=True,
        native_SU_unavailable_zero_gradient=True, shared_old_native_rng_exact=True, extra_equal_rng_algorithm_distinct_populations=True,
        extra_generator_consumption_exact_invented100=True, all512_TRAIN_pairs_ignore_obsolete_area_flags=True,
        generic_new40960_ledger_to20480_rows_exact_identity_and_malformed_rejection=True,
        explicit11_primary_groups22_gates_missing_support_ineligible=True, full_known_U_and_teacher_SU_denominators=True,
        original_helper_NNATIVE_unchanged6144=m.NNATIVE == NATIVE, actual_source_feature_checkpoint_RGB_model_optimizer_reads=0))
    print(json.dumps(dict(completed=True, generated=bind(GENERATED), actual_fit=False)), flush=True)


def prepare():
    assert sha(POLICY) == POLICY_SHA and sha(PARENT) == PARENT_SHA
    policy = read(POLICY); assert policy['arms'] == list(ARMS)
    # Policy explicitly requires these completions before protocol/entry creation.
    render = read(NEW_RENDER); init = read(NEW_INIT_SEAL); nf = read(NEW_FEATURE_SEAL); diag = read(DIAGNOSIS)
    assert all(x['completed'] and x['PASS'] for x in (render, init, nf, diag))
    assert init['GT_read'] is False and nf['GT_read'] is False and diag['RGB_read'] is False
    assert render['scenes'] == 2560 and render['bases'] == 640 and nf['rows'] == NEWN
    assert same(nf['initializer_seal'], bind(NEW_INIT_SEAL)) and same(diag['prediction_seal'], bind(NEW_INIT_SEAL))
    source = read(check(policy['source_binding_policy']))
    freeze = read(check(source['fixed_initialization'])); assert freeze['update'] == 4500
    ledger = next(b for b in diag['outputs'] if Path(b['path']).name == 'ALL20480_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl')
    proof = read(GENERATED); assert proof['PASS'] and same(proof['code'], bind(__file__))
    oldtrace = C26 / 'TRAIN_DRAWS_AND_LOSSES.jsonl'; oldfit = read(C26 / 'FIT_COMPLETE.json')
    assert oldfit['completed'] and oldfit['optimizer_updates'] == 8000
    p = dict(completed=True, prepared_only=True, code=bind(__file__), policy=bind(POLICY), generated=bind(GENERATED),
        parent_trainer=bind(PARENT), source_binding_policy=policy['source_binding_policy'], source_helper=bind(q.SOURCE_HELPER),
        historical_C24_fit=bind(q.C24_FIT), historical_C24_trace=bind(q.C24_TRACE), historical_C26_fit=bind(C26 / 'FIT_COMPLETE.json'), historical_C26_trace=bind(oldtrace),
        pair_coverage=bind(q.PAIR_COVERAGE), C16_freeze=source['fixed_initialization'], C16_checkpoint=freeze['checkpoint'],
        new_data_policy=policy['new_data_policy'], new_render=bind(NEW_RENDER), new_initializer_seal=bind(NEW_INIT_SEAL),
        new_feature_seal=bind(NEW_FEATURE_SEAL), new_diagnosis=bind(DIAGNOSIS), new_GT_ledger=ledger,
        arms=list(ARMS), parameters_each=224513, geometry_all_zero=True, actual_G42_packet_read=False,
        old_occurrences=NOLD, old_native_occurrences=NATIVE, new_native_occurrences=NEWN,
        old192_and_native64_exact_historical_C26=True, extra_pair_seed=1374, extra_uniform_seed=1372,
        source_selection='old42 plus new22, min of original27+new22 recalls; control tie; no eligible => null',
        fixed_updates=8000, VAL_checkpoint_choice=False, threshold_logit=0, runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',
            torch='2.1.1+cu118', numpy='1.26.4', CUDA=True, FP32=True, TF32=False), run_directory=str(RUN), actual_fit=False)
    new = load_new(p); primary = new['meta'][CONVENTIONS[0]]
    p['new_TRAIN_support'] = dict(pair_count=len(new['pairs']), uniform_D=len(new['pools'][0]), uniform_E=len(new['pools'][1]),
                                  historical_area_flags_not_a_filter=True, no_qerror_or_score_or_VAL_filter=True)
    baseline_new = new_report(new['baseline']['raw_head_logit_before_U'], new['packet']['query_available'], primary)
    p['new_frozen_C16_primary_baseline'] = {g: baseline_new[g] for g in new_primary_groups()}
    write(PROTOCOL, p)
    template = dict(approved_source_fit=False, current223=False, V112=False, PnP=False, q_correction=False, automatic_retry=False,
        code=p['code'], protocol=bind(PROTOCOL), policy=p['policy'], generated=p['generated'])
    write(H / 'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json', template)
    print(json.dumps(dict(completed=True, code=bind(__file__), protocol=bind(PROTOCOL), template=bind(H / 'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'), actual_fit=False)), flush=True)


def run(entry_path):
    e = read(entry_path); assert e['approved_source_fit'] is True
    assert all(e[k] is False for k in ('current223', 'V112', 'PnP', 'q_correction', 'automatic_retry'))
    assert sha(POLICY) == POLICY_SHA and same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL))
    p = read(PROTOCOL)
    for key in ('code', 'policy', 'generated'):
        assert same(e[key], p[key])
    for key in ('policy', 'generated', 'parent_trainer', 'source_binding_policy', 'source_helper', 'historical_C24_fit', 'historical_C24_trace',
                'historical_C26_fit', 'historical_C26_trace', 'pair_coverage', 'C16_freeze', 'C16_checkpoint', 'new_data_policy', 'new_render',
                'new_initializer_seal', 'new_feature_seal', 'new_diagnosis', 'new_GT_ledger'):
        check(p[key])
    assert read(GENERATED)['PASS'] and same(read(GENERATED)['code'], e['code']) and not RUN.exists()
    source = read(p['source_binding_policy']['path']); data = q.load_source_data_zero_geometry(source)
    oldmeta = data['nmeta'][CONVENTIONS[0]]; oldpairs = q.make_pairpool(oldmeta, data['npack']['query_available']); q.assert_pair_coverage(oldpairs)
    new = load_new(p); newmeta = new['meta'][CONVENTIONS[0]]
    baseline_old = q.old_reports(data, data['baseline']['query_logit']); baseline_native = q.native_report(data['nbaseline']['raw_head_logit_before_U'], data['npack']['query_available'], oldmeta)
    baseline_new = new_report(new['baseline']['raw_head_logit_before_U'], new['packet']['query_available'], newmeta)
    assert m.NNATIVE == NATIVE and len(oldpairs) == 125
    assert torch.__version__ == p['runtime']['torch'] and np.__version__ == p['runtime']['numpy'] and torch.cuda.is_available()
    assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    xo = torch.as_tensor(data['op']['features'], device='cuda'); xf = torch.as_tensor(data['npk']['features'], device='cuda'); xt = torch.cat((xo, xf))
    units = torch.as_tensor(data['units'], device='cuda', dtype=torch.long)
    oldtarget = torch.as_tensor(data['oldmeta']['external_target'], device='cuda', dtype=torch.float32); oldav = torch.as_tensor(data['available'], device='cuda')
    xn = torch.as_tensor(data['npack']['features'], device='cuda'); xx = torch.as_tensor(new['packet']['features'], device='cuda')
    nt = torch.as_tensor(oldmeta['target'], device='cuda', dtype=torch.float32); na = torch.as_tensor(data['npack']['query_available'], device='cuda')
    et = torch.as_tensor(newmeta['target'], device='cuda', dtype=torch.float32); ea = torch.as_tensor(new['packet']['query_available'], device='cuda')
    zeros = torch.zeros((max(len(xo), len(xf), 512), 42), device='cuda')
    ni_all = np.flatnonzero(data['npack']['feature_constructed']); ei_all = np.flatnonzero(new['packet']['feature_constructed'])
    RUN.mkdir(parents=True); write(RUN / 'STARTED.json', dict(entry=bind(entry_path), protocol=bind(PROTOCOL), actual_updates=0,
        retained_old51004_native6144_new20480=True, old_pair_count125=len(oldpairs), new_pair_count=len(new['pairs']), actual_geometry_input_used=False))
    write(RUN / 'BASELINE_REPORT_BEFORE_FIT.json', dict(old=baseline_old, native_CF=baseline_native, new_native_CF=baseline_new))
    write(RUN / 'NEW_NATIVE_TRAIN_POOL_BEFORE_FIT.json', dict(completed=True, pairs=new['pairs'].tolist(), uniform_D=new['pools'][0].tolist(),
        uniform_E=new['pools'][1].tolist(), only_TRAIN_CF_state_canonical_availability=True, historical_area_qerror_modelscore_VAL_not_filters=True))
    started = time.monotonic(); arms = {}; shared_reference = None; extra_rng_reference = None
    try:
        for arm in ARMS:
            directory = RUN / arm; directory.mkdir(); torch.manual_seed(1330); torch.cuda.manual_seed_all(1330)
            model = q.GeometryQueryHead().cuda().float(); payload = torch.load(p['C16_checkpoint']['path'], map_location='cpu', weights_only=True)
            assert payload['update'] == 4500
            missing, unexpected = model.load_state_dict(payload['state_dict'], strict=False)
            assert missing == ['geometry.weight'] and not unexpected and not model.geometry.weight.any()
            assert all(torch.equal(v.cpu(), payload['state_dict'][k]) for k, v in model.state_dict().items() if k != 'geometry.weight'); del payload
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.001)
            gens = [torch.Generator().manual_seed(s) for s in (208, 214, 215, 216, 217)]; flatgen = torch.Generator().manual_seed(218)
            pg = torch.Generator().manual_seed(1334); ug = torch.Generator().manual_seed(1332)
            epg = torch.Generator().manual_seed(1374); eug = torch.Generator().manual_seed(1372)
            extra_pool, extra_uniform = (oldpairs, data['native_pools']) if arm == ARMS[0] else (new['pairs'], new['pools'])
            ex, extarget, exav = (xn, nt, na) if arm == ARMS[0] else (xx, et, ea)
            shared_hash = hashlib.sha256(); extra_hash = hashlib.sha256()
            exposure_old = np.zeros(NOLD, np.int64); exposure_native = np.zeros(NATIVE, np.int64); exposure_new = np.zeros(NEWN, np.int64)

            def checkpoint(step):
                return dict(state_dict=q.clone_cpu(model.state_dict()), optimizer_state=q.clone_cpu(optimizer.state_dict()), update=step,
                    C16_parent_update=4500, arm=arm, threshold_logit=0, policy=bind(POLICY), trainer=bind(__file__), RNG=dict(
                    old=[g.get_state() for g in gens], flat=flatgen.get_state(), native_pair=pg.get_state(), native_uniform=ug.get_state(),
                    extra_pair=epg.get_state(), extra_uniform=eug.get_state(), torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all()))

            @torch.no_grad()
            def predict():
                model.eval(); rawunits = torch.cat((model(xo, zeros[:len(xo)]), model(xf, zeros[:len(xf)])))
                oldz = np.where(data['available'], rawunits[units].cpu().numpy(), 0).astype(np.float32)
                outputs = []
                for tensor, constructed, size in ((xn, ni_all, NATIVE), (xx, ei_all, NEWN)):
                    raw = np.full(size, np.nan, np.float32)
                    for lo in range(0, len(constructed), 512):
                        indices = constructed[lo:lo + 512]; ix = torch.as_tensor(indices, device='cuda', dtype=torch.long)
                        raw[indices] = model(tensor[ix], zeros[:len(ix)]).cpu().numpy()
                    outputs.append(raw)
                return oldz, outputs[0], outputs[1]

            @torch.no_grad()
            def evaluate(step):
                oldz, raw, eraw = predict()
                _, parts = data['h'].selector14(data['c12'], oldz, data['available'], data['meta'], data['new'], data['d'])
                nr = q.native_report(raw, data['npack']['query_available'], oldmeta)
                en = new_report(eraw, new['packet']['query_available'], newmeta)
                with (directory / 'HISTORY.jsonl').open('a') as f:
                    f.write(json.dumps(dict(update=step, old14_BCE_parts=parts, old14=q.recall14(oldz, data['available'], data['oldmeta']),
                        native_TRAIN=nr['TRAIN'], native_VAL=nr['VALIDATION'], new_native_TRAIN=en['TRAIN'], new_native_VAL=en['VALIDATION']), allow_nan=False) + '\n')
                return oldz, raw, eraw

            old0, native0, new0 = evaluate(0)
            parity = dict(completed=True, all51004_old_scores_bitexact=np.array_equal(old0, data['baseline']['query_logit']),
                all6144_native_raw_scores_bitexact=np.array_equal(native0, data['nbaseline']['raw_head_logit_before_U'], equal_nan=True),
                all20480_new_raw_scores_bitexact=np.array_equal(new0, new['baseline']['raw_head_logit_before_U'], equal_nan=True),
                original_constructed_compact512_batches_and_last_lengths_exact=True, available_only_compaction=False, geometry_weights_zero=True, actual_updates=0)
            parity['PASS'] = bool(parity['all51004_old_scores_bitexact'] and parity['all6144_native_raw_scores_bitexact'] and parity['all20480_new_raw_scores_bitexact'])
            # Convert NumPy booleans before saving any replay receipt.
            parity = {k: bool(v) if isinstance(v, np.bool_) else v for k, v in parity.items()}
            write(directory / 'UPDATE0_C16_EXACT_PARITY.json', parity); assert parity['PASS'], 'Initialization parity failure; preserve before any fit.'
            torch.save(checkpoint(0), directory / 'INITIAL_CHECKPOINT.pt')
            with (directory / 'TRAIN_DRAWS_AND_LOSSES.jsonl').open('x') as trace, Path(p['historical_C26_trace']['path']).open() as previous, Path(p['historical_C24_trace']['path']).open() as c24:
                for step in range(1, 8001):
                    ix, pairs, audit, *_ = data['h'].draw192(data['c12'], data['c10'], data['pools'], data['mp'], data['wp'], data['fp'], gens, flatgen)
                    ni, npi = q.draw_native64(oldpairs, data['native_pools'], pg, ug)
                    ei, epi = q.draw_native64(extra_pool, extra_uniform, epg, eug)
                    prior = json.loads(next(previous)); original = json.loads(next(c24))
                    assert prior['update'] == original['update'] == step
                    for history in (prior, original):
                        assert np.array_equal(ix, history['old_occurrences']) and np.array_equal(pairs, history['old_matched_cover_sham_positions'])
                    assert np.array_equal(ni, prior['native_occurrences']) and np.array_equal(npi, prior['native_pair_pool_positions'])
                    assert (data['oldmeta']['split'][ix] == 'TRAIN').all() and data['available'][ix].all()
                    assert (oldmeta['split'][ni] == 'TRAIN').all() and data['npack']['query_available'][ni].all()
                    assert np.array_equal(oldmeta['target'][ni], np.r_[np.ones(16, np.int8), np.zeros(32, np.int8), np.ones(16, np.int8)])
                    extra_meta = oldmeta if arm == ARMS[0] else newmeta
                    assert (extra_meta['split'][ei] == 'TRAIN').all()
                    extra_available = data['npack']['query_available'] if arm == ARMS[0] else new['packet']['query_available']
                    assert extra_available[ei].all()
                    assert np.array_equal(extra_meta['target'][ei], np.r_[np.ones(16, np.int8), np.zeros(32, np.int8), np.ones(16, np.int8)])
                    for value in (ix, pairs, ni): shared_hash.update(value.astype('<i8').tobytes())
                    extra_hash.update(ei.astype('<i8').tobytes()); extra_hash.update(epi.astype('<i8').tobytes())
                    np.add.at(exposure_old, ix, 1); np.add.at(exposure_native, ni, 1)
                    np.add.at(exposure_native if arm == ARMS[0] else exposure_new, ei, 1)
                    oi = torch.as_tensor(ix, device='cuda', dtype=torch.long); nn = torch.as_tensor(ni, device='cuda', dtype=torch.long)
                    ee = torch.as_tensor(ei, device='cuda', dtype=torch.long); pp = torch.as_tensor(pairs, device='cuda', dtype=torch.long)
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    zo = model(xt[units[oi]], zeros[:192]); zn = model(xn[nn], zeros[:64]); ze = model(ex[ee], zeros[:64])
                    oldloss, bce, hinge, flatbce = data['h'].loss192(zo, oldtarget[oi], oldav[oi], pp)
                    nl = q.native_bce(zn, nt[nn], na[nn]); el = q.native_bce(ze, extarget[ee], exav[ee]); loss = oldloss + nl + el
                    assert torch.isfinite(loss); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5); assert torch.isfinite(grad); optimizer.step()
                    assert not model.geometry.weight.any()
                    trace.write(json.dumps(dict(update=step, old_occurrences=ix.tolist(), old_matched_cover_sham_positions=pairs.tolist(),
                        native_occurrences=ni.tolist(), native_pair_pool_positions=npi.tolist(), extra_native_occurrences=ei.tolist(),
                        extra_pair_pool_positions=epi.tolist(), extra_source='OLD_NATIVE' if arm == ARMS[0] else 'V136_NATIVE',
                        loss=float(loss.detach()), old_BCE=float(bce.detach()), old_hinge=float(hinge.detach()), flat_BCE=float(flatbce.detach()),
                        native_BCE=float(nl.detach()), extra_native_BCE=float(el.detach()), native_hinge_weight=0,
                        gradient_before_clip=float(grad)), allow_nan=False) + '\n')
                    if step % 100 == 0: evaluate(step)
                    if step % 1000 == 0: print(json.dumps(dict(arm=arm, updates=step, seconds=time.monotonic() - started)), flush=True)
                assert next(previous, None) is None and next(c24, None) is None
            shared = shared_hash.hexdigest(); rng_hash = hashlib.sha256(epg.get_state().numpy().tobytes() + eug.get_state().numpy().tobytes()).hexdigest()
            if shared_reference is None: shared_reference, extra_rng_reference = shared, rng_hash
            else: assert shared == shared_reference and rng_hash == extra_rng_reference
            torch.save(checkpoint(8000), directory / 'FINAL_RESUME_CHECKPOINT.pt'); oldz, raw, eraw = predict()
            report = q.old_reports(data, oldz); nr = q.native_report(raw, data['npack']['query_available'], oldmeta)
            en = new_report(eraw, new['packet']['query_available'], newmeta)
            gates = q.make_gates(report, nr, baseline_old['old14'], baseline_native); gates.update(new_gates(en, baseline_new)); assert len(gates) == 64
            rank = [v['value'] for k, v in gates.items() if not k.startswith('HISTORICAL15::')]; assert len(rank) == 49
            worst = None if any(x is None for x in rank) else min(rank)
            eligible = all(g['PASS'] for g in gates.values())
            np.savez_compressed(directory / 'FINAL_OLD51004_SOURCE_SCORES.npz', query_logit=oldz,
                prediction=np.where(data['available'], (oldz >= 0).astype(np.int8), -1), query_available=data['available'],
                global_occurrence_index=np.arange(NOLD), input_unit_index=data['units'], external_target=data['oldmeta']['external_target'],
                raw_teacher_state=data['oldmeta']['typed_state'], split=data['oldmeta']['split'], domain=data['oldmeta']['domain'], source=data['oldmeta']['source'], variant=data['oldmeta']['variant'])
            saved_native(directory / 'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz', raw, data['npack'], oldmeta, NATIVE)
            saved_native(directory / 'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz', eraw, new['packet'], newmeta, NEWN)
            np.savez_compressed(directory / 'SAMPLING_EXPOSURE.npz', old51004=exposure_old, native6144=exposure_native, new_native20480=exposure_new)
            write(directory / 'SOURCE_RESULTS.json', dict(completed=True, arm=arm, selected_update=8000, VAL_checkpoint_choice=False, threshold_logit=0,
                old=report, native_CF=nr, native_V4_sensitivity_only=q.native_report(raw, data['npack']['query_available'], data['nmeta'][CONVENTIONS[1]]),
                new_native_CF=en, new_native_obsolete_area_V4_sensitivity_only=new_report(eraw, new['packet']['query_available'], new['meta'][CONVENTIONS[1]]),
                gates=gates, source_eligible=eligible, worst_required49_class_recall=worst, shared_old192_native64_sha256=shared,
                extra_native_draw_sha256=extra_hash.hexdigest(), extra_rng_consumption_sha256=rng_hash,
                old_pair_count=125, new_pair_count=len(new['pairs']), historical_old192_and_native64_all8000_exact=True,
                all_old51004_native6144_new20480_preserved=True, teacher_SU_no_external_gradient=True, actual_geometry_input_used=False,
                semantic_target='Registered native physical corner visibility, separate from actual query coordinate error; no error filtering.', current_V112_PnP=0))
            write(directory / 'MODEL_FREEZE.json', dict(completed=True, arm=arm, update=8000, selected_update=8000,
                checkpoint=bind(directory / 'FINAL_RESUME_CHECKPOINT.pt'), threshold_logit=0, parameters=224513, source_eligible=eligible,
                policy=bind(POLICY), trainer=bind(__file__), C16_freeze=p['C16_freeze'], all_geometry_input_zero=True,
                encoder_trainable_parameters=0, current_allowed=False, V112_allowed=False, PnP=0))
            write(directory / 'FIT_COMPLETE.json', dict(completed=True, optimizer_updates=8000, arm=arm,
                results=bind(directory / 'SOURCE_RESULTS.json'), freeze=bind(directory / 'MODEL_FREEZE.json'),
                initialization_parity=bind(directory / 'UPDATE0_C16_EXACT_PARITY.json'), shared_draw_sha256=shared,
                source_eligible=eligible, worst_required49_class_recall=worst))
            arms[arm] = dict(source_eligible=eligible, worst_required49_class_recall=worst,
                fit=bind(directory / 'FIT_COMPLETE.json'), freeze=bind(directory / 'MODEL_FREEZE.json'))
            del model, optimizer; torch.cuda.empty_cache()
        eligible = [a for a in ARMS if arms[a]['source_eligible']]
        selected = max(eligible, key=lambda a: (arms[a]['worst_required49_class_recall'], -ARMS.index(a))) if eligible else None
        write(RUN / 'CONTROL_COMPLETE.json', dict(completed=True, PASS=True, arms=arms, selected_arm_source_only=selected,
            same_all8000_old192_native64_draws=True, shared_draw_sha256=shared_reference, extra_same_algorithm_rng_consumption=True,
            extra_rng_consumption_sha256=extra_rng_reference, extra_occurrence_ids_not_claimed_equal=True, final_update8000_both=True,
            old42_and_new22_required_gates=True, original27_plus_new22_rank=True, application_solved=False,
            private_current_V112_PnP_q_correction_calls=0, seconds=time.monotonic() - started, code=bind(__file__), policy=bind(POLICY), protocol=bind(PROTOCOL)))
        print(json.dumps(read(RUN / 'CONTROL_COMPLETE.json')), flush=True)
    except BaseException:
        write(RUN / 'FAILED_EXECUTION.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--generated', action='store_true'); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    if args.generated and not args.prepare and args.entry is None: generated_checks()
    elif args.prepare and not args.generated and args.entry is None: prepare()
    elif args.entry and not args.generated and not args.prepare: run(args.entry)
    else: raise SystemExit('Use --generated, --prepare after all new seals, or root --entry')
