"""V142 single C32 matched AUX-gradient control; inert except --entry.

The entire pinned C29 run is preserved by count-checked source edits. Preparation
reads code and existing JSON metadata only. No feature, GT ledger, source score,
RGB or checkpoint body is opened until the separately reviewed root entry.
"""
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import ast
import copy
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
POLICY = V / 'ROOT_NATIVE_GRADIENT_CONTROL_POLICY_BEFORE_FIT.json'
POLICY_SHA = '8364a66eba156bbc01d5789fb790471ce68702dfc02b687060a7dbad9a27520d'
PARENT = P / 'query_visibility_continuous_source_coverage_control_20261024_v137/source_fit_preparation/train_matched_coverage.py'
PARENT_SHA = '9751b158c14cf0e0ba1cc2b8cb41e761f798c6ff8fe6b6dcdd4d69d2b5f9d2b5'
REPLAY_CODE = P / 'query_visibility_broad_source_geometry_control_20261024_v139/source_fit_preparation/train_broad_geometry.py'
REPLAY_CODE_SHA = 'ef297c1c4b9cd90abc7d077092e7ba22c6a715ab398f35befc7af4de1764f690'
PARENT_PROTOCOL = PARENT.parent / 'PROTOCOL_BEFORE_SOURCE_FIT.json'
REPLAY_PROTOCOL = REPLAY_CODE.parent / 'PROTOCOL_BEFORE_SOURCE_FIT.json'
C30_REPLAY = REPLAY_CODE.parent.parent / 'runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000/C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL/C30_EXACT_C29_REPLAY.json'
FIXED_CASES = REPLAY_CODE.parent.parent / 'saved_result_preparation/FIXED_CASES_BEFORE_NEW_SCORES.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_SOURCE_FIT.json'
GENERATED = H / 'GENERATED_MEANINGFUL_CHECKS.json'
AUDIT = V / 'EXISTING_METHOD_AND_TARGET_ONTOLOGY_AUDIT.json'
RUN = V / 'runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000'
ARMS = ('C32_AUX_TO_NATIVE_GRAD0_FIXED8000',)
INTERPRETATION_LIMITS = [
    'C32 preserves the C16 update4500 initialization already trained on mixed AUX sources; this is matched fine-tuning, not native-only training from scratch.',
    'The intervention estimates the net effect of removing AUX-to-native gradients. It cannot establish label conflict as the unique failure cause or isolate a single causal mechanism.',
    'Native BCE absolute weights and 8000 draws stay fixed. Changes in total-gradient direction, magnitude, global clip scaling and subsequent optimizer trajectory are part of this removal effect; no extra clipping correction or compensating weight is introduced.',
    'All original64 gates and split AUX29/native35 reports remain unchanged. This diagnostic phase selects no model, runs no private evaluation and does not establish real-world success.'
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(PARENT) == PARENT_SHA
spec = importlib.util.spec_from_file_location('v142_pinned_V137', PARENT)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
bind, check, same, read, write = h.bind, h.check, h.same, h.read, h.write
q, m = h.q, h.m


def equal_array(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')


def qualification_split(gates):
    """No qualification changed or waived; native-only is additional reporting."""
    aux = {k: v for k, v in gates.items() if k.startswith(('OLD14::', 'HISTORICAL15::'))}
    native = {k: v for k, v in gates.items() if k not in aux}
    assert len(gates) == 64 and len(aux) == 29 and len(native) == 35
    def summary(group):
        return dict(passed=sum(bool(v['PASS']) for v in group.values()), required=len(group),
                    all_pass=all(v['PASS'] for v in group.values()),
                    failed=[k for k, v in group.items() if not v['PASS']])
    return dict(all64=summary(gates), AUX29=summary(aux), registered_native35=summary(native),
                original64_unchanged=True, retrospective_qualification_changed=False,
                selected_model=None, private_evaluation_authorized=False)


def assert_draw_replay(previous, step, ix, pairs, ni, npi, ei, epi):
    assert previous['update'] == step and previous['extra_source'] == 'V136_NATIVE'
    for a, key in ((ix, 'old_occurrences'), (pairs, 'old_matched_cover_sham_positions'),
                   (ni, 'native_occurrences'), (npi, 'native_pair_pool_positions'),
                   (ei, 'extra_native_occurrences'), (epi, 'extra_pair_pool_positions')):
        assert np.array_equal(a, previous[key]), key


def validate_extra_bindings(p, entry):
    for key in ('effective_run', 'parent_run_diff', 'audit'):
        assert same(entry[key], p[key]); check(p[key])
    for key in ('parent_V137_code', 'parent_V137_protocol', 'prior_C30_replay_code', 'prior_C30_replay_protocol',
                'prior_C30_exact_C29_replay', 'historical_C29_fit', 'historical_C29_results', 'historical_C29_trace', 'fixed_cases'):
        check(p[key])
    assert same(p['parent_V137_code'], bind(PARENT)) and sha(REPLAY_CODE) == REPLAY_CODE_SHA
    assert p['arms'] == list(ARMS) and p['run_directory'] == str(RUN)
    assert p['parameters_each'] == 224513 and p['fixed_updates'] == 8000 and p['AUX_old192_coefficient'] == 0
    assert p['geometry_all_zero'] is True and p['actual_G42_packet_read'] is False
    assert p['model_selection'] == 'NONE_DIAGNOSTIC_ONLY' and p['new_baseline_fits'] == 0
    assert p['interpretation_limits'] == INTERPRETATION_LIMITS
    replay = read(p['prior_C30_exact_C29_replay']['path'])
    assert replay['completed'] and replay['PASS'] and replay['all8000_draws_losses_trace_bitexact']
    assert all(replay['final_state_optimizer_RNG_exact'].values())
    assert read(p['fixed_cases']['path'])['no_future_candidate_selection'] is True
    assert m.NNATIVE == h.NATIVE == 6144 and h.NEWN == 20480


def fixed_case_scores(raw, packet, meta, p):
    """Use previously fixed IDs. Read no RGB or candidate-selected cases."""
    registry = read(p['fixed_cases']['path'])
    def case(i):
        r = meta['rows'][i]; available = bool(packet['query_available'][i])
        score = float(raw[i]) if np.isfinite(raw[i]) else None
        return dict(global_index=i, base_index=r['base_index'], native_slot=r['native_slot'], variant=r['variant'],
                    teacher=r['teacher_state'], position_band=r['position_band'], q_error_px=r['error_px'],
                    query_xy=r['query_xy'], L=float(packet['L'][i]) if np.isfinite(packet['L'][i]) else None,
                    available=available, head_forwarded=bool(packet['feature_constructed'][i]),
                    raw_hidden_logit_before_U=score, prediction=('E' if score >= 0 else 'D') if available else 'U')
    cases = [case(int(i)) for i in registry['fixed8_indices']]
    strips = [dict(base_index=s['base_index'], native_slot=s['native_slot'],
                   variants=[case(int(r['global_index'])) for r in s['variants']]) for s in registry['strips']]
    return dict(completed=True, registry=p['fixed_cases'], cases=cases, strips=strips,
                new_case_selection=False, RGB_read=False, GT_q_L_correction=False, prediction_is_actual_original_query=True)


def adapted_run_source():
    source = inspect.getsource(h.run)
    edits = []
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new); edits.append(dict(before=old, after=new))
    replace('p = read(PROTOCOL)', 'p = read(PROTOCOL)\n    validate_extra_bindings(p, e)')
    replace("extra_pool, extra_uniform = (oldpairs, data['native_pools']) if arm == ARMS[0] else (new['pairs'], new['pools'])", "extra_pool, extra_uniform = new['pairs'], new['pools']")
    replace('ex, extarget, exav = (xn, nt, na) if arm == ARMS[0] else (xx, et, ea)', 'ex, extarget, exav = xx, et, ea')
    replace("np.array_equal(old0, data['baseline']['query_logit'])", "equal_array(old0, data['baseline']['query_logit'])")
    replace("np.array_equal(native0, data['nbaseline']['raw_head_logit_before_U'], equal_nan=True)", "equal_array(native0, data['nbaseline']['raw_head_logit_before_U'])")
    replace("np.array_equal(new0, new['baseline']['raw_head_logit_before_U'], equal_nan=True)", "equal_array(new0, new['baseline']['raw_head_logit_before_U'])")
    replace("Path(p['historical_C24_trace']['path']).open() as c24:", "Path(p['historical_C24_trace']['path']).open() as c24, Path(p['historical_C29_trace']['path']).open() as c29:")
    replace('prior = json.loads(next(previous)); original = json.loads(next(c24))', 'prior = json.loads(next(previous)); original = json.loads(next(c24))\n                    historical = json.loads(next(c29)); assert_draw_replay(historical, step, ix, pairs, ni, npi, ei, epi)')
    replace('extra_meta = oldmeta if arm == ARMS[0] else newmeta', 'extra_meta = newmeta')
    replace("extra_available = data['npack']['query_available'] if arm == ARMS[0] else new['packet']['query_available']", "extra_available = new['packet']['query_available']")
    replace('np.add.at(exposure_native if arm == ARMS[0] else exposure_new, ei, 1)', 'np.add.at(exposure_new, ei, 1)')
    replace('nl = q.native_bce(zn, nt[nn], na[nn]); el = q.native_bce(ze, extarget[ee], exav[ee]); loss = oldloss + nl + el',
            'nl = q.native_bce(zn, nt[nn], na[nn]); el = q.native_bce(ze, extarget[ee], exav[ee]); loss = oldloss * 0. + nl + el\n                    zo.retain_grad()')
    replace('assert torch.isfinite(loss); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5); assert torch.isfinite(grad); optimizer.step()',
            'assert torch.isfinite(loss); loss.backward()\n                    assert zo.grad is not None and torch.count_nonzero(zo.grad).item() == 0\n                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5); assert torch.isfinite(grad); optimizer.step()')
    replace("extra_source='OLD_NATIVE' if arm == ARMS[0] else 'V136_NATIVE',", "extra_source='V136_NATIVE', AUX_old192_loss_coefficient=0, AUX_old192_output_gradient_nonzero=0, AUX_old192_full_loss_value=float(oldloss.detach()),")
    replace('assert next(previous, None) is None and next(c24, None) is None', 'assert next(previous, None) is None and next(c24, None) is None and next(c29, None) is None')
    replace('shared = shared_hash.hexdigest(); rng_hash =', "assert shared_hash.hexdigest() == p['historical_C29_draw_hash'] and extra_hash.hexdigest() == p['historical_C29_extra_draw_hash']\n            shared = shared_hash.hexdigest(); rng_hash =")
    replace('eligible = all(g[\'PASS\'] for g in gates.values())', "eligible = all(g['PASS'] for g in gates.values())\n            split_report = qualification_split(gates)\n            write(directory / 'QUALIFICATION_SPLIT.json', split_report)\n            write(directory / 'FIXED8_CASES_AND_TWO_STRIPS.json', fixed_case_scores(eraw, new['packet'], newmeta, p))")
    replace('gates=gates, source_eligible=eligible, worst_required49_class_recall=worst,', "gates=gates, qualification_split=split_report, AUX_old192_loss_coefficient=0, selected_model=None, diagnostic_only=True, interpretation_limits=p['interpretation_limits'], source_eligible=eligible, worst_required49_class_recall=worst,")
    replace("semantic_target='Registered native physical corner visibility, separate from actual query coordinate error; no error filtering.'", "semantic_target='Declared CF registered native reference visibility; constructed anchors coincide with support mesh, separate from actual query coordinate error; no error filtering.'")
    replace('encoder_trainable_parameters=0, current_allowed=False,', 'encoder_trainable_parameters=0, selected_model=None, diagnostic_only=True, AUX_old192_loss_coefficient=0, current_allowed=False,')
    replace("source_eligible=eligible, worst_required49_class_recall=worst))\n            arms[arm]", "source_eligible=eligible, worst_required49_class_recall=worst, qualification_split=bind(directory / 'QUALIFICATION_SPLIT.json'), fixed_cases=bind(directory / 'FIXED8_CASES_AND_TWO_STRIPS.json'), selected_model=None, diagnostic_only=True))\n            arms[arm]")
    replace("selected = max(eligible, key=lambda a: (arms[a]['worst_required49_class_recall'], -ARMS.index(a))) if eligible else None", 'selected = None  # Root policy: no model admission/selection in this diagnostic phase.')
    replace('extra_occurrence_ids_not_claimed_equal=True, final_update8000_both=True,', 'extra_occurrence_ids_all8000_exact_C29=True, final_update8000_single_arm=True, new_baseline_fits=0, AUX_old192_loss_coefficient=0, model_selection_prohibited=True, selected_model=None,')
    ast.parse(source)
    return source, edits


EFFECTIVE_SOURCE, EDITS = adapted_run_source()


def run(entry):
    namespace = dict(h.__dict__)
    namespace.update(__file__=str(Path(__file__).resolve()), POLICY=POLICY, POLICY_SHA=POLICY_SHA,
                     PROTOCOL=PROTOCOL, GENERATED=GENERATED, RUN=RUN, ARMS=ARMS,
                     validate_extra_bindings=validate_extra_bindings, equal_array=equal_array,
                     assert_draw_replay=assert_draw_replay, qualification_split=qualification_split,
                     fixed_case_scores=fixed_case_scores)
    exec(compile(EFFECTIVE_SOURCE, str(H / 'EFFECTIVE_V142_RUN.py'), 'exec'), namespace)
    namespace['run'](entry)


def generated_checks():
    torch.set_num_threads(1); torch.manual_seed(1330)
    a = q.GeometryQueryHead().float(); b = copy.deepcopy(a)
    assert sum(p.numel() for p in a.parameters()) == 224513 and not a.geometry.weight.any()
    xo, xn, xe = torch.randn(192, 1928), torch.randn(64, 1928), torch.randn(64, 1928)
    zero = torch.zeros(192, 42)
    yo = torch.tensor([1., 0.] * 96); yn = torch.tensor([1.]*16 + [0.]*32 + [1.]*16)
    ye = yn.clone(); ao = torch.ones(192, dtype=torch.bool); an = torch.ones(64, dtype=torch.bool); ae = an.clone()
    yo[-1] = -1.; ao[-2] = False; yn[-1] = -1.; an[-2] = False; ye[-3] = -1.; ae[-4] = False
    pairs = torch.arange(48).reshape(24, 2)
    # Actual old loss operator on invented data, with meaningful nonzero gradient.
    z0 = a(xo, zero); oldloss, *_ = h.q.m.old_helpers().loss192(z0, yo, ao, pairs)
    assert torch.isfinite(oldloss)
    oldgrad = torch.autograd.grad(oldloss, tuple(a.parameters()), retain_graph=True)
    assert any(torch.count_nonzero(g) for g in oldgrad)
    aux0grad = torch.autograd.grad(oldloss * 0., tuple(a.parameters()))
    assert all(torch.count_nonzero(g) == 0 for g in aux0grad)
    # Including entire AUX*0 is bit-exact to native-only through grads, clip, AdamW.
    oa = torch.optim.AdamW(a.parameters(), lr=1e-4, weight_decay=.001)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-4, weight_decay=.001)
    zo = a(xo, zero); zo.retain_grad()
    zl1, zl2 = a(xn, zero[:64]), a(xe, zero[:64]); zl1.retain_grad(); zl2.retain_grad()
    oldloss, *_ = h.q.m.old_helpers().loss192(zo, yo, ao, pairs)
    nl, el = q.native_bce(zl1, yn, an), q.native_bce(zl2, ye, ae)
    total = oldloss * 0. + nl + el
    native_only = q.native_bce(b(xn, zero[:64]), yn, an) + q.native_bce(b(xe, zero[:64]), ye, ae)
    assert torch.equal(total, native_only)
    total.backward(); native_only.backward()
    assert zo.grad is not None and not torch.count_nonzero(zo.grad)
    assert not zl1.grad[-2:].any() and zl1.grad[:-2].abs().sum() > 0
    assert zl2.grad[-3] == 0 and zl2.grad[-4] == 0 and zl2.grad[:60].abs().sum() > 0
    assert all(torch.equal(x.grad, y.grad) for x, y in zip(a.parameters(), b.parameters()))
    ca = torch.nn.utils.clip_grad_norm_(a.parameters(), 5); cb = torch.nn.utils.clip_grad_norm_(b.parameters(), 5)
    assert torch.equal(ca, cb)
    oa.step(); ob.step()
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())) and not a.geometry.weight.any()
    # Original native sampler seeds/pools/position order unchanged; no actual rows.
    oldpairs = np.c_[np.arange(125)*2, np.arange(125)*2+1]
    newpairs = np.c_[np.arange(473)*2+10000, np.arange(473)*2+10001]
    uniform = [np.arange(1500)+20000, np.arange(120)+30000]
    def plans():
        pg, ug = torch.Generator().manual_seed(1334), torch.Generator().manual_seed(1332)
        epg, eug = torch.Generator().manual_seed(1374), torch.Generator().manual_seed(1372)
        return [(q.draw_native64(oldpairs, uniform, pg, ug), q.draw_native64(newpairs, uniform, epg, eug)) for _ in range(100)]
    pa, pb = plans(), plans()
    assert all(np.array_equal(x, y) for aa, bb in zip(pa, pb) for a1, b1 in zip(aa, bb) for x, y in zip(a1, b1))
    # Mandatory draw guard rejects changed AUX/native/pair position, independent of loss.
    xs = (np.arange(192), np.arange(48).reshape(24, 2), np.arange(64), np.arange(16), np.arange(64)+1000, np.arange(16)+200)
    keys = ('old_occurrences','old_matched_cover_sham_positions','native_occurrences','native_pair_pool_positions','extra_native_occurrences','extra_pair_pool_positions')
    previous = dict(update=1, extra_source='V136_NATIVE', **{k: x.tolist() for k, x in zip(keys, xs)})
    assert_draw_replay(previous, 1, *xs)
    for j in range(6):
        changed = copy.deepcopy(previous); changed[keys[j]][0] = -999 if j != 1 else [-999, -999]
        try: assert_draw_replay(changed, 1, *xs)
        except AssertionError: pass
        else: raise AssertionError('Changed draw accepted')
    gates = {**{'OLD14::'+str(i):dict(PASS=True) for i in range(14)}, **{'HISTORICAL15::'+str(i):dict(PASS=True) for i in range(15)}, **{'NATIVE::'+str(i):dict(PASS=True) for i in range(35)}}
    gates['NATIVE::0']['PASS'] = False
    summary = qualification_split(gates); assert summary['AUX29']['all_pass'] and not summary['registered_native35']['all_pass'] and not summary['all64']['all_pass']
    # Fixed gallery consumes the actual GT-only ledger schema (error_px), never
    # a guessed alternate field or a new candidate-dependent selection.
    fixture = H / 'GENERATED_FIXED_CASES_METADATA.json'
    write(fixture, dict(fixed8_indices=list(range(8)), strips=[dict(base_index=0, native_slot=0,
          variants=[dict(global_index=i) for i in (0,8,16,24)]), dict(base_index=0, native_slot=1,
          variants=[dict(global_index=i) for i in (1,9,17,25)])]))
    rows = [dict(base_index=0, native_slot=i%8, variant=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')[i//8],
                 teacher_state='D' if i//8 != 1 else 'E', position_band='LE3_PX', error_px=2., query_xy=[1.,2.]) for i in range(32)]
    packet = dict(query_available=np.ones(32,bool), feature_constructed=np.ones(32,bool), L=np.full(32,16.))
    invented_raw = np.linspace(-2,2,32,dtype=np.float32); invented_raw[0] = np.nan
    packet['query_available'][0] = packet['feature_constructed'][0] = False
    rows[0].update(position_band='NO_QUERY', error_px=None, query_xy=None); packet['L'][0] = np.nan
    casefacts = fixed_case_scores(invented_raw, packet, dict(rows=rows), dict(fixed_cases=bind(fixture)))
    assert len(casefacts['cases']) == 8 and len(casefacts['strips']) == 2
    assert casefacts['cases'][0]['prediction'] == 'U' and casefacts['cases'][0]['q_error_px'] is None
    assert casefacts['strips'][0]['variants'][2]['variant'] == 'NEAR_SHAM' and casefacts['cases'][1]['q_error_px'] == 2.
    assert equal_array(np.array([np.nan, 0.], np.float32), np.array([np.nan, 0.], np.float32)) and not equal_array(np.array([0.], np.float32), np.array([-0.], np.float32))
    ast.parse(EFFECTIVE_SOURCE)
    assert h.RUN != RUN and h.ARMS != ARMS and h.POLICY != POLICY and m.NNATIVE == 6144
    assert EFFECTIVE_SOURCE.count('for arm in ARMS:') == 1 and len(ARMS) == 1
    assert 'loss = oldloss * 0. + nl + el' in EFFECTIVE_SOURCE and 'selected = None' in EFFECTIVE_SOURCE
    write(GENERATED, dict(completed=True, PASS=True, code=bind(__file__), parent_code=bind(PARENT), invented_CPU_arrays_only=True,
        parameters224513_G42_zero=True, AUX_loss_positive_coefficient_meaningful_gradient=True,
        entire_AUX_coefficient0_zero_parameter_and_output_gradient=True, native_only_full_gradient_clip_AdamW_step_bitexact=True,
        native_old64_new64_separate_mean_weights_unchanged=True, teacher_SU_missing_zero_gradient=True,
        exact100_seeded_native_pair_plans_and_all6_draw_guard_tamper_rejections=True,
        all64_AUX29_native35_no_waiver=True, one_fit_only_model_selection_null=True,
        fixed8_two_strips_actual_ledger_schema_and_NOQUERY_nulls=True,
        explicit_parent_run_edits=len(EDITS), parent_globals_not_mutated=True,
        source_feature_GT_score_weight_RGB_bodies_or_actual_fit=0))
    print(json.dumps(dict(completed=True, generated=bind(GENERATED), actual_fit=False)), flush=True)


def prepare():
    assert sha(POLICY) == POLICY_SHA and sha(PARENT) == PARENT_SHA and sha(REPLAY_CODE) == REPLAY_CODE_SHA
    policy = read(POLICY); assert policy['arm'] == ARMS[0] and policy['model_selection'].startswith('None')
    proof = read(GENERATED); assert proof['PASS'] and same(proof['code'], bind(__file__))
    parent = read(PARENT_PROTOCOL); prior = read(REPLAY_PROTOCOL)
    assert same(parent['code'], bind(PARENT)) and same(prior['code'], bind(REPLAY_CODE))
    assert parent['fixed_updates'] == 8000 and parent['geometry_all_zero'] and parent['new_TRAIN_support']['pair_count'] == 473
    replay = read(C30_REPLAY); assert replay['completed'] and replay['PASS'] and replay['all8000_draws_losses_trace_bitexact']
    # Every non-JSON body binding below is copied from existing sealed metadata;
    # preparation neither hashes nor opens those bodies.
    p = dict(parent)
    for key in ('historical_C29_fit', 'historical_C29_results', 'historical_C29_trace', 'historical_C29_draw_hash', 'historical_C29_extra_draw_hash'):
        p[key] = prior[key]
    effective = H / 'EFFECTIVE_V142_RUN.py'; effective.write_text(EFFECTIVE_SOURCE)
    diff = H / 'EXACT_PARENT_RUN_DIFF.patch'; diff.write_text(''.join(difflib.unified_diff(inspect.getsource(h.run).splitlines(True), EFFECTIVE_SOURCE.splitlines(True), fromfile='PINNED_V137_RUN', tofile='EFFECTIVE_V142_RUN')))
    p.update(code=bind(__file__), policy=bind(POLICY), generated=bind(GENERATED), audit=bind(AUDIT),
        parent_V137_code=bind(PARENT), parent_V137_protocol=bind(PARENT_PROTOCOL), prior_C30_replay_code=bind(REPLAY_CODE),
        prior_C30_replay_protocol=bind(REPLAY_PROTOCOL), prior_C30_exact_C29_replay=bind(C30_REPLAY), fixed_cases=bind(FIXED_CASES),
        effective_run=bind(effective), parent_run_diff=bind(diff), arms=list(ARMS), run_directory=str(RUN),
        source_selection='All original64 unchanged, additional AUX29/native35 split; NO MODEL SELECTION.', model_selection='NONE_DIAGNOSTIC_ONLY',
        AUX_old192_coefficient=0, native_old64_BCE_coefficient=1, native_new64_BCE_coefficient=1, native_paired_hinge_coefficient=0,
        new_baseline_fits=0, new_public_native_proposals_in_fit=0, selected_model=None,
        interpretation_limits=INTERPRETATION_LIMITS,
        preparation_reads='CODE_AND_EXISTING_JSON_METADATA_ONLY; body hashes inherited without body read',
        source_feature_GT_score_weight_RGB_body_reads_during_preparation=0, actual_fit=False)
    write(PROTOCOL, p)
    template = dict(approved_source_fit=False, current223=False, V112=False, PnP=False, q_correction=False, automatic_retry=False,
        code=p['code'], protocol=bind(PROTOCOL), policy=p['policy'], generated=p['generated'], audit=p['audit'],
        effective_run=p['effective_run'], parent_run_diff=p['parent_run_diff'])
    write(H / 'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json', template)
    receipt = dict(completed=True, prepared_only=True, code=bind(__file__), policy=bind(POLICY), protocol=bind(PROTOCOL), generated=bind(GENERATED),
        audit=bind(AUDIT), effective_run=p['effective_run'], parent_run_diff=p['parent_run_diff'], template=bind(H / 'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'),
        run_directory=str(RUN), command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json',
        source_feature_GT_score_weight_RGB_body_reads_during_preparation=0, actual_fit=False)
    write(H / 'PREPARATION_READY.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--generated', action='store_true'); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    if args.generated and not args.prepare and args.entry is None: generated_checks()
    elif args.prepare and not args.generated and args.entry is None: prepare()
    elif args.entry and not args.generated and not args.prepare: run(args.entry)
    else: raise SystemExit('Use --generated, --prepare or explicit root --entry')
