"""Synthetic accounting contracts only; no actual scores/journals are read."""
from copy import deepcopy

import pytest

from . import build_repair_campaign_progress_v12_v31 as B


def _summary(count):
    return dict(count=count, translation_le5=count-1, successes_5cm_5deg=count-2,
        translation_cm=dict(mean=2., max=9.), rotation_deg=dict(mean=3., max=8.),
        rotation_unresolved=1, fallback_count=1, max_translation_id='synthetic<maximum>')


def _method(name):
    return dict(method=name, label_KO='synthetic '+name, current_default=False,
        original_TEST49_all_translation_le5=False, numerical_primary192_target_met=False,
        populations={p: _summary(n) for p, n in {**B.P.COUNTS, 'PRIMARY192': 192}.items()})


def test_missing_GO_rejects_before_any_data_or_hash_IO(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No actual score/prediction/reference IO before GO')
    monkeypatch.setattr(B.C, 'read', forbidden)
    monkeypatch.setattr(B.C, 'sha', forbidden)
    monkeypatch.setattr(B.P, '_load_run', forbidden)
    with pytest.raises(ValueError, match='root accounting GO'):
        B.build()


def test_policy_and_containment_contracts_keep_raw_generation_and_no_e_pose():
    contracts = B._profile_contracts()
    assert list(contracts) == list(B.V.ARMS[1:])
    assert [c['containment_weight'] for c in contracts.values()] == [0., 1., 0., 1.]
    for arm, contract in contracts.items():
        assert contract['selection_profile'] == B.V.G.SELECTION_PROFILE
        assert contract['raw_observation_generation'] == B.V.G.RAW_OBSERVATION_GENERATION
        assert contract['noise_observation_family'] == B.V.G.OBSERVATION_FAMILY
        assert contract['original_named_BI_to_actual_infinite_line_normal_reach_px'] == 43.
        assert contract['actual_original_BI_q_query_anchor_required'] is True
        assert contract['fixed_signed_native_ray_required'] is True
        assert contract['query_anchor_metadata_tolerance_px'] == 1e-6
        assert contract['minimum_fixed_reference_ray_length_px'] == 10.
        assert contract['model_anchor_queries_relocated_to_q'] is False
        assert contract['old_V30_noise_relabelled'] is False
        assert contract['new_RGB_NN_NMS_or_pose_guide_fits'] == 0
        assert contract['full_pose_TP_topnormal_prior_weight'] == 0.
        assert contract['e_numeric_pose_target_prior_or_fallback'] is False
        assert contract['S_image_2D_or_regions'] == 0


def test_four_cross_generation_pairs_match_policy_AND_selector_index():
    pairs = B._paired_definitions()
    assert len(pairs) == 8
    expected = [(old, new) for policy in B.V.G.SOURCE_POLICIES
        for old, new in zip(B.PREVIOUS.V.POLICY_ARMS[policy], B.V.POLICY_ARMS[policy])]
    assert [(p['before'], p['after']) for p in pairs[:4]] == expected
    assert [p['selector_index'] for p in pairs[:4]] == [0, 1, 0, 1]
    assert all('단일 요소 효과 아님' in p['interpretation_KO'] for p in pairs[:4])
    assert all(p['before_run']=='V30' and p['after_run']=='V31' for p in pairs[:4])
    assert all('q 탐색 앵커' in p['interpretation_KO'] and 'CAL' in p['interpretation_KO'] for p in pairs[:4])
    assert [(p['before'], p['after']) for p in pairs[4:]] == [
        (B.V.ARMS[1], B.V.ARMS[2]), (B.V.ARMS[3], B.V.ARMS[4]),
        (B.V.ARMS[1], B.V.ARMS[3]), (B.V.ARMS[2], B.V.ARMS[4])]


def test_previous37_copied_exactly_with_four_distinct_nondefault_controls():
    previous = dict(method_count=37, methods=[_method('old_'+str(i)) for i in range(37)])
    methods = deepcopy(previous['methods']) + [_method(a) for a in B.V.ARMS[1:]]
    B._same_previous_methods(previous, methods)
    methods[0]['label_KO'] = 'modified history'
    with pytest.raises(ValueError, match='previous37'):
        B._same_previous_methods(previous, methods)
    methods = deepcopy(previous['methods']) + [_method(a) for a in B.V.ARMS[1:]]
    methods[-1]['current_default'] = True
    with pytest.raises(ValueError, match='default adoption'):
        B._same_previous_methods(previous, methods)


def test_fixed_reference_identity_and_exact_e_acceptance_are_required():
    before = {p: dict(reference_binding=dict(path='same/'+p, sha256=p), rows=[dict(
        id=p, metrics={B.P.BASELINE: {'synthetic_metric': 1}}, accepted={B.P.BASELINE: False})])
        for p in B.P.COUNTS}
    after = deepcopy(before)
    B.REFERENCE._require_same_reference_bindings(before, after)
    B.COMMON._same_e(before, after)
    after['REMAINDER']['reference_binding']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='REMAINDER'):
        B.REFERENCE._require_same_reference_bindings(before, after)
    after = deepcopy(before)
    after['TEST']['rows'][0]['accepted'][B.P.BASELINE] = True
    with pytest.raises(ValueError, match='fallback changed'):
        B.COMMON._same_e(before, after)


def test_paired_translation_joint_and_fallback_keep_all_rows_and_order():
    old = [dict(id='a', metrics={'old': dict(translation_cm=6., success_5cm_5deg=False)}, accepted={'old': True}),
           dict(id='b', metrics={'old': dict(translation_cm=4., success_5cm_5deg=True)}, accepted={'old': False})]
    new = [dict(id='a', metrics={'new': dict(translation_cm=4., success_5cm_5deg=True)}, accepted={'new': False}),
           dict(id='b', metrics={'new': dict(translation_cm=7., success_5cm_5deg=False)}, accepted={'new': True})]
    q = B.COMMON.PAIR._paired(old, new, 'old', 'new')
    assert q['count'] == 2 and q['mean_translation_delta_cm'] == .5
    assert q['translation_le5_gains'] == ['a'] and q['translation_le5_losses'] == ['b']
    assert q['joint_gains'] == ['a'] and q['joint_losses'] == ['b']
    assert q['fallback_count_before'] == q['fallback_count_after'] == 1
    with pytest.raises(ValueError, match='order'):
        B.COMMON.PAIR._paired(old, new[::-1], 'old', 'new')


def test_HTML_has164_method_rows_and_all_four_populations_with_limits():
    methods = [_method('old_'+str(i)) for i in range(37)] + [_method(a) for a in B.V.ARMS[1:]]
    rendered = B._render(dict(methods=methods))
    assert rendered.count('<tr>') + rendered.count('<tr class="new">') == 168
    assert rendered.count('<tr class="new">') == 16
    assert 'synthetic&lt;maximum&gt;' in rendered and 'synthetic<maximum>' not in rendered
    for text in ('회전 미확정', 'fallback', '기존49 보존', '41개 방법×4집단=164행',
            '별도 CAL46', '43px', '단일 요인 효과가 아닙니다', 'annotation/PnP 유도값',
            'PROGRESS_V12_V31.json', 'RESULTS_V11_V30.html'):
        assert text in rendered


def test_five_arm_profile_and_score_loader_uses_only_fake_records(tmp_path, monkeypatch):
    protocol = dict(ids=[], source_sha256={}, input_sha256={}, protected_user_sha256={},
        selection_profile=B.V.G.SELECTION_PROFILE, noise_observation_family=B.V.G.OBSERVATION_FAMILY,
        raw_observation_generation=B.V.G.RAW_OBSERVATION_GENERATION,
        central_plan_sha256='PREDECLARATION.json',declaration=dict(sha256='PREDECLARATION.json'))
    pred = dict(rows=238, arms=list(B.V.ARMS), arm_count=5, all_declared_arms_complete=True,
        before_new_control_metric_read=True, predictions_sha256='PREDICTIONS.jsonl.gz',
        protocol_sha256='PROTOCOL.json', input_seal_sha256='INPUT_SEAL.json')
    score = dict(prediction_seal=dict(sha256='PREDICTION_SEAL.json'), score_sha256={})
    files = {'PROTOCOL.json': protocol, 'PREDICTION_SEAL.json': pred, 'SCORE_SEAL.json': score}
    for part, count in B.P.COUNTS.items():
        rows = [dict(id=part+str(i), metrics={a: {} for a in B.V.ARMS},
            accepted={a: False for a in B.V.ARMS}) for i in range(count)]
        protocol['ids'].extend(r['id'] for r in rows)
        files[part+'_SCORE.json'] = dict(count=count, rows=rows)
        score['score_sha256'][part] = part+'_SCORE.json'
    monkeypatch.setattr(B.C, 'read', lambda p: files[p.name])
    monkeypatch.setattr(B.C, 'sha', lambda p: p.name)
    monkeypatch.setattr(B.P, '_verify', lambda mapping: None)
    _, reports = B._load_new_run(tmp_path)
    assert sum(len(q['rows']) for q in reports.values()) == 238
    protocol['declaration']['sha256']='different central generation'
    with pytest.raises(ValueError,match='five-arm'):
        B._load_new_run(tmp_path)
    protocol['declaration']['sha256']=protocol['central_plan_sha256']
    pred['all_declared_arms_complete'] = False
    with pytest.raises(ValueError, match='five-arm'):
        B._load_new_run(tmp_path)
    pred['all_declared_arms_complete'] = True
    files['TEST_SCORE.json']['rows'][0]['accepted'].pop(B.V.ARMS[-1])
    with pytest.raises(ValueError, match='fallback'):
        B._load_new_run(tmp_path)
