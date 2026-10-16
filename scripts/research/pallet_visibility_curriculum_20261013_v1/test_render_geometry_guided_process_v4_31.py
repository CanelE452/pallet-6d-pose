"""Synthetic display/profile/facade tests; actual primary and images remain unread."""
from copy import deepcopy
import gzip
import json
import numpy as np
import pytest

from . import render_geometry_guided_process_v4_31 as R
from .test_joint_geometry_guided_lines_v4_28 import Poison


def completion():
    ids = ['frame:' + str(i) for i in range(238)]
    tags = dict(selection_profile=R.G.SELECTION_PROFILE, raw_observation_generation=R.G.RAW_OBSERVATION_GENERATION)
    protocol = dict(ids=ids, arms=list(R.ARMS), central_plan_sha256='a'*64, noise_observation_family=R.G.OBSERVATION_FAMILY, **tags)
    prediction = dict(rows=238, ordered_ids=ids, arms=list(R.ARMS), arm_count=5, all_declared_arms_complete=True,
        CAL=46, TEST=49, REMAINDER=143, severe=0)
    audit = dict(rows=238, failure_count=0, failures=[])
    return protocol, prediction, audit, dict(ids=ids, central_plan_sha256='a'*64, **tags), dict(rows=238, ordered_ids=ids, **tags)


def cost_plan():
    tags = dict(observation_family=R.G.OBSERVATION_FAMILY, selection_profile=R.G.SELECTION_PROFILE,
        raw_observation_generation=R.G.RAW_OBSERVATION_GENERATION, noise_population=R.G.POPULATION, source_policy='ALL_BI_REPLACE')
    return dict(**tags, population=R.G.POPULATION, active_groups=[], new_owner_ids=[],
        CAL_noise_binding=dict(available=True, noise=dict(noise_provenance=tags), tangent=tags))


def test_render_GO_precedes_any_IO(monkeypatch):
    def blocked(*a, **kw): raise AssertionError('No new scores/models/images before GO')
    monkeypatch.setattr(R.C, 'read', blocked); monkeypatch.setattr(R.C, 'sha', blocked)
    with pytest.raises(ValueError, match='explicit root'): R.render()


def test_private_renderer_uses_only_new_paths_arms_and_no_parent_global_change():
    before = dict(R.PARENT.render.__globals__)
    new = R._renderer_function(central_sha256='a'*64, model_seal_sha256='b'*64); namespace = new.__globals__
    assert namespace['OUTPUT'] == R.OUTPUT and namespace['PLAN'] == R.PLAN
    assert namespace['PREDICTIONS'] == R.EXP.OUTPUT and namespace['OBS'].OUTPUT == R.PREP.OBS
    assert namespace['ARMS'] == R.EXP.ARMS and len(namespace['ARMS']) == 5
    assert callable(namespace['_completed'])
    assert namespace['__file__'] == R.__file__
    assert all(R.PARENT.render.__globals__[k] is v for k, v in before.items())


def test_private_renderer_schema_and_page_text_are_separate_V31():
    fn = R._renderer_function(central_sha256='a'*64, model_seal_sha256='b'*64); strings = [c for c in fn.__code__.co_consts if isinstance(c, str)]
    assert 'FIXED_TWO_CASES_GEOMETRY_GUIDED_PROCESS_DATA_V31' in strings
    assert 'FIXED_TWO_CASES_GEOMETRY_GUIDED_RENDER_SEAL_V31' in strings
    assert not any('FIXED_TWO_CASES_GEOMETRY_GUIDED_PROCESS_DATA_V29' == c for c in strings)
    page = next(c for c in strings if c.startswith('<!doctype html>'))
    assert 'v31 원 q 앵커' in page and 'bounded coarse-S' in page
    assert any('43px 후보 검사 통과' in c for c in strings)


def test_complete238_profile_guard_and_partial_fivearm_rejection():
    items = completion(); R.validate_completed_metadata(*items, expected_central_plan_sha256='a'*64)
    for index, key, value in [(1, 'rows', 237), (1, 'arm_count', 4),
                              (1, 'all_declared_arms_complete', False), (2, 'failure_count', 1),
                              (4, 'rows', 46), (1, 'TEST', 48), (1, 'severe', 1)]:
        bad = deepcopy(items); bad[index][key] = value
        with pytest.raises(ValueError): R.validate_completed_metadata(*bad, expected_central_plan_sha256='a'*64)


def test_old_raw_family_or_profile_cannot_be_relabelled():
    items = completion()
    for index, key, value in [(0, 'noise_observation_family', 'old'), (3, 'selection_profile', 'old'),
                              (4, 'raw_observation_generation', 'FRESH_WRONG')]:
        bad = deepcopy(items); bad[index][key] = value
        with pytest.raises(ValueError): R.validate_completed_metadata(*bad, expected_central_plan_sha256='a'*64)
    plan = cost_plan(); R._require_plan(plan)
    for key in ['observation_family', 'population', 'selection_profile', 'raw_observation_generation']:
        bad = deepcopy(plan); bad[key] = 'old'
        with pytest.raises(ValueError): R._require_plan(bad)


def test_active_vs_fallback_plan_and_no_precise_new_KP():
    row = dict(states=['B', 'U'] + ['S'] * 6, initial_geometry=dict(initial_self_hidden=[False, False] + [True] * 6))
    plan = cost_plan(); group = dict(native_corner=0, native_neighbor=1, actual_inlier_xy=[[10.,20.],[22.,20.]],
        normal_unit_xy=[0.,1.], offset_px=-20., proposal_grade=R.PARENT.P.PARTIAL_GRADE, physical_native_edge=[0,1])
    plan['active_groups'] = [group]; plan['new_owner_ids'] = [0]
    before = deepcopy(plan)
    result = dict(solution=dict(v29_cost_plan=plan, v31_cost_plan=plan))
    shown, lines = R._active_plan_lines(result, row, R.ARMS[1])
    assert shown is plan and np.array_equal(lines[0]['xy'], group['actual_inlier_xy']) and lines[0]['partial']
    fallback = dict(solution={}, WD_family='ORIGINAL_WD', new_geometry_edge_cost_plans={'ORIGINAL_WD': plan})
    assert R._active_plan_lines(fallback, row, R.ARMS[1])[0] is plan
    assert R._active_plan_lines(Poison(), row, R.ARMS[0]) == ({}, []) and plan == before


def test_v31_model_policy_profile_binding_and_wire_alias_disagreement():
    plan = cost_plan(); plan['CAL_noise_binding']['tangent'] = {**plan['CAL_noise_binding']['tangent'], 'source_policy': 'wrong'}
    with pytest.raises(ValueError, match='policy-specific'): R._require_plan(plan)
    first = cost_plan(); other = deepcopy(first); other['new_owner_ids'] = [0]
    with pytest.raises(ValueError, match='wire aliases'):
        R._active_plan_lines(dict(solution=dict(v29_cost_plan=first, v31_cost_plan=other)), Poison(), R.ARMS[1])


def test_inherited_S_U_markers_and_selected_S_source_are_still_gated():
    row = dict(states=['I', 'U'] + ['S'] * 6, initial_geometry=dict(initial_self_hidden=[False, False] + [True] * 6),
        visible_measurements=[dict(native_corner=0, state='I', xy=[10.,20.], source='RGB'),
            dict(native_corner=1, state='U', xy=Poison(), source=Poison()),
            dict(native_corner=2, state='S', xy=Poison(), source=Poison())])
    assert R.PARENT._gated_visible(row) == [row['visible_measurements'][0]]
    bad = dict(states=['S'] * 8, initial_geometry=dict(initial_self_hidden=[True] * 8),
        observation=dict(selected_chain_refs=[dict(native_corner=0, native_neighbor=1)], chain_observation=Poison()))
    with pytest.raises(ValueError, match='permanent state gates'): R.PARENT._selected(bad)


def test_only_the_two_fixed_cases_decode_not_whole_journal(tmp_path):
    path = tmp_path / 'rows.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        stream.write('not requested and not JSON\n')
        for ident in R.CASES: stream.write(json.dumps(dict(id=ident)) + '\n')
    rows, digests = R.PARENT._exact_rows(path, R.CASES)
    assert list(rows) == list(R.CASES) and len(digests) == 2


def test_product_metadata_binding_is_V31_without_mutating_value(monkeypatch, tmp_path):
    saved = []; monkeypatch.setattr(R.E, '_binding', lambda p: dict(path=str(p), sha256='fixture'))
    monkeypatch.setattr(R.C, 'write', lambda p, v: saved.append((p, v)))
    original = dict(schema='FIXED_TWO_CASES_GEOMETRY_GUIDED_PROCESS_DATA_V31', cases=[])
    R._write_product(tmp_path / 'PROCESS_DATA.json', original, central_sha256='a'*64, model_seal_sha256='b'*64)
    assert original == dict(schema='FIXED_TWO_CASES_GEOMETRY_GUIDED_PROCESS_DATA_V31', cases=[])
    value = saved[0][1]
    assert value['selection_profile'] == R.G.SELECTION_PROFILE and value['population'] == R.G.POPULATION
    assert value['raw_observation_generation'] == R.G.RAW_OBSERVATION_GENERATION
    assert value['final_observation_protocol']['path'].endswith('FINAL_PROTOCOL.json')
    assert value['policy_specific_CAL_model_seal_sha256'] == 'b'*64
    assert value['explicit_central_method_sha256'] == 'a'*64


def test_explicit_central_and_model_hashes_fail_before_prepare_IO(monkeypatch):
    monkeypatch.setattr(R.C, 'read', lambda *a: pytest.fail('No IO without explicit hashes'))
    monkeypatch.setattr(R.C, 'sha', lambda *a: pytest.fail('No IO without explicit hashes'))
    for central, model in [(None, None), ('a'*64, None), ('A'*64, 'b'*64)]:
        with pytest.raises(ValueError, match='Explicit exact'):
            R.prepare(expected_central_plan_sha256=central, expected_model_seal_sha256=model)
    with pytest.raises(ValueError, match='Explicit exact'):
        R.render(root_postscore_audit_memory_GO=True)


def test_complete_observation_and_eval_must_bind_same_central():
    items = completion()
    items[3]['central_plan_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='explicitly pinned central'):
        R.validate_completed_metadata(*items, expected_central_plan_sha256='a'*64)


def test_unavailable_new_noise_cannot_show_active_factors_or_owners():
    plan = cost_plan(); plan['CAL_noise_binding'] = dict(available=False)
    R._require_plan(plan)
    for key, value in [('active_groups', [{}]), ('new_owner_ids', [0])]:
        bad = deepcopy(plan); bad[key] = value
        with pytest.raises(ValueError, match='uniformly abstain'): R._require_plan(bad)


def model_metadata(monkeypatch):
    # Metadata-only fake documents: no numerical fitted MODEL.json is decoded.
    root = R.CAL.OUTPUT
    central_key = str(R.EXP.PLAN.relative_to(R.C.ROOT))
    tags = dict(observation_generation=R.G.OBSERVATION_FAMILY,
        raw_observation_generation=R.G.RAW_OBSERVATION_GENERATION,
        noise_population=R.G.POPULATION, selection_profile=R.G.SELECTION_PROFILE,
        input_sha256={central_key: 'a'*64}, source_sha256={})
    model = dict(**deepcopy(tags), fit_partition='CAL', CAL_inventory=46, compatible_inventory=44,
        old_model_relabelled=False, default_adoption=False, preparation_seal_sha256='c'*64,
        assets={p+'/'+name: 'e'*64 for p in R.G.SOURCE_POLICIES for name in ('MODEL.json', 'TANGENT_MODEL.json')})
    protocol = dict(**deepcopy(tags), protected_user_sha256={})
    prep = dict(protocol_sha256='d'*64)
    docs = {root/'MODEL_SEAL.json': model, root/'PROTOCOL.json': protocol, root/'PREPARATION_SEAL.json': prep}
    hashes = {root/'MODEL_SEAL.json': 'b'*64, root/'PROTOCOL.json': 'd'*64, root/'PREPARATION_SEAL.json': 'c'*64}
    hashes.update({root/name: 'e'*64 for name in model['assets']})
    monkeypatch.setattr(R.C, 'read', lambda p: docs[p])
    monkeypatch.setattr(R.C, 'sha', lambda p: hashes[p])
    monkeypatch.setattr(R, '_verify_map', lambda value: None)
    return model, protocol


def test_model_seal_binding_has_both_policies_and_typed_population_without_numeric_model_read(monkeypatch):
    model, _ = model_metadata(monkeypatch)
    assert R._model_metadata('b'*64, 'a'*64) is model


@pytest.mark.parametrize('change', ['family', 'profile', 'population', 'raw', 'central', 'policy_asset'])
def test_old_or_incomplete_model_metadata_is_rejected(monkeypatch, change):
    model, protocol = model_metadata(monkeypatch)
    if change == 'family': model['observation_generation'] = R.ADAPTER.G.OBSERVATION_FAMILY
    elif change == 'profile': model['selection_profile'] = R.ADAPTER.G.SELECTION_PROFILE
    elif change == 'population': model['noise_population'] = 'V30'
    elif change == 'raw': model['raw_observation_generation'] = 'relabelled'
    elif change == 'central': protocol['input_sha256'] = {}
    else: model['assets'].pop(R.G.SOURCE_POLICIES[0]+'/TANGENT_MODEL.json')
    with pytest.raises(ValueError): R._model_metadata('b'*64, 'a'*64)
