"""Source-only synthetic fixtures; no real CAL row, mask or label is read."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from . import run_observed_anchor_native_ray_CAL_v4_31 as R


def _entries():
    return [dict(id=ident, partition='CAL') for ident in R.CASES]


def test_fixed_cases_keep_order_partition_and_duplicates():
    assert R._fixed_cases(_entries()) == R.CASES
    for bad in (_entries()[::-1], _entries()[:2], [*_entries()[:2], _entries()[0]]):
        with pytest.raises(ValueError, match='ordered three CAL'):
            R._fixed_cases(bad)
    bad = _entries()
    bad[0]['partition'] = 'TEST'
    with pytest.raises(ValueError, match='primary'):
        R._fixed_cases(bad)


def test_actual_GO_precedes_all_IO_and_source_hashes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No IO before explicit cached-CAL execution GO')
    monkeypatch.setattr(R.C, 'read', forbidden)
    monkeypatch.setattr(R.C, 'sha', forbidden)
    monkeypatch.setattr(R, '_source_pins', forbidden)
    with pytest.raises(PermissionError, match='before any IO'):
        R.run()


def test_real_source_pins_and_new_population_do_not_transfer_old_noise():
    R._source_pins()
    assert R.F.SELECTION_PROFILE != R.OLD.F.SELECTION_PROFILE
    assert R.F.RECIPE['old_V29_or_V30_noise_validated_for_new_population'] is False
    assert R.F.RECIPE['cached_query_anchor_or_pixels_not_relocated'] is True
    assert R.F.RECIPE['finite_alongline_identity_certified'] is False


def test_single_prior_graph_requires_exact_profile_and_one_record(monkeypatch):
    graph = dict(selected_line_names=[])
    item = dict(selection_profile=R.OLD.F.SELECTION_PROFILE, graph=graph)
    monkeypatch.setattr(R.C, 'rows', lambda path: iter([item]))
    assert R._single_graph(Path('synthetic')) is graph
    monkeypatch.setattr(R.C, 'rows', lambda path: iter([item, item]))
    with pytest.raises(ValueError, match='One exact'):
        R._single_graph(Path('synthetic'))
    monkeypatch.setattr(R.C, 'rows', lambda path: iter([{**item, 'selection_profile': 'wrong'}]))
    with pytest.raises(ValueError, match='V30 graph'):
        R._single_graph(Path('synthetic'))


def test_selected_audits_keep_empty_abstention_and_no_unselected_entries():
    result = dict(graph=dict(selected_line_names=[]), node_association_ledger=[
        dict(line_name='unused', passes=False, reason='model anchor')])
    assert R._selected_audits(result) == []
    result['graph']['selected_line_names'] = ['selected']
    result['node_association_ledger'].append(dict(line_name='selected', passes=True))
    assert R._selected_audits(result) == [dict(line_name='selected', passes=True)]


def test_immutable_view_contains_full_raw_arrays_and_all_input_bindings():
    row = dict(states=['B']*8, visible_measurements=[dict(native_corner=0, record=dict(xy=[1., 2.]))],
        initial_geometry=dict(K=[[1.]], native_xyz=[[1., 2., 3.]]), mask_binding=dict(path='mask', sha256='x'),
        observation=dict(chain_observation=dict(edge_records=[dict(chains=[dict(observed_points=[[1., 2.]])])]),
            guide_bank=dict(a=1), guide_geometries=dict(b=2)))
    original = deepcopy(row)
    before = R.R._canonical_sha(R._immutable_observation_view(row))
    assert row == original
    row['observation']['chain_observation']['edge_records'][0]['chains'][0]['observed_points'][0][0] += 1.
    assert R.R._canonical_sha(R._immutable_observation_view(row)) != before


def test_declare_binds_fake_bytes_but_never_decodes_rows_masks_RGB_or_labels(tmp_path, monkeypatch):
    """Marker spools deliberately are not JSON; declaration can only hash them."""
    real_read, real_sha = R.C.read, R.C.sha
    raw, old, new = tmp_path/'raw', tmp_path/'previous', tmp_path/'new_plan'
    raw.mkdir(); (old/'graphs').mkdir(parents=True)
    protected = tmp_path/'protected.txt'; protected.write_text('unchanged')
    source = tmp_path/'fake_source.py'; source.write_text('# synthetic source only\n')
    entries = []
    forbidden_paths = set()
    assets = {}
    for number, ident in enumerate(R.CASES):
        path = raw/('spool'+str(number)+'.gz')
        path.write_bytes(b'NOT JSON; DECLARATION MUST NOT DECODE THIS CAL ROW')
        graph = old/'graphs'/('case'+str(number)+'.jsonl.gz')
        graph.write_bytes(b'NOT JSON; DECLARATION MUST NOT DECODE THIS GRAPH')
        forbidden_paths.update((path, graph))
        entries.append(dict(id=ident, partition='CAL', sha256=real_sha(path),
            canonical_row_sha256='canonical'+str(number), unchanged_observation_math_sha256='math'+str(number)))
        assets[str(graph.relative_to(old))] = real_sha(graph)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    previous_plan = old/'PREDECLARATION.json'
    write(previous_plan, dict(cases=entries, configuration={'synthetic': True}))
    write(old/'PLAN_SEAL.json', dict(plan_sha256=real_sha(previous_plan)))
    write(old/'RERANK_SEAL.json', dict(assets=assets))
    write(raw/'PROTOCOL.json', dict(configuration={'synthetic': True},
        protected_user_sha256={'protected.txt': real_sha(protected)}))
    write(raw/'CAL_ROW_INDEX.json', dict(entries=entries))
    for name in ('PREPARATION_SEAL.json', 'CAL_OBSERVATION_SEAL.json'):
        write(raw/name, {'metadata_only': True})
    ids = {ident: raw/('spool'+str(i)+'.gz') for i, ident in enumerate(R.CASES)}
    monkeypatch.setattr(R.C, 'ROOT', tmp_path)
    monkeypatch.setattr(R, 'PLAN', new/'PREDECLARATION.json')
    monkeypatch.setattr(R.OLD, 'PLAN', previous_plan)
    monkeypatch.setattr(R.OLD, 'OUTPUT', old)
    monkeypatch.setattr(R.OBS, 'OUTPUT', raw)
    monkeypatch.setattr(R.OBS, '_row_path', lambda output, ident: ids[ident])
    monkeypatch.setattr(R, '_source_pins', lambda: None)
    monkeypatch.setattr(R, '_sources', lambda: {'fake_source.py': real_sha(source)})
    def metadata_read(path):
        assert Path(path) not in forbidden_paths, 'Actual CAL graph/row decode before GO'
        return real_read(path)
    monkeypatch.setattr(R.C, 'read', metadata_read)
    monkeypatch.setattr(R.C, 'rows', lambda *args: pytest.fail('No actual row decode in declare'))
    monkeypatch.setattr(R.OBS, '_read_single', lambda *args: pytest.fail('No actual row decode in declare'))
    monkeypatch.setattr(R.np, 'load', lambda *args, **kwargs: pytest.fail('No mask/image decode in declare'))
    monkeypatch.setattr(R.F, 'rank_observed_anchor_native_ray_graph', lambda *args, **kwargs: pytest.fail('No rerank in declare'))
    R.declare()
    plan = real_read(new/'PREDECLARATION.json')
    assert [c['id'] for c in plan['cases']] == list(R.CASES)
    assert plan['no_actual_row_mask_RGB_reference_or_score_decode_in_declaration'] is True
    assert plan['validated_noise_model_available'] is False
    assert plan['all_three_rows_and_zero_candidates_or_selected_abstentions_retained'] is True
    assert all(real_sha(tmp_path/path) == digest for path, digest in plan['input_sha256'].items())
    assert (new/'code_snapshot/fake_source.py').read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        R.declare()
