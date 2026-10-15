"""Pure controller closure and streaming inventory tests, no actual CAL IO."""
from copy import deepcopy
import pytest

from . import prepare_geometry_guided_native_metadata_v4_31b as P


def closure():
    ids = ['synthetic:' + str(i) for i in range(46)]
    hashes = {name: str(i)*64 for i, name in enumerate(P.CAL_FILES)}
    protocol = dict(ids=ids + ['primary'], CAL_ids=ids, central_plan_sha256='a'*64,
        source_sha256={'synthetic.py': 'b'*64}, selection_profile=P.G.SELECTION_PROFILE,
        raw_observation_generation=P.G.RAW_OBSERVATION_GENERATION)
    index = [dict(id=ident, partition='CAL') for ident in ids]
    summary = dict(rows=46, records=[dict(id=ident) for ident in ids], selection_profile=P.G.SELECTION_PROFILE)
    seal = dict(rows=46, ordered_ids=ids, source_sha256=deepcopy(protocol['source_sha256']),
        selection_profile=P.G.SELECTION_PROFILE, raw_observation_generation=P.G.RAW_OBSERVATION_GENERATION,
        canonical_validation=dict(rows=46, canonical_row_digests_verified=46, missing_duplicate_or_extra_rows=0))
    for field, name in dict(protocol_sha256='PROTOCOL.json', preparation_seal_sha256='PREPARATION_SEAL.json',
        execution_protocol_sha256='CAL_EXECUTION_PROTOCOL.json', row_index_sha256='CAL_ROW_INDEX.json',
        observations_sha256='CAL_OBSERVATIONS.jsonl.gz', summary_sha256='CAL_SUMMARY.json').items():
        seal[field] = hashes[name]
    return protocol, seal, index, summary, dict(central_plan_sha256='a'*64), hashes, 'a'*64


def test_root_GO_and_explicit_central_SHA_precede_all_IO(monkeypatch):
    monkeypatch.setattr(P.C, 'read', lambda *a: pytest.fail('No IO before GO/hash'))
    monkeypatch.setattr(P, '_observation_closure', lambda *a: pytest.fail('No closure IO before explicit hash'))
    with pytest.raises(PermissionError): P.main()
    for value in (None, '', 'A'*64, 'a'*63):
        with pytest.raises(ValueError, match='Explicit exact central31'):
            P.main(root_metadata_GO=True, expected_central_plan_sha256=value)


def test_exact_CAL46_source_seal_index_exec_and_canonical_links():
    args = closure()
    assert P.validate_CAL_closure(*args) == args[0]['CAL_ids']


@pytest.mark.parametrize('change', ['source', 'preparation', 'execution', 'summary', 'index',
    'raw', 'profile', 'duplicate', 'missing', 'nonCAL', 'canonical', 'central'])
def test_bad_producer_CAL_closure_is_fail_closed(change):
    protocol, seal, index, summary, execution, hashes, expected = closure()
    if change == 'source': seal['source_sha256'] = {}
    elif change == 'preparation': seal['preparation_seal_sha256'] = 'changed'
    elif change == 'execution': seal['execution_protocol_sha256'] = 'changed'
    elif change == 'summary': summary['records'][0]['id'] = 'changed'
    elif change == 'index': seal['row_index_sha256'] = 'changed'
    elif change == 'raw': seal['raw_observation_generation'] = 'V31-relabeled-raw'
    elif change == 'profile': seal['selection_profile'] = 'V30'
    elif change == 'duplicate': index[0]['id'] = index[1]['id']
    elif change == 'missing': index.pop()
    elif change == 'nonCAL': index[0]['partition'] = 'TEST'
    elif change == 'canonical': seal['canonical_validation']['canonical_row_digests_verified'] = 45
    else: execution['central_plan_sha256'] = 'c'*64
    with pytest.raises(ValueError):
        P.validate_CAL_closure(protocol, seal, index, summary, execution, hashes, expected)


def stream_fixture():
    ids = ['CAL:' + str(i) for i in range(46)]
    tasks = [dict(id=ident, partition='CAL') for ident in ids]
    rows = [dict(id=ident, partition='CAL', selection_profile=P.G.SELECTION_PROFILE,
        observation=dict(selected_chain_refs=[], selected_physical_edges=0,
                         joint_graph=dict(selected_line_names=[])),
        summary=dict(selected_physical_edges=0, zero_selected_edges=True)) for ident in ids]
    entries = [dict(id=row['id'], canonical_row_sha256=P.OP.D.R._canonical_sha(row)) for row in rows]
    return ids, tasks, rows, entries


def test_streams_all46_zero_rows_both_policies_without_objective_or_label_calls(monkeypatch):
    ids, tasks, rows, entries = stream_fixture(); records, errors = [], []
    def prepare(task, row):
        assert task['id'] == row['id']
        return dict(id=task['id'], partition='CAL', profiles={p: dict(groups=[], tangent_records=[])
                    for p in P.G.SOURCE_POLICIES}), dict(id=task['id'], profiles={})
    monkeypatch.setattr(P.M, 'prepare_one', prepare)
    prepared = list(P.prepared_CAL_rows(iter(tasks), iter(rows), iter(entries), ids, records, errors))
    assert [item['id'] for item in prepared] == ids and len(records) == 46 and not errors
    assert all(set(item['profiles']) == set(P.G.SOURCE_POLICIES) for item in prepared)


def test_changed_canonical_row_or_missing_inventory_fails_before_prepare(monkeypatch):
    monkeypatch.setattr(P.M, 'prepare_one', lambda *a: pytest.fail('Invalid row before prepare'))
    ids, tasks, rows, entries = stream_fixture(); rows[0]['summary']['zero_selected_edges'] = False
    with pytest.raises(ValueError, match='canonically equal'):
        list(P.prepared_CAL_rows(tasks, rows, entries, ids, [], []))
    ids, tasks, rows, entries = stream_fixture()
    with pytest.raises(ValueError, match='counts'):
        list(P.prepared_CAL_rows(tasks, [], entries, ids, [], []))


def test_preparation_failure_preserved_without_dropping_remaining_zero_inventory(monkeypatch):
    ids, tasks, rows, entries = stream_fixture(); records, errors = [], []
    def fail(task, row): raise ValueError('synthetic metadata failure')
    monkeypatch.setattr(P.M, 'prepare_one', fail)
    assert list(P.prepared_CAL_rows(tasks, rows, entries, ids, records, errors)) == []
    assert not records and [e['id'] for e in errors] == ids


def test_immutable_source_union_has_no_silent_override():
    assert P._union({'a': 'x'}, {'a': 'x', 'b': 'y'}) == {'a': 'x', 'b': 'y'}
    with pytest.raises(ValueError, match='ancestry conflict'): P._union({'a': 'x'}, {'a': 'y'})
