"""Pure storage/admission orchestration tests; no real dataset is decoded."""
from concurrent.futures import Future
from copy import deepcopy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from . import rerank_observed_anchor_native_ray_population_v4_31 as R


def _row(ident, part='CAL', selected=False):
    refs = [dict(line_name='line', native_edge=[0, 1])] if selected else []
    return dict(id=ident, partition=part, selection_profile=R.SELECTION_PROFILE,
        observation=dict(selected_chain_refs=refs, selected_physical_edges=len(refs),
            joint_graph=dict(selected_line_names=[ref['line_name'] for ref in refs]),
            partial_line_ledger=[dict(raw_alternative='keep all, 선정 여부 무관')]),
        summary=dict(selected_physical_edges=len(refs), zero_selected_edges=not refs),
        unicode='팔레트 / 木 / 😀', values=[1., None, True, -0.])


def _entry(row):
    return dict(id=row['id'], partition=row['partition'], canonical_row_sha256=R.D.R._canonical_sha(row))


def test_batched_storage_has_exact_legacy_decompressed_bytes_and_finite_values(tmp_path):
    values = [dict(id='unicode', label='팔레트\n木😀', ordered={'z': 2, 'a': 1},
        array=np.array([1., np.nan, np.inf, -np.inf]), scalar=np.float64(2.5)), _row('empty')]
    old, new = tmp_path/'old.gz', tmp_path/'new.gz'
    R.C.save_rows(old, values);R.save_rows_large(new, values)
    assert gzip.decompress(old.read_bytes()) == gzip.decompress(new.read_bytes())
    assert list(R.C.rows(new)) == R.C.finite(values)
    assert R.D.R._canonical_sha(list(R.C.rows(old))) == R.D.R._canonical_sha(list(R.C.rows(new)))
    with pytest.raises(FileExistsError):
        R.save_rows_large(new, values)


def test_storage_uses_one_large_write_per_row_not_iterencode_micro_writes(monkeypatch):
    writes = []
    class Stream:
        def __enter__(self):return self
        def __exit__(self, *args):return False
        def write(self, text):writes.append(text)
    seen = []
    def opened(path, mode, **kwargs):
        seen.append((mode, kwargs));return Stream()
    monkeypatch.setattr(R.gzip, 'open', opened)
    rows = [_row('a'), _row('b')]
    R.save_rows_large(Path('synthetic'), rows)
    assert len(writes) == 2 and all(text.endswith('\n') for text in writes)
    assert [json.loads(text) for text in writes] == rows
    assert seen == [('xt', dict(encoding='utf-8', compresslevel=1))]


@pytest.mark.parametrize('member_count', [2, 3])
def test_multimember_binary_concat_exact_bytes_order_unicode_finites_and_canonical_digests(tmp_path, member_count):
    rows = [_row(str(i), selected=i%2 == 0) for i in range(member_count+2)]
    rows[0]['values'] = np.array([np.nan, 4., np.inf])
    members, original = [], []
    for number in range(member_count):
        path = tmp_path/('member'+str(number)+'.gz')
        selected = rows[number::member_count]
        R.save_rows_large(path, selected);original.append(path.read_bytes())
        members.append(dict(path=path, sha256=R.C.sha(path)))
    ordered = [r for number in range(member_count) for r in rows[number::member_count]]
    final = tmp_path/'all.gz';copies = R.concatenate_gzip_members(final, members)
    assert final.read_bytes() == b''.join(original)
    assert [Path(m['path']).read_bytes() for m in copies] == original
    assert list(R.C.rows(final)) == R.C.finite(ordered)
    assert R.validate_journal(final, [_entry(r) for r in ordered]) == dict(
        rows=len(rows), canonical_row_digests_verified=len(rows), missing_duplicate_or_extra_rows=0)
    assert sum(m['bytes'] for m in copies) == final.stat().st_size


def test_concat_rejects_changed_member_and_never_overwrites_existing_final(tmp_path):
    members = []
    for i in range(2):
        path = tmp_path/(str(i)+'.gz');R.save_rows_large(path, [_row(str(i))])
        members.append(dict(path=path, sha256=R.C.sha(path)))
    destination = tmp_path/'final.gz'
    bad = deepcopy(members);bad[0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='before binary copy'):
        R.concatenate_gzip_members(destination, bad)
    assert not destination.exists()
    R.concatenate_gzip_members(destination, members)
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        R.concatenate_gzip_members(destination, members)
    assert destination.read_bytes() == before


def test_journal_rejects_duplicate_missing_extra_reordered_and_changed_values(tmp_path):
    rows = [_row('a'), _row('b', selected=True)]
    path = tmp_path/'rows.gz';R.save_rows_large(path, rows)
    entries = [_entry(r) for r in rows]
    for wrong in (entries[:1], entries+[entries[0]], entries[::-1], [entries[0], {**entries[1], 'canonical_row_sha256': 'wrong'}]):
        with pytest.raises(ValueError):R.validate_journal(path, wrong)
    duplicate = tmp_path/'duplicate.gz';R.save_rows_large(duplicate, [rows[0], rows[0]])
    with pytest.raises(ValueError, match='unique IDs'):
        R.validate_journal(duplicate, [entries[0], entries[0]])


def test_new_topcount_parent_legacy_and_entire_partial_ledger_are_distinct():
    parent = dict(selected_physical_edges=99, partial_line_ledger=[dict(name='unselected raw partial')])
    original = deepcopy(parent);new = _row('x', selected=True)
    new['observation']['partial_line_ledger'] = deepcopy(parent['partial_line_ledger'])
    new['observation']['selected_physical_edges'] = 99
    fixed = R._correct_selection_metadata(new, parent)
    assert fixed['observation']['selected_physical_edges'] == 1
    assert fixed['observation']['parent_legacy_selected_physical_edges'] == 99
    assert fixed['observation']['partial_line_ledger'] == original['partial_line_ledger']
    assert parent == original
    bad = deepcopy(fixed);bad['observation']['partial_line_ledger'] = []
    with pytest.raises(ValueError, match='raw partial'):
        R._correct_selection_metadata(bad, parent)
    bad = deepcopy(fixed);bad['summary']['selected_physical_edges'] = 2
    with pytest.raises(ValueError, match='authoritative summary'):
        R._validate_selection_counts(bad)


def test_private_alias_uses_F31_API_without_parent_global_mutation(monkeypatch):
    calls = []
    def new_rank(*args, **kwargs):calls.append((args, kwargs));return {'new_rank': True}
    def parent_row(row, configuration, mask):
        # F is intentionally only present in the private execution namespace.
        result = F.rank_observed_BI_associated_graph('raw', configuration=configuration)
        assert result['new_rank'] is True
        return deepcopy(row), dict(zero_selected=True)
    monkeypatch.setattr(R.F, 'rank_observed_anchor_native_ray_graph', new_rank)
    monkeypatch.setattr(R.PARENT, 'rerank_cached_row', parent_row)
    monkeypatch.setattr(R, '_pins', lambda: None)
    globals_before = dict(parent_row.__globals__)
    row = _row('zero');row['observation']['selected_physical_edges'] = 12
    result, summary = R.rerank_cached_row(row, {'cfg': True}, 'not a mask query')
    assert result['observation']['selected_physical_edges'] == 0
    assert result['observation']['parent_legacy_selected_physical_edges'] == 12
    assert summary['zero_selected'] is True and len(calls) == 1
    assert dict(parent_row.__globals__) == globals_before


def test_GO_guards_before_any_real_IO(monkeypatch):
    def forbidden(*args, **kwargs):raise AssertionError('No actual IO before root GO')
    monkeypatch.setattr(R.C, 'read', forbidden);monkeypatch.setattr(R.C, 'sha', forbidden)
    for action in (R.prepare, R.finalize):
        with pytest.raises(PermissionError):action()
    for part in ('CAL', 'REST'):
        with pytest.raises(PermissionError):R.rerank(part)


def test_central_SHA_has_no_default_and_rejects_before_json_decode(monkeypatch):
    def forbidden(*args, **kwargs):raise AssertionError('No central JSON before exact SHA')
    monkeypatch.setattr(R.C, 'read', forbidden)
    for bad in (None, '', 'xyz', 'z'*64):
        with pytest.raises(ValueError, match='no default'):
            R._central(bad)
    monkeypatch.setattr(R.C, 'sha', lambda path: 'b'*64)
    with pytest.raises(ValueError, match='no default'):
        R._central('a'*64)


def test_worker_is_module_importable_and_failed_future_metadata_is_serializable(tmp_path, monkeypatch):
    assert pickle.loads(pickle.dumps(R._worker)) is R._worker
    instances = []
    class Process:
        terminated = False
        def is_alive(self):return True
        def terminate(self):self.terminated = True
    class Pool:
        def __init__(self, **kwargs):
            self.futures=[];self._processes={1:Process()};self.shutdown_calls=[];instances.append(self)
        def submit(self, fn, case, cfg, output):
            future=Future();self.futures.append(future)
            if len(self.futures)==1:future.set_exception(RuntimeError('synthetic worker failure'))
            return future
        def shutdown(self, **kwargs):self.shutdown_calls.append(kwargs)
    monkeypatch.setattr(R, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(R, 'wait', lambda pending, **kwargs: ({list(pending)[0]}, set()))
    cases=[dict(id='id'+str(i))for i in range(3)]
    with pytest.raises(RuntimeError, match='synthetic worker failure'):
        R._bounded_pool(cases, {}, tmp_path, 2, 'CAL')
    receipt=R.C.read(tmp_path/'CAL_FAILURE_RECEIPT.json')
    assert receipt['pending_ids']==['id1'] and receipt['failed_id']=='id0'
    assert len(instances[0].futures)==2
    assert instances[0]._processes[1].terminated is True
    assert instances[0].shutdown_calls==[dict(wait=False,cancel_futures=True)]


def test_CAL53_snapshot_rejects_changed_spool_instead_of_resealing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(R.C,'ROOT',tmp_path)
    entries=[]
    for i in range(46):
        path=R.OLD._row_path(tmp_path,'CAL'+str(i));path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'synthetic frozen spool')
        entries.append(dict(id='CAL'+str(i),sha256=R.C.sha(path)))
    for name in ('PROTOCOL.json','PREPARATION_SEAL.json','CAL_EXECUTION_PROTOCOL.json',
            'CAL_OBSERVATIONS.jsonl.gz','CAL_SUMMARY.json','CAL_OBSERVATION_SEAL.json'):
        (tmp_path/name).write_bytes(b'synthetic metadata')
    R.C.write(tmp_path/'CAL_ROW_INDEX.json',dict(entries=entries))
    before=R._cal_immutable(tmp_path)
    assert len(before)==53
    R.OLD._row_path(tmp_path,entries[0]['id']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='sealed CAL spool bytes changed'):
        R._cal_immutable(tmp_path)


def test_final238_is_exact_two_member_copy_with_perrow_canonical_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(R.C, 'ROOT', tmp_path)
    rows=[_row('CAL'+str(i))for i in range(46)]
    rows += [_row('TEST'+str(i),'TEST',i%2==0)for i in range(49)]
    rows += [_row('R'+str(i),'REMAINDER',i%3==0)for i in range(143)]
    for name in ('PROTOCOL.json','PREPARATION_SEAL.json','REST_PROTOCOL.json','REST_PREPARATION_SEAL.json',
            'CAL_EXECUTION_PROTOCOL.json','REST_EXECUTION_PROTOCOL.json'):
        R.C.write(tmp_path/name,dict(synthetic=name))
    expected=[]
    for part, selected in (('CAL',rows[:46]),('REST',rows[46:])):
        journal=tmp_path/(part+'_OBSERVATIONS.jsonl.gz');R.save_rows_large(journal,selected)
        expected.append(journal.read_bytes())
        index=tmp_path/(part+'_ROW_INDEX.json');R.C.write(index,dict(entries=[_entry(r)for r in selected]))
        R.C.write(tmp_path/(part+'_OBSERVATION_SEAL.json'),dict(rows=len(selected),ordered_ids=[r['id']for r in selected],
            row_index_sha256=R.C.sha(index),observations_sha256=R.C.sha(journal),
            selection_profile=R.SELECTION_PROFILE,raw_observation_generation=R.RAW_OBSERVATION_GENERATION,
            protocol_sha256=R.C.sha(tmp_path/('PROTOCOL.json'if part=='CAL'else'REST_PROTOCOL.json')),
            preparation_seal_sha256=R.C.sha(tmp_path/('PREPARATION_SEAL.json'if part=='CAL'else'REST_PREPARATION_SEAL.json')),
            execution_protocol_sha256=R.C.sha(tmp_path/(part+'_EXECUTION_PROTOCOL.json'))))
    protocol=dict(ids=[r['id']for r in rows],CAL_ids=[r['id']for r in rows[:46]],REST_ids=[r['id']for r in rows[46:]],
        source_sha256={},input_sha256={},protected_user_sha256={},original_CAL_file_sha256={},original_parent_immutable_file_sha256={})
    monkeypatch.setattr(R,'_load_rest',lambda output,sha:protocol)
    R.finalize(root_finalize_GO=True,expected_central_plan_sha256='a'*64,output=tmp_path)
    assert (tmp_path/'OBSERVATIONS.jsonl.gz').read_bytes()==b''.join(expected)
    seal=R.C.read(tmp_path/'OBSERVATION_SEAL.json')
    assert seal['rows']==238 and seal['canonical_validation']['canonical_row_digests_verified']==238
    assert seal['final_JSON_reencoding_or_compression_calls']==0
    assert [r['id']for r in R.C.rows(tmp_path/'OBSERVATIONS.jsonl.gz')]==protocol['ids']
    assert seal['partition_seals']=={part:R.C.sha(tmp_path/(part+'_OBSERVATION_SEAL.json'))for part in ('CAL','REST')}
    with pytest.raises(FileExistsError):
        R.finalize(root_finalize_GO=True,expected_central_plan_sha256='a'*64,output=tmp_path)
