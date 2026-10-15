"""CAL46 manifest/operator contracts; no real arrays, models or labels."""
import json
from pathlib import Path

import pytest

from . import capture_generic_instance_mask_population_CAL_v1 as M


def fixture():
    ids=[f'CAL:{i:02d}'for i in range(46)]
    index={'entries':[dict(id=ident,partition='CAL',unused_line_summary={'not':'read'})for ident in ids]}
    seal=dict(rows=46,ordered_ids=ids)
    records=[dict(id=ident,partition='CAL',image=f'data/cal{i}.jpg',image_sha256='a'*64,
        mask_npz=f'cal{i}.npz',mask_file_sha256='b'*64,
        provider_provenance=dict(raw_points=['FORBIDDEN_PAYLOAD'],states=['S','U'],geometry=['FORBIDDEN_PAYLOAD']))
        for i,ident in enumerate(ids)]
    return ids,index,seal,records


def test_uniform_CAL46_metadata_preserves_index_order_ignoring_manifest_order():
    ids,index,seal,records=fixture()
    rows,audit=M.join_CAL_bindings(index,seal,json.dumps(list(reversed(records))))
    assert [r['id']for r in rows]==ids and len(rows)==46
    assert rows[0]['image_binding']==dict(path='data/cal0.jpg',sha256='a'*64)
    assert set(rows[0])=={'id','partition','image_binding','target_mask_binding'}
    assert audit['provider_provenance_raw_points_states_or_line_payloads_decoded']==0


def test_nonCAL_skips_image_mask_strings_and_all_payload_decoding(monkeypatch):
    ids,index,seal,records=fixture()
    records.append(dict(id='TEST:never_query',partition='TEST',image='FORBIDDEN_PAYLOAD',
        image_sha256='FORBIDDEN_PAYLOAD',mask_npz='FORBIDDEN_PAYLOAD',mask_file_sha256='FORBIDDEN_PAYLOAD',
        nested={'q':['FORBIDDEN_PAYLOAD']}))
    original=M.json.JSONDecoder
    class Decoder(original):
        def raw_decode(self,text,idx=0):
            if text[idx:].startswith('"FORBIDDEN_PAYLOAD"'):raise AssertionError('Unrequested value decoded')
            return super().raw_decode(text,idx)
    monkeypatch.setattr(M.json,'JSONDecoder',Decoder)
    rows,audit=M.join_CAL_bindings(index,seal,json.dumps(records))
    assert len(rows)==46 and audit['nonCAL_records_skipped_after_id']==1
    assert audit['nonCAL_image_mask_or_payload_values_decoded']==0


@pytest.mark.parametrize('change',['missing','duplicate','wrong_partition','wrong_order','wrong_count'])
def test_population_identity_failclosed(change):
    ids,index,seal,records=fixture()
    if change=='missing':records.pop()
    elif change=='duplicate':records.append(records[0])
    elif change=='wrong_partition':records[0]['partition']='TEST'
    elif change=='wrong_order':seal['ordered_ids']=list(reversed(ids))
    else:seal['rows']=45
    with pytest.raises(ValueError):M.join_CAL_bindings(index,seal,json.dumps(records))


def test_extra_nonCAL_metadata_does_not_add_cases():
    ids,index,seal,records=fixture()
    records += [dict(id=f'TEST:{i}',partition='TEST',image='unread')for i in range(49)]
    rows,audit=M.join_CAL_bindings(index,seal,json.dumps(records))
    assert len(rows)==46 and audit['manifest_records_scanned']==95
    assert audit['nonCAL_records_skipped_after_id']==49


def test_mask_archive_path_traversal_rejected():
    ids,index,seal,records=fixture();records[0]['mask_npz']='../other.npz'
    with pytest.raises(ValueError,match='inside'):M.join_CAL_bindings(index,seal,json.dumps(records))


def test_skipped_nested_strings_brackets_escapes_unicode_and_scalar_not_decoded():
    raw=json.dumps(dict(id='CAL:01',unused=dict(s='[} escaped\\"\\é',a=[1,None,False,{'z':'x'}]),partition='CAL'))
    assert M.flat_fields(raw,{'id','partition'})==dict(id='CAL:01',partition='CAL')
    assert list(M._records('['+raw+']'))==[raw]
    with pytest.raises(ValueError):M.flat_fields('{"id":"a","id":"b"}',{'id'})
    with pytest.raises(ValueError):list(M._records('['+raw+'] trailing'))
    with pytest.raises(ValueError):M.flat_fields('{"id":[1,2]}',{'id'})


def test_metadata_parser_does_not_decode_unrequested_numeric_arrays():
    # Deliberately non-JSON scalar cannot be passed to JSONDecoder unnoticed.
    raw='{"id":"CAL:01","unused":[POISON_NOT_DECODED,{"x":"} ["}],"partition":"CAL"}'
    assert M.flat_fields(raw,{'id','partition'})==dict(id='CAL:01',partition='CAL')


def test_infer_case_config_and_factory_bytecode_are_frozen_parent_exact():
    original=M.P.OUTPUT;namespace=dict(M.P._load_models.__globals__)
    assert M.infer_case is M.P.infer_case and M.CONFIGURATION is M.P.CONFIGURATION
    assert M.CONFIGURATION['conf']==.25 and M.CONFIGURATION['max_det']==20
    factory=M.model_factory(Path('/new-output'))
    assert factory.__code__ is M.P._load_models.__code__
    assert factory.__globals__['OUTPUT']==Path('/new-output')
    assert M.P.OUTPUT==original and M.P._load_models.__globals__==namespace


def test_runtime_config_directory_exists_before_factory_import(tmp_path):
    output=tmp_path/'new_attempt';output.mkdir()
    path=M.runtime_config_directory(output)
    assert path==output/'ultralytics_runtime_config/Ultralytics' and path.is_dir()
    with pytest.raises(FileExistsError):M.runtime_config_directory(output)


def test_all_GO_guards_before_IO(monkeypatch):
    def poison(*args,**kwargs):raise AssertionError('No IO or model construction')
    monkeypatch.setattr(M.C,'read',poison);monkeypatch.setattr(M.C,'sha',poison)
    monkeypatch.setattr(M,'model_factory',poison)
    with pytest.raises(PermissionError):M.prepare()
    with pytest.raises(PermissionError):M.run(expected_plan_sha256='a'*64)
    with pytest.raises(ValueError):M.run(expected_plan_sha256=None,root_CAL_actual_inference_GO=True)
