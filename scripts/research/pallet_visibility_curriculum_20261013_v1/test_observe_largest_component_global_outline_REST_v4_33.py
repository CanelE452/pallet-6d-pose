"""Toy metadata/masks/storage only; no live CAL/REST dataset IO."""
import ast
from copy import deepcopy
import gzip
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from . import observe_largest_component_global_outline_REST_v4_33 as M
from .test_observe_largest_component_global_outline_CAL_v4_33 import toy


def rest_toy(index=0, *, empty=False, tie=False):
    packet,mask,binding=toy(index,empty=empty,tie=tie)
    partition='TEST' if index<49 else 'REMAINDER'
    identifier=f'{partition}:목재:{index:03d}'
    packet['identifiers'].update(id=identifier,partition=partition)
    binding.update(id=identifier,partition=partition)
    binding['positive_packet_sha256']=M._digest(packet)
    return packet,mask,binding


def test_process_source_exact_CAL_AST_except_two_partition_metadata_expressions():
    source=ast.parse(inspect.getsource(M.process_one))
    parent=ast.parse(inspect.getsource(M.CAL.process_one));changed=0
    # The first comparison's RHS and returned partition are the only changes.
    compare=next(node for node in ast.walk(source) if isinstance(node,ast.Compare)
        and ast.unparse(node.left)=="packet['identifiers']['partition']")
    assert ast.unparse(compare.comparators[0])=="binding['partition']"
    compare.comparators[0]=ast.Constant('CAL');changed+=1
    keyword=next(node for node in ast.walk(source) if isinstance(node,ast.keyword) and node.arg=='partition')
    assert ast.unparse(keyword.value)=="binding['partition']"
    keyword.value=ast.Constant('CAL');changed+=1
    assert changed==2
    assert ast.dump(source,include_attributes=False)==ast.dump(parent,include_attributes=False)


def test_plain_source_has_no_dynamic_AST_compile_FunctionType_or_parent_patch():
    source=Path(M.__file__).read_text()
    tree=ast.parse(source)
    calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not set(calls)&{'compile','exec','eval','FunctionType','types.FunctionType'}
    before=M.CAL.process_one
    packet,mask,binding=rest_toy();M.process_one(packet,mask,binding)
    assert M.CAL.process_one is before


def test_all192_uniform_rows_include_TEST_REMAINDER_empty_and_tie():
    results=[]
    for i in range(192):
        packet,mask,binding=rest_toy(i,empty=i==63,tie=i==113)
        results.append(M.process_one(packet,mask,binding))
    assert len(results)==192
    assert [r['partition'] for r in results]==['TEST']*49+['REMAINDER']*143
    assert all(row['row_retained'] and row['union_exact_match'] for row in results)
    assert not results[63]['largest_component']['supported']
    assert len(results[113]['largest_component']['foreground_component_selection']['maximum_area_tied_labels'])==2


def test_original_packet_and_mask_unchanged_and_union_mismatch_retained():
    packet,mask,binding=rest_toy();before=deepcopy(packet);pixels=mask.tobytes();mask.setflags(write=False)
    row=M.process_one(packet,mask,binding)
    assert row['union_exact_match'] and packet==before and mask.tobytes()==pixels
    packet['regularizers']['outline_observation']['observed_boundary_samples_xy'][0][0]+=.1
    binding['positive_packet_sha256']=M._digest(packet)
    row=M.process_one(packet,mask,binding)
    assert not row['union_exact_match'] and row['stored_union_on_mismatch'] is not None


def test_REST_reader_skips_CAL_numeric_decode_and_reads_exact192(tmp_path,monkeypatch):
    path=tmp_path/'positive.jsonl.gz';bindings=[]
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for _ in range(46):stream.write('NOT_JSON_CAL_PACKET_MUST_NOT_BE_DECODED\n')
        for i in range(192):
            packet,_,binding=rest_toy(i);bindings.append(binding);stream.write(json.dumps(packet)+'\n')
    original=json.loads;calls=[]
    def decode(text,*a,**k):calls.append(text);return original(text,*a,**k)
    monkeypatch.setattr(M.json,'loads',decode)
    rows=list(M.REST_packets(path,bindings))
    assert len(rows)==len(calls)==192
    assert rows[0]['identifiers']['partition']=='TEST' and rows[-1]['identifiers']['partition']=='REMAINDER'
    assert not any('NOT_JSON_CAL' in value for value in calls)


def test_REST_reader_rejects_extra_or_changed_order_packets(tmp_path):
    path=tmp_path/'positive.jsonl.gz';fixtures=[rest_toy(i) for i in range(192)];bindings=[b for _,_,b in fixtures]
    with gzip.open(path,'wt')as stream:
        stream.write('opaque CAL\n'*46)
        for packet,_,_ in fixtures:stream.write(json.dumps(packet)+'\n')
        stream.write('{}\n')
    with pytest.raises(ValueError,match='extra'):list(M.REST_packets(path,bindings))
    bindings[0],bindings[1]=bindings[1],bindings[0]
    with pytest.raises(ValueError,match='identity/order'):list(M.REST_packets(path,bindings))


def metadata():
    ids=[f'CAL:{i}' for i in range(46)]+[f'TEST:{i}' for i in range(49)]+[f'REMAINDER:{i}' for i in range(143)]
    records=[dict(id=identifier,partition='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER',
        positive_packet_sha256=hashlib.sha256(identifier.encode()).hexdigest()) for i,identifier in enumerate(ids)]
    original=[];fresh=[]
    for i,record in enumerate(records):
        value=dict(id=record['id'],image=f'image{i}.png',image_sha256='a'*64,mask_npz=f'mask{i}.npz',mask_file_sha256='b'*64)
        if i<95:value['partition']=record['partition'];original.append(value)
        else:fresh.append(value)  # Actual fresh143 manifest lacks the partition field.
    original[0].update(image={'unrequested_CAL':'poison'},mask_npz={'unrequested_CAL':'poison'})
    return dict(ordered_ids=ids,records=records),[(Path('original'),json.dumps(original)),(Path('fresh'),json.dumps(fresh))]


def test_two_flat_manifests_join192_skip_CAL_payload_and_support_actual143_shape(monkeypatch):
    monkeypatch.setattr(M,'_relative',lambda path:str(path))
    protocol,manifests=metadata();cases,audit=M.join_REST_bindings(protocol,manifests)
    assert len(cases)==192 and [row['id'] for row in cases]==protocol['ordered_ids'][46:]
    assert [row['partition'] for row in cases]==['TEST']*49+['REMAINDER']*143
    assert audit['unrequested_records_skipped_after_id']==46 and audit['nonREST_payload_fields_decoded']==0


def test_metadata_duplicate_and_missing_REST_mask_rejected(monkeypatch):
    monkeypatch.setattr(M,'_relative',lambda path:str(path))
    protocol,manifests=metadata()
    with pytest.raises(ValueError,match='Duplicate'):M.join_REST_bindings(protocol,manifests+[manifests[1]])
    with pytest.raises(ValueError,match='Every REST'):M.join_REST_bindings(protocol,manifests[:1])


@pytest.mark.parametrize('stage',['prepare','run','finalize'])
def test_stage_GO_and_central_format_before_IO(monkeypatch,stage):
    def forbidden(*a,**k):raise AssertionError('IO before ROOT stage GO')
    monkeypatch.setattr(M.C,'sha',forbidden);monkeypatch.setattr(M.C,'read',forbidden)
    kwargs=dict(central_plan='no-open.json',expected_central_sha256='a'*64,gate_binding=dict(path='gate.json',sha256='b'*64))
    if stage=='prepare':
        with pytest.raises(PermissionError):M.prepare(**kwargs,source_peer_binding={})
        kwargs['expected_central_sha256']='invalid'
        with pytest.raises(ValueError):M.prepare(**kwargs,source_peer_binding={},root_metadata_GO=True)
    else:
        kwargs['expected_preparation_sha256']='c'*64
        with pytest.raises(PermissionError):getattr(M,stage)(**kwargs)


def gate_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(M.C,'ROOT',tmp_path)
    output=tmp_path/'out';output.mkdir()
    M.C.write(output/'CAL_OBSERVATION_SEAL.json',dict(central_sha256='a'*64))
    audit=tmp_path/'audit.json';audit.write_text('{"source":"toy"}')
    seal=tmp_path/'auditseal.json';seal.write_text('{"source":"toy"}')
    gate=dict(schema='ROOT_CAL46_LCC33_INDEPENDENT_GATE_V1',status='PASS',rows=46,failure_count=0,
        union_exact_match_rows=46,central_plan_sha256='a'*64,
        CAL_observation_seal=dict(path='out/CAL_OBSERVATION_SEAL.json',sha256=M.C.sha(output/'CAL_OBSERVATION_SEAL.json')),
        independent_audit=dict(path='audit.json',sha256=M.C.sha(audit)),
        independent_audit_seal=dict(path='auditseal.json',sha256=M.C.sha(seal)))
    path=tmp_path/'gate.json';M.C.write(path,gate)
    return output,path,gate


def test_ROOT_gate_requires_exact_current_CAL_central_and_independent_refs(tmp_path,monkeypatch):
    output,path,gate=gate_fixture(tmp_path,monkeypatch)
    binding=dict(path='gate.json',sha256=M.C.sha(path))
    assert M._gate(binding,'a'*64,output)['union_exact_match_rows']==46
    for field,value in [('failure_count',1),('union_exact_match_rows',45),('central_plan_sha256','b'*64)]:
        broken={**gate,field:value};M.C.write(path,broken)
        with pytest.raises(ValueError):M._gate(dict(path='gate.json',sha256=M.C.sha(path)),'a'*64,output)
    M.C.write(path,gate);(tmp_path/'audit.json').write_text('changed')
    with pytest.raises(ValueError,match='audit byte'):M._gate(dict(path='gate.json',sha256=M.C.sha(path)),'a'*64,output)


def full_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(M.C,'ROOT',tmp_path)
    output=tmp_path/'outline';output.mkdir();ids=[];allentries=[]
    for partition,count in [('CAL',46),('REST',192)]:
        entries=[];rows=[];folder=output/('rows' if partition=='CAL' else 'rows_REST');folder.mkdir()
        for i in range(count):
            if partition=='CAL':
                packet,mask,binding=toy(i,empty=i==3);row=M.CAL.process_one(packet,mask,binding)
            else:
                packet,mask,binding=rest_toy(i,empty=i==13);row=M.process_one(packet,mask,binding)
            path=folder/f'{i:03d}.json';M.C.write(path,row);rows.append(row);ids.append(row['id'])
            entries.append(dict(id=row['id'],partition=row['partition'],path=M._relative(path),sha256=M.C.sha(path),canonical_row_sha256=M._digest(row)))
        with gzip.open(output/(partition+'_OUTLINE_OBSERVATIONS.jsonl.gz'),'wt',encoding='utf-8',compresslevel=1)as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        M.C.write(output/(partition+'_ROW_INDEX.json'),dict(entries=entries))
        protocol=output/('PREPARATION.json'if partition=='CAL'else'REST_PREPARATION.json');M.C.write(protocol,dict(source='toy'))
        M.C.write(output/(partition+'_OBSERVATION_SEAL.json'),dict(rows=count,ordered_ids=[row['id']for row in rows],
            selection_profile=M.SELECTION_PROFILE,union_exact_match_rows=count,protocol_sha256=M.C.sha(protocol),
            observations_sha256=M.C.sha(output/(partition+'_OUTLINE_OBSERVATIONS.jsonl.gz')),
            row_index_sha256=M.C.sha(output/(partition+'_ROW_INDEX.json'))))
        allentries.extend(entries)
    immutable={}
    for group in ('oldCAL','newCAL'):
        mapping={}
        for i in range(53):
            path=tmp_path/group/f'{i}.dat';path.parent.mkdir(exist_ok=True);path.write_text('immutable toy')
            mapping[M._relative(path)]=M.C.sha(path)
        immutable[group]=mapping
    prepared=dict(source_sha256={},input_sha256={},protected_user_sha256={},
        original_CAL_file_sha256=immutable['oldCAL'],immutable_LCC_CAL_file_sha256=immutable['newCAL'])
    central=dict(ordered_ids=ids)
    monkeypatch.setattr(M,'_prepared',lambda *a,**k:(central,prepared))
    return output,ids,allentries


def test_fake_full238_two_member_concat_exact_index_and_CAL_bytes_immutable(tmp_path,monkeypatch):
    output,ids,entries=full_fixture(tmp_path,monkeypatch)
    cal=(output/'CAL_OUTLINE_OBSERVATIONS.jsonl.gz').read_bytes();rest=(output/'REST_OUTLINE_OBSERVATIONS.jsonl.gz').read_bytes()
    result=M.finalize(central_plan='unused',expected_central_sha256='a'*64,gate_binding={},expected_preparation_sha256='b'*64,root_finalize_GO=True,output=output)
    assert result['rows']==238
    assert (output/'OUTLINE_OBSERVATIONS.jsonl.gz').read_bytes()==cal+rest
    assert (output/'CAL_OUTLINE_OBSERVATIONS.jsonl.gz').read_bytes()==cal
    assert M.C.read(output/'OUTLINE_ROW_INDEX.json')['entries']==entries
    seal=M.C.read(output/'OUTLINE_OBSERVATION_SEAL.json')
    assert seal['ordered_ids']==ids and seal['partitions']==dict(CAL=46,TEST=49,REMAINDER=143)
    assert seal['canonical_validation']==dict(rows=238,canonical_row_digests_verified=238,missing_duplicate_or_extra_rows=0)
    with pytest.raises(FileExistsError):M.finalize(central_plan='unused',expected_central_sha256='a'*64,gate_binding={},expected_preparation_sha256='b'*64,root_finalize_GO=True,output=output)


def test_finalizer_detects_stored_row_change_before_issuing_full_seal(tmp_path,monkeypatch):
    output,_,entries=full_fixture(tmp_path,monkeypatch)
    M._resolve(entries[61]['path']).write_text('{"corrupted":true}')
    with pytest.raises(ValueError,match='row changed'):
        M.finalize(central_plan='unused',expected_central_sha256='a'*64,gate_binding={},expected_preparation_sha256='b'*64,root_finalize_GO=True,output=output)
    assert not (output/'OUTLINE_OBSERVATION_SEAL.json').exists()
