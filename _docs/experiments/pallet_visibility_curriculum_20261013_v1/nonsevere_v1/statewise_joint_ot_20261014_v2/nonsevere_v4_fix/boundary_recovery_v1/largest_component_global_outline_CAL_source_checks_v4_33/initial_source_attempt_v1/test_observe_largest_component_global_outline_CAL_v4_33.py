"""Synthetic metadata and masks only; no existing dataset IO or model loads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from . import observe_largest_component_global_outline_CAL_v4_33 as M


def toy(index=0, *, empty=False, tie=False):
    mask=np.zeros((48,64),np.float32)
    if not empty:
        mask[26:35,24:38]=1.
        if tie: mask[9:18,40:54]=1.
        else: mask[10:12,10:13]=1.
    identifier=f'CAL:{index:02d}'
    binding=dict(id=identifier,partition='CAL',image_binding=dict(path='images/toy.png',sha256='a'*64),
        target_mask_binding=dict(path=f'masks/toy{index}.npz',sha256='b'*64))
    image=dict(image_sha256='a'*64,mask_file_sha256='b'*64,mask_npz=f'toy{index}.npz')
    observation=M.LCC.frozen_union_observer(mask,mask.shape,image_binding=image)
    packet=dict(schema='SYNTHETIC_POSITIVE_SEVEN_KEY_PACKET',
        identifiers=dict(id=identifier,partition='CAL',image_sha256='a'*64),
        clean_input=dict(unused_states=['B','S','O','U']),
        physics=dict(image_hw=[48,64]),typed_bundles=None,
        regularizers=dict(o_regions=[],o_parameters=None,original_S_prepared=None,
            outline_observation=observation,outline_configuration=deepcopy(observation['configuration'])),
        source_contract=dict(ready=True))
    assert set(packet)=={'schema','identifiers','clean_input','physics','typed_bundles','regularizers','source_contract'}
    assert 'partition' not in packet
    binding['positive_packet_sha256']=M._digest(packet)
    return packet,mask,binding


def test_exact_seven_key_packet_joins_and_all_union_fields_match():
    packet,mask,binding=toy()
    result=M.process_one(packet,mask,binding)
    assert result['id']==packet['identifiers']['id'] and result['partition']=='CAL'
    assert result['union_exact_match'] and all(result['union_per_field_equal'].values())
    assert result['stored_union_sha256']==result['regenerated_union_sha256']
    assert result['selection_profile']==M.LCC.SELECTION_PROFILE
    assert result['baseline_profile']==M.LCC.BASELINE_PROFILE
    assert result['largest_component']['mask_area_px'] < result['original_union']['mask_area_px']


def test_whole_CAL46_is_processed_in_order_including_empty_and_tied_cases():
    fixtures=[toy(i,empty=i==17,tie=i==31) for i in range(46)]
    packets=[packet for packet,_,_ in fixtures];ids=[packet['identifiers']['id'] for packet in packets]
    records=[M.process_one(packet,fixtures[i][1],fixtures[i][2])
        for i,packet in enumerate(M.CAL_prefix(iter(packets),ids))]
    assert len(records)==46 and [row['id'] for row in records]==ids
    assert all(row['row_retained'] and row['union_exact_match'] for row in records)
    assert not records[17]['original_union']['supported']
    assert not records[17]['largest_component']['supported']
    assert len(records[31]['largest_component']['foreground_component_selection']['maximum_area_tied_labels'])==2


def test_prefix_closes_without_requesting_TEST_payload_47():
    packet,_,_=toy();seen=[];closed=[]
    def stream():
        try:
            for i in range(46):
                row=deepcopy(packet);row['identifiers']['id']=f'CAL:{i:02d}';seen.append(i);yield row
            raise AssertionError('A TEST/REMAINDER numeric payload was requested')
        finally: closed.append(True)
    assert len(list(M.CAL_prefix(stream(),[f'CAL:{i:02d}' for i in range(46)])))==46
    assert seen==list(range(46)) and closed==[True]


def test_nonCAL_or_wrong_order_packet_fails_without_routing():
    packet,_,_=toy();packet['identifiers']['partition']='TEST'
    with pytest.raises(ValueError,match='CAL46 prefix'):
        list(M.CAL_prefix(iter([packet]),[f'CAL:{i:02d}' for i in range(46)]))


def test_union_mismatch_is_preserved_in_row_instead_of_silently_accepted():
    packet,mask,binding=toy()
    packet['regularizers']['outline_observation']['observed_boundary_samples_xy'][0][0]+=.25
    binding['positive_packet_sha256']=M._digest(packet)
    row=M.process_one(packet,mask,binding)
    assert not row['union_exact_match']
    assert not row['union_per_field_equal']['observed_boundary_samples_xy']
    assert row['stored_union_on_mismatch']==packet['regularizers']['outline_observation']


def test_different_cached_pixels_fail_full_union_hash_comparison():
    packet,mask,binding=toy();mask[10,10]=0.
    row=M.process_one(packet,mask,binding)
    assert not row['union_exact_match']
    assert not row['union_per_field_equal']['mask_array_sha256']


@pytest.mark.parametrize('field',['packet_digest','image','mask','canvas','configuration'])
def test_exact_positive_mask_image_canvas_and_config_join_rejects_mismatch(field):
    packet,mask,binding=toy()
    if field=='packet_digest': binding['positive_packet_sha256']='c'*64
    if field=='image': binding['image_binding']['sha256']='c'*64
    if field=='mask': binding['target_mask_binding']['sha256']='c'*64
    if field=='canvas': packet['physics']['image_hw']=[47,64]
    if field=='configuration': packet['regularizers']['outline_configuration']['normal_sigma_px']=2.
    if field!='packet_digest': binding['positive_packet_sha256']=M._digest(packet)
    with pytest.raises(ValueError): M.process_one(packet,mask,binding)


def test_readonly_cached_mask_and_original_packet_are_not_modified():
    packet,mask,binding=toy();before=deepcopy(packet);pixels=mask.tobytes();mask.setflags(write=False)
    M.process_one(packet,mask,binding)
    assert packet==before and mask.tobytes()==pixels


def test_supported_union_can_become_unsupported_LCC_without_fallback_or_row_drop():
    packet,mask,binding=toy();mask[:]=0.
    mask[10,10]=mask[20,30]=mask[34,17]=1.
    image=packet['regularizers']['outline_observation']['image_binding']
    saved=M.LCC.frozen_union_observer(mask,mask.shape,image_binding=image)
    packet['regularizers']['outline_observation']=saved
    packet['regularizers']['outline_configuration']=deepcopy(saved['configuration'])
    binding['positive_packet_sha256']=M._digest(packet)
    row=M.process_one(packet,mask,binding)
    assert row['row_retained'] and row['union_exact_match']
    assert row['original_union']['supported']
    assert not row['largest_component']['supported']
    assert row['largest_component']['foreground_component_selection']['retained_foreground_area_px']==1


def metadata():
    ids=[f'CAL:{i:02d}' for i in range(46)]
    entries=[dict(id=ident,partition='CAL') for ident in ids]
    seal=dict(rows=46,ordered_ids=ids)
    allids=ids+[f'PRIMARY:{i}' for i in range(192)]
    records=[dict(id=ident,partition='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER',
        positive_packet_sha256=hashlib.sha256(ident.encode()).hexdigest()) for i,ident in enumerate(allids)]
    protocol=dict(ordered_ids=allids,records=records)
    positive=dict(rows=238,ordered_ids=allids)
    mask=[dict(id=ident,partition='CAL',image=f'images/{i}.png',image_sha256='a'*64,
        mask_npz=f'{i}.npz',mask_file_sha256='b'*64,provenance={'raw':{'forbidden':'never decoded'}}) for i,ident in enumerate(ids)]
    # This nonCAL record's requested fields are objects and would fail scalar decoding.
    mask.insert(3,dict(id='PRIMARY:skip',image={'poison':'unrequested'},mask_npz={'poison':'unrequested'}))
    return dict(entries=entries),seal,json.dumps(mask),protocol,positive


def test_flat_manifest_join_keeps_exact46_and_skips_nonCAL_after_ID():
    index,seal,manifest,protocol,positive=metadata()
    cases,inventory=M.join_bindings(index,seal,manifest,protocol,positive)
    assert len(cases)==46 and [row['id'] for row in cases]==seal['ordered_ids']
    assert inventory['nonCAL_image_mask_or_payload_values_decoded']==0
    assert inventory['provider_provenance_raw_points_states_or_line_payloads_decoded']==0
    assert inventory['nonCAL_records_skipped_after_id']==1
    assert [row['positive_packet_sha256'] for row in cases]==[row['positive_packet_sha256'] for row in protocol['records'][:46]]


def test_metadata_join_rejects_altered_prefix_or_packet_record_partition():
    args=list(metadata());args[3]['records'][7]['partition']='TEST'
    with pytest.raises(ValueError):M.join_bindings(*args)
    args=list(metadata());args[4]['ordered_ids']=list(reversed(args[4]['ordered_ids']))
    with pytest.raises(ValueError):M.join_bindings(*args)


@pytest.mark.parametrize('stage',['prepare','run'])
def test_ROOT_GO_and_central_SHA_checks_precede_file_IO(monkeypatch,stage):
    def forbidden(*args,**kwargs):raise AssertionError('IO was attempted before GO/central format')
    monkeypatch.setattr(M.C,'sha',forbidden);monkeypatch.setattr(M.C,'read',forbidden)
    kwargs=dict(central_plan='must-not-open.json',expected_central_sha256='a'*64)
    if stage=='prepare':
        with pytest.raises(PermissionError):M.prepare(**kwargs)
        kwargs['expected_central_sha256']='invalid'
        with pytest.raises(ValueError):M.prepare(**kwargs,root_metadata_GO=True)
    else:
        kwargs['expected_preparation_sha256']='b'*64
        with pytest.raises(PermissionError):M.run(**kwargs)
        kwargs['expected_central_sha256']='invalid'
        with pytest.raises(ValueError):M.run(**kwargs,root_CAL_mask_GO=True)


def test_source_map_union_conflicts_fail_closed():
    assert M._union({'same':'a'*64},{'same':'a'*64,'other':'b'*64})=={'same':'a'*64,'other':'b'*64}
    with pytest.raises(ValueError):M._union({'same':'a'*64},{'same':'b'*64})


def test_CALS53_inventory_and_each_spool_hash_are_checked_without_decoding(monkeypatch):
    ids=[f'CAL:{i:02d}' for i in range(46)]
    index=dict(entries=[dict(id=ident,sha256='a'*64) for ident in ids])
    monkeypatch.setattr(M,'_relative',lambda path:str(path))
    monkeypatch.setattr(M.C,'sha',lambda path:'a'*64)
    result=M._CAL53(Path('synthetic'),index)
    assert len(result)==53
    index['entries'][9]['sha256']='b'*64
    with pytest.raises(ValueError,match='spool changed'):M._CAL53(Path('synthetic'),index)


def test_process_has_no_pose_state_model_or_E_inputs_in_its_interface():
    import inspect
    assert list(inspect.signature(M.process_one).parameters)==['packet','mask','binding']
    source=inspect.getsource(M.process_one)
    for token in ("['clean_input']","['typed_bundles']",'PnP','optimizer','new_model','E_COMPARISON'):
        assert token not in source
