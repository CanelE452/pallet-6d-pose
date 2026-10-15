from copy import deepcopy

import pytest

from . import audit_largest_component_outline_REST_v4_33 as V


def fixture():
    ids=[f'SYNTHETIC:{i}' for i in range(238)]
    rows=[dict(id=i,partition='CAL' if n<46 else 'TEST' if n<95 else 'REMAINDER') for n,i in enumerate(ids)]
    central=dict(ordered_ids=ids,mask_source_packet_records=rows)
    final=dict(ordered_ids=ids,selection_profile=V.PROFILE,protected_user_sha256={str(i):'a'*64 for i in range(6)},
        original_CAL_file_sha256={str(i):'b'*64 for i in range(53)},immutable_LCC_CAL_file_sha256={str(i):'c'*64 for i in range(53)},source_sha256={'source':'d'*64})
    central.update({key:deepcopy(final[key]) for key in ('protected_user_sha256','original_CAL_file_sha256','source_sha256')})
    full=dict(rows=238,ordered_ids=ids,partitions=dict(CAL=46,TEST=49,REMAINDER=143),selection_profile=V.PROFILE,partition_seals=dict(CAL='e'*64))
    rest=dict(rows=192,ordered_ids=ids[46:],selection_profile=V.PROFILE,union_exact_match_rows=192)
    return central,final,full,rest,rows,dict(rows=46,failure_count=0,CAL_observation_seal_sha256='e'*64)


def test_gate_and_SHA_validation_before_IO(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs): raise AssertionError('IO before GO/SHA')
    monkeypatch.setattr(V.A,'sha_file',forbidden)
    args=dict(folder=tmp_path,expected_full_seal_sha256='a'*64,expected_central_sha256='b'*64,expected_cal_audit_sha256='c'*64)
    with pytest.raises(PermissionError): V.audit_rest(**args)
    args['expected_full_seal_sha256']='invalid'
    with pytest.raises(ValueError,match='before IO'): V.audit_rest(**args,root_REST_observation_audit_GO=True)


def test_exact_ordered_inventory_retains_every_row():
    values=fixture()
    ids,records,entries=V.validate_metadata(*values)
    assert len(ids)==len(records)==len(entries)==192
    assert ids==values[0]['ordered_ids'][46:]


@pytest.mark.parametrize('which',range(10))
def test_metadata_disagreement_rejected(which):
    values=list(deepcopy(fixture()))
    if which==0: values[0]['mask_source_packet_records'][46]['partition']='CAL'
    elif which==1: values[1]['immutable_LCC_CAL_file_sha256'].pop('0')
    elif which==2: values[2]['rows']=237
    elif which==3: values[3]['ordered_ids']=values[3]['ordered_ids'][:-1]
    elif which==4: values[4][95]['partition']='CAL'
    elif which==5: values[5]['failure_count']=1
    elif which==6: values[5]['CAL_observation_seal_sha256']='f'*64
    elif which==7: values[1]['protected_user_sha256']['0']='f'*64
    elif which==8: values[1]['original_CAL_file_sha256']['0']='f'*64
    else: values[1]['source_sha256']['source']='f'*64
    with pytest.raises(ValueError): V.validate_metadata(*values)
