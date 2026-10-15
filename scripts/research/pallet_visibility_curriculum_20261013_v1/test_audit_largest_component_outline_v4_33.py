from copy import deepcopy

import numpy as np
import pytest

from . import audit_largest_component_outline_v4_33 as A
from . import largest_component_global_outline_v4_33 as P


def row(mask):
    value=P.observe_union_and_largest_component(mask, mask.shape)
    union={k:v for k,v in value['original_union'].items() if k!='selection_profile'}
    return dict(value,selection_profile=P.SELECTION_PROFILE,baseline_profile=P.BASELINE_PROFILE,
                stored_union_sha256=A.digest(union),regenerated_union_sha256=A.digest(union),
                union_exact_match=True,union_per_field_equal={k:True for k in union})


@pytest.mark.parametrize('kind',['empty','one','diagonal','tie','hole','frame','single','triangle'])
def test_independent_component_and_segment_replay(kind):
    mask=np.zeros((48,64),float)
    if kind=='one':mask[12:26,13:32]=1;mask[4:6,42:46]=1
    elif kind=='diagonal':mask[12:17,12:17]=1;mask[17:22,17:22]=1
    elif kind=='tie':mask[13:17,13:17]=1;mask[31:35,41:45]=1
    elif kind=='hole':mask[12:34,14:48]=1;mask[17:29,20:43]=0
    elif kind=='frame':mask[0:35,0:35]=1
    elif kind=='single':mask[13,13]=.5
    elif kind=='triangle':
        for y in range(10,31):mask[y,10:10+y]=1
    result=A.audit_row(mask,row(mask))
    assert result['failure_count']==0,result['checks']
    assert result['maximum_coordinate_difference']<1e-9


@pytest.mark.parametrize('field',['label','pixels','samples'])
def test_corrupt_stored_result_is_detected(field):
    mask=np.zeros((48,64),float);mask[12:25,15:37]=1;mask[32:34,43:46]=1
    value=deepcopy(row(mask))
    if field=='label':value['largest_component']['foreground_component_selection']['selected_label']=2
    elif field=='pixels':value['largest_component']['foreground_component_selection']['selected_binary_pixel_sha256']='0'*64
    else:value['largest_component']['observed_boundary_samples_xy'][0][0]+=1
    assert A.audit_row(mask,value)['failure_count']>0


def test_audit_uses_neither_new_producer_nor_old_observer(monkeypatch):
    mask=np.zeros((48,64),float);mask[12:25,15:37]=1
    value=row(mask)
    def forbidden(*args,**kwargs):raise AssertionError('Producer/observer must not certify itself')
    monkeypatch.setattr(P,'observe_union_and_largest_component',forbidden)
    monkeypatch.setattr(P,'frozen_union_observer',forbidden)
    monkeypatch.setattr(P.FROZEN_OUTLINE,'observe_global_outline',forbidden)
    assert A.audit_row(mask,value)['failure_count']==0


def test_actual_audit_GO_and_explicit_pins_before_IO(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No IO before ROOT GO and pins')
    monkeypatch.setattr(A,'sha_file',forbidden)
    monkeypatch.setattr(A,'read',forbidden)
    with pytest.raises(PermissionError):
        A.audit_CAL(folder='unused',expected_seal_sha256=None,expected_central_sha256=None)
    with pytest.raises(ValueError):
        A.audit_CAL(folder='unused',expected_seal_sha256=None,expected_central_sha256=None,
                    root_CAL_observation_audit_GO=True)
