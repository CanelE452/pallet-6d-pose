from copy import deepcopy
import numpy as np
import pytest
from . import diagnose_CAL_geometry_guided_noise_v4_29 as P


def records():
    groups=lambda e,c:dict(physical_native_edge=[0,1],components=[dict(normal_error_px=e,component_id=c)])
    values=[dict(id='b',partition='CAL',compatible=True,session='one',groups=[groups(-10,'b')]),
        dict(id='a',partition='CAL',compatible=True,session='one',groups=[groups(10,'a')]),
        dict(id='c',partition='CAL',compatible=False,session='two',groups=[groups(999,'x')]),
        dict(id='d',partition='CAL',compatible=True,session='two',groups=[groups(3,'d'),groups(2,'e')])]
    return {policy:deepcopy(values)for policy in P.D.M.SOURCE_POLICIES}


def test_fixed_top3_abs_errors_ties_use_metadata_and_incompatible_excluded():
    data=records();before=deepcopy(data);result=P.selection(data)
    assert len(result)==6 and [r['id']for r in result]==['a','b','d']*2
    assert data==before and [r['policy_rank']for r in result]==[1,2,3]*2


def test_primary_row_rejected_before_diagnostic_selection():
    data=records();data[P.D.M.SOURCE_POLICIES[0]][0]['partition']='TEST'
    with pytest.raises(ValueError,match='CAL-only'):P.selection(data)


def test_segment_uses_existing_measured_TLS_actual_extent_only():
    edge=dict(normal_unit_xy=[0.,1.],offset_px=-8.,actual_inlier_xy=[[2.,8.1],[4.,7.9],[9.,8.]])
    result=P.segment(edge)
    assert np.array_equal(result,np.array([[9.,8.],[2.,8.]]))
