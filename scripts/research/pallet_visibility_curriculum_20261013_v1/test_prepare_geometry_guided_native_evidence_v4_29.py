"""CAL/runtime metadata join and new fitted-population schema, synthetic only."""
from copy import deepcopy
import numpy as np
import pytest
from . import prepare_geometry_guided_native_evidence_v4_29 as P
from . import calibrate_geometry_guided_edge_noise_v4_29 as CAL
from .test_geometry_guided_edge_authentication_v4_29 import fixture


def rows():
    base,obs,kw,_=fixture();geometry=kw['initial_geometry']
    family=dict(name='ORIGINAL_WD',native_xyz=base.xyz.tolist(),physical_registration_Q=base.Q.tolist(),
                coarse_pose=dict(R=geometry['R_cf'],t=geometry['centroid']))
    visible=[deepcopy(a['source_record'])for a in base.prepartial_bound_source_archives.values()]
    task=dict(id='synthetic:cal',partition='CAL',session='synthetic',states=base.states,
        initial_self_hidden=geometry['initial_self_hidden'],image_sha256='a'*64,
        spec=dict(K=base.K.tolist(),states=base.states,visible=visible,domains=[],image_hw=base.image_hw,
                  config=dict(prior_weight=0.,analytic_jacobian=True)),boundary_merged=[],u_regions=[],
        families=dict(ORIGINAL_WD=family),complete_BI_evidence=dict(initial_projection_xy=geometry['initial_projection_xy']))
    row=dict(id=task['id'],partition='CAL',session=task['session'],states=task['states'],visible_measurements=deepcopy(visible),
        domains=[],u_regions=[],image_sha256='a'*64,initial_geometry=deepcopy(geometry),observation=obs)
    return task,row


def test_prepared_profiles_match_actual_fitter_population_contract_without_labels():
    task,row=rows();prepared,metadata=P.prepare_one(task,row)
    checked=CAL.validate_prepared_row(prepared)
    assert set(checked['profiles'])==set(P.G.SOURCE_POLICIES)
    assert metadata['profiles']['ALL_BI_REPLACE']['groups']==1
    assert metadata['profiles']['ALL_BI_REPLACE']['tangents']==2
    assert metadata['profiles']['RGB_NONFALLBACK_PRESERVE']['groups']==0
    for tangent in prepared['profiles']['ALL_BI_REPLACE']['tangent_records']:
        component=prepared['profiles']['ALL_BI_REPLACE']['groups'][0]['components'][0]
        assert tangent['selected_line_ref']==component['component_id']
        assert abs(np.asarray(tangent['selected_line_tangent_unit_xy'])@component['normal_unit_xy'])<1e-12


@pytest.mark.parametrize('change',['rotation','center','q','U','image','projection'])
def test_cal_runtime_input_changes_are_rejected_before_planning(change):
    task,row=rows()
    if change=='rotation':row['initial_geometry']['R_cf'][0][0]+=.01
    elif change=='center':row['initial_geometry']['centroid'][2]+=.01
    elif change=='q':row['visible_measurements'][0]['xy'][0]+=.01
    elif change=='U':row['u_regions']=[dict(native_corner=1)]
    elif change=='image':row['image_sha256']='b'*64
    else:row['initial_geometry']['initial_projection_xy'][0][0]+=.01
    with pytest.raises(ValueError):P.prepare_one(task,row)


def test_CAL_preparer_refuses_primary_row_before_coordinate_fields():
    with pytest.raises(ValueError,match='CAL46'):
        P.prepare_one(dict(id='x',partition='TEST'),dict(id='x',partition='TEST'))
