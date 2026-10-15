import pytest
from . import prepare_geometry_guided_native_evidence_v4_30 as P


def test_metadata_ROOT_GO_precedes_all_reads(monkeypatch):
    monkeypatch.setattr(P.C,'read',lambda path:pytest.fail('No metadata IO before explicit root GO'))
    with pytest.raises(PermissionError):P.main()


def test_CAL_population_rawgen_and_new_family_are_distinct():
    assert P.AUTH.PRODUCER_GENERATION=='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
    assert P.G.OBSERVATION_FAMILY=='OBSERVED_BI_ASSOCIATED_MULTIGUIDE_ACTUAL_NATIVE_EDGES_V30'
    assert P.G.POPULATION=='V30_ASSOCIATION_GATED_RUNTIME_ACTIVE_DETERMINISTIC_ONE_LINE_PER_PHYSICAL_EDGE'
    assert P.DECLARATION_SHA256=='1598f5ac0a4308ce758f792656f05bfba7f351adcfd2fc1b4000f564a33d22b1'


def synthetic_rows():
    from .test_prepare_geometry_guided_native_evidence_v4_29 import rows
    task,row=rows();row['selection_profile']=P.G.SELECTION_PROFILE
    obs=row['observation'];obs['selection_profile']=P.G.SELECTION_PROFILE
    obs['association_configuration']=dict(normal_search_radius_px=43.)
    obs['chain_observation']['configuration']=dict(obs['association_configuration'])
    return task,row


def test_two_source_profiles_and_actual_measured_tangent_axes_before_labels():
    import numpy as np
    task,row=synthetic_rows();prepared,meta=P.prepare_one(task,row)
    assert prepared['selection_profile']==P.G.SELECTION_PROFILE
    assert prepared['noise_population']==P.G.POPULATION
    assert prepared['observation_generation']!=prepared['raw_observation_generation']
    assert meta['profiles']['ALL_BI_REPLACE']['groups']==1
    assert meta['profiles']['ALL_BI_REPLACE']['tangents']==2
    assert meta['profiles']['RGB_NONFALLBACK_PRESERVE']['groups']==0
    line=prepared['profiles']['ALL_BI_REPLACE']['groups'][0]['components'][0]
    for t in prepared['profiles']['ALL_BI_REPLACE']['tangent_records']:
        assert abs(np.asarray(t['selected_line_tangent_unit_xy'])@line['normal_unit_xy'])<1e-12


def test_zero_selected_rows_stay_in_both_profiles_and_primary_gate_precedes_payload():
    task,row=synthetic_rows();obs=row['observation']
    obs['selected_chain_refs']=[];obs['joint_graph']['selected_line_names']=[]
    prepared,meta=P.prepare_one(task,row)
    assert set(prepared['profiles'])==set(P.G.SOURCE_POLICIES)
    assert all(not v['groups']and not v['tangent_records']for v in prepared['profiles'].values())
    with pytest.raises(ValueError,match='CAL46'):
        P.prepare_one(dict(partition='TEST'),dict(partition='TEST'))


def test_missing_new_selection_or_changed_q_fails_before_plan():
    task,row=synthetic_rows();row['selection_profile']='OLD'
    with pytest.raises(ValueError,match='V30'):P.prepare_one(task,row)
    task,row=synthetic_rows();row['visible_measurements'][0]['xy'][0]+=.1
    with pytest.raises(ValueError):P.prepare_one(task,row)
