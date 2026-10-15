"""Focused state/coordinate/factor gates; no RGB or pose fitting."""
from copy import deepcopy
import numpy as np
import pytest
from . import common as C
from . import joint_ot_pnp as P
from . import grazing_BI_uncertainty_v4_18 as G
from .test_geometric_visibility_uncertainty_v4_9 import fixture


def test_geometry_dot_product_uses_original_model_and_preserves_input():
    geometry,states=fixture();before=deepcopy(geometry);states[7]='I'
    result=G.apply_grazing_BI_uncertainty(geometry,states,geometry_tolerance_deg=2.)
    xyz=np.asarray(geometry['native_xyz']);ray=-np.asarray(geometry['centroid'])-xyz[7]
    cosine=max(P.registered_incident_normals(xyz)[7]@ray/np.linalg.norm(ray))
    assert result['records'][7]['maximum_incident_outward_normal_cosine']==cosine
    assert result['states'][7]=='U'and result['states'][6]=='S'
    assert geometry==before and result['raw_coordinate_reads']==result['RGB_queries']==0


@pytest.mark.parametrize('state',['O','T','U'])
def test_existing_nonBI_state_preserved_even_inside_the_same_band(state):
    geometry,states=fixture();states[7]=state
    result=G.apply_grazing_BI_uncertainty(geometry,states,geometry_tolerance_deg=2.)
    assert result['states'][7]==state and 7 not in result['new_U_native_ids']


def test_initialS_cannot_be_reopened():
    geometry,states=fixture();states[6]='I'
    with pytest.raises(ValueError,match='Initial S'):
        G.apply_grazing_BI_uncertainty(geometry,states,geometry_tolerance_deg=2.)


def test_receiver_and_source_query_gates_before_line_or_coordinate_payload():
    geometry,states=fixture();states=G.apply_grazing_BI_uncertainty(geometry,states,geometry_tolerance_deg=2.)['states'];states[0]='I'
    complete=[dict(native_corner=7,xy=object()),dict(native_corner=6,xy=object()),
        dict(native_corner=0,xy=[10,20],fit_covariance_px2=[[9,0],[0,9]])]
    kept_line=dict(provenance=dict(source_query_native_corner=0,native_neighbor=7),normal_unit_xy=[1,0])
    partial=[dict(native_corner=7,lines=object()),dict(native_corner=6,lines=object()),
        dict(native_corner=0,state='I',lines=[dict(provenance=dict(source_query_native_corner=7),normal_unit_xy=object()),kept_line])]
    result=G.filter_grazing_BI_reuse(complete,partial,states,[7],initial_self_hidden=geometry['initial_self_hidden'])
    assert result['visible']==[complete[2]] and result['visible'][0]is complete[2]
    assert result['boundary_observations'][0]['lines']==[kept_line]
    assert result['boundary_observations'][0]['state']=='I'


def test_new_U_factory_reads_only_new_nonS_center_keeps_old_region_exact():
    geometry,states=fixture();states[1]='U';states[7]='U'
    class Raw:
        def __len__(self):return 9
        def __getitem__(self,k):
            assert k==7,'Existing U/S/nonU coordinate unexpectedly read'
            return[-30.,20.]
    path=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/truncation_interval_model_CAL_v4_2/MODEL.json'
    old=dict(native_corner=1,state='U',payload=object())
    result=G.append_grazing_U_regions(geometry,states,[7],raw_native_points=Raw(),
        nonself_prediction_scale_px=100.,localization_model=C.read(path),existing_u_regions=[old])
    assert result['u_regions'][0]is old and result['new_U_raw_coordinate_reads']==1
    region=result['new_u_regions'][0]
    assert region['center_xy']==[-30.,20.]and region['halfplanes']==[]
    assert region['half_axes_px']==pytest.approx([54.4750754221928]*2)
    assert result['S_coordinate_reads']==0 and not result['U_canvas_or_visibility_guards_added']


def test_old_whole_edge_at_retained_BI_does_not_reuse_new_U_precise_mate():
    geometry,states=fixture();states[7]='U'
    full=dict(provenance=dict(native_neighbor=7,line_record=dict(native_corner=0,native_neighbor=7,
        whole_edge_search=True)),normal_unit_xy=object())
    local=dict(provenance=dict(native_neighbor=7,line_record=dict(native_corner=0,native_neighbor=7,
        whole_edge_search=False,neighbor_coordinate_reads=0,neighbor_corner_queries=0)),normal_unit_xy=[1,0])
    record=dict(native_corner=0,state='B',lines=[full,local])
    result=G.filter_grazing_BI_reuse([], [record],states,[7],initial_self_hidden=geometry['initial_self_hidden'])
    assert result['boundary_observations'][0]['lines']==[local]
    assert result['discarded_partial_records_or_lines'][0]['reason']=='historical_full_edge_used_new_U_precise_endpoint_not_reused'
