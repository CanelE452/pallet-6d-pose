"""Geometry/state gates and cached observation reuse; no GT/image fits."""
from copy import deepcopy

import numpy as np
import pytest

from . import common as C
from . import joint_ot_pnp as P
from .geometric_visibility_uncertainty_v4_9 import apply_geometric_visibility_uncertainty as apply
from .geometric_visibility_uncertainty_v4_9 import filter_grazing_observation_reuse as filter_reuse


def fixture():
    xyz=C.cuboid(1.1,.11,1.3)
    # Camera lies almost on x=min face plane. Vertex7 faces away in y/z and
    # has a small positive x-normal cosine, so visibility is near grazing.
    camera=np.array([-.551,-.4,-2.]);t=-camera
    return {'native_xyz':xyz.tolist(),'R_cf':np.eye(3).tolist(),'centroid':t.tolist(),
            'initial_self_hidden':[False]*6+[True,False]},['B']*6+['S','B']


def test_initial_geometry_angle_matches_incident_normal_dot_and_newU():
    g,s=fixture();before=deepcopy(g);a=apply(g,s,geometry_tolerance_deg=2.)
    xyz=np.asarray(g['native_xyz']);ray=-np.asarray(g['centroid'])-xyz[7]
    expected=float(max(P.registered_incident_normals(xyz)[7]@ray/np.linalg.norm(ray)))
    assert a['records'][7]['maximum_incident_outward_normal_cosine']==expected
    assert np.isclose(a['records'][7]['signed_incident_angle_deg'],np.degrees(np.arcsin(expected)))
    assert a['states'][7]=='U' and a['states'][6]=='S' and g==before
    assert a['raw_coordinate_reads']==a['RGB_queries']==0


def test_S_state_record_not_read_and_no_S_reopening():
    g,s=fixture();records=[{} for _ in range(8)];records[6]=object()
    assert apply(g,s,geometry_tolerance_deg=2.,state_records=records)['states'][6]=='S'
    s[6]='B'
    with pytest.raises(ValueError,match='Initial S'):
        apply(g,s,geometry_tolerance_deg=2.)


def test_certified_T_preserved_despite_conditional_geometry_uncertainty():
    g,s=fixture();s[7]='T';records=[{} for _ in range(8)];records[7]['physical_truncation_certified']=True
    assert apply(g,s,geometry_tolerance_deg=2.,state_records=records)['states'][7]=='T'
    records[7]['physical_truncation_certified']=False
    assert apply(g,s,geometry_tolerance_deg=2.,state_records=records)['states'][7]=='U'


def test_new_U_precise_records_and_propagated_query_costs_removed_before_fields():
    g,s=fixture();s=apply(g,s,geometry_tolerance_deg=2.)['states']
    visible=[{'native_corner':7,'xy':object(),'fit_covariance_px2':object()},
             {'native_corner':0,'xy':[10.,20.],'fit_covariance_px2':[[9.,0.],[0.,9.]]}]
    partial=[{'native_corner':7,'lines':object()},
             {'native_corner':0,'lines':[{'provenance':{'source_query_native_corner':7},'normal_unit_xy':object()},
                 {'provenance':{'source_query_native_corner':0,'native_neighbor':7},'normal_unit_xy':[1.,0.]}]}]
    result=filter_reuse(visible,partial,s,[7])
    assert result['visible']==[visible[1]]
    assert result['boundary_observations'][0]['lines']==[partial[1]['lines'][1]]
    assert result['new_U_exact_point_or_line_costs']==0 and result['raw_or_RGB_queries']==0
