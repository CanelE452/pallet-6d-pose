"""Shared-edge association/dedup/gating checks; no GT or new line fitting."""
from copy import deepcopy

import numpy as np
import pytest

from . import boundary_shared_edge_observations_v4_7 as NEW
from . import boundary_partial_observations_v4_6 as BASE
from .test_statewise_partial_boundary_u_solver_v4_6 import producer_arguments


def arguments():
    c,g,s,f,kw=producer_arguments()
    g['initial_projection_xy']=[[0.,0.],[100.,0.],[100.,100.],[0.,100.],
                                [30.,30.],[40.,30.],[40.,40.],object()]
    c['records'].append(dict(native_corner=1,state='B',corrected=False,rays=[]))
    f.append(dict(native_corner=1,state='B',xy=[80.,90.],source='VISIBLE_BACKBONE_FALLBACK',
        fit_covariance_px2=[[16.,0.],[0.,9.]],fit_reliability_weight=.7))
    return c,g,s,f,kw


def test_shared_edge_line_applies_to_both_endpoints_without_new_receiver_query():
    c,g,s,f,kw=arguments();saved=deepcopy(f)
    result=NEW.observe_shared_supported_b_lines(c,g,s,f,**kw)
    assert [o['native_corner'] for o in result['boundary_observations']]==[0,1]
    source,receiver=result['boundary_observations'];a,b=source['lines'][0],receiver['lines'][0]
    assert a['normal_unit_xy']==b['normal_unit_xy']==[0.,1.] and a['offset_px']==b['offset_px']==-100.
    assert a['normal_variance_px2']==b['normal_variance_px2']==18.
    assert b['provenance']['source_query_native_corner']==0 and b['provenance']['receiver_native_corner']==1
    assert b['provenance']['RGB_queried_at_receiver_for_this_line'] is False
    assert b['provenance']['physical_native_edge']==[0,1]
    assert receiver['association_provenance']['normal_projected_display_xy']==[80.,100.]
    assert receiver['replaces_raw_B_likelihood'] is True and result['propagated_factor_count']==1
    assert f==saved and result['RGB_queries']==result['new_line_fits']==result['S_coordinate_reads']==0


def test_repeat_rays_one_physical_edge_dedup_and_strongest_only():
    c,g,s,f,kw=arguments();ray=c['boundary_recovery_records'][0]['expanded_rays'][0]
    repeated=deepcopy(ray);repeated['inlier_rms_px']=.1
    c['boundary_recovery_records'][0]['observed_line_recentering']=[{'remeasured_ray':repeated} for _ in range(5)]
    result=NEW.observe_shared_supported_b_lines(c,g,s,f,**kw)
    assert result['propagated_factor_count']==1 and len(result['shared_edge_records'])==1
    assert all(len(o['lines'])==1 for o in result['boundary_observations'])
    assert all(o['lines'][0]['provenance']['inlier_rms_px']==.1 for o in result['boundary_observations'])


@pytest.mark.parametrize('state',['S','U','O','T','I'])
def test_no_shared_factor_or_coordinate_read_for_nonB_receiver(state):
    c,g,s,f,kw=arguments();s[1]=state;f[1]['xy']=object();f[1]['fit_covariance_px2']=object()
    c['records'][-1]['rays']=object()
    if state=='S':g['initial_self_hidden'][1]=True;g['initial_projection_xy'][1]=object()
    result=NEW.observe_shared_supported_b_lines(c,g,s,f,**kw)
    assert [o['native_corner'] for o in result['boundary_observations']]==[0]
    assert result['propagated_factor_count']==0 and result['records'][1]['coordinate_read'] is False


def test_complete_corrected_B_preserved_as_source_and_never_replaced_as_receiver():
    c,g,s,f,kw=arguments();c['records'][0]['corrected']=True;f[0]['source']='B_ACTUAL_COMPLETE_CORRECTION'
    saved=deepcopy(f);result=NEW.observe_shared_supported_b_lines(c,g,s,f,**kw)
    assert [o['native_corner'] for o in result['boundary_observations']]==[1]
    assert f==saved and result['records'][0]['accepted_partial'] is False
    assert result['shared_edge_records'][0]['complete_B_native_ids_preserved']==[0]
    c,g,s,f,kw=arguments();c['records'][-1]['corrected']=True;f[1]['source']='B_ACTUAL_COMPLETE_CORRECTION'
    result=NEW.observe_shared_supported_b_lines(c,g,s,f,**kw)
    assert [o['native_corner'] for o in result['boundary_observations']]==[0]
    assert result['propagated_factor_count']==0


def test_existing_own_line_never_duplicated_no_shared_noise_change():
    c,g,s,f,kw=arguments();other=deepcopy(c['boundary_recovery_records'][0]['expanded_rays'][0])
    other.update(native_corner=1,native_neighbor=0)
    c['boundary_recovery_records'].append(dict(native_corner=1,expanded_rays=[other],observed_line_recentering=[]))
    own=BASE.observe_single_supported_b_lines(c,g,s,f,**kw)
    result=NEW.observe_shared_supported_b_lines(c,g,s,f,**kw)
    assert result['propagated_factor_count']==0
    for a,b in zip(own['boundary_observations'],result['boundary_observations']):
        assert a['lines'][0]['normal_unit_xy']==b['lines'][0]['normal_unit_xy']
        assert a['lines'][0]['offset_px']==b['lines'][0]['offset_px']
        assert a['lines'][0]['normal_variance_px2']==b['lines'][0]['normal_variance_px2']==9.
        assert a['lines'][0]['support_tangent_interval_px']==b['lines'][0]['support_tangent_interval_px']


def test_nonincident_cube_edge_is_not_propagated():
    c,g,s,f,kw=arguments()
    # 0/2 share the projected hull here but are a diagonal, not a physical
    # registered cuboid edge. No full/native point is invented from the line.
    g['initial_projection_xy'][1]=[50.,50.];s[2]='B'
    c['records'].append(dict(native_corner=2,state='B',corrected=False,rays=[]))
    f.append(dict(native_corner=2,state='B',xy=[80.,90.],source='VISIBLE_BACKBONE_FALLBACK',
        fit_covariance_px2=[[16.,0.],[0.,9.]],fit_reliability_weight=.7))
    for ray in c['boundary_recovery_records'][0]['expanded_rays']:
        ray['native_neighbor']=2
    c['boundary_recovery_records'][0]['observed_line_recentering']=[]
    result=NEW.observe_shared_supported_b_lines(c,g,s,f,**kw)
    assert result['propagated_factor_count']==0
