import numpy as np
import pytest
from . import render_observed_bi_association_CAL_v4_30 as R


def test_render_GO_precedes_IO(monkeypatch):
    monkeypatch.setattr(R.C,'read',lambda p:pytest.fail('No reads before root render GO'))
    with pytest.raises(PermissionError):R.render()


def test_actual_support_extent_not_virtual_corner_or_bridge():
    node=dict(normal=[0.,1.],offset=-19.,actual_inliers=[[1.,19.],[4.,19.],[9.,19.]])
    segment=R.measured_support_segment(node)
    assert sorted(segment[:,0])==[1.,9.] and np.all(segment[:,1]==19.)


def test_one_factor_per_physical_edge_and_exact_cached_support():
    line=dict(normal=[0.,2.],offset=-38.,observed_points=[[1.,19.],[4.,19.],[9.,19.]],inlier_mask=[True,False,True])
    node=dict(name='0-1/0:0',native_edge=[0,1],normal=[0.,1.],offset=-19.,actual_inliers=[[1.,19.],[9.,19.]],chain_references=[dict(ray_index=0,chain_index=0)])
    assert R.verify_actual_origin(node,dict(edge_records=[dict(chains=[line])]))
    assert R.selected_actual_lines(dict(line_candidates=[node],selected_line_names=[node['name']]))==[node]
    line['offset']=-37.
    with pytest.raises(ValueError):R.verify_actual_origin(node,dict(edge_records=[dict(chains=[line])]))
    with pytest.raises(ValueError):R.selected_actual_lines(dict(line_candidates=[node],selected_line_names=[node['name'],node['name']]))
