from copy import deepcopy
import numpy as np
import pytest
from .statewise_outline_noise_calibration_v4_10 import (
    signed_nearest_segment_distances,fit_outline_noise_model,outline_noise_parameters)


def _records():
    return [dict(id=f'{s}{i}',partition='CAL',session=s,supported=True,
        signed_distances_normalized=[.01,.02,.03,.04]) for s in ('a','b','c','d') for i in range(3)]


def test_signed_distance_and_finite_segment_corner_have_correct_geometry():
    polygon=[[0.,0.],[10.,0.],[10.,10.],[0.,10.]]
    points=[[5.,5.],[12.,5.],[12.,12.],[0.,3.]]
    result=signed_nearest_segment_distances(points,polygon)
    assert np.allclose(result['signed_distances_px'],[-5.,2.,np.sqrt(8.),0.])
    assert np.allclose(result['nearest_xy'][2],[10.,10.])


def test_unordered_vertices_orientation_do_not_change_silhouette_error():
    vertices=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.],[2.,2.]])
    samples=np.array([[5.,-2.],[7.,7.],[14.,8.]])
    first=signed_nearest_segment_distances(samples,vertices)['signed_distances_px']
    second=signed_nearest_segment_distances(samples,vertices[[4,2,1,3,0]])['signed_distances_px']
    assert np.array_equal(first,second)


def test_equal_frame_weight_is_invariant_to_boundary_pixel_duplication():
    records=_records();baseline=fit_outline_noise_model(records);more=deepcopy(records)
    more[0]['signed_distances_normalized']*=1000
    fitted=fit_outline_noise_model(more)
    assert fitted['bias_ratio']==baseline['bias_ratio']
    assert fitted['total_normal_sigma_ratio']==baseline['total_normal_sigma_ratio']


def test_LOGO_excludes_tail_from_all_stats_and_TEST_is_rejected():
    records=_records()
    for r in records:
        if r['session']=='d':r['signed_distances_normalized']=[100.,100.]
    fitted=fit_outline_noise_model(records,heldout_session='d')
    assert not any(r.startswith('d') for r in fitted['fit_ids'])
    assert np.isclose(fitted['bias_ratio'],.025)
    assert fitted['total_normal_sigma_ratio']<.1
    p=outline_noise_parameters(fitted,100.)
    assert np.isclose(p['outward_mask_bias_px'],2.5) and p['total_normal_sigma_px']>=1.
    records[0]['partition']='TEST'
    with pytest.raises(ValueError,match='CAL only'):fit_outline_noise_model(records)
