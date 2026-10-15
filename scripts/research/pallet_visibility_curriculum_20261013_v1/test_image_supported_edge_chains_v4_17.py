"""Small image-gradient contracts; no dataset, network, reference or pose fit."""
from copy import deepcopy
import numpy as np
import pytest

from . import image_supported_edge_chains_v4_17 as P
from .test_face_aware_partial_bi_observations_v4_13d import fixture, Poison


def call(args, kw):
    return P.observe_image_supported_edge_chains(*args, image_binding=kw['image_binding'])


def supported(out):
    return [r for r in out['edge_records'] if r.get('supported')]


def test_actual_image_chain_angle_is_not_the_proposal_angle_and_noise_has_floor():
    args, kw = fixture()
    args[2]['initial_projection_xy'][1] = [130., 65.]
    before = deepcopy(args[-1])
    out = call(args, kw)
    rays = supported(out)
    assert len(rays) == 1
    ray = rays[0]; line = ray['chains'][ray['selected_chain_index']]
    assert abs(line['normal'][0]) < 1e-6
    assert line['angle_change_deg'] > 2.
    assert line['conditional_uncertainty']['angle_sigma_rad'] > 0
    assert line['conditional_uncertainty']['normal_sigma_px'] >= 1
    assert args[-1] == before
    assert not out['full_2D_corner_measurements_created']


def test_blank_image_abstains_without_fake_line_or_corner():
    args, kw = fixture(); args[0][:] = 50
    out = call(args, kw)
    assert not supported(out)
    assert out['full_2D_corner_measurements_created'] == out['old_corner_observations_changed'] == 0


def test_S_projection_and_excluded_measurement_fields_are_never_read():
    args, kw = fixture()
    for k in range(1, 8):
        args[-1].append(dict(native_corner=k,state=args[3][k],xy=Poison(),covariance_px2=Poison()))
    out = call(args,kw)
    assert supported(out) and out['S_coordinate_reads'] == out['U_coordinate_reads'] == 0
    args[3][1]='S';args[2]['initial_self_hidden'][1]=True
    args[2]['initial_projection_xy'][1]=Poison()
    out=call(args,kw)
    assert not supported(out)
    assert all(not r['pixel_queried'] for r in out['edge_records'])


def test_unknown_mate_exclusion_precedes_all_pixel_probes(monkeypatch):
    args,kw=fixture();mate=np.array([65.,60.])
    args[2]['initial_projection_xy'][1]=mate.tolist()
    seen=[];original=P._sample
    def sample(array,xy):
        seen.append(np.asarray(xy).copy());return original(array,xy)
    monkeypatch.setattr(P,'_sample',sample)
    call(args,kw)
    assert seen
    # Lattice centres are 14px away; all actual NMS/color probes remain >7px.
    assert min(np.linalg.norm(x-mate,axis=-1).min() for x in seen)>7


def test_multiple_supported_actual_chains_remain_explicit():
    args,kw=fixture()
    args[0][:]=50;args[0][60:70]=180;args[0][70:]=50
    args[-1][0]['source']='VISIBLE_BACKBONE_FALLBACK'
    out=call(args,kw);ray=supported(out)[0]
    assert ray['multiple_supported_chains'] and len(ray['chains'])>=2
    assert ray['selected_chain_is_not_independently_certified']
    assert not out['calibrated_confidence_or_future_coverage_claimed']


def test_bound_image_and_configuration_are_checked_before_search():
    args,kw=fixture();kw['image_binding']['image_pixel_sha256']='f'*64
    with pytest.raises(ValueError,match='pixel'):call(args,kw)
    args,kw=fixture()
    with pytest.raises(ValueError,match='Unknown'):
        P.observe_image_supported_edge_chains(*args,image_binding=kw['image_binding'],configuration={'GT':[1,2]})


def test_uncertainty_does_not_shrink_to_zero_with_perfect_collinearity():
    xy=np.column_stack([np.arange(15.),np.ones(15)*60])
    u=P._chain_uncertainty(xy,np.array([0.,1.]),0.,P.DEFAULT_CONFIGURATION)
    assert u['normal_sigma_px']==1
    assert u['angle_sigma_rad']==pytest.approx(np.arctan2(np.sqrt(2),14))
    assert not u['calibrated'] and not u['normal_angle_independence_claimed']
