"""Geometric/display checks; these never open a reference or run a predictor."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.research.pallet_guarded_bi_final_v1 import gallery as G
from scripts.research.pallet_guarded_bi_final_v1.test_guard import fixture


def test_projection_preserves_CF_basis_instead_of_applying_physical_rotation_to_CF_cube():
    o, s, p, K = fixture()
    f = dict(xyz=o['initial_geometry']['native_xyz'], K=K.tolist())
    q = G.projection(f, p)
    X = np.asarray(f['xyz'])
    camera = X @ np.asarray(p['R_cf']).T + p['centroid']
    homogeneous = camera @ K.T
    expected = homogeneous[:,:2] / homogeneous[:,2,None]
    assert np.array_equal(q, expected)
    dimensions = deepcopy(f); dimensions['xyz'] = [.8,.2,1.2]
    assert np.array_equal(G.projection(dimensions, p), expected)
    wrong = deepcopy(p); wrong['R_cf'] = p['R_physical']
    assert not np.allclose(G.projection(f, wrong), expected)
    f['xyz'][0][0] += .1
    with pytest.raises(ValueError, match='vertex/CF basis'):
        G.projection(f, p)


@pytest.mark.parametrize('first,second,expected', [([-20,20],[120,20],[[0,20],[99,20]]),
    ([20,-50],[20,100],[[20,0],[20,79]]), ([0,0],[99,79],[[0,0],[99,79]]),
    ([-20,-10],[-20,90],None), ([float('nan'),0],[10,10],None)])
def test_model_edges_clip_to_the_exact_RGB_canvas(first,second,expected):
    actual = G.clip_segment(first,second,100,80)
    if expected is None:
        assert actual is None
    else:
        assert np.allclose(actual, expected, atol=1e-12, rtol=0)


def test_selection_is_deterministic_preserves_absent_categories_and_never_selects_CAL():
    records = []
    for fid, partition, translation in [('z','TEST',8.),('a','REMAINDER',8.),('cal','CAL',1000.)]:
        ms = {a:dict(success_le5=False,translation_cm=translation) for a in ('E0','EA','EB','N0','NB')}
        if fid == 'a': ms['EA']['success_le5'] = True
        if fid == 'z': ms['N0']['success_le5'] = True
        records.append(dict(id=fid,partition=partition,metrics=ms))
    selection = G.select_cases(records, 'E0')
    assert selection == G.select_cases(list(reversed(records)), 'E0')
    assert [r['id'] for r in selection['cases']] == ['a','z']
    assert 'cal' not in [r['id'] for r in selection['cases']]
    assert len(selection['cases']) <= selection['maximum_cases'] == 8
    assert any('EB' in c for c in selection['absent_categories'])
    assert selection['categories'][1]['id'] == 'a'  # max-T tie breaks by ID


def test_only_real_BI_supported_native_incident_rays_are_shown():
    o, s, p, K = fixture()
    original = deepcopy(o)
    good, excluded = G._supported_rays(o)
    assert len(good) == 1 and not excluded
    ray = deepcopy(good[0][1]); ray['measurement_source'] = 'MODEL_ONLY_CURVE'
    o['visible_correction']['records'][0]['rays'].append(ray)
    good, excluded = G._supported_rays(o)
    assert len(good) == 1 and len(excluded) == 1
    assert original['visible_correction']['records'][0]['rays'][0] == good[0][1]
    o['states'][0] = 'S'
    assert not G._supported_rays(o)[0]


def test_four_actual_stage_panels_keep_hidden_states_out_of_measured_markers(tmp_path):
    from PIL import Image
    from scripts.research.pallet_guarded_bi_final_v1.guard import apply_guard
    o, s, p, K = fixture()
    image = tmp_path/'toy.png'
    Image.fromarray(np.full((120,160,3), 180, dtype=np.uint8)).save(image)
    frame = dict(id='TOY_ONLY', image='toy.png',image_sha256=G.sha(image),raw_hw=[120,160],
        K=K.tolist(),xyz=o['initial_geometry']['native_xyz'],
        arms={'N3_SUBPIX':dict(native_points=o['original_raw_points'],prediction_support=[True]*9,coarse_pose=p)})
    guarded = {m:dict(result=apply_guard(o,s,p,K,m)) for m in ('A','B')}
    entries = {'N0':dict(actual_pose=p),'N1':dict(actual_pose=s['actual_pose']),
               **{'N'+m:guarded[m]['result'] for m in ('A','B')}}
    audit = {'N'+m:guarded[m] for m in ('A','B')}
    metrics = {arm:dict(translation_cm=1.,rotation_deg=1.,success_le5=True) for arm in entries}
    proof = G._draw(frame,dict(predictions=entries),o,s,audit,dict(metrics=metrics),
        dict(prefix='N',categories=['TOY_SCHEMATIC_TEST']),tmp_path/'case',tmp_path)
    assert len(proof['tiles']) == 4
    assert all((tmp_path/'case'/name).is_file() for name in proof['tiles'])
    assert {m['native_corner'] for m in proof['actual_BI_measurements']} == {0,1,2,3}
    assert proof['S_O_T_U_actual_measurement_markers'] == 0
    assert proof['GT_wire_drawn'] is False
    assert all(r['state'] in ('B','I') for r in proof['measurement_display'])
