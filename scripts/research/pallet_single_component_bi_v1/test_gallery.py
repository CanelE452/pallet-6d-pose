"""JPEG/display and fixed disagreement selection; no actual predictor calls."""
from copy import deepcopy

import numpy as np

from scripts.research.pallet_guarded_bi_final_v1.test_guard import fixture
from scripts.research.pallet_single_component_bi_v1 import gallery as G


def test_fixed_disagreement_groups_keep_both_directions_and_deduplicate():
    records=[]
    for fid,partition,n3,sp,ns in [('b','TEST',True,True,False),('a','REMAINDER',True,False,False),
                                  ('c','TEST',False,False,True),('cal','CAL',False,True,True)]:
        records.append(dict(id=fid,partition=partition,metrics={
            'N3_BI':{'success_le5':n3},'SUBPIX_BI':{'success_le5':sp},'N1':{'success_le5':ns}}))
    s=G.selection(records)
    assert s==G.selection(list(reversed(records)))
    assert [r['id'] for r in s['cases']]==['a','c','b']
    assert len(s['cases'])<=s['maximum_cases']==4
    assert len(next(r for r in s['cases']if r['id']=='c')['categories'])==2
    assert 'cal' not in [r['id']for r in s['cases']]


def test_absent_groups_are_explicit_and_never_substituted():
    r=dict(id='a',partition='TEST',metrics={a:{'success_le5':True}for a in ('N3_BI','SUBPIX_BI','N1')})
    s=G.selection([r])
    assert s['cases']==[] and len(s['absent_categories'])==4
    assert all(c['absent']and c['id'] is None for c in s['categories'])


def test_eight_actual_JPEG_stage_panels_source_RGB_and_excluded_state_markers(tmp_path):
    from PIL import Image
    obs,solved,coarse,K=fixture()
    rgb=tmp_path/'toy.png'
    Image.fromarray(np.full((120,160,3),180,dtype=np.uint8)).save(rgb)
    source=dict(native_points=deepcopy(obs['original_raw_points']),prediction_support=[True]*9,coarse_pose=coarse)
    frame=dict(id='TOY_ONLY',image='toy.png',image_sha256=G.sha(rgb),raw_hw=[120,160],
        K=K.tolist(),xyz=[.8,.2,1.2],arms={branch:deepcopy(source)for branch,_,_,_ in G.BRANCHES})
    predictions={}
    for branch,before,after,_ in G.BRANCHES:
        predictions[before]=dict(actual_pose=deepcopy(coarse))
        predictions[after]=dict(actual_pose=deepcopy(solved['actual_pose']))
    metrics={arm:dict(translation_cm=1.,rotation_deg=1.,success_le5=True)for arm in predictions}
    observations={b:dict(observation=deepcopy(obs))for b,_,_,_ in G.BRANCHES}
    solutions={b:deepcopy(solved)for b,_,_,_ in G.BRANCHES}
    proof=G.draw_case(frame,dict(predictions=predictions),observations,solutions,
        dict(metrics=metrics),tmp_path/'case',tmp_path)
    assert len(proof['tiles'])==8
    assert proof['same_full_RGB_canvas_every_panel']
    assert proof['S_O_T_U_actual_measurement_markers']==0
    assert G.sha(tmp_path/'case/source_rgb.png')==G.sha(rgb)
    for tile in proof['tiles']:
        assert tile.endswith('.jpg')
        with Image.open(tmp_path/'case'/tile)as image:
            assert image.format=='JPEG' and image.size[0]>1000
    for b,_,_,_ in G.BRANCHES:
        d=proof['branches'][b]
        assert {m['native_corner']for m in d['actual_BI_measurements']}=={0,1,2,3}
        assert all(m['state']in ('B','I')for m in d['movement'])
        assert np.array_equal(np.asarray(d['initial_projection_xy']),G.GEOMETRY.projection(frame,coarse))
        assert d['synthetic_midstage_pose_or_GT_wire'] is False
