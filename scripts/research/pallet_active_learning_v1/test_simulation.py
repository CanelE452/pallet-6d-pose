import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('al_simulation',Path(__file__).with_name('simulation.py'))
sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)

def test_group_split_label_blind_and_disjoint():
    rows=[dict(frame_id=f'{k}-{j}',capture_session=f'g{k}',object_type='wood' if k>3 else 'plastic') for k in range(8) for j in range(4)]
    a,b=sim.split_records(rows)
    assert not {r['capture_session'] for r in a}&{r['capture_session'] for r in b}
    assert len(a)+len(b)==len(rows)
    assert {r['object_type'] for r in b}=={'plastic','wood'}
    changed=[dict(**r,GT=np.ones((9,2))*123) for r in rows]
    c,d=sim.split_records(changed)
    assert [r['frame_id'] for r in a]==[r['frame_id'] for r in c]

def test_gt_v1_visibility_remains_supervised_stock_not_pseudo_ignore():
    xy=np.ones((9,2))*50;xy[2]=[110,40]
    vis=np.array([1,2,2,0,2,2,2,2,1])
    t=SimpleNamespace(box_xyxy=np.array([-1,0,110,80.]),keypoints_xy=xy,visibility=vis,keypoint_supervision_mask=vis>0)
    line,masked=sim.export_target(t,100,100);v=np.array(list(map(float,line.split())))
    p=v[5:].reshape(9,3)
    assert len(v)==32 and p[0,2]==1 and p[8,2]==1 and p[2,2]==0 and masked==1
    assert np.all((p[:,:2]>=0)&(p[:,:2]<=1))
    assert vis[2]==2,'Never mutate source visibility'

def test_retained_fit_blocks_before_loading_model(tmp_path,monkeypatch):
    monkeypatch.setattr(sim,'RAW',tmp_path)
    path=tmp_path/'runs/random_seed1';path.mkdir(parents=True);(path/'last_unverified.pt').write_bytes(b'saved')
    monkeypatch.setattr(sim,'load_model',lambda:pytest.fail('Must not initialize another fit'))
    with pytest.raises(AssertionError,match='Never repeat'):sim.train('random',1)

def test_stock_loss_factory_not_true_ignore():
    from ultralytics.utils.loss import PoseLoss26
    m=sim.load_model();m.args=sim.get_cfg(overrides=sim.HYP)
    criterion=m.init_criterion()
    assert type(criterion.one2many) is PoseLoss26
    assert type(criterion.one2one) is PoseLoss26

def test_restricted_canonical_2d_interface_without_real_GT():
    pair=SimpleNamespace(positive=SimpleNamespace(items=[SimpleNamespace(frame_id='fixture')]),negative=SimpleNamespace(items=[]))
    target=SimpleNamespace(keypoints_xy=np.zeros((9,2)),keypoint_xy_present=np.ones(9,bool),
        keypoint_supervision_mask=np.ones(9,bool),visibility=np.full(9,2))
    pred=sim.P.E.DetectionCandidate(frame_id='fixture',is_positive=True,score=.9,
        box_xyxy=np.array([0.,0.,10.,10.]),keypoints_xy=np.zeros((9,2)),target_iou=1.)
    m=sim.P.E._evaluate_2d_collected(pair,{'fixture':target},[pred],{'fixture':pred})
    assert m['positive_count']==1 and m['keypoint_location_median_px']==0

def test_pose_manifest_joins_image_not_frame_id_spelling():
    spec=importlib.util.spec_from_file_location('sim_eval_test',Path(__file__).with_name('simulation_evaluate.py'))
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    evaluation=[dict(frame_id='eval:123',image_path='data/fixture.png',image_sha256='fixture-sha')]
    manifest=[dict(frame_id='eval__123',image='data/fixture.png'),dict(frame_id='other__456',image='data/other.png')]
    selected,binding=evaluator.join_pose_manifest(evaluation,manifest)
    assert selected==manifest[:1]
    assert binding[0]['evaluation_frame_id']=='eval:123'
    assert binding[0]['pose_frame_id']=='eval__123'
    with pytest.raises(AssertionError):evaluator.join_pose_manifest(evaluation,manifest[1:])
    with pytest.raises(AssertionError):evaluator.join_pose_manifest(evaluation,manifest+manifest[:1])
