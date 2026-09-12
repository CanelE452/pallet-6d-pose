import sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import R0,R0_SHA,sha,tensor_sha
from track_c.wiring import load_model,detection_parameters,freeze_detection_only
from track_c.train import batches,HYP,DATA,frozen_buffers
from track_d.mechanism import split_for,selected_stats,SIDE_EDGES
from track_b.jacobian import delta
from track_e.alignment import summarize
from track_c.verify_saved import detection_state_keys

def test_checkpoint_sha():assert sha(R0)==R0_SHA

def test_baseline_reproducibility_and_init():
    torch.set_num_threads(4)
    a=load_model().eval();b=load_model().eval()
    assert tensor_sha(a.state_dict())==tensor_sha(b.state_dict())
    x=torch.zeros(1,3,64,64)
    with torch.no_grad():assert torch.equal(a(x)[0],b(x)[0])

def test_optimizer_allowlist():
    m=load_model();ids=freeze_detection_only(m)
    assert ids==detection_parameters(m)
    assert {id(p) for p in m.parameters() if p.requires_grad}==ids
    # Ultralytics shares one stateless SiLU instance across branches. Its mode flag
    # has no forward effect. Stateful frozen BN/dropout must remain in eval.
    assert all(not mod.training for layer in list(m.model.children())[:-1]
               for mod in layer.modules() if not isinstance(mod,torch.nn.SiLU))

def test_seed_order():
    assert batches(1440,24,1,0)==batches(1440,24,1,0)
    assert batches(1440,24,1,0)!=batches(1440,24,2,0)
    assert len(batches(273,8,1,0))==90

def test_center_contract():
    assert DATA['kpt_shape']==[9,3]
    assert all(8 not in edge for edge in SIDE_EDGES)
    assert DATA['flip_idx'][8]==8

def test_pose_task_metadata():assert HYP['task']=='pose'

def test_frozen_buffer_names_are_actual_state_keys():
    m=load_model();freeze_detection_only(m)
    buffers=frozen_buffers(m)
    assert 'model.0.bn.running_mean' in buffers
    assert 'model.0.bn.running_var' in buffers
    assert 'model.0.bn.num_batches_tracked' in buffers
    assert all(k in m.state_dict() for k in buffers)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in buffers.items())

def test_independent_detection_state_ownership():
    m=load_model();mutable=detection_state_keys(m)
    assert mutable<=set(m.state_dict())
    assert 'model.0.bn.running_mean' not in mutable
    assert 'model.23.cv2.0.0.bn.running_mean' in mutable
    assert not any('.cv4.' in key or '.one2one_cv4.' in key for key in mutable)

def test_retained_unverified_checkpoint_prevents_refit(tmp_path,monkeypatch):
    import track_c.train as training
    monkeypatch.setattr(training,'RAW',tmp_path)
    folder=tmp_path/'C_geometry_preserving_da/C2_seed1'
    folder.mkdir(parents=True)
    (folder/'last_unverified.pt').write_bytes(b'persisted before audit')
    def forbidden_load():raise RuntimeError('Must not initialize another fit')
    monkeypatch.setattr(training,'load_model',forbidden_load)
    with pytest.raises(AssertionError,match='never repeat its fit'):
        training.train('C2',1)

def test_pose_probe_does_not_change_detection_training_output():
    import copy
    a=load_model();b=copy.deepcopy(a)
    freeze_detection_only(a);freeze_detection_only(b)
    torch.manual_seed(13)
    with torch.no_grad():
        b(torch.rand(2,3,64,64))
        x=torch.rand(2,3,64,64);left=a(x);right=b(x)
    # Non-frozen detection BN updates running statistics during the C2 probe,
    # but training uses current batch statistics. Frozen feature/pose BN is eval.
    assert all(torch.equal(left[branch][key],right[branch][key])
               for branch in left for key in ['boxes','scores','kpts'])

def test_group_split_deterministic():
    assert split_for('scenario1')==split_for('scenario1')
    assert {split_for(str(i)) for i in range(100)}=={'trust_train','trust_cal','trust_test'}

def test_trust_no_selection_is_failure():
    s=selected_stats(np.ones(10),np.zeros(10,bool),np.arange(10))
    assert not s['PASS'] and s['coverage']==0 and s['mean_gain_px'] is None

def test_trust_harm_and_cluster_gate():
    gain=np.array([1.,-1.]*10)
    s=selected_stats(gain,np.ones(20,bool),np.arange(20))
    assert s['harm_fraction']==.5 and not s['PASS']

def test_c2_yaw_wrap():
    a=np.array([1.,2.,np.deg2rad(179.)]);b=np.array([1.,2.,np.deg2rad(1.)])
    assert np.allclose(delta(a,b),[0,0,np.deg2rad(-2.)])

def test_depth_missing_denominator():
    rows=[dict(session='s',signed_px=0.)]*7+[dict(session='s',signed_px=None)]*3
    s=summarize(rows)
    assert s['coverage']==.7 and s['status']=='FAIL'

def test_depth_directional_bias_failure():
    s=summarize([dict(session='s',signed_px=4.)]*10)
    assert s['status']=='FAIL'
