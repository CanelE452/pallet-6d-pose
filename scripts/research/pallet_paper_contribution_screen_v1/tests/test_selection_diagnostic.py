import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track_c.selection_diagnostic import combine
from track_c.wiring import load_model

def tensors():
    a=torch.arange(32*8,dtype=torch.float32).reshape(1,32,8)/256
    b=a.clone();b[:,:4]+=2;b[:,4]=torch.flip(a[:,4],[-1])
    return a,b

def test_intervention_identity_and_channel_ownership():
    a,b=tensors()
    assert torch.equal(combine(a,b,'R0'),a)
    assert torch.equal(combine(a,b,'C2'),b)
    for kind,box,score in [('R0box_C2score',a,b),('C2box_R0score',b,a)]:
        y=combine(a,b,kind)
        assert torch.equal(y[:,:4],box[:,:4])
        assert torch.equal(y[:,4:5],score[:,4:5])
        assert torch.equal(y[:,5:],a[:,5:])

def test_mismatched_dense_pose_rejected():
    a,b=tensors();b[:,5,0]+=1
    with pytest.raises(AssertionError,match='pose changed'):combine(a,b,'R0box_C2score')

def test_stock_topk_uses_score_source_and_preserves_anchor_pose():
    a,b=tensors();head=load_model().model[-1];head.max_det=4
    def selected(x):return head.postprocess(x.permute(0,2,1))
    r,c=selected(a),selected(b)
    bs,cb=selected(combine(a,b,'R0box_C2score')),selected(combine(a,b,'C2box_R0score'))
    assert torch.equal(bs[:,:,4:],c[:,:,4:])
    assert torch.equal(cb[:,:,4:],r[:,:,4:])
    assert not torch.equal(r[:,:,6:],c[:,:,6:]),'Fixture must change selected anchor'
