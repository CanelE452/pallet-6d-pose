import numpy as np
from .recovery_agreement import gate,count_control,masked_label,C2


def fixture():
    q=np.array([[10,10],[90,10],[90,30],[10,30],[20,15],[80,15],[80,25],[20,25],[50,20]],float)
    return dict(box_xyxy=[0,0,100,50],keypoints_xy=q.tolist(),keypoints_conf=[.9]*9)


def test_disagreement_masks_without_moving_target():
    r=fixture();t=fixture();before=np.array(r['keypoints_xy']).copy()
    t['keypoints_xy'][0][0]+=20
    mask,_=gate(r,np.ones(9,bool),[fixture(),t])
    assert not mask[0] and mask[1:].all()
    np.testing.assert_array_equal(r['keypoints_xy'],before)


def test_only_whole_C2_alignment_and_missing_teacher():
    r=fixture();t=fixture();t['keypoints_xy']=np.array(t['keypoints_xy'])[C2[1]].tolist()
    mask,details=gate(r,np.ones(9,bool),[t,fixture()])
    assert mask.all() and details[0]['branch']==1
    mask,_=gate(r,np.ones(9,bool),[None,fixture()])
    assert not mask[:8].any() and mask[8]


def test_count_matched_deterministic_control():
    v=np.array([1,0,1,1,0,1,1,1,1],bool);s=v.copy();s[[0,3]]=False
    a=count_control(v,s,'frame');b=count_control(v,s,'frame')
    assert np.array_equal(a,b) and a[:8].sum()==s[:8].sum() and not (a&~v).any()


def test_surviving_tokens_and_box_exact():
    tokens=['0','0.51','0.49','0.75','0.8']+['0.3','0.7','2']*9
    mask=np.ones(9,bool);mask[2]=False
    out=masked_label(' '.join(tokens),mask).split()
    assert out[:11]==tokens[:11] and out[14:]==tokens[14:]
    assert out[11:14]==['0.500000000','0.500000000','1.000000000']
