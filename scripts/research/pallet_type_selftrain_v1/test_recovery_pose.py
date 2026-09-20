import numpy as np
from .recovery_pose import paired_labels
from .recovery_pose_trainer import pose_parameter


def test_only_pose_parameters_allowed():
    assert pose_parameter('model.23.one2one_cv4.0.0.bn.weight')
    assert pose_parameter('model.23.cv4_kpts.0.bias')
    assert not pose_parameter('model.23.one2one_cv2.0.0.conv.weight')
    assert not pose_parameter('model.0.conv.weight')


def test_matched_target_masks_and_boxes():
    candidate=dict(box_xyxy=[10,20,100,90],keypoints_xy=[[50,40]]*9,keypoints_conf=[.9]*9)
    changed={**candidate,'keypoints_xy':[[51,41]]*8+[[-5,40]]}
    record=dict(raw_hw=[100,200],raw=dict(selected_index=0,candidates=[candidate]),refined=dict(selected_index=0,candidates=[changed]))
    labels,count=paired_labels(record);assert count==8
    a,b=[np.array(labels[k].split(),float) for k in ['RAW','REF']]
    np.testing.assert_array_equal(a[:5],b[:5]);np.testing.assert_array_equal(a[5:].reshape(9,3)[:,2],b[5:].reshape(9,3)[:,2])
    assert (a[5:].reshape(9,3)[8]==[.5,.5,1]).all()
