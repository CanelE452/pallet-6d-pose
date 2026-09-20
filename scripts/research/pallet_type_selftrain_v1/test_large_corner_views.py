import copy
import numpy as np
from .large_corner_views import restore,transform,choose_medoid,replace_corners,VARIANTS

IDENTITY=list(range(9))
C2=[IDENTITY,[5,4,7,6,1,0,3,2,8]]


def candidate(q):
    return dict(keypoints_xy=np.asarray(q).tolist(),keypoints_conf=[.9]*9,box_xyxy=[10,20,90,80],score=.95,candidate_index=0)


def test_restore_crop_translation_and_flip_twice():
    q=np.arange(18).reshape(9,2).astype(float);p=candidate(q)
    z=restore(p,dict(offset=[13,21],flip=False,original_width=100))
    np.testing.assert_array_equal(z['keypoints_xy'],q+[13,21])
    meta=dict(offset=[0,0],flip=True,original_width=100)
    np.testing.assert_array_equal(restore(restore(p,meta),meta)['keypoints_xy'],q)
    np.testing.assert_array_equal(restore(restore(p,meta),meta)['box_xyxy'],p['box_xyxy'])


def test_crop_uses_only_predicted_box_and_bounds():
    im=np.zeros((100,200,3),np.uint8)
    view,meta,size=transform(im,[50,20,150,80],'CROP125')
    assert meta['offset']==[37,12] and view.shape[:2]==(76,126) and size==640


def test_medoid_returns_actual_pose_not_average_and_respects_c2():
    q=np.arange(18).reshape(9,2).astype(float)
    views={n:None for n in VARIANTS};views['R0']=candidate(q+100)
    views['FULL960']=candidate(q);views['FULL1280']=candidate(q[np.array(C2[1])]);views['CROP125']=candidate(q+1)
    name,info=choose_medoid(views,C2,[0,0,200,100])
    assert name in ['FULL960','FULL1280']
    assert info['distance_matrix'][1][2]==0


def test_replace_only_first8_coordinates_and_never_input_mutation():
    p=candidate(np.zeros((9,2)));original=dict(candidates=[p],selected_index=0);before=copy.deepcopy(original)
    out=replace_corners(original,candidate(np.ones((9,2))*30))
    assert original==before
    assert out['candidates'][0]['keypoints_xy'][8]==[0,0]
    for key in ['score','box_xyxy','keypoints_conf']:assert out['candidates'][0][key]==p[key]
    assert replace_corners(original,None)==original
