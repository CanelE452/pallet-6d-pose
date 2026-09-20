import copy
import numpy as np
from ultralytics.data.augment import RandomFlip
from ultralytics.utils.instance import Instances
from .pseudo import PERM
from . import common as C


def fixture():
    q=np.zeros((1,9,3),np.float32)
    q[0,:,0]=np.linspace(.1,.9,9);q[0,:,1]=np.linspace(.2,.8,9);q[0,:,2]=[2,1,0,2,2,1,2,0,2]
    labels=dict(img=np.arange(8*10*3,dtype=np.uint8).reshape(8,10,3),
        instances=Instances(bboxes=np.array([[.3,.4,.2,.3]],np.float32),
            segments=np.zeros((0,0,2),np.float32),keypoints=q.copy(),bbox_format='xywh',normalized=True))
    return labels,q


def test_stock_horizontal_transform_and_ignore_identity():
    labels,q=fixture();oldimage=labels['img'].copy()
    out=RandomFlip(p=1,direction='horizontal',flip_idx=PERM)(labels)
    expected=q[:,PERM].copy();expected[...,0]=1-expected[...,0]
    np.testing.assert_allclose(out['instances'].keypoints,expected,atol=1e-7)
    np.testing.assert_array_equal(out['img'],oldimage[:,::-1])
    np.testing.assert_array_equal(out['instances'].keypoints[...,2],q[:,PERM,2])


def test_double_flip_restores_image_coordinates_box_and_masks():
    labels,q=fixture();before=copy.deepcopy(labels)
    flip=RandomFlip(p=1,direction='horizontal',flip_idx=PERM)
    out=flip(flip(labels))
    np.testing.assert_allclose(out['instances'].keypoints,q,atol=1e-7)
    np.testing.assert_allclose(out['instances'].bboxes,before['instances'].bboxes,atol=1e-7)
    np.testing.assert_array_equal(out['img'],before['img'])


def test_mapping_is_object_x_reflection_not_quarter_turn_symmetry():
    rows=C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    for row in rows:
        q=np.array(row['corners_centroid']);mirrored=q.copy();mirrored[:,0]*=-1
        np.testing.assert_allclose(q[PERM],mirrored,atol=1e-12)
        assert PERM not in row['permutations']
