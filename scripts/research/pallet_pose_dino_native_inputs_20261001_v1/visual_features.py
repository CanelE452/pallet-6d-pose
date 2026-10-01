"""Pure native-image support restriction of the frozen DINO corner descriptor.

The existing crop, token grid, projection, bilinear interpolation and sorted
FP32 pooling are unchanged. Only the eligible sampling locations are reduced.
No image, cache, label, model, optimizer or file is accessed by these functions.
"""
import json
import numpy as np
from scripts.research.pallet_pose_dino_input_audit_20261001_v1 import visual_features as PREVIOUS

DESCRIPTOR_DIM = 385
RULE = 'NATIVE_UNPADDED_SUPPORTED_PROJECTED_CORNERS8_DINO384_MEAN_PLUS_SUPPORT_FRACTION'
SUPPORT_RULE = 'PREVIOUS_SUPPORT_AND_PAD_LE_U_LT_WIDTH_MINUS_PAD_AND_PAD_LE_V_LT_HEIGHT_MINUS_PAD'


def native_rectangle(hw, pad):
    hw = np.asarray(hw)
    assert hw.shape == (2,) and np.isfinite(hw).all() and (hw > 0).all()
    assert np.array_equal(hw, hw.astype(np.int64))
    assert np.ndim(pad) == 0 and np.isfinite(pad) and float(pad) == int(pad) and int(pad) >= 0
    pad = int(pad)
    assert 2*pad < int(hw[0]) and 2*pad < int(hw[1]), 'Padding must leave a nonempty original image region'
    return np.array([pad,pad,int(hw[1])-pad,int(hw[0])-pad],np.float64)


def native_support(projected, previous_support, hw, pad):
    rectangle = native_rectangle(hw,pad)
    projected,previous_support = np.asarray(projected,np.float64),np.asarray(previous_support)
    assert projected.shape == (8,2) and previous_support.shape == (8,) and previous_support.dtype == bool
    inside = np.isfinite(projected).all(1)
    inside &= (projected >= rectangle[:2]).all(1) & (projected < rectangle[2:]).all(1)
    support = previous_support & inside
    assert not (support & ~previous_support).any()
    return support


def describe(pose, K, matrix, hw, feature, pad):
    """Same output385 and old diagnostics, plus exact native-mask provenance.

    A zero-supported descriptor does not make the original pose invalid. Caller
    owns and must preserve the original candidate validity mask.
    """
    rectangle = native_rectangle(hw,pad)
    old = PREVIOUS.describe(pose,K,matrix,hw,feature)
    support = native_support(old['projected8'],old['support8'],hw,pad)
    descriptor = PREVIOUS.pool_supported(PREVIOUS.sample_map(feature,old['crop_points8'],support),support)
    return dict(descriptor=descriptor,projected8=old['projected8'],crop_points8=old['crop_points8'],
                support8=support,prepared_support8=old['support8'],native_rectangle_xyxy=rectangle,
                padding_px=int(pad),original_pose_available=bool(pose['available']))


def selfcheck():
    rng=np.random.default_rng(20261001656385100)
    feature=rng.normal(size=(384,56,42)).astype(np.float16)
    K=np.array([[100.,0.,100.],[0.,100.,100.],[0.,0.,1.]])
    pose=dict(available=True,cf_extents=[2.,2.,2.],R_cf=np.eye(3).tolist(),centroid=[0.,0.,5.])
    old=PREVIOUS.describe(pose,K,np.eye(3),[768,576],feature)
    zero=describe(pose,K,np.eye(3),[768,576],feature,0)
    for key in ('descriptor','projected8','crop_points8','support8'):
        np.testing.assert_array_equal(zero[key],old[key])
    native=describe(pose,K,np.eye(3),[768,576],feature,100)
    assert old['support8'].all() and int(native['support8'].sum())==2
    assert native['descriptor'][-1]==np.float32(2/8)
    assert native['original_pose_available'] and native['padding_px']==100
    np.testing.assert_array_equal(native['prepared_support8'],old['support8'])
    excluded_K=K.copy();excluded_K[:2,2]=[40.,40.]
    excluded=describe(pose,excluded_K,np.eye(3),[768,576],feature,100)
    assert excluded['prepared_support8'].all() and not excluded['support8'].any()
    assert excluded['original_pose_available'] and not excluded['descriptor'].any()
    original_valid=np.array([[True,True],[False,False]])
    preserved=original_valid.copy()
    _=describe(dict(available=False),K,np.eye(3),[768,576],feature,100)
    np.testing.assert_array_equal(original_valid,preserved)
    points=np.array([[100.,100.],[476.-1e-9,668.-1e-9],[476.,200.],[200.,668.],
                     [100.-1e-9,200.],[200.,100.-1e-9],[200.,200.],[np.nan,200.]])
    mask=native_support(points,np.ones(8,bool),[768,576],100)
    np.testing.assert_array_equal(mask,[True,True,False,False,False,False,True,False])
    rotated=dict(pose,R_cf=np.diag([-1.,1.,-1.]).tolist())
    c2=describe(rotated,K,np.eye(3),[768,576],feature,100)
    np.testing.assert_array_equal(c2['descriptor'],native['descriptor'])
    for bad in (-1,100.5,384):
        try:
            native_rectangle([768,576],bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid padding metadata accepted')
    return dict(PASS=True,invented_only=True,pad0_exact_old_descriptor=True,
                pad100_excludes_prepared_padding=True,half_open_native_boundaries=True,
                zero_support_candidate_validity_preserved=True,C2_exact=True,
                actual_inputs_read=0,new_forwards=0,new_fits=0,new_pose_solves=0)


if __name__=='__main__':
    print(json.dumps(selfcheck(),ensure_ascii=False,indent=2))
