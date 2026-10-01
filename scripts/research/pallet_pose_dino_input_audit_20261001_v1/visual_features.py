"""Pure frozen-pose DINO corner pooling; no files, labels or model inference.

Only ``selfcheck`` imports Torch, to compare the NumPy sampler independently.
Unsupported visual points never change the caller's original pose validity.
"""
import json
import numpy as np

CHANNELS=384
GRID_H,GRID_W=56,42
CROP_H,CROP_W=768,576
DESCRIPTOR_DIM=385
RULE='SUPPORTED_PROJECTED_CORNERS8_DINO384_MEAN_PLUS_SUPPORT_FRACTION'
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],np.float64)


def sample_map(feature,crop_points,support):
    """FP32 bilinear border samples; unsupported rows are exactly zero.

    The pixel-center transform is (crop+.5)*grid/crop_size-.5. Clamping is
    permitted only after support rejects points outside the crop itself.
    """
    feature=np.asarray(feature)
    assert feature.shape==(CHANNELS,GRID_H,GRID_W)
    assert feature.dtype in (np.dtype(np.float16),np.dtype(np.float32))
    assert np.isfinite(feature).all()
    points=np.asarray(crop_points,np.float64);support=np.asarray(support)
    assert points.ndim==2 and points.shape[1]==2
    assert support.shape==(len(points),) and support.dtype==bool
    q=points[support]
    assert np.isfinite(q).all()
    assert ((q>=0.) & (q<np.array([CROP_W,CROP_H],np.float64))).all()
    out=np.zeros((len(points),CHANNELS),np.float32)
    if not len(q):return out
    fmap=feature.astype(np.float32,copy=False)
    xy=(q+.5)*np.array([GRID_W/CROP_W,GRID_H/CROP_H],np.float64)-.5
    xy=np.minimum(np.maximum(xy,0.),np.array([GRID_W-1,GRID_H-1],np.float64))
    lower=np.floor(xy).astype(np.int64)
    upper=np.minimum(lower+1,np.array([GRID_W-1,GRID_H-1]))
    frac=(xy-lower).astype(np.float32)
    fx,fy=frac[:,0,None],frac[:,1,None]
    a=fmap[:,lower[:,1],lower[:,0]].T
    b=fmap[:,lower[:,1],upper[:,0]].T
    c=fmap[:,upper[:,1],lower[:,0]].T
    d=fmap[:,upper[:,1],upper[:,0]].T
    top=a*(np.float32(1.)-fx)+b*fx
    bottom=c*(np.float32(1.)-fx)+d*fx
    out[support]=top*(np.float32(1.)-fy)+bottom*fy
    assert out.dtype==np.float32 and np.isfinite(out).all() and not out[~support].any()
    return out


def pool_supported(sampled,support):
    """Channelwise sorted FP32 reduction gives exact permutation invariance.

    Sorting only changes the floating-point summation order; the mathematical
    feature remains the mean of supported corner tokens, plus count/8.
    """
    sampled,support=np.asarray(sampled),np.asarray(support)
    assert sampled.shape==(8,CHANNELS) and sampled.dtype==np.float32
    assert support.shape==(8,) and support.dtype==bool and np.isfinite(sampled).all()
    assert not sampled[~support].any()
    descriptor=np.zeros(DESCRIPTOR_DIM,np.float32)
    count=int(support.sum())
    if count:
        descriptor[:CHANNELS]=np.sort(sampled[support],axis=0).mean(axis=0,dtype=np.float32)
    descriptor[-1]=np.float32(count/8.)
    assert np.isfinite(descriptor).all()
    return descriptor


def describe(pose,K,matrix,hw,feature):
    """Return descriptor385, projected8, crop_points8 and support8.

    Coordinates are float64, support boolean, descriptor float32. An unavailable
    pose returns zeros. Nonpositive-depth points have zero display coordinates
    and no support; their division is not evaluated. ``hw`` is the dimensions
    of the already prepared image on which pose/K/crop affine are defined.
    """
    K,matrix=np.asarray(K,np.float64),np.asarray(matrix,np.float64)
    hw=np.asarray(hw)
    assert K.shape==matrix.shape==(3,3) and np.isfinite(K).all() and np.isfinite(matrix).all()
    np.testing.assert_array_equal(K[2],[0.,0.,1.])
    np.testing.assert_array_equal(matrix[2],[0.,0.,1.])
    assert K[0,0]>0 and K[1,1]>0 and np.linalg.det(matrix[:2,:2])!=0.
    assert hw.shape==(2,) and np.isfinite(hw).all() and (hw>0).all()
    assert np.array_equal(hw,hw.astype(np.int64))
    descriptor=np.zeros(DESCRIPTOR_DIM,np.float32)
    projected=np.zeros((8,2),np.float64);crop=np.zeros((8,2),np.float64);support=np.zeros(8,bool)
    if not pose['available']:
        # Validate the supplied feature contract even when no pose is available.
        sample_map(feature,crop,support)
        return dict(descriptor=descriptor,projected8=projected,crop_points8=crop,support8=support)
    ext=np.asarray(pose['cf_extents'],np.float64)
    rotation=np.asarray(pose['R_cf'],np.float64);translation=np.asarray(pose['centroid'],np.float64)
    assert ext.shape==(3,) and rotation.shape==(3,3) and translation.shape==(3,)
    assert np.isfinite(ext).all() and np.isfinite(rotation).all() and np.isfinite(translation).all()
    assert (ext>0).all()
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-6)
    assert abs(np.linalg.det(rotation)-1.)<=1e-6
    camera=(SIGNS*(ext/2.))@rotation.T+translation
    assert np.isfinite(camera).all()
    positive=camera[:,2]>0.
    homogeneous=camera[positive]@K.T
    projected[positive]=homogeneous[:,:2]/homogeneous[:,2,None]
    finite=positive & np.isfinite(projected).all(1)
    crop[finite]=np.column_stack([projected[finite],np.ones(int(finite.sum()))])@matrix[:2].T
    support=finite & np.isfinite(crop).all(1)
    support&=(projected>=0.).all(1)&(projected<np.array([hw[1],hw[0]])).all(1)
    support&=(crop>=0.).all(1)&(crop<np.array([CROP_W,CROP_H])).all(1)
    sampled=sample_map(feature,crop,support)
    descriptor=pool_supported(sampled,support)
    assert descriptor.dtype==np.float32 and np.isfinite(descriptor).all()
    return dict(descriptor=descriptor,projected8=projected,crop_points8=crop,support8=support)


def selfcheck():
    """Invented CPU arrays only: no caches, RGB, weights, labels or fits."""
    import torch
    from torch.nn import functional as F
    from scripts.research.pallet_pose_residual_direction_audit_20261001_v1 import direction_features as O
    xx,yy=np.meshgrid(np.arange(GRID_W),np.arange(GRID_H))
    # Integer-valued FP16 linear grid has an exact analytical bilinear value.
    feature=np.broadcast_to((xx+2*yy)[None],(CHANNELS,GRID_H,GRID_W)).astype(np.float16).copy()
    def crop_at(token):return (np.asarray(token,np.float64)+.5)*[CROP_W/GRID_W,CROP_H/GRID_H]-.5
    token=np.array([[0.,0.],[41.,55.],[17.,23.],[3.25,4.5]],np.float64)
    points=crop_at(token)
    sampled=sample_map(feature,points,np.ones(len(points),bool))
    np.testing.assert_allclose(sampled[:,0],token[:,0]+2*token[:,1],rtol=0,atol=1e-6)
    for i in range(3):np.testing.assert_array_equal(sampled[i],feature[:,int(token[i,1]),int(token[i,0])].astype(np.float32))
    boundary=np.array([[0.,0.],[CROP_W-1e-6,CROP_H-1e-6],[-.01,100.],[100.,-.01],[CROP_W,5.],[5.,CROP_H],[np.nan,np.inf]])
    supported=np.array([1,1,0,0,0,0,0],bool)
    sampled_boundary=sample_map(feature,boundary,supported)
    np.testing.assert_array_equal(sampled_boundary[0],feature[:,0,0].astype(np.float32))
    np.testing.assert_array_equal(sampled_boundary[1],feature[:,-1,-1].astype(np.float32))
    assert not sampled_boundary[~supported].any()
    wrong=supported.copy();wrong[2]=True
    try:sample_map(feature,boundary,wrong)
    except AssertionError:pass
    else:raise AssertionError('Out-of-crop point incorrectly accepted for border sampling')
    rng=np.random.default_rng(20261001656385)
    random_feature=rng.normal(size=(CHANNELS,GRID_H,GRID_W)).astype(np.float16)
    random_points=np.vstack([rng.uniform([0.,0.],[CROP_W,CROP_H],(64,2)),points,boundary[:2]])
    numpy_sample=sample_map(random_feature,random_points,np.ones(len(random_points),bool))
    grid=2*(random_points+.5)/np.array([CROP_W,CROP_H])-1
    torch_sample=F.grid_sample(torch.from_numpy(random_feature.astype(np.float32))[None],
        torch.tensor(grid,dtype=torch.float32)[None,None],mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T.numpy()
    np.testing.assert_allclose(numpy_sample,torch_sample,rtol=2e-5,atol=2e-5)
    pose=dict(available=True,cf_extents=[1.2,.4,.8],R_cf=np.eye(3).tolist(),centroid=[.1,-.1,4.])
    K=np.array([[500.,0.,280.],[0.,510.,350.],[0.,0.,1.]])
    affine=np.eye(3);hw=[CROP_H,CROP_W]
    result=describe(pose,K,affine,hw,random_feature)
    original=O.project_direction(pose,np.zeros((9,2)),[0.,0.,576.,768.],K)
    np.testing.assert_array_equal(result['projected8'],original['projected'][:8])
    assert result['support8'].all() and result['descriptor'][-1]==1.
    assert result['projected8'].shape==(8,2)  # No center token.
    a,b=.27,-.19
    Ry=np.array([[np.cos(a),0.,np.sin(a)],[0.,1.,0.],[-np.sin(a),0.,np.cos(a)]])
    Rz=np.array([[np.cos(b),-np.sin(b),0.],[np.sin(b),np.cos(b),0.],[0.,0.,1.]])
    rotated=dict(pose,R_cf=(Ry@Rz).tolist())
    rotated_result=describe(rotated,K,affine,hw,random_feature)
    rotated_old=O.project_direction(rotated,np.zeros((9,2)),[0.,0.,576.,768.],K)
    np.testing.assert_allclose(rotated_result['projected8'],rotated_old['projected'][:8],rtol=0,atol=1e-12)
    perm=np.array([5,4,7,6,1,0,3,2])
    symmetry=np.diag([-1.,1.,-1.]);half=dict(pose,R_cf=symmetry.tolist())
    half_result=describe(half,K,affine,hw,random_feature)
    np.testing.assert_array_equal(half_result['projected8'],result['projected8'][perm])
    np.testing.assert_array_equal(half_result['support8'],result['support8'][perm])
    np.testing.assert_array_equal(half_result['descriptor'],result['descriptor'])
    rotated_half=describe(dict(rotated,R_cf=(Ry@Rz@symmetry).tolist()),K,affine,hw,random_feature)
    np.testing.assert_allclose(rotated_half['projected8'],rotated_result['projected8'][perm],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(rotated_half['descriptor'],rotated_result['descriptor'])
    # Image bounds remain independent of a crop affine that maps outside image
    # points into the crop; these points must still have no visual support.
    moved=dict(pose,centroid=[20.,0.,4.]);move_affine=np.eye(3);move_affine[0,2]=-2500.
    unsupported=describe(moved,K,move_affine,hw,random_feature)
    assert not unsupported['support8'].any() and not unsupported['descriptor'].any()
    with np.errstate(invalid='raise',divide='raise',over='raise'):
        behind=describe(dict(pose,centroid=[0.,0.,-.4]),K,affine,hw,random_feature)
    assert not behind['support8'].any() and not behind['descriptor'].any()
    invalid=describe(dict(available=False),K,affine,hw,random_feature)
    assert not any(v.any() for v in invalid.values())
    zero=describe(pose,K,affine,hw,np.zeros_like(random_feature))
    assert not zero['descriptor'][:384].any() and zero['descriptor'][-1]==1.
    # Crop-boundary clipping is per point; the fraction is independent of the
    # channel amplitudes and pooling is over supported points only.
    shift=np.eye(3);shift[0,2]=300.
    partial=describe(pose,K,shift,hw,random_feature)
    assert 0<int(partial['support8'].sum())<8
    partial_samples=sample_map(random_feature,partial['crop_points8'],partial['support8'])
    expected=np.sort(partial_samples[partial['support8']],axis=0).mean(0,dtype=np.float32)
    np.testing.assert_array_equal(partial['descriptor'][:384],expected)
    assert partial['descriptor'][-1]==partial['support8'].sum()/8.
    # Arbitrary permutations include unsupported rows and cancellation-prone
    # FP32 values. Exact equality is stronger than a tolerance-based C2 check.
    permutation_trials=32
    cancellation=rng.normal(size=(8,384)).astype(np.float32)
    cancellation[0]=np.float32(1e8);cancellation[1]=np.float32(-1e8)
    full=np.ones(8,bool);reference=pool_supported(cancellation,full)
    for _ in range(permutation_trials):
        order=rng.permutation(8)
        np.testing.assert_array_equal(pool_supported(cancellation[order],full[order]),reference)
        np.testing.assert_array_equal(pool_supported(partial_samples[order],partial['support8'][order]),partial['descriptor'])
    return dict(PASS=True,invented_CPU_arrays_only=True,descriptor_dim=DESCRIPTOR_DIM,
        analytic_linear_grid=True,exact_token_centers=True,border_and_no_negative_wrap=True,
        Torch_grid_sample_max_abs=float(np.abs(numpy_sample-torch_sample).max()),
        C2_descriptor_max_abs=float(np.abs(half_result['descriptor']-result['descriptor']).max()),
        C2_descriptor_bit_exact_in_invented_fixtures=True,
        same_sample_corner_permutation_bit_exact=True,permutation_trials=permutation_trials,
        projection_vs_frozen_operator_exact=True,nonidentity_rotation_projection_checked=True,
        partial_support_count=int(partial['support8'].sum()),
        unavailable_and_zero_support_preserved=True,original_pose_validity_not_modified=True,
        actual_input_files_read=0,images_read=0,model_forwards=0,fit_count=0,policy_selections=0)


if __name__=='__main__':print(json.dumps(selfcheck(),allow_nan=False))
