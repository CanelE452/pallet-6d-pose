"""Independent invented NumPy layout/loss/RNG contracts; no producer/model import."""
import json
import numpy as np


def local_input(rgb,valid,sam,availability,use_sam):
    assert rgb.dtype==np.uint8 and rgb.ndim==4 and rgb.shape[1:]==(96,96,3)
    n=len(rgb);assert valid.dtype==bool and valid.shape==(n,96,96)
    axis=((np.arange(96,dtype=np.float32)+np.float32(.5))/np.float32(96))*np.float32(2)-np.float32(1)
    yy,xx=np.meshgrid(axis,axis,indexing='ij');xy=np.broadcast_to(np.stack([xx,yy]),(n,2,96,96))
    if use_sam:
        assert sam.dtype==availability.dtype==bool and sam.shape==availability.shape==(n,96,96)
        extra=np.stack((sam&availability,availability),axis=1).astype(np.float32)
    else:extra=np.zeros((n,2,96,96),np.float32)
    return np.concatenate([rgb.transpose(0,3,1,2).astype(np.float32)/np.float32(255),valid[:,None].astype(np.float32),xy,extra],axis=1)


def balanced(logits,targets,available):
    assert logits.shape==targets.shape==available.shape
    assert np.isin(targets,[-1,0,1]).all() and (targets[~available]==-1).all()
    loss=0.;g=np.zeros_like(logits)
    for c in (0,1):
        m=available&(targets==c);assert m.any() and np.isfinite(logits[m]).all()
        loss+=.5*np.mean(np.logaddexp(0,logits[m])-c*logits[m])
        g[m]=(1/(1+np.exp(-logits[m]))-c)*(.5/m.sum())
    return loss,g


def add(base,direct_callable,weight):
    if weight==0:return base
    return base+weight*direct_callable()


def run():
    checks={}
    rgb=np.zeros((2,96,96,3),np.uint8);rgb[0,:,:,0]=255;rgb[0,:,:,1]=127;rgb[1,:,:,2]=64
    valid=np.ones((2,96,96),bool);valid[1,0,0]=False;sam=np.ones_like(valid);av=np.ones_like(valid);av[:,10,10]=False
    raw=local_input(rgb,valid,sam,av,False);seg=local_input(rgb,valid,sam,av,True)
    checks['exact8channel_originalRGB_div255_valid_XY_SAM_layout']=raw.dtype==np.float32 and raw.shape==(2,8,96,96) and np.array_equal(raw[:,:6],seg[:,:6]) and raw[0,0,0,0]==1 and raw[0,1,0,0]==np.float32(127)/np.float32(255)
    checks['XY_centres_fixed_q47P5_no_q_native_or_GT_input']=np.allclose([raw[0,4,0,0],raw[0,4,0,95],raw[0,5,95,0]],[-95/96,95/96,95/96],rtol=0,atol=np.finfo(np.float32).eps)
    checks['SAMmask_availability_distinct_invalid_maskzero_no_RGB_mutation']=not seg[:,6,10,10].any() and not seg[:,7,10,10].any() and np.all(seg[:,6,11,11]==1) and not raw[:,6:].any() and rgb[0,0,0,0]==255
    z=np.array([-.8,.4,1.2,np.nan,2.],np.float64);t=np.array([0,0,1,-1,-1],np.int8);avail=np.array([True,True,True,True,False]);loss,grad=balanced(z,t,avail)
    manual=.5*((np.logaddexp(0,z[:2])).mean()+np.logaddexp(0,-z[2]))
    checks['unequal_class_balanced_direct_mean_not_plain_unbalanced_BCE']=loss==manual and loss!=np.mean(np.r_[np.logaddexp(0,z[:2]),np.logaddexp(0,-z[2])]) and np.array_equal(grad[3:],np.zeros(2))
    numeric=[]
    for i in range(3):
        zp=z.copy();zm=z.copy();zp[i]+=1e-5;zm[i]-=1e-5;numeric.append((balanced(zp,t,avail)[0]-balanced(zm,t,avail)[0])/2e-5)
    checks['analytic_gradient_knownonly_UNKNOWN_and_U_zero']=np.allclose(numeric,grad[:3],atol=2e-11,rtol=0)
    for name,tar,aa in [('missing_class',np.array([0,0,-1,-1,-1]),avail),('known_on_inputU',np.array([0,0,1,-1,1]),avail)]:
        try:balanced(z,tar,aa);checks[name+'_rejected']=False
        except AssertionError:checks[name+'_rejected']=True
    sentinel=object()
    def forbidden():raise AssertionError('Lambda0 must never call direct branch')
    checks['lambda0_identity_no_direct_loss_or_loader_invocation']=add(sentinel,forbidden,0) is sentinel
    old=np.random.default_rng(1810);extended=np.random.default_rng(1810);direct=np.random.default_rng(91810)
    old_hist=[];new_hist=[]
    for _ in range(50):
        old_hist.append(old.integers(0,1000,size=192));new_hist.append(extended.integers(0,1000,size=192));d0=direct.choice([2,4,8],16,replace=True);d1=direct.choice([1,3],16,replace=True);assert len(d0)==len(d1)==16
    checks['separate_direct32_rng_preserves_all_original192_draws']=all(np.array_equal(a,b) for a,b in zip(old_hist,new_hist)) and old.bit_generator.state==extended.bit_generator.state
    # Same original source identity can appear in multiple occurrence rows; missing
    # statuses have no physical group rather than a fake all-zero input row.
    status=[dict(row_id='A/native0',origin=7,physical=0),dict(row_id='B/native0',origin=7,physical=1),dict(row_id='missing/native0',origin=8,physical=None)]
    checks['occurrence_unique_repeated_origin_allowed_missing_null_not_fakezero']=len({x['row_id'] for x in status})==3 and status[-1]['physical'] is None and [x['physical'] for x in status if x['physical'] is not None]==[0,1]
    assert all(checks.values()),checks
    return dict(completed=True,PASS=True,invented_only=True,checks=checks,count=len(checks),source_asset_header_cache_model_or_evaluation_reads=0,production_model_instances=0)
if __name__=='__main__':print(json.dumps(run()),flush=True)
