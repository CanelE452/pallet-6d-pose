"""Independent stock-loss/gradient, geometry and pretrained-output checks."""
from core import *

def main():
    from ultralytics.utils.loss import PoseLoss26
    torch.set_num_threads(2)
    m=base_model();h=m.model[-1];flow=h.flow_model.eval();stock=PoseLoss26(m)
    z=torch.load(OUT/'train_smoke.pt',weights_only=True)
    values=[];grads=[]
    for i in range(min(16,len(z['features']))):
        pred=z['stock_grid'][i:i+1].detach().clone().requires_grad_()
        gt=z['target'][i:i+1];area=z['area'][i:i+1,None];mask=gt[:,:,2]!=0
        ref=(12*stock.keypoint_loss(pred,gt,mask,area)+stock.bce_pose(pred[:,:,2],mask.float())+
             stock.calculate_rle_loss(pred,gt,mask).clamp_min(0))
        ours=loss_terms(pred,gt,area[:,0],flow)[0].sum()
        values.append(float((ref-ours).abs()))
        g1=torch.autograd.grad(ref,pred,retain_graph=True)[0]
        g2=torch.autograd.grad(ours,pred)[0]
        grads.append(float((g1-g2).abs().max()))
    assert max(values)<.0001 and max(grads)<.0001,(values,grads)
    parity={}
    for arm,(dims,sym) in ARMS.items():
        model=OutputAdapter(h,1)
        pred=model(z,dims)
        parity[arm]=float((pred-z['stock_grid']).abs().max())
    assert max(parity.values())<.0001,parity
    # A physically rotated box must preserve lengths and incidence, including non-square footprints.
    import itertools
    for dims in [[1.2,.15,.8],[.7,.3,1.5],[1.,.2,1.]]:
        X=np.array(list(itertools.product([-1,1],repeat=3)))*np.array(dims)/2
        # Use the actual camera-dynamic ordering, not the product ordering for edge validation.
        signs=np.array([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]])
        X=signs*np.array(dims)/2;p=c2_permutation(X)
        assert np.allclose(X[p[:8]],X@np.diag([-1,1,-1]))
        assert p[8]==8
    # A perfect C2-reordered prediction must choose one shared alternate target;
    # identical coordinates on C1 must NOT gain access to that branch.
    zz=subset(z,np.arange(2));zz['order']=torch.tensor([2,1])
    gt=zz['target'].gather(1,zz['perm'][:,:,None].expand(-1,-1,3))
    pred=torch.cat([gt[:,:,:2],torch.where(gt[:,:,2:3]>0,10.,-10.),torch.zeros(2,9,2)],-1)
    vals,terms,branches=objective(pred,zz,flow,True)
    assert branches.tolist()==[1,0],branches
    indexed=objective(pred,zz,flow,False)[0]
    assert vals[0]<indexed[0] and vals[1]==indexed[1]
    # Actual dimension path must train, rather than merely accepting an unused argument.
    model=OutputAdapter(h,1);opt=torch.optim.AdamW(model.parameters(),lr=.0001)
    before=state_sha(model.state_dict())
    for _ in range(2):
        opt.zero_grad();loss=objective(model(z,True),z,flow,True)[0].mean();loss.backward()
        assert torch.isfinite(model.condition[-1].weight.grad).all()
        assert model.condition[-1].weight.grad.abs().sum()>0
        opt.step()
    changed=state_sha(model.state_dict())!=before;assert changed
    shuffled=dict(z,dims_whd=z['dims_whd'].roll(1,0))
    response=float((model(z,True)-model(shuffled,True)).abs().max());assert response>0
    write(OUT/'PREFLIGHT.json',dict(PASS=True,stock_loss_max_abs=max(values),stock_gradient_max_abs=max(grads),
        pretrained_output_max_abs=parity,physical_c2_cases=3,shared_branch_and_C1_guard=True,
        smoke_optimizer_steps=2,dimension_gradient_nonzero=True,dimension_response_after_training=response))
    print(read(OUT/'PREFLIGHT.json'))

if __name__=='__main__':main()
