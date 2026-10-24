"""Invented tensors/checkpoint only, exact learned Torch2.1.1+cu118 runtime."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,inspect,json,tempfile
import torch
from local_rgb_residual import (LocalRGBResidualC5,load_frozen_c1,file_sha256,state_sha256,
    sample_query_feature,known_binary_loss,decode_binary,_BASE,BASE_CODE,BASE_CODE_SHA256)

H=Path(__file__).resolve().parent
def binding(p):return dict(path=str(Path(p).resolve()),sha256=file_sha256(p))
def inputs(n):
    p=torch.randn(n,65,9,9);p[:,64]=1
    r=torch.randint(0,256,(n,3,96,96),dtype=torch.uint8)
    return p,r,torch.ones(n,1,96,96,dtype=torch.bool)

def tests():
    assert torch.__version__=='2.1.1+cu118',torch.__version__
    torch.set_num_threads(1);torch.manual_seed(20261023);torch.use_deterministic_algorithms(True)
    base=_BASE.QueryCenterScalar()
    with torch.no_grad():base.readout[-1].weight.normal_(0,.2);base.readout[-1].bias.fill_(.17)
    original=state_sha256(base.state_dict())
    with tempfile.TemporaryDirectory(prefix='v101_generated_C1_')as tmp:
        path=Path(tmp)/'INVENTED_C1_FORMAT.pt'
        torch.save(dict(state_dict=base.state_dict(),update=100,seed=20261023),path)
        model,receipt=load_frozen_c1(path,file_sha256(path))
        assert receipt['loaded_base_state_sha256']==original and receipt['checkpoint_update']==100
        try:load_frozen_c1(path,'0'*64)
        except RuntimeError:pass
        else:raise AssertionError('Hash mismatch accepted')
    assert sum(p.numel()for p in model.parameters())==29794
    assert sum(p.numel()for p in model.parameters()if p.requires_grad)==3221
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.residual_encoder.parameters())
    assert all(torch.equal(a,b)for a,b in zip(base.local_encoder.state_dict().values(),model.residual_encoder.state_dict().values()))
    assert tuple(inspect.signature(model.forward).parameters)==('p3_roi','local_rgb96','local_valid96')
    x=inputs(8);base.eval()
    with torch.no_grad():expected=base(*x)
    model.train();out=model(*x)
    assert not model.base.training and model.residual_encoder.training
    assert all(not m.training for m in model.base.modules())
    assert torch.equal(out['query_logit'],expected['query_logit'])and out['query_logit'].abs().sum()>0
    assert torch.equal(out['query_available'],expected['query_available'])and torch.equal(out['delta_logit'],torch.zeros(8))
    assert not out['base_logit'].requires_grad and out['delta_logit'].requires_grad
    model.eval();evaluation=model(*x)
    assert all(torch.equal(out[k],evaluation[k])for k in out)
    # q's exact fixed feature phase, not the centre2x2/wholemap average.
    yy,xx=torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij')
    ramp=(xx+10*yy).float()[None,None].expand(2,16,-1,-1).clone()
    sampled=sample_query_feature(ramp)
    expected_value=15.5/8*11
    assert torch.equal(sampled,torch.full((2,16),expected_value))
    assert not torch.equal(sampled,ramp[:,:,1:3,1:3].mean((2,3)))
    # One generated update changes only the new branch; base state stays exact.
    model.train();before={n:p.detach().clone()for n,p in model.named_parameters()}
    opt=torch.optim.SGD([p for p in model.parameters()if p.requires_grad],lr=.03)
    opt.zero_grad(set_to_none=True)
    labels=torch.tensor([0.,1.,0.,1.,0.,1.,0.,1.]);known=torch.ones(8,dtype=torch.bool)
    loss=known_binary_loss(model(*x),labels,known)['loss'];loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    first_encoder_grad=sum(float(p.grad.abs().sum())for p in model.residual_encoder.parameters()if p.grad is not None)
    assert first_encoder_grad==0  # Expected with final layer exactlyzero.
    opt.step();model.assert_base_unchanged()
    changed=[n for n,p in model.named_parameters()if not torch.equal(before[n],p)]
    assert changed and all(n.startswith('residual_')for n in changed)
    # Once zero head opens, a subsequent backward reaches copied encoder (no2ndupdate).
    opt.zero_grad(set_to_none=True);known_binary_loss(model(*x),labels,known)['loss'].backward()
    copied_encoder_grad=sum(float(p.grad.abs().sum())for p in model.residual_encoder.parameters()if p.grad is not None)
    assert copied_encoder_grad>0
    model.assert_base_unchanged()
    # Nonzero delta avoids a vacuous locality test.
    with torch.no_grad():model.residual_readout[-1].weight.normal_(0,.2);model.residual_readout[-1].bias.fill_(.07)
    p,r,v=inputs(3);r2=255-r;r2[:,:,32:64,32:64]=r[:,:,32:64,32:64]
    v2=v.clone();v2[:,:,:16,:16]=False
    a=model(p,r,v);b=model(p,r2,v2)
    assert a['delta_logit'].abs().sum()>0 and torch.equal(a['delta_logit'],b['delta_logit'])
    assert torch.equal(a['query_available'],b['query_available'])
    # Hidden invalid RGB contents cannot leak through the residual input.
    v[:,:,36:42,36:42]=False;r3=r.clone();r3[:,:,36:42,36:42]=255-r3[:,:,36:42,36:42]
    assert torch.equal(model(p,r,v)['delta_logit'],model(p,r3,v)['delta_logit'])
    # Same exact C1 availability, including conservative47:49 and native9centre.
    p,r,v=inputs(4);p[0,64]=0;v[0]=False;v[1,0,47,47]=False;p[2,64,4,4]=0
    invalid=model(p,r,v)
    assert invalid['query_available'].tolist()==[False,False,False,True]
    for key in ('query_logit','base_logit','delta_logit'):assert torch.equal(invalid[key][:3],torch.zeros(3))
    assert decode_binary(invalid)[:3].tolist()==[-1,-1,-1]
    # Unknown and unavailable labels remain losszero, no U->hidden relabeling.
    z=torch.tensor([-.3,.7,1.,-2.],requires_grad=True)
    lo=known_binary_loss(dict(query_logit=z,query_available=torch.tensor([True,True,False,True])),torch.tensor([0.,1.,1.,float('nan')]),torch.tensor([True,True,True,False]))
    lo['loss'].backward();assert torch.equal(z.grad[2:],torch.zeros(2))and z.grad[:2].abs().min()>0
    rejected=0
    for field in ('reference_xy','query_xy','native_id','GT_region','source_id','mask','pose','target_id'):
        try:model(*x,**{field:torch.zeros(8,2)})
        except TypeError:rejected+=1
        else:raise AssertionError('Forbidden input accepted')
    try:model({'p3_roi':x[0]},x[1],x[2])
    except TypeError:rejected+=1
    else:raise AssertionError('Dictionary accepted')
    model.assert_base_unchanged()
    return dict(completed=True,PASS=True,model_code=binding(H/'local_rgb_residual.py'),test_code=binding(__file__),
        base_code=dict(path=str(BASE_CODE),sha256=BASE_CODE_SHA256),total_parameters=29794,frozen_base_parameters=26573,trainable_new_parameters=3221,
        generated_nonzero_base_zero_residual_parity_exact=True,base_eval_even_during_train=True,base_state_hash_before_and_after=original,
        strict_generated_C1_payload_load_and_SHA_guard=True,copied_encoder_values_exact_and_requires_grad_restored=True,
        fixed_query_phase=dict(crop32_input_index=15.5,feature4_index=1.9375,align_corners=True,normalized_grid=7/24,ramp_actual=float(sampled[0,0]),ramp_expected=expected_value),
        residual_outside_central32_invariance_nonzero_exact=True,masked_local_RGB_invariance=True,
        one_generated_update_changed_parameters=changed,base_gradients_absent=True,first_encoder_gradient_zero_due_to_final_zero=first_encoder_grad,
        copied_encoder_gradient_nonzero_after_head_opens=copied_encoder_grad,C1_availability_and_safezero_exact=True,
        loss_unknown_and_unavailable_gradient_zero=True,forbidden_input_rejections=rejected,predictive_output='query_logit scalar only; components loss/diagnostic only; no xy/map',
        runtime=dict(torch=torch.__version__,device='CPU',threads=1,deterministic=True,seed=20261023),generated_optimizer_updates=1,
        actual_source_current_RGB_arrays_checkpoint_parameter_reads=0,actual_source_forward_fit_features_private_current_PnP=0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();result=tests()
    with a.output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))
