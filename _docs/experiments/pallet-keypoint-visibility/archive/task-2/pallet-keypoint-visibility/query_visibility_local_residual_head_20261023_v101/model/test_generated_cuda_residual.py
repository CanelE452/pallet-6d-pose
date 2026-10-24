"""Generated-only CUDA2.1.1 strict-deterministic forward/backward preflight."""
import os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,json
import torch
from local_rgb_residual import LocalRGBResidualC5,_BASE,sample_query_feature,state_sha256,file_sha256

H=Path(__file__).resolve().parent
def binding(p):return dict(path=str(Path(p).resolve()),sha256=file_sha256(p))
def tests():
    assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
    torch.set_num_threads(1);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=_BASE.QueryCenterScalar()
    with torch.no_grad():base.readout[-1].weight.normal_(0,.2);base.readout[-1].bias.fill_(.17)
    model=LocalRGBResidualC5(base.state_dict()).cuda().train()
    p=torch.randn(4,65,9,9,device='cuda');p[:,64]=1
    r=torch.randint(0,256,(4,3,96,96),dtype=torch.uint8,device='cuda')
    v=torch.ones(4,1,96,96,dtype=torch.bool,device='cuda')
    with torch.no_grad():expected=model.base(p,r,v)
    out=model(p,r,v)
    assert torch.equal(out['query_logit'],expected['query_logit'])and out['query_logit'].abs().sum()>0
    assert torch.equal(out['delta_logit'],torch.zeros(4,device='cuda'))and not model.base.training
    yy,xx=torch.meshgrid(torch.arange(4,device='cuda'),torch.arange(4,device='cuda'),indexing='ij')
    feature=(xx+10*yy).float()[None,None].expand(4,16,-1,-1).clone().requires_grad_()
    sampled=sample_query_feature(feature)
    assert torch.equal(sampled,torch.full_like(sampled,21.3125))
    sampled.sum().backward();assert feature.grad.abs().sum()>0
    # Exercise an open residual backward in the exact fit runtime, without an update.
    with torch.no_grad():model.residual_readout[-1].weight.normal_(0,.2);model.residual_readout[-1].bias.fill_(.07)
    out=model(p,r,v);torch.nn.functional.binary_cross_entropy_with_logits(out['query_logit'],torch.tensor([0.,1.,0.,1.],device='cuda')).backward()
    encoder_grad=sum(float(a.grad.abs().sum())for a in model.residual_encoder.parameters()if a.grad is not None)
    assert encoder_grad>0 and all(a.grad is None for a in model.base.parameters())
    model.assert_base_unchanged()
    # Known invalid centre keeps safezero and U, despite residual nonzero parameters.
    v[0,0,47,47]=False;p[1,64,4,4]=0
    invalid=model(p,r,v)
    assert invalid['query_available'].tolist()==[False,False,True,True]
    assert torch.equal(invalid['query_logit'][:2],torch.zeros(2,device='cuda'))
    return dict(completed=True,PASS=True,model_code=binding(H/'local_rgb_residual.py'),test_code=binding(__file__),
        runtime=dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(0),deterministic=True,TF32=False,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG']),
        nonzero_base_zero_residual_parity_exact=True,CUDA_fixed_query_phase_ramp_exact=True,
        CUDA_strict_deterministic_grid_sample_backward_supported=True,copied_encoder_backward_abs_sum=encoder_grad,
        frozen_base_gradients_absent=True,frozen_base_value_hash_exact=True,invalid_centres_safezero_U=True,
        generated_optimizer_updates=0,actual_source_RGB_arrays_weights_current_reads=0,actual_source_forward_fit_features_private_current_PnP=0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=tests()
    with a.output.open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(r))
