"""Independent generated comparison with the frozen V80 loss implementation."""
from pathlib import Path
import hashlib,importlib.util,json
import torch
from torch.nn import functional as F
from verify_loss_math_generated import balanced_query_loss

R=Path(__file__).resolve().parent
CODE=R.parent/'model/dense_auxiliary_loss.py'
EXPECTED='11cf2fa9faa674e248c7cc896710980df0abb281bb2f8a260e4eaa363d7c710a'
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    assert b(CODE)['sha256']==EXPECTED
    spec=importlib.util.spec_from_file_location('bound_generated_dense_loss',CODE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(1);torch.manual_seed(20261023)
    native=torch.rand(96,9,9)>.2;cached=torch.rand(96,9,9)>.2
    valid=native&cached;A=torch.rand(96,9,9)>.4;V=A&(torch.rand(96,9,9)>.3)
    valid[0]=False;valid[1:5]=True
    A[1]=False;V[1]=False
    A[2:4]=True;V[2]=False;V[3]=True
    A[4]=False;A[4,4,4]=True;V[4]=A[4]
    z=torch.randn(96,2,9,9,dtype=torch.float64,requires_grad=True)
    out=module.dense_support_losses(z,valid,A,V)
    ra=balanced_query_loss(z[:,0],A,valid)
    rv=balanced_query_loss(z[:,1],V,valid&A)
    assert torch.allclose(out['M0_loss'],ra,atol=1e-15,rtol=1e-14)
    assert torch.allclose(out['VgivenM0_loss'],rv,atol=1e-15,rtol=1e-14)
    assert out['M0_present_classes'][0]==0 and out['VgivenM0_present_classes'][1]==0
    gz=torch.autograd.grad(out['auxiliary_mean'],z,retain_graph=True)[0]
    gr=torch.autograd.grad((ra+rv)/2,z,retain_graph=True)[0]
    assert torch.allclose(gz,gr,atol=1e-16,rtol=1e-13)
    assert not gz[:,0][~valid].any() and not gz[:,1][~(valid&A)].any()
    # Flat96 and whole12x8 retain identical perquery reduction.
    whole=module.dense_support_losses(z.reshape(12,8,2,9,9),valid.reshape(12,8,9,9),A.reshape(12,8,9,9),V.reshape(12,8,9,9))
    assert torch.equal(out['auxiliary_mean'],whole['auxiliary_mean'])
    primary=torch.randn(96,dtype=torch.float64,requires_grad=True);target=torch.randint(0,2,(96,))
    both=module.combined_support_loss(primary,target,z,valid,A,V)
    expected=F.binary_cross_entropy_with_logits(primary,target.double())+.125*(ra+rv)
    assert torch.allclose(both['loss'],expected,atol=1e-15,rtol=1e-14)
    gt=torch.autograd.grad(both['loss'],z,retain_graph=True)[0]
    assert torch.allclose(gt,.25*gz,atol=1e-16,rtol=1e-13)
    zero=module.combined_support_loss(primary,target,z,valid,A,V,aux_weight=0.)
    assert torch.equal(zero['loss'],F.binary_cross_entropy_with_logits(primary,target.double()))
    zgrad=torch.autograd.grad(zero['loss'],z)[0];assert not zgrad.any()
    # Unknown primary is rejected, not clamped into a new negative.
    bad=target.clone();bad[0]=-1
    rejects=[]
    try:module.combined_support_loss(primary,bad,z,valid,A,V)
    except ValueError:rejects.append('primary_U')
    else:raise AssertionError('Primary U admitted')
    vbad=V.clone();vbad[1,0,0]=True
    try:module.dense_support_losses(z,valid,A,vbad)
    except ValueError:rejects.append('valid_M4_not_subset_M0')
    else:raise AssertionError('Invalid mask subset admitted')
    for weight in (.1,.5,1.):
        try:module.combined_support_loss(primary,target,z,valid,A,V,aux_weight=weight)
        except ValueError:rejects.append('weight_'+str(weight))
        else:raise AssertionError('Unfrozen weight admitted')
    assert b(CODE)['sha256']==EXPECTED
    result=dict(PASS=True,test=b(Path(__file__).resolve()),implementation=b(CODE),reference=b(R/'verify_loss_math_generated.py'),
        generated_queries=96,classes_and_empty_queries=True,flat_whole_exact=True,
        independent_loop_M0_absdiff=float(abs(out['M0_loss']-ra)),independent_loop_VgivenM0_absdiff=float(abs(out['VgivenM0_loss']-rv)),
        independent_loop_gradient_maxdiff=float((gz-gr).abs().max()),fixed_lambda_perhead_gradient_match=True,
        outside_M0_conditional_gradient_zero=True,native_cache_invalid_gradient_zero=True,
        primary_U_rejected=True,lambda0_aux_gradient_zero=True,rejections=rejects,
        actual_source_RGB_masks_scores_features_GT_weights_read=0,head_backbone_forward=0,optimizer_updates=0,current_reads=0,PnP=0)
    with (R/'GENERATED_LOSS_IMPLEMENTATION_TESTS.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,report=b(R/'GENERATED_LOSS_IMPLEMENTATION_TESTS.json'))))
if __name__=='__main__':main()
