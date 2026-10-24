"""Generated tensor tests only. Never opens source arrays, images or weights."""
from pathlib import Path
import hashlib,json,inspect,sys
import numpy as np
import torch
from torch.nn import functional as F
from native_context_proxy_head import NativeContextProxyHead,FORWARD_FIELDS,PROXY_STATES

HERE=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())


def main():
    result=HERE/'GENERATED_NATIVE_CONTEXT_HEAD_TESTS_V2.json';assert not result.exists()
    torch.set_num_threads(1);torch.manual_seed(20261023);torch.use_deterministic_algorithms(True)
    m=NativeContextProxyHead().cpu()
    f=2
    p3=torch.randn(f,8,65,9,9);p3[:,:,64]=1
    local=torch.randint(0,256,(f,8,3,96,96),dtype=torch.uint8)
    localvalid=torch.ones(f,8,1,96,96,dtype=torch.bool)
    target=torch.randint(0,256,(f,3,128,128),dtype=torch.uint8)
    targetvalid=torch.ones(f,1,128,128,dtype=torch.bool)
    context=torch.randn(f,8,42)
    whole=dict(p3_roi=p3,local_rgb96=local,local_valid96=localvalid,target_rgb128=target,target_valid128=targetvalid,context42=context)
    flat=dict(p3_roi=p3.reshape(16,65,9,9),local_rgb96=local.reshape(16,3,96,96),local_valid96=localvalid.reshape(16,1,96,96),target_rgb128=target.repeat_interleave(8,0),target_valid128=targetvalid.repeat_interleave(8,0),context42=context.reshape(16,42))
    assert tuple(inspect.signature(m.forward).parameters)==FORWARD_FIELDS
    m.eval()
    with torch.no_grad():
        a=m(**whole);b=m(**flat)
        assert a['proxy_logits'].shape==(2,8,3) and a['reference_response_map'].shape==(2,8,9,9)
        assert all(torch.equal(a[k].reshape(b[k].shape),b[k]) for k in a)
        assert torch.allclose(b['reference_attention_weights'].sum((1,2)),torch.ones(16))
        # Pixels marked outside the native crop cannot affect the branch input.
        mv=dict(flat);mv['local_valid96']=flat['local_valid96'].clone();mv['local_valid96'][:,:,:20,:20]=False
        mv['target_valid128']=flat['target_valid128'].clone();mv['target_valid128'][:,:,-24:,-24:]=False
        c=m(**mv);changed=dict(mv);changed['local_rgb96']=flat['local_rgb96'].clone();changed['target_rgb128']=flat['target_rgb128'].clone()
        changed['local_rgb96'][:,:,:20,:20]=255-changed['local_rgb96'][:,:,:20,:20];changed['target_rgb128'][:,:,-24:,-24:]=255-changed['target_rgb128'][:,:,-24:,-24:]
        d=m(**changed);assert all(torch.equal(c[k],d[k]) for k in c)
        null={k:v.clone() for k,v in flat.items()};null['p3_roi'][:,64]=0;null['local_valid96'].zero_();null['target_valid128'].zero_()
        z=m(**null);assert not z['input_available'].any() and not z['reference_attention_weights'].any() and torch.isfinite(z['proxy_logits']).all()
    blocked=[]
    for name in ['GT','reference_xy','perm','matched','source_id','semantic_GT','mask','cueID']:
        try:m(**flat,**{name:torch.zeros(16)})
        except TypeError:blocked.append(name)
        else:raise AssertionError('Forbidden field accepted: '+name)
    try:m(**dict(flat,context42=flat['context42'].tolist()))
    except TypeError:pass
    else:raise AssertionError('NonTensor context accepted')
    try:m(**dict(flat,local_rgb96=flat['local_rgb96'].float()))
    except TypeError:pass
    else:raise AssertionError('Silently normalized floatRGB')
    try:m(**dict(flat,context42=torch.full_like(flat['context42'],float('nan'))))
    except ValueError:pass
    else:raise AssertionError('Nonfinite context accepted')
    # Class loss must train every input branch and the predicted attention path.
    m.train();m.zero_grad(set_to_none=True);pred=m(**flat);y=torch.arange(16)%3
    ce=F.cross_entropy(pred['proxy_logits'],y);ce.backward()
    class_grad={name:float(sum(p.grad.abs().sum() for p in mod.parameters() if p.grad is not None)) for name,mod in [('P3',m.p3_encoder),('localRGB',m.local_encoder),('targetRGB',m.target_encoder),('context42',m.context_encoder),('fusion',m.fusion),('reference_localizer',m.reference_localizer),('classifier',m.classifier)]}
    assert all(np.isfinite(v) and v>0 for v in class_grad.values())
    m.zero_grad(set_to_none=True);pred=m(**flat);generated_heat=torch.zeros(16,9,9);generated_heat[:,4,4]=1
    auxiliary=-(generated_heat*F.log_softmax(pred['reference_response_map'].flatten(1),1).reshape(16,9,9)).sum((1,2)).mean()
    auxiliary.backward();aux_grad=float(m.reference_localizer.weight.grad.abs().sum());assert aux_grad>0
    assert all(p.grad is None for p in m.classifier.parameters())
    # Shape/ordering path parity includes gradients under the same generated loss.
    m.zero_grad(set_to_none=True);pw=m(**whole);F.cross_entropy(pw['proxy_logits'].reshape(-1,3),y).backward();wg={k:p.grad.clone() if p.grad is not None else None for k,p in m.named_parameters()}
    m.zero_grad(set_to_none=True);pf=m(**flat);F.cross_entropy(pf['proxy_logits'],y).backward()
    assert all((wg[k] is None and p.grad is None) or (wg[k] is not None and p.grad is not None and torch.equal(wg[k],p.grad)) for k,p in m.named_parameters())
    report=dict(schema='V74_GENERATED_NATIVE_CONTEXT_HEAD_TESTS',completed=True,PASS=True,code=bind(HERE/'native_context_proxy_head.py'),test_code=bind(__file__),
        parameter_count=sum(p.numel() for p in m.parameters()),proxy_output_order=list(PROXY_STATES),forward_fields=list(FORWARD_FIELDS),generated_frames=2,generated_queries=16,
        whole8_vs_flat_all_output_bit_differences=0,whole8_vs_flat_parameter_gradient_bit_differences=0,forbidden_GT_fields_rejected=blocked,Tensor_only=True,
        raw_uint8_RGB_required=True,nonfinite_context_rejected=True,invalid_RGB_pixel_changes_output=False,all_invalid_finite_scores_and_zero_attention=True,
        class_gradient_L1_by_branch=class_grad,auxiliary_reference_response_gradient_L1=aux_grad,auxiliary_alone_classifier_gradient=False,
        runtime=dict(executable=sys.executable,torch=str(torch.__version__),device='cpu',threads=1,seed=20261023),
        actual_source_RGB_features_or_checkpoint_reads=0,actual_source_forward=0,fit_optimizer_steps=0,current_private_reads_or_forward=0,PnP=0,
        status='UNTRAINED_COMPONENT_ONLY_NOT_NATIVE8_PHYSICAL_GOLD')
    result.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(bind(result)))


if __name__=='__main__':main()
