"""Generated CPU tensors only; no source/cache/model-checkpoint/current reads."""
import hashlib
import json
from pathlib import Path

import torch
from cue_visibility_head import CueVisibilityHead, semantic_cue_loss, uncertainty_heatmap_target


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    model = CueVisibilityHead()
    n = 4
    x = torch.randn(n,65,9,9);x[:,64]=1
    rgb = torch.randint(0,256,(n,3,96,96),dtype=torch.uint8)
    valid = torch.ones(n,1,96,96,dtype=torch.bool)
    q = torch.tensor([[100.,100.]]*n,dtype=torch.float64)
    box = torch.tensor([[50.,50.,150.,150.]]*n,dtype=torch.float64)
    side = torch.ones(n,dtype=torch.float64)*90
    outputs = model(x,rgb,valid,q,box,side)
    assert outputs['logit_D_given_DE'].shape == (n,)
    assert outputs['cue_heatmap_logits'].shape == (n,9,9)
    assert torch.allclose(outputs['cue_attention_weights'].sum((1,2)),torch.ones(n),atol=1e-6,rtol=0)
    params = sum(p.numel() for p in model.parameters())
    assert params < 100000
    # The center native9 cell is [95,105]^2 for q=(100,100), L=90.
    heat = uncertainty_heatmap_target(torch.tensor([[99.,99.,101.,101.]]*n),q,side)
    assert torch.equal(heat[:,4,4],torch.ones(n)) and torch.count_nonzero(heat).item()==n
    full = torch.ones(n,dtype=torch.bool);iv=full.clone();source=torch.tensor([0,0,1,1])
    target=torch.tensor([1,0,-1,-1])
    result=semantic_cue_loss(outputs,target,full,iv,source,heat)
    result['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # E has no heatmap teacher, but class BCE must train predicted attention.
    model.zero_grad()
    epred=model(x,rgb,valid,q,box,side)
    eloss=semantic_cue_loss(epred,torch.zeros(n,dtype=torch.long),full,iv,source,heat)
    assert eloss['D_heatmap_CE'].item()==0
    eloss['loss'].backward()
    assert model.localizer.weight.grad is not None and model.localizer.weight.grad.abs().sum().item()>0
    # Test analytical masks directly; U class and U/E heatmap get exact zero gradients.
    z=torch.tensor([-.3,.7,1.2,-2.],requires_grad=True)
    h=torch.randn(n,9,9,requires_grad=True)
    raw={'logit_D_given_DE':z,'cue_heatmap_logits':h,'heatmap_input_valid':torch.ones(n,9,9,dtype=torch.bool)}
    loss=semantic_cue_loss(raw,target,full,iv,source,heat)
    loss['loss'].backward()
    assert torch.equal(z.grad[2:],torch.zeros(2)) and torch.count_nonzero(h.grad[1:]).item()==0
    assert z.grad[0]<0 and z.grad[1]>0
    # Source means are averaged over informative sources, not U count.
    expected=torch.nn.functional.binary_cross_entropy_with_logits(z[:2],torch.tensor([1.,0.]))
    assert torch.equal(loss['binary_BCE'],expected)
    uz=torch.randn(n,requires_grad=True);uh=torch.randn(n,9,9,requires_grad=True)
    unknown={'logit_D_given_DE':uz,'cue_heatmap_logits':uh,'heatmap_input_valid':raw['heatmap_input_valid']}
    ul=semantic_cue_loss(unknown,torch.full((n,),-1),~full,iv,source,heat)
    assert ul['loss'].item()==0
    ul['loss'].backward()
    assert torch.count_nonzero(uz.grad).item()==torch.count_nonzero(uh.grad).item()==0
    # E heatmap content, even NaNs, cannot influence the D-only loss.
    altered=heat.clone();altered[1:]=float('nan')
    e_ignored=semantic_cue_loss(raw,target,full,iv,source,altered)
    assert torch.equal(e_ignored['loss'],loss['loss'])
    rejected=0
    for badfull,badinput in [(torch.zeros(n,dtype=torch.bool),iv),(full,torch.zeros(n,dtype=torch.bool))]:
        try:semantic_cue_loss(raw,target,badfull,badinput,source,heat)
        except ValueError:rejected+=1
    assert rejected==2
    # Input geometry co-translation leaves geometry encoding unchanged.
    model.eval()
    a=model(x,rgb,valid,q,box,side)
    b=model(x,rgb,valid,q+20,box+20,side)
    assert torch.equal(a['logit_D_given_DE'],b['logit_D_given_DE'])
    # All-invalid RGB sampling remains finite; outer inference flags it U.
    x[:,64]=0
    c=model(x,rgb,torch.zeros_like(valid),q,box,side)
    assert torch.isfinite(c['logit_D_given_DE']).all() and not c['heatmap_input_valid'].any()
    assert torch.count_nonzero(c['cue_attention_weights']).item()==0
    root=Path(__file__).resolve().parent
    binding=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    report={'completed':True,'status':'PASS','generated_only':True,'parameters':params,
            'code':binding(root/'cue_visibility_head.py'),'test_code':binding(Path(__file__).resolve()),
            'source_cache_RGB_weights_current_reads':0,'optimizer_updates':0,'actual_source_forward_calls':0,
            'unknown_loss_and_gradient_zero':True,'E_and_U_heatmap_gradient_zero':True,
            'positive_D_negative_E_BCE_gradient_signs':True,'FULL_invalid_label_rejections':2,
            'source_informative_denominator_checked':True,'native9_heatmap_cell_phase_checked':True,
            'predicted_attention_normalized_inputonly':True,'E_class_gradient_reaches_localizer_without_E_heatmap_teacher':True,
            'input_cotranslation_invariance':True,'invalid_input_finite_and_outer_U_required':True}
    dest=root/'GENERATED_CUE_HEAD_TESTS_V3.json'
    assert not dest.exists(),dest
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
