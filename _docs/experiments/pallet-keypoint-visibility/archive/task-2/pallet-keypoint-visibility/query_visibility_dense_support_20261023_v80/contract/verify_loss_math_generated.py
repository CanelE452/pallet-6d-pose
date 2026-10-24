"""Independent V80 loss equations on artificial CPU logits only."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F

R=Path(__file__).resolve().parent

def balanced_query_loss(logits,target,eligible):
    """Transparent loop reference: mean(present-class means), then all queries."""
    values=[]
    for i in range(len(logits)):
        classes=[]
        for c in (0,1):
            take=eligible[i]&(target[i]==c)
            if take.any():
                classes.append(F.binary_cross_entropy_with_logits(logits[i][take],target[i][take].to(logits.dtype)))
        values.append(torch.stack(classes).mean() if classes else logits[i].sum()*0)
    return torch.stack(values).mean()

def explicit_gradient(logits,target,eligible):
    out=torch.zeros_like(logits)
    for i in range(len(logits)):
        present=[c for c in (0,1) if (eligible[i]&(target[i]==c)).any()]
        for c in present:
            take=eligible[i]&(target[i]==c)
            out[i][take]=(logits[i][take].sigmoid()-target[i][take])/len(logits)/len(present)/take.sum()
    return out

def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    torch.set_num_threads(1);torch.manual_seed(20261023)
    # Query0 has imbalanced A/V classes; query1 only outside A;
    # query2 has no native/cache intersection despite nonempty individual masks.
    A=torch.tensor([[1,1,1,0,1],[0,0,0,0,0],[1,0,1,0,1]])
    V=torch.tensor([[1,0,1,0,1],[0,0,0,0,0],[1,0,0,0,1]])
    native=torch.tensor([[1,1,1,1,0],[1,1,1,1,1],[1,0,1,0,1]],dtype=torch.bool)
    cached=torch.tensor([[1,1,1,1,1],[1,1,1,1,1],[0,1,0,1,0]],dtype=torch.bool)
    ea=native&cached;ev=ea&A.bool()
    za=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
    zv=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
    LA=balanced_query_loss(za,A,ea);LV=balanced_query_loss(zv,V,ev)
    combined=.25*(LA+LV)/2
    ga,gv=torch.autograd.grad(combined,(za,zv),retain_graph=True)
    expectedA=.125*explicit_gradient(za,A,ea)
    expectedV=.125*explicit_gradient(zv,V,ev)
    assert torch.allclose(ga,expectedA,atol=1e-16,rtol=1e-14)
    assert torch.allclose(gv,expectedV,atol=1e-16,rtol=1e-14)
    assert not ga[~ea].any() and not gv[~ev].any()
    # Arbitrary conditional logits outside M0 cannot alter V loss/gradient.
    changed=zv.detach().clone();changed[~A.bool()]=123.
    assert torch.equal(balanced_query_loss(changed,V,ev),LV.detach())
    assert not gv[1].any() and not ga[2].any() and not gv[2].any()
    # Empty query remains graph-connected and contributes 0 to the fixed batch denominator.
    invalid=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    empty=balanced_query_loss(invalid,torch.zeros((2,3),dtype=torch.long),torch.zeros((2,3),dtype=torch.bool))
    empty.backward();assert empty==0 and invalid.grad is not None and not invalid.grad.any()
    ea2=ea.clone();ea2[2]=False
    withzero=balanced_query_loss(za,A,ea2)
    withoutzero=balanced_query_loss(za[:2],A[:2],ea2[:2])
    assert torch.allclose(withzero,withoutzero*2/3)
    # A sole present class receives its full mean, not an artificial half weight.
    single=torch.tensor([[.2,-.7,1.3]],dtype=torch.float64)
    assert torch.equal(balanced_query_loss(single,torch.ones_like(single),torch.ones_like(single,dtype=torch.bool)),F.binary_cross_entropy_with_logits(single,torch.ones_like(single)))
    # Duplicating identical sites of one class preserves balanced loss.
    one=torch.tensor([[.6,-.4]],dtype=torch.float64);t=torch.tensor([[1,0]])
    many=torch.tensor([[.6]*30+[-.4]],dtype=torch.float64);tm=torch.tensor([[1]*30+[0]])
    assert torch.allclose(balanced_query_loss(one,t,torch.ones_like(t,dtype=torch.bool)),balanced_query_loss(many,tm,torch.ones_like(tm,dtype=torch.bool)))
    # The specified reduction differs from a global site-weighted BCE.
    example=torch.tensor([[2.,2.,2.,-2.]],dtype=torch.float64);target=torch.tensor([[1,1,1,1]])
    # Make opposite targets for a strongly asymmetric present-class example.
    target=torch.tensor([[1,1,1,0]])
    ex=torch.tensor([[2.,2.,2.,2.]],dtype=torch.float64)
    balanced=balanced_query_loss(ex,target,torch.ones_like(target,dtype=torch.bool))
    assert abs(float(balanced-F.binary_cross_entropy_with_logits(ex,target.double())))>.1
    # Factorized diagnostic probabilities, never the primary support output.
    qa=torch.randn(10000,dtype=torch.float64).sigmoid();qs=torch.randn(10000,dtype=torch.float64).sigmoid()
    pv=qa*qs;pc=qa*(1-qs);po=1-qa
    assert (pv<=qa).all() and torch.allclose(pv+pc+po,torch.ones_like(pv))
    report=dict(PASS=True,test=b(Path(__file__).resolve()),reference_definition='Loop class means within each query, mean present classes, fixed allquery mean; no data/model imports.',
        native_cache_intersection=True,outside_M0_V_loss_gradient_zero=True,
        invalid_cell_gradient_zero=True,allinvalid_graph_connected_zero=True,
        empty_query_keeps_batch_denominator=True,present_singleclass_fullmean=True,
        withinclass_duplicate_invariance=True,not_global_site_weighted_BCE=True,
        exact_aux_weight_per_channel=.125,analytic_gradient_maxdiff=max(float((ga-expectedA).abs().max()),float((gv-expectedV).abs().max())),
        factorized_probabilities_sum_to_one=True,actual_implementation_tested=False,
        actual_source_RGB_masks_scores_features_weights_read=0,model_forward=0,optimizer_updates=0,current_reads=0,PnP=0)
    p=R/'GENERATED_LOSS_MATH_TESTS.json'
    with p.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,report=b(p))))

if __name__=='__main__':main()
