"""Actual YOLO two-head costs/gradients, complete-loss orbits, and tuple-mask wiring."""
import itertools,copy
import cv2,torch,numpy as np
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
import env as E
from a_data import records,sample,collate,cuda,validate_permutations
from a_loss import GenericSymmetryPoseLoss,stable_target_order
from a_joint import objective

def leaves(x):
    if isinstance(x,dict):return {k:leaves(v) for k,v in x.items()}
    if isinstance(x,list):return [leaves(v) for v in x]
    if torch.is_tensor(x):return x.detach().clone().requires_grad_(x.is_floating_point())
    return x
def tensors(x):
    if isinstance(x,dict):return sum([tensors(v) for v in x.values()],[])
    if isinstance(x,list):return sum([tensors(v) for v in x],[])
    return [x] if torch.is_tensor(x) and x.requires_grad else []
def selected(batch,branches):
    out=dict(batch);p=batch['permutations'][torch.arange(len(branches),device='cuda'),torch.tensor(branches,device='cuda')]
    out['keypoints']=batch['keypoints'].gather(1,p[:,:,None].expand_as(batch['keypoints']))
    return out
def gradients(loss,parameters):return torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True)
def grad_delta(a,b):
    assert all((x is None)==(y is None) for x,y in zip(a,b))
    return max([float((x-y).abs().max()) for x,y in zip(a,b) if x is not None]+[0])

def main():
    E.gpu();torch.set_num_threads(4);cv2.setNumThreads(1)
    model=YOLO(str(E.R0)).model.cuda().float();model.args=get_cfg();model.requires_grad_(True).train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    # Include the corrected missing-annotation training frame and a fully annotated one.
    square=records('SQUARE','train');missing=next(r for r in square if r['id']=='capture_20260902_manual_gt__010583')
    b=cuda(collate([sample(missing),sample(square[0])]))
    with torch.no_grad():p=leaves(model(b['img']))
    params=tensors(p)+list(model.model[-1].flow_model.parameters())
    fn=GenericSymmetryPoseLoss(model,True)
    costs,heads=fn.costs(p,b);det=sum(sum(h['det_loss'])*w for h,w in zip(heads,[fn.stock.o2m,fn.stock.o2o]))
    errors=[];totals=[]
    for branch in itertools.product(range(4),repeat=2):
        actual=float(fn.stock(p,selected(b,branch))[0].sum())/2
        expected=det+objective(costs,np.array(branch));errors.append(abs(actual-expected));totals.append(actual)
        assert abs(actual-expected)<2e-5*max(1,abs(actual)),(actual,expected)
    actual=fn(p,b)[0].sum();chosen=fn.last_audit['branches'];g=gradients(actual,params)
    reference=fn.stock(p,selected(b,chosen))[0].sum();reference_grad=gradients(reference,params)
    assert torch.equal(actual,reference) and grad_delta(g,reference_grad)==0
    assert float(actual)/2<=min(totals)+2e-5*max(1,abs(min(totals)))
    base=float(actual);symmetry=[]
    for k in range(4):
        altered=selected(b,[k,k]);loss=fn(p,altered)[0].sum();gg=gradients(loss,params)
        delta=grad_delta(g,gg)
        assert torch.allclose(actual,loss,atol=2e-5,rtol=2e-5) and delta<2e-5
        symmetry.append(dict(quarter_turn=k,loss=float(loss),gradient_max_abs=delta))
    # Coordinates-only and visibility-only relabeling must NOT masquerade as the valid tuple action.
    good=selected(b,[1,1]);coord=dict(b);coord['keypoints']=b['keypoints'].clone();coord['keypoints'][...,:2]=good['keypoints'][...,:2]
    vis=dict(b);vis['keypoints']=b['keypoints'].clone();vis['keypoints'][...,2]=good['keypoints'][...,2]
    badloss=[float(fn(p,x)[0].sum()) for x in [coord,vis]]
    assert all(abs(x-base)>1e-5 for x in badloss),(badloss,base)
    c2=dict(b);c2['permutations']=b['permutations'][:,[0,2,0,0]];c2['group_valid']=torch.tensor([[1,1,0,0]]*2,device='cuda',dtype=torch.bool)
    l2=fn(p,c2)[0].sum();rot180=dict(c2);rot180['keypoints']=selected(b,[2,2])['keypoints'];l180=fn(p,rot180)[0].sum()
    rot90=dict(c2);rot90['keypoints']=selected(b,[1,1])['keypoints'];l90=fn(p,rot90)[0].sum()
    assert torch.allclose(l2,l180,atol=2e-5,rtol=2e-5) and abs(float(l2-l90))>1e-4
    absent=dict(b);absent['keypoints']=b['keypoints'].clone();absent['keypoints'][...,2]=0
    la=fn(p,absent)[0].sum();ga=gradients(la,params)
    assert torch.isfinite(la) and all(x is None or torch.isfinite(x).all() for x in ga)
    # Duplicate distinct GT rows in image 0 and scrambled image grouping; identity parity exercises stock slot mapping.
    multi=dict(b)
    for key in ['batch_idx','cls','bboxes','keypoints','permutations','group_valid']:multi[key]=b[key][[0,0,1]]
    multi['bboxes']=multi['bboxes'].clone();multi['bboxes'][1,0]=.8;multi['bboxes'][1,2]=.15
    scrambled=dict(multi)
    for key in ['batch_idx','cls','bboxes','keypoints','permutations','group_valid']:scrambled[key]=multi[key][[2,0,1]]
    lm=fn(p,multi)[0].sum();gm=gradients(lm,params);ls=fn(p,scrambled)[0].sum();gs=gradients(ls,params)
    assert torch.equal(lm,ls) and grad_delta(gm,gs)==0
    invalid=np.arange(9);invalid[[0,1]]=invalid[[1,0]]
    rejected=False
    try:validate_permutations((tuple(range(9)),tuple(invalid)))
    except ValueError:rejected=True
    assert rejected
    E.write(E.DOC/'A/JOINT_WIRING_AUDIT.json',dict(PASS=True,actual_detector_forward_images=2,optimizer_updates=0,
      exhaustive_combinations=16,max_complete_stock_loss_reconstruction_error=max(errors),
      selected_gradient_bit_exact=True,shared_branch_per_GT_row=True,heads=heads,C4=symmetry,
      mask_bug_losses=badloss,correct_tuple_loss=base,C2=dict(identity=float(l2),rotation180=float(l180),rotation90=float(l90)),
      all_unannotated_loss=float(la),all_unannotated_gradient_finite=True,multi_GT_shuffled_rows_bit_exact=True,
      invalid_partial_permutation_rejected=True,selection_no_grad=True,
      inference_missing_outputs='Training heads are dense and finite; no positive anchors uses unchanged stock zero-pose path. Evaluation missing detections retain full penalty.'))
    print('JOINT WIRING PASS max stock reconstruction',max(errors),flush=True)
if __name__=='__main__':main()
