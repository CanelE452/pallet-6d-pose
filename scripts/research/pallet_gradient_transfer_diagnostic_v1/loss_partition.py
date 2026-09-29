"""Differentiate real/source contributions of ONE unchanged full-batch graph.

This is not a sum of independently normalized real-only/source-only criteria.
Installed KeypointLoss/RLELoss forward methods are reused on selected rows,
then their ORIGINAL full-batch denominators and shared RLE clamp gate restored.
Flow ``log_phi`` tensors stay attached, retaining the direct flow-parameter path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def actual_clamp_gate(raw):
    """Use PyTorch's real min0 derivative, including derivative1 at raw==0."""
    if raw.requires_grad:
        return torch.autograd.grad(raw.clamp(min=0), raw, retain_graph=True)[0].detach()
    # No differentiable upstream at all: preserve value partition at the boundary.
    return (raw >= 0).to(raw.dtype)


def _weighted_subset(module, inputs, select, size_average=True):
    """Call the actual module.forward, avoiding extra diagnostic hook callbacks."""
    n=int(select.numel());k=int(select.sum())
    if not k:
        # log_phi, not only sigma/error, must remain graph-connected for flow.
        return sum((x.sum()*0. for x in inputs if torch.is_tensor(x)),inputs[0].new_zeros(()))
    values=tuple(x[select] if torch.is_tensor(x) and x.ndim and len(x)==n else x for x in inputs)
    result=module.forward(*values)
    return result*(k/n if size_average else 1.)


def partition(model, criterion, hook, real_image_mask):
    """Return ``real/source/combined`` differentiable per-image objectives.

    Required hook API (the existing V2 SignalHooks satisfies row fields):
      hook.rows[one2many|one2one] = branch,args,location_inputs,terms,
                                  rle_inputs,rle_before_clamp
      hook.full_loss = criterion(predictions,batch)[0]  # original batch-summed6vector

    ``args[0]`` is the original foreground mask (B,A); point identity and TAL
    assignment never change. ``real_image_mask`` is a boolean B-vector on either
    device. Both branches retain their actual E2E weights and hyp gains.

    Detector terms are constant over the allowed pose+flow parameter space.
    Their detached scalar is allocated by IMAGE fraction only to preserve full
    criterion value parity. This is NOT detector real/source attribution.
    Consumers must compare autograd(real)+autograd(source) with the original
    full criterion gradient on every trainable parameter (including flow).
    """
    full=hook.full_loss
    assert torch.is_tensor(full) and full.ndim==1 and len(full)==6
    real_image_mask=torch.as_tensor(real_image_mask,device=full.device,dtype=torch.bool)
    assert real_image_mask.ndim==1 and len(real_image_mask)>0
    batch_size=len(real_image_mask)
    forbidden=[n for n,p in model.named_parameters() if p.requires_grad and not n.startswith((
        'model.23.cv4.','model.23.cv4_kpts.','model.23.cv4_sigma.',
        'model.23.one2one_cv4.','model.23.one2one_cv4_kpts.','model.23.one2one_cv4_sigma.',
        'model.23.flow_model.'))]
    assert not forbidden,('Only frozen-backbone pose+flow diagnostic allowed',forbidden)
    zero=full.sum()*0.
    real=zero;source=zero;branches={}
    for name,branch_weight in [('one2many',criterion.o2m),('one2one',criterion.o2o)]:
        row=hook.rows[name]
        if 'location_inputs' not in row:
            branches[name]=dict(status='NO_FOREGROUND',branch_weight=float(branch_weight));continue
        branch=row['branch'];pred,gt,supervise,area=row['location_inputs']
        fg=row['args'][0]
        assert fg.shape[0]==batch_size and len(pred)==int(fg.sum())
        assert torch.equal(supervise,gt[...,2]==2)
        anchor_real=real_image_mask[fg.nonzero()[:,0]]
        keep=gt[...,2]!=1
        # Same installed BCE kernel and full non-ignore denominator as true-ignore.
        bce=F.binary_cross_entropy_with_logits(pred[...,2],supervise.to(pred.dtype),reduction='none')
        keep_count=keep.sum()
        location={};visibility={};rle={};gate=pred.new_zeros(())
        raw_rle=row.get('rle_before_clamp')
        rle_roles=None
        if 'rle_inputs' in row:
            inputs=row['rle_inputs'];assert len(inputs)==4
            # Preserve installed finite filtering in calculate_rle_loss before clamp.
            with torch.no_grad():
                visible=pred[supervise];truth=gt[supervise]
                sigma=visible[:,-2:].sigmoid()
                error=(visible[:,:2]-truth[:,:2])/(sigma+1e-9)
                finite=~(torch.isnan(error)|torch.isinf(error)).any(-1)
                rle_roles=anchor_real[:,None].expand_as(supervise)[supervise][finite]
                assert len(rle_roles)==len(inputs[0])
                assert torch.equal(sigma[finite],inputs[0].detach())
                assert torch.equal(error[finite].clamp(-100,100),inputs[2].detach())
            assert torch.is_tensor(raw_rle)
            gate=actual_clamp_gate(raw_rle)
        for role,select in [('real',anchor_real),('source',~anchor_real)]:
            n=int(select.sum())
            if n:
                location[role]=branch.keypoint_loss.forward(pred[select],gt[select],supervise[select],area[select])*(n/len(pred))
            else:location[role]=pred.sum()*0.
            visibility[role]=(bce*keep*select[:,None]).sum()/keep_count if int(keep_count) else pred.sum()*0.
            if rle_roles is not None:
                role_points=rle_roles if role=='real' else ~rle_roles
                rle[role]=_weighted_subset(branch.rle_loss,row['rle_inputs'],role_points,
                    size_average=branch.rle_loss.size_average)*gate
            else:rle[role]=pred.sum()*0.
        actual_terms=row['terms']
        errors={}
        for index,(term,parts) in enumerate([('location',location),('visibility',visibility),('rle',rle)]):
            actual=actual_terms[index]
            reconstructed=parts['real']+parts['source']
            actual=torch.as_tensor(actual,device=pred.device,dtype=pred.dtype)
            assert torch.allclose(reconstructed.detach(),actual.detach(),atol=2e-6,rtol=2e-5),(name,term,float(reconstructed),float(actual))
            errors[term]=float((reconstructed.detach()-actual.detach()).abs())
        def objective(role):
            return branch_weight*(branch.hyp.pose*location[role]+branch.hyp.kobj*visibility[role]+branch.hyp.rle*rle[role])
        real=real+objective('real');source=source+objective('source')
        branches[name]=dict(branch_weight=float(branch_weight),foreground_anchors=len(pred),
            real_foreground_anchors=int(anchor_real.sum()),source_foreground_anchors=int((~anchor_real).sum()),
            supervised_points=int(supervise.sum()),visibility_denominator=int(keep_count),
            RLE_finite_supervised_points=len(rle_roles) if rle_roles is not None else 0,
            RLE_real_points=int(rle_roles.sum()) if rle_roles is not None else 0,
            RLE_raw=float(raw_rle.detach()) if raw_rle is not None else None,RLE_shared_gate=float(gate),
            gains=dict(pose=float(branch.hyp.pose),kobj=float(branch.hyp.kobj),rle=float(branch.hyp.rle)),
            term_value_reconstruction_abs_error=errors,
            per_image_role_terms={role:{term:float(parts[role].detach()) for term,parts in [('location',location),('visibility',visibility),('rle',rle)]} for role in ('real','source')})
    detector=full[[0,3,4]].sum()/batch_size
    fraction=real_image_mask.to(full.dtype).mean()
    real=real+detector.detach()*fraction
    source=source+detector.detach()*(1-fraction)
    combined=real+source;expected=full.sum()/batch_size
    assert torch.allclose(combined.detach(),expected.detach(),atol=3e-6,rtol=3e-5),(float(combined),float(expected))
    return dict(real=real,source=source,combined=combined,diagnostics=dict(branches=branches,
        batch_size=batch_size,real_images=int(real_image_mask.sum()),source_images=int((~real_image_mask).sum()),
        combined_value_abs_error=float((combined.detach()-expected.detach()).abs()),
        detector_constant_per_image=float(detector.detach()),detector_constant_allocation='Image fractions only; no role attribution claimed. Detector/backbone frozen.',
        role_semantics='Contributions to original FULL mixed graph; NOT independently renormalized standalone losses.',
        reduction='Location global foreground-anchor denominator, visibility global nonignore-point denominator, RLE global finite supervised-point denominator and actual shared clamp gate.',
        flow='Captured log_phi retains direct shared-flow gradients; no detach of per-role RLE terms.',
        original_objective='criterion full six-vector sum / batch_size; detached logging items never used'))
