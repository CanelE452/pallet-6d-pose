"""Independent V270 source-scope review plus artificial head-gradient checks.

Loads the declared R0 weights and forwards generated neck-shaped tensors on
CPU only. It reads no SYN/current images/labels and performs no optimizer step.
"""
from pathlib import Path
import hashlib
import json
import sys
import types

import torch

HERE=Path(__file__).absolute().parent
MODEL_SHA='b6d27f1aa9ef1f8a602baa557a4b6792bd31d18e2705e629ca98a8108966eb5f'


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def norm(grads):
    return float(sum((x.detach().double().pow(2).sum() for x in grads if x is not None),torch.zeros((),dtype=torch.float64)).sqrt())


def main():
    torch.set_num_threads(1);torch.manual_seed(27017);torch.use_deterministic_algorithms(True)
    p=safe(HERE/'model_initial_joint_v270.py');raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==MODEL_SHA
    module=types.ModuleType('v270_independent_gradient_model');module.__file__=str(p);sys.modules[module.__name__]=module
    exec(compile(raw,str(p),'exec'),module.__dict__)
    model=module.InitialKPJoint('JOINT','cpu');model.train()
    assert all(not x.training for x in model.base.modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm))
    shared=list(model.head.one2one_cv4.parameters());eparams=list(model.e_heads.parameters())
    assert all(x.requires_grad for x in shared+eparams)
    shared_ids={id(x) for x in shared}
    assert all(not x.requires_grad for x in model.base.parameters() if id(x) not in shared_ids)
    assert all(not x.requires_grad for x in model.head.one2one_cv4_kpts.parameters())
    before_buffers=module.state_sha(dict(model.base.named_buffers()))
    shapes=[(2,block[0].conv.in_channels,s,s) for block,s in zip(model.head.one2one_cv4,[8,4,2])]
    features=[torch.randn(shape) for shape in shapes]
    packet=model.forward_pyramid(features,with_det=False)
    assigned=torch.tensor([2,7],dtype=torch.int64)
    reference=module.selected(packet,assigned)['xy_px'].detach()
    targets={'xy_px':reference+torch.tensor([4.,-6.]),
        'loc_valid9':torch.ones((2,9),dtype=torch.bool),
        'canvas_obj9':torch.tensor([[1,1,1,1,0,1,1,1,1],[1,1,0,1,1,1,1,1,1]],dtype=torch.float32),
        'E9':torch.tensor([[0,1,0,1,-1,0,1,0,-1],[1,0,1,0,1,-1,0,1,-1]],dtype=torch.float32),
        'Emask9':torch.tensor([[1,1,1,1,0,1,1,1,0],[1,1,1,1,1,0,1,1,0]],dtype=torch.bool)}
    bbox=torch.tensor([[0,0,200,150],[10,15,210,165]],dtype=torch.float32)
    weights=[0.6150186181227141,2.673561151079137]
    losses={arm:module.losses(packet,targets,assigned,bbox,weights,arm) for arm in module.ARMS}
    common=losses['JOINT'];assert all(torch.isfinite(v) for arm in losses.values() for v in arm.values())
    components={key:torch.autograd.grad(common[key],shared+eparams,retain_graph=True,allow_unused=True) for key in ('canvas','xy','E')}
    component_norm={key:{'shared':norm(g[:len(shared)]),'new_E':norm(g[len(shared):])} for key,g in components.items()}
    assert component_norm['xy']['shared']>0 and component_norm['xy']['new_E']==0
    assert component_norm['E']['shared']>0 and component_norm['E']['new_E']>0
    assert component_norm['canvas']['shared']>0 and component_norm['canvas']['new_E']==0
    results={}
    coefficients={'XY_ONLY':{'canvas':1,'xy':12,'E':0},'VIS_ONLY':{'canvas':1,'xy':0,'E':1},'JOINT':{'canvas':1,'xy':12,'E':1}}
    for arm in module.ARMS:
        coefficients_arm=coefficients[arm]
        expected=sum(common[k]*v for k,v in coefficients_arm.items())
        assert torch.allclose(losses[arm]['total'],expected,rtol=0,atol=1e-7)
        g=torch.autograd.grad(losses[arm]['total'],shared+eparams,retain_graph=True,allow_unused=True)
        maximum=0.0
        for j,actual in enumerate(g):
            expected_grad=sum((components[k][j]*v for k,v in coefficients_arm.items() if components[k][j] is not None),
                torch.zeros_like(actual) if actual is not None else torch.zeros(()))
            if actual is not None:
                maximum=max(maximum,float((actual-expected_grad).abs().max()))
                assert torch.allclose(actual,expected_grad,rtol=2e-5,atol=2e-6),(arm,j,maximum)
        results[arm]={'total':float(losses[arm]['total'].detach()),'shared_gradient_norm':norm(g[:len(shared)]),
            'new_E_gradient_norm':norm(g[len(shared):]),'component_coefficients':coefficients_arm,
            'max_linear_component_gradient_error':maximum}
    assert results['XY_ONLY']['new_E_gradient_norm']==0
    altered={k:v.clone() for k,v in targets.items()};altered['E9'][~altered['Emask9']]=1
    unchanged=module.losses(packet,altered,assigned,bbox,weights,'JOINT')['E']
    assert torch.equal(unchanged,common['E'])
    # Position supervision remains present for a native point whose E is unknown.
    loc_unknown=targets['xy_px'].clone();loc_unknown[0,4]+=torch.tensor([5.,4.])
    second={**targets,'xy_px':loc_unknown}
    changed_xy=module.losses(packet,second,assigned,bbox,weights,'JOINT')['xy']
    assert float(abs(changed_xy-common['xy']))>0
    assert module.state_sha(dict(model.base.named_buffers()))==before_buffers
    xy_only=module.InitialKPJoint('XY_ONLY','cpu')
    assert all(not x.requires_grad for x in xy_only.e_heads.parameters())
    model_code=binding(HERE/'model_initial_joint_v270.py')
    extract=safe(HERE/'extract_syn_initial_v270.py').read_text()
    training=safe(HERE/'train_joint_v270.py').read_text()
    assert "raw_model_targets9_v2/COMPLETE.json" in extract
    assert "torch.optim" not in extract and 'M4' not in extract and 'mask_visible' not in extract
    assert "('xy_px','loc_valid9','canvas_obj9','E9','Emask9')" in training
    assert "for step,ids in enumerate(draws,1)" in training and "draws.shape==(300,8)" in training
    assert "lr=1e-4" in training and "lr=1e-3" in training and "eps=1e-8" in training
    assert "clip_grad_norm_(params,10)" in training and "scheduler" not in training
    assert training.index("ALL_CHECKPOINTS_FROZEN_BEFORE_DEV")<training.index("dev=np.flatnonzero")
    assert "protocol['code'].values()" in training and "m.verify(" in training
    sources={n:binding(HERE/n) for n in ['model_initial_joint_v270.py','extract_syn_initial_v270.py','train_joint_v270.py']}
    result={'completed':True,'PASS':True,'schema':'V270_INDEPENDENT_SCOPE_AND_ARTIFICIAL_GRADIENT_AUDIT_V1',
        'audit_code':binding(Path(__file__).absolute()),'reviewed_code_bindings':sources,
        'scope_checks':{'no_current_or_real_inputs':True,'M4_not_supervision':True,
            'fixed_three_terminal_300_fits':True,'shared_cv4_trainable_terminal_frozen':True,
            'independent_artificial_gradient_graph_PASS':True},
        'generated_fixture_feature_shapes':[list(s) for s in shapes],
        'component_gradient_norms':component_norm,'arm_gradient_checks':results,
        'centroid_and_unknown_E_mask_invariance':True,'unknown_E_point_still_has_independent_position_loss':True,
        'BN_running_buffers_unchanged':True,'XY_ONLY_E_parameters_frozen':True,
        'canvas_scope':'Known finite point within reflect100 padded canvas, separate from raw raster loc_valid and semantic external E mask.',
        'official_decoder_and_OKS_source_review':'Compared installed Pose26.kpts_decode and KeypointLoss implementation; plus-anchor/stride and uniform9sigmas/area-normalized OKS agree.',
        'prototype_scope_limits':['Frozen full-image backbone/neck, shared existing pose stem only; not whole-network finetuning.','VIS_ONLY has zero explicit xy loss but shared-stem updates may change xy; report drift separately.','Source virtual-cuboid external-ray proxy, not physical native D/E gold.','Baseline TAL assignment frozen across arms; unassigned source cases must be disclosed.'],
        'current_or_real_image_reads':0,'SYN_image_reads':0,'GT_label_reads':0,'optimizer_steps':0,
        'audit_implementation_repair':'Initial audit source-string check unnecessarily required loop variable named bound. Actual trainer used a different variable and performs the same binding verification. Narrowed the check; no model/label/gradient change.',
        'backbone_forwards':0,'generated_head_fixture_forwards':1,'R0_checkpoint_read_only':True}
    out=safe(HERE/'INDEPENDENT_SCOPE_AND_GRADIENT_AUDIT.json');assert not out.exists(),out
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'audit':binding(out),'gradient_norms':component_norm,'arms':results}),flush=True)


if __name__=='__main__':main()
