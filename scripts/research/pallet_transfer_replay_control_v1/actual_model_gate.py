"""GPU gate: real augmented batches, YOLO gradients, frozen BN, and roundtrip."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from ultralytics.cfg import get_cfg

from runtime import (ROOT,DOC,R0,HYP,ARM_WEIGHTS,atomic_json,batch_digest,bn_buffers,
    device_batch,epoch_streams,freeze_bn,load_model,optimizer,sha,state_sha)


def grads(stream_batches, coefficients):
    torch.manual_seed(991); model=load_model(); freeze_bn(model); before=bn_buffers(model)
    criterion=model.init_criterion(); model.zero_grad(set_to_none=True); values={}
    for name, coefficient in coefficients.items():
        batch=device_batch(stream_batches[name]); value=criterion(model(batch['img']),batch)[0].sum()
        (coefficient*value).backward(); values[name]=float(value.detach())
    result={name:(p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu')) for name,p in model.named_parameters()}
    bn_affine={name:dict(present=module.weight.grad is not None,
        finite=bool(module.weight.grad is not None and torch.isfinite(module.weight.grad).all()),
        nonzero=bool(module.weight.grad is not None and module.weight.grad.abs().sum()>0))
        for name,module in model.named_modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)}
    assert before.keys()==bn_buffers(model).keys() and all(torch.equal(v,bn_buffers(model)[k]) for k,v in before.items())
    norm=float(torch.sqrt(sum((v.double().square().sum() for v in result.values()))))
    del model,criterion; torch.cuda.empty_cache()
    return result,values,norm,bn_affine


def close(left,right,rtol=2e-5,atol=2e-6):
    failures=[]; max_abs=0.; max_rel=0.
    for name in left:
        a,b=left[name],right[name]; delta=(a-b).abs(); max_abs=max(max_abs,float(delta.max()))
        scale=torch.maximum(a.abs(),b.abs()); rel=delta/(scale+1e-12); max_rel=max(max_rel,float(rel.max()))
        if not torch.allclose(a,b,rtol=rtol,atol=atol): failures.append(name)
    return dict(passed=not failures,failures=failures,max_abs=max_abs,max_relative=max_rel,rtol=rtol,atol=atol)


def combine(*values):
    return {name:sum(value[name] for value in values) for name in values[0]}


def scale(value,factor):
    return {name:tensor*factor for name,tensor in value.items()}


def main():
    assert torch.cuda.is_available(); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True,warn_only=True)
    gpu=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    # Four independently constructed arm streams must produce identical base tensors.
    all_batches={}; all_records={}
    for arm in ARM_WEIGHTS:
        streams=epoch_streams(arm,1,0); batches={};records={}
        for name,stream in streams.items():batches[name],records[name]=stream.batch(0)
        all_batches[arm],all_records[arm]=batches,records
    base_hashes={arm:batch_digest(value['target_base']) for arm,value in all_batches.items()}
    assert len(set(base_hashes.values()))==1
    target=all_batches['T8_FULL']['target_base']
    source={name:all_batches['REPLAY'][name] for name in ('source_1','source_2','source_3')}
    full,full_values,full_norm,full_bn=grads({'target_base':target},{'target_base':4.})
    quarter,quarter_values,quarter_norm,quarter_bn=grads({'target_base':target},{'target_base':1.})
    source_grads=[]; source_values={}
    for name,batch in source.items():
        value,component,_,_=grads({name:batch},{name:1.}); source_grads.append(value);source_values.update(component)
    mix,mix_values,mix_norm,mix_bn=grads({'target_base':target,**source},{name:1. for name in ('target_base','source_1','source_2','source_3')})
    sequential=combine(quarter,*source_grads)
    checks=dict(quarter_equals_full_over4=close(quarter,scale(full,.25)),
        replay_minus_quarter_equals_sources=close({n:mix[n]-quarter[n] for n in mix},combine(*source_grads)),
        sequential_backward_equals_component_sum=close(mix,sequential),
        zero_source_equals_quarter=close(quarter,combine(quarter,*[scale(v,0.) for v in source_grads])))
    assert all(c['passed'] for c in checks.values())
    assert all(v['present'] and v['finite'] for result in (full_bn,quarter_bn,mix_bn) for v in result.values())
    nonzero_bn={label:sum(v['nonzero'] for v in result.values()) for label,result in
                [('full',full_bn),('quarter',quarter_bn),('mix',mix_bn)]}
    assert all(value>0 for value in nonzero_bn.values())
    # One disposable optimizer update, then exact state/CPU checkpoint roundtrip.
    model=load_model(); freeze_bn(model); before=bn_buffers(model); opt=optimizer(model); criterion=model.init_criterion()
    opt.zero_grad(set_to_none=True); b=device_batch(target); loss=criterion(model(b['img']),b)[0].sum(); (4*loss).backward()
    preclip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)); opt.step()
    after=bn_buffers(model); assert all(torch.equal(v,after[k]) for k,v in before.items())
    probe=torch.linspace(0,1,3*64*64,device='cuda').reshape(1,3,64,64); model.eval()
    with torch.no_grad(): output=model(probe)
    with tempfile.TemporaryDirectory(dir='/tmp') as folder:
        path=Path(folder)/'smoke.pt'; torch.save(dict(model=copy.deepcopy(model).cpu().eval()),path)
        restored=torch.load(path,map_location='cpu')['model'].float().cuda().eval()
        with torch.no_grad(): output2=restored(probe)
    def flatten(value):
        if torch.is_tensor(value): return [value]
        if isinstance(value,dict): return sum((flatten(value[k]) for k in sorted(value)),[])
        if isinstance(value,(list,tuple)): return sum((flatten(v) for v in value),[])
        return []
    tensors1,tensors2=flatten(output),flatten(output2); assert len(tensors1)==len(tensors2) and all(torch.equal(a,b) for a,b in zip(tensors1,tensors2))
    atomic_json(DOC/'ACTUAL_MODEL_GATE.json',dict(status='PASS',gpu=gpu,torch_version=torch.__version__,
        base_input_tensor_parity=base_hashes,stream_records=all_records,
        gradient_checks=checks,C8=dict(full=full_values,quarter=quarter_values,sources=source_values,mix=mix_values),
        gradient_norms=dict(full=full_norm,quarter=quarter_norm,mix=mix_norm),
        BN_modules=len(full_bn),BN_affine_gradient_present_finite_all=True,
        BN_affine_nonzero_module_counts=nonzero_bn,
        BN_affine_nonzero_all_not_required='A particular augmented batch may give exact-zero gradient in an otherwise trainable BN affine parameter',
        BN_running_buffers_bit_exact=True,
        smoke_optimizer_updates=1,smoke_preclip_gradient_norm=preclip,smoke_clipped=preclip>10,
        checkpoint_roundtrip_output_bit_exact=True,main_optimizer_updates=0,
        augmentation_auxiliary_domain_allowlist=True,workers=0,
        caveat='Step0 actual augmented batches. Full 300-step base parity and per-batch Mosaic provenance remain mandatory in training traces.'))
    print('ACTUAL_MODEL_GATE_PASS; SMOKE_UPDATES=1; MAIN_UPDATES=0',flush=True)


if __name__=='__main__':main()
