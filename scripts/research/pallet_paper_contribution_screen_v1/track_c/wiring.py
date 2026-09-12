"""Trace actual Pose26 output dependencies, then test a detection-only update.

This is a wiring gate, not evidence of post-selection geometry preservation.
"""
import copy
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contracts import R0, R0_SHA, DOC, sha, tensor_sha, write

def load_model():
    assert sha(R0) == R0_SHA
    ck = torch.load(R0, map_location='cpu')
    m = (ck.get('ema') or ck['model']).float()
    m.requires_grad_(True)
    return m

def detection_parameters(m):
    head = m.model[-1]
    modules = [branch[key] for branch in (head.one2many, head.one2one)
               for key in ('box_head', 'cls_head')]
    return {id(p) for module in modules for p in module.parameters()}

def freeze_detection_only(m):
    allowed = detection_parameters(m)
    m.eval()
    # Return training tensors and sigma without updating frozen BN buffers.
    m.training = True
    head = m.model[-1]
    head.training = True
    for branch in (head.one2many, head.one2one):
        for key in ('box_head', 'cls_head'):
            branch[key].train()
    for p in m.parameters():
        p.requires_grad_(id(p) in allowed)
    return allowed

def outputs(m, x):
    out = m(x)
    assert set(out) == {'one2many', 'one2one'}, out.keys()
    return out

def objective(out, keys):
    return sum(v[key].square().mean() for v in out.values() for key in keys)

def main():
    torch.set_num_threads(4)
    torch.manual_seed(817)
    m = load_model()
    # eval BN + training output: no running-stat mutation during dependency trace.
    m.eval(); m.training = True; m.model[-1].training = True
    x = torch.rand(2, 3, 64, 64)
    params = list(m.named_parameters())
    out = outputs(m, x)
    traced = {}
    for label, keys in [('detection', ('boxes', 'scores')), ('pose', ('kpts',))]:
        grads = torch.autograd.grad(objective(out, keys), [p for _,p in params],
                                    allow_unused=True, retain_graph=True)
        traced[label] = {name for (name,p), g in zip(params, grads) if g is not None}
    allowed = detection_parameters(m)
    names = {n for n,p in params if id(p) in allowed}
    assert names <= traced['detection']
    assert not names & traced['pose']
    shared = traced['detection'] & traced['pose']
    freeze_detection_only(m)
    frozen = {n:v.clone() for n,v in m.state_dict().items()
              if not any(n == a or n.startswith(a.rsplit('.',1)[0]+'.') for a in names)}
    # Exact frozen buffer inventory by module ownership, not string-based training.
    train_modules = set()
    for branch in (m.model[-1].one2many, m.model[-1].one2one):
        for key in ('box_head', 'cls_head'):
            train_modules.update(id(mod) for mod in branch[key].modules())
    mutable = set(names)
    for prefix, mod in m.named_modules():
        if id(mod) in train_modules:
            mutable.update(prefix+'.'+n for n,_ in mod.named_buffers(recurse=False))
    frozen = {n:v.clone() for n,v in m.state_dict().items() if n not in mutable}
    before = outputs(m, x)
    raw_before = {k:v['kpts'].detach().clone() for k,v in before.items()}
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    assert {id(p) for g in opt.param_groups for p in g['params']} == allowed
    loss = objective(before, ('boxes', 'scores'))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    opt.step()
    after = outputs(m, x)
    assert all(torch.equal(v, m.state_dict()[n]) for n,v in frozen.items())
    assert all(torch.equal(raw_before[k], after[k]['kpts']) for k in after)
    receipt = dict(status='PASS', evidence_level='MECHANISM_ONLY',
        checkpoint_sha256=sha(R0), synthetic_probe_shape=list(x.shape),
        allowlist=sorted(names), trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        shared_detection_pose=sorted(shared), dependency_graph={k:sorted(v) for k,v in traced.items()},
        frozen_state_sha256=tensor_sha(frozen), frozen_tensors_equal=True,
        raw_keypoints_bit_exact=True, optimizer_allowlist_equal=True,
        finite_gradient=True, probe_optimizer_updates=1, student_optimizer_updates=0,
        caveat='Class-dependent final detection selection can change; dense pose invariance does not prove matched-frame geometry non-regression.')
    write(DOC/'C_geometry_preserving_da'/'MECHANISM_RESULT.json',receipt)
    print({k:v for k,v in receipt.items() if k not in ('allowlist','shared_detection_pose','dependency_graph')})

if __name__ == '__main__':
    main()
