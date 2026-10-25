"""Invented CPU tensors only; no SOURCE/model/image payloads."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).parent
P = H.parent.parent
OLD = H.parent/'V222_PAIRED_BRIDGE_TRAINING_B_V2'


def source(path, sha=None):
    path = Path(path)
    assert 'v112' not in str(path).lower()
    for p in [path, *path.parents]:
        assert not p.is_symlink()
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        b = f.read()
    if sha:
        assert hashlib.sha256(b).hexdigest() == sha
    return b


def module(path, name, sha=None):
    s = source(path, sha)
    m = type(sys)(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(s, str(path), 'exec'), m.__dict__)
    return m


def state_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    C = module(H/'source_training_contract.py', 'v223_invented_contract')
    S = module(H/'split_point_part_head.py', 'v223_invented_head')
    R = module(H/'train_split_point_part_source.py', 'v223_invented_runner')
    R.C = C
    kernels = {
        'base': (P/'query_visibility_local_support_raw_head_20261024_v181/head_preparation/local_support_head.py', '14a40c4dd14d416c1648e62b408fd7acdf74f06d4113c69785bd737b45aa848b'),
        'candidate': (P/'query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/synthetic_direct_end_candidate.py', '1668550225434f609ed70d2d62093e35bea6bd8471b9c1f5692f7cfb0b358341'),
        'spatial': (P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/fit_preparation_b/spatial_q8_conditioned_head.py', 'a03f2f4f531d78ef8a2a4733ab89026e09727907dd91991464192dd799c56ad2')}
    M = {k: module(p, 'v223_invented_'+k, sha) for k, (p, sha) in kernels.items()}
    oldclass = M['spatial'].make_spatial_q8_head_class(M['base'], M['candidate'])
    torch.manual_seed(1810)
    loaded = oldclass(spatial_q8=True)
    # Invented nonzero final weights make gradient checks meaningful.
    with torch.no_grad():
        loaded.direct_end_head.weight.normal_(0, .02)
        loaded.spatial_residual[-1].weight.normal_(0, .02)
    invented_C65 = copy.deepcopy(loaded.state_dict())
    rng = torch.random.get_rng_state().clone()
    expanded = S.make_split_target_head_class(M['spatial'])(loaded)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert sum(p.numel() for p in expanded.parameters()) == 121561
    assert len(expanded.state_dict()) == 42
    assert all(torch.equal(expanded.state_dict()[k], v) for k, v in invented_C65.items())
    other = oldclass(spatial_q8=True)
    other.load_state_dict(invented_C65, strict=True)
    duplicate = S.make_split_target_head_class(M['spatial'])(other)
    assert state_digest(expanded) == state_digest(duplicate)
    x = torch.randn(2, 8, 96, 96)
    native = torch.tensor([1, 6], dtype=torch.int64)
    mask = torch.ones(2, 8);mask[torch.arange(2), native] = 0
    xy = torch.randn(2, 8, 2)*.2;xy[mask == 0] = 0;xy = xy.reshape(2, 16)
    calls = []
    hook = expanded.encoder.register_forward_hook(lambda *args: calls.append(1))
    out = expanded(x, native, mask, xy)
    hook.remove()
    assert len(calls) == 1
    oldout = loaded(x, native, mask, xy)
    assert torch.equal(out['point_logit'], out['part_logit'])
    assert torch.equal(out['point_logit'], oldout['direct_end_external_logit'])
    assert torch.equal(out['support_logits'], oldout['support_logits'])
    gpoint = torch.autograd.grad(out['point_logit'].sum(), expanded.encoder[0].weight, retain_graph=True)[0]
    gpart = torch.autograd.grad(out['part_logit'].sum(), expanded.encoder[0].weight, retain_graph=True)[0]
    assert gpoint.abs().sum() > 0 and torch.allclose(gpoint, gpart, rtol=1e-6, atol=1e-7)

    # Actual new objective on independent logit leaves verifies each slice/route.
    point = torch.tensor([0]*16+[1]*16, dtype=torch.int8)
    natural = torch.tensor([0]*8+[1]*8, dtype=torch.int8)
    initial = torch.randn(252)
    routing = {}
    losses = []
    for separate in (False, True):
        po = initial.clone().requires_grad_();pa = initial.clone().requires_grad_()
        a = dict(point_logit=po, part_logit=pa, support_logits=torch.randn(252, 2, requires_grad=True), owner_logits=torch.randn(252, 3, requires_grad=True))
        def five_losses(z):
            base = z['support_logits'].square().mean()+z['owner_logits'].square().mean()
            return base, [base/5]*5
        loss, parts, strong, local, cf = C.routed_objective(a, point, natural, separate, five_losses)
        assert torch.allclose(loss, sum(parts)+strong+.25*local+.25*cf)
        loss.backward()
        if separate:
            assert po.grad is not None and po.grad[192:224].abs().sum() > 0
            assert po.grad[:192].abs().sum() == po.grad[224:].abs().sum() == 0
            assert pa.grad[192:224].abs().sum() == 0
        else:
            assert po.grad is None and pa.grad[192:224].abs().sum() > 0
        assert pa.grad[224:240].abs().sum() > 0 and pa.grad[240:252].abs().sum() > 0
        assert pa.grad[:192].abs().sum() == 0
        routing[str(separate)] = True

    # Control POINT's six tensors are correctly unused, not missing optimizer steps.
    expanded.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(expanded.parameters(), lr=.001, weight_decay=.0001)
    out = expanded(x, native, mask, xy)
    (out['part_logit'].square().mean()+out['support_logits'].square().mean()+out['owner_logits'].square().mean()).backward()
    unused = list(expanded.direct_end_head.parameters())+list(expanded.spatial_residual.parameters())
    assert len(unused) == 6 and all(p.grad is None for p in unused)
    opt.step()
    assert all(p not in opt.state for p in unused)

    # Common draws include CF in both arms; separate RNG state remains paired.
    def olddraw(rng, pools):
        return {k: rng.choice(v, 3, replace=True) for k, v in pools.items()}
    streams = []
    for _ in (0, 1):
        rngs = [np.random.default_rng(seed) for seed in (1810, 1811, 1812, 1813)]
        trace = []
        for step in range(5):
            d, p, w, cf = C.draw_common_and_cf(olddraw, *rngs, {'x':np.arange(8)}, [np.arange(4),np.arange(4,8)], [np.arange(3),np.arange(3,6)], 39, True)
            assert len(p) == 32 and len(w) == 16 and len(cf) == 4
            trace.append([d['x'].tolist(), p.tolist(), w.tolist(), cf.tolist()])
        streams.append(trace)
    assert streams[0] == streams[1]

    # Same decoder boundary fingerprint, including validity and condition drift.
    rgb = np.zeros((96,96,3),np.uint8);valid = np.ones((96,96),bool)
    c = np.zeros(32,np.float32);c[2] = 1
    digest = R.input_fingerprint(rgb,valid,c)
    altered = rgb.copy();altered[0,0,0] = 1
    assert digest != R.input_fingerprint(altered,valid,c)
    vv = valid.copy();vv[0,0] = False
    assert digest != R.input_fingerprint(rgb,vv,c)
    cc = c.copy();cc[8] = 1
    assert digest != R.input_fingerprint(rgb,valid,cc)
    try:
        R.run(str(H/'ROOT_ACTUAL_V223_SPLIT_TARGET_FALSE_TEMPLATE.json'))
        raise AssertionError('false entry accepted')
    except ValueError as exc:
        assert 'root actual source entry' in str(exc)
    old_contract = ast.parse(source(OLD/'source_training_contract_v2.py').decode())
    new_contract = ast.parse(source(H/'source_training_contract.py').decode())
    a = {n.name:ast.dump(n, include_attributes=False) for n in old_contract.body if isinstance(n,ast.FunctionDef)}
    b = {n.name:ast.dump(n, include_attributes=False) for n in new_contract.body if isinstance(n,ast.FunctionDef)}
    assert all(b[name] == node for name, node in a.items())
    rs = source(H/'train_split_point_part_source.py').decode()
    assert rs.index("freeze=write(out/'ALL2_FINAL_FROZEN") < rs.index("dev_manifest=js") < rs.index("fingerprint=write")
    assert 'THROUGHPUT_PROBE' not in rs and "steps==1100" in rs and "use_cf=True" in rs
    result = dict(completed=True,PASS=True,groups=['strict118359_then_clone121561_noRNG_sharedfeatures_once', 'actual_objective252_slice_and_gradient_routing', 'control_POINT6_unused_no_AdamWstate_expected', 'four_common_RNGs_CFboth', 'canonical_fingerprint_drift_and_false_entry_gate', 'inherited_contract_function_AST_and_barrier_order'],
                  inherited_contract_functions_AST_exact=len(a), state_tensors=42, parameters=121561,
                  operational_payload_or_model_weights_read=0, invented_CPU_only=True)
    with (H/'INVENTED_CHECKS.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
