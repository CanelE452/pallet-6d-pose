"""Independent CPU leaf-logit audit. No source/cache/weights/model forward."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'query_visibility_controlled_real_20261022_v66'
TRAINER = ROOT / 'model/fit_partial_head.py'
MODEL = ROOT / 'model/query_global_amodal_head.py'
OUT = ROOT / 'source_audit/INDEPENDENT_OBJECTIVE_CONTROL_REVIEW.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ref(p):
    return dict(path=str(p.resolve()), sha256=sha(p))

def prohibited(*args, **kwargs):
    raise AssertionError('No cache, weights, forward, optimizer or GPU operation is allowed')

def load_code(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def manual_reduce(value, known, valid, codes, per_target):
    eligible = known & valid
    terms = []
    for domain in sorted(set(codes.tolist())):
        members = [i for i in range(len(codes)) if int(codes[i]) == domain and bool(eligible[i].any())]
        if not members:
            continue
        if per_target:
            terms.append(torch.stack([value[i][eligible[i]].mean() for i in members]).mean())
        else:
            terms.append(torch.cat([value[i].reshape(-1)[eligible[i].reshape(-1)] for i in members]).mean())
    return torch.stack(terms).mean() if terms else value.sum() * 0

def code_function(path, name):
    tree = ast.parse(path.read_text())
    found = [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name]
    assert len(found) == 1, name
    return ast.dump(found[0], include_attributes=False)

def main():
    assert not OUT.exists(), 'Preserve an earlier independent audit'
    torch.set_num_threads(1)
    torch.manual_seed(6701)
    np.load = prohibited
    torch.load = prohibited
    torch.cuda.init = prohibited
    torch.optim.AdamW.step = prohibited
    sys.path.insert(0, str(ROOT / 'model'))
    new = load_code('independent_v67_math', TRAINER)
    old = load_code('independent_v66_math', BASE / 'model/fit_partial_head.py')
    new.load_cache = prohibited
    old.load_cache = prohibited
    new.QueryGlobalAmodalHead.forward = prohibited
    old.QueryGlobalAmodalHead.forward = prohibited
    built_in = new.visible_smoke()
    assert built_in['PASS']

    # Enumerate all 63 nonempty subsets of the six legal joint states.
    masks = torch.arange(1, 64, dtype=torch.long)
    exact_is_visible = (True, False, False, False, False, False)
    semantic_known, semantic_positive = [], []
    for mask in masks.tolist():
        possibilities = {exact_is_visible[j] for j in range(6) if mask & (1 << j)}
        semantic_known.append(len(possibilities) == 1)
        semantic_positive.append(possibilities == {True})
    known_ref = torch.tensor(semantic_known)
    positive_ref = torch.tensor(semantic_positive)
    z = torch.randn(63, 6, dtype=torch.float64, requires_grad=True)
    values, known = new.visible_marginal_nll(z, masks)
    assert torch.equal(known, known_ref)
    log_odds = z[:, 0] - torch.logsumexp(z[:, 1:], 1)
    independent = F.binary_cross_entropy_with_logits(log_odds, positive_ref.double(), reduction='none')
    independent = torch.where(known_ref, independent, torch.zeros_like(independent))
    max_value_error = float((values - independent).abs().max())
    assert torch.allclose(values, independent, atol=2e-15, rtol=2e-15)
    actual_grad = torch.autograd.grad(values.sum(), z, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(independent.sum(), z)[0]
    max_gradient_error = float((actual_grad - reference_grad).abs().max())
    assert torch.allclose(actual_grad, reference_grad, atol=2e-15, rtol=2e-15)
    assert torch.equal(actual_grad[~known], torch.zeros_like(actual_grad[~known]))
    p = z.detach().softmax(1)
    analytic = torch.zeros_like(p)
    analytic[0] = p[0]
    analytic[0, 0] -= 1
    for i, mask in enumerate(masks.tolist()):
        if not (mask & 1):
            analytic[i] = p[i]
            analytic[i, 1:] -= z.detach()[i, 1:].softmax(0)
    assert torch.allclose(actual_grad, analytic, atol=2e-15, rtol=2e-15)

    normalization_cases = 0
    zero_cases = 0
    # Independent domain means, sparse known bits, masked cells, empty targets.
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        zz = torch.randn(7, 6, 3, 4, generator=g, dtype=torch.float64, requires_grad=True)
        aa = torch.randint(1, 64, (7, 3, 4), generator=g)
        aa[-1] = 63
        vv = torch.rand((7, 3, 4), generator=g) > .3
        cc = torch.tensor([5, 5, 8, 8, 8, 9, 11])
        lv, kk = new.visible_marginal_nll(zz, aa)
        for per_target in (False, True):
            got = new.visible_domain_loss(zz, aa, vv, cc, per_target=per_target)
            want = manual_reduce(lv, kk, vv, cc, per_target)
            assert torch.allclose(got, want, atol=2e-15, rtol=2e-15)
            normalization_cases += 1
        # All odd>1 are visibility-ambiguous even if partial NLL informative.
        odd = torch.full((7, 3, 4), 3, dtype=torch.long)
        nofact = new.visible_domain_loss(zz, odd, vv, cc, per_target=True)
        invalid = new.visible_domain_loss(zz, aa, torch.zeros_like(vv), cc)
        for value in (nofact, invalid):
            assert value.item() == 0
            assert torch.equal(torch.autograd.grad(value, zz, retain_graph=True)[0], torch.zeros_like(zz))
            zero_cases += 1

    # Direct V66 formula on generated center/local/global logits. No fit call.
    lambda0_cases = 0
    for seed in range(8):
        torch.manual_seed(100 + seed)
        values = [torch.randn(5, 6, dtype=torch.float64), torch.randn(5, 6, 9, 9, dtype=torch.float64), torch.randn(3, 6, 4, 4, dtype=torch.float64)]
        za = [v.clone().requires_grad_() for v in values]
        zb = [v.clone().requires_grad_() for v in values]
        aa = torch.randint(1, 64, (5, 9, 9)); vv = torch.rand(5, 9, 9) > .2
        ga = torch.randint(1, 64, (3, 4, 4)); gv = torch.rand(3, 4, 4) > .2
        cc = torch.tensor([0, 0, 1, 1, 2]); gc = torch.tensor([0, 1, 2])
        expected = old.domain_loss(old.partial_nll(za[0], aa[:, 4, 4]), aa[:, 4, 4], vv[:, 4, 4], cc)
        expected = expected + .25 * old.domain_loss(old.partial_nll(za[1], aa), aa, vv, cc)
        expected = expected + .25 * old.domain_loss(old.partial_nll(za[2], ga), ga, gv, gc, per_target=True)
        terms = new.objective_terms(*zb, aa, vv, cc, ga, gv, gc, visible_bce_weight=0.)
        assert torch.equal(expected, terms['total'])
        for go, gn in zip(torch.autograd.grad(expected, za), torch.autograd.grad(terms['total'], zb)):
            assert torch.equal(go, gn)
        lambda0_cases += 1

    # Invalid compatible sets rejected rather than becoming fabricated negatives.
    bad_rejections = 0
    for mask in (0, 64, -1):
        try:
            new.visible_marginal_nll(torch.zeros(1, 6), torch.tensor([mask]))
        except ValueError:
            bad_rejections += 1
        else:
            raise AssertionError(mask)

    baseline_model = BASE / 'model/query_global_amodal_head.py'
    assert sha(MODEL) == sha(baseline_model)
    unchanged = {}
    for name in ('partial_nll', 'domain_loss', 'load_cache', 'tensor', 'inputs', 'validation'):
        unchanged[name] = code_function(TRAINER, name) == code_function(BASE / 'model/fit_partial_head.py', name)
        assert unchanged[name], name
    protocol_path = ROOT / 'FIT_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    if protocol is not None:
        assert protocol['visible_bce_weight'] == 1.
        assert protocol['updates'] == 2000 and protocol['validation_every'] == 100 and protocol['seed'] == 20261022
    report = dict(
        schema='v67_independent_objective_control_math_review_v1', PASS=True,
        scope='Code plus generated CPU leaf logits only. No source/cache/checkpoint/current data opened.',
        trainer=ref(TRAINER), model=ref(MODEL), baseline_trainer=ref(BASE / 'model/fit_partial_head.py'),
        baseline_model=ref(baseline_model), reviewer_code=ref(Path(__file__)),
        fit_protocol=ref(protocol_path) if protocol is not None else None,
        built_in_generated_smoke=built_in,
        independent_tests=dict(
            all_nonempty_joint_subsets=63, known_positive_masks=1, known_nonvisible_masks=31, unknown_masks=31,
            marginal_BCE_max_abs_error=max_value_error, leaf_logit_gradient_max_abs_error=max_gradient_error,
            analytic_gradient_verified=True, unknown_gradient_bit_exact_zero=True,
            manual_source_and_target_normalization_cases=normalization_cases,
            all_unknown_or_invalid_zero_gradient_cases=zero_cases,
            lambda0_bit_exact_scalar_and_three_leaf_gradient_cases=lambda0_cases,
            invalid_mask_rejections=bad_rejections, unchanged_function_AST=unchanged,
            baseline_architecture_byte_identical=True),
        findings=[], blockers=[],
        mathematical_interpretation=[
            'Exact V is only joint VV: allowed1. Every nonempty even mask excludes V and certifies nonvisible; every odd mask greater than1 leaves exact visibility unknown.',
            'Added BCE supervises exact-visible versus exact-nonvisible at all three existing scales. It adds no radius2-visible BCE and invents no E/O/self label.',
            'This is not a positive-only V-preservation objective: known negatives also receive BCE. It remains a legitimate one-factor known-marginal objective change.',
            'BCE uses its own known-and-valid denominator within each source, and unique-target means at global scale. Thus known-V weights are not generally exactly doubled relative to V66.',
            'At a fully known isolated VV target, BCE equals its partial NLL; the isolated generated fixture doubles. This special case does not generalize to mixed partial masks.',
            'Architecture, data sampling, V66 partial-NLL checkpoint selector, fixed binary threshold, and budget are unchanged. Source release gates must not be used for checkpoint selection.',
            'Generated loss/gradient checks do not establish a feature-branch or parameter-level causal mechanism.'],
        source_admission_status='Fit authorization and immutable source input/admission bindings are owned by root; this math audit does not authorize fitting or current evaluation.',
        actual_cache_reads=0, actual_weight_reads=0, actual_current_reads=0,
        model_forward_calls=0, optimizer_updates=0, GPU_calls=0)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(PASS=True, report=ref(OUT), independent_tests=report['independent_tests'])))

if __name__ == '__main__':
    main()
