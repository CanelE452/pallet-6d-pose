"""Generated tensors only: no source files, features, RGB, weights or GT."""
from pathlib import Path
import argparse, hashlib, inspect, json
import numpy as np
import torch
from torch.nn import functional as F
from dino_pooled_query_support_head import DinoPooledQuerySupportHead, FORWARD_FIELDS

R = Path(__file__).resolve().parent

def binding(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main(output):
    assert not output.exists()
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    model = DinoPooledQuerySupportHead()
    assert sum(p.numel() for p in model.parameters()) == 48929
    assert tuple(inspect.signature(model.forward).parameters) == FORWARD_FIELDS
    x = dict(local_features=torch.randn(3, 8, 2, 384).half(),
             target_features=torch.randn(3, 384).half(),
             context42=torch.randn(3, 8, 42),
             validity_fractions=torch.rand(3, 8, 3).half())
    untouched = {k:v.clone() for k,v in x.items()}
    assert torch.equal(model(**x)['support_logit'], torch.zeros(3, 8))
    for forbidden in ('reference_GT', 'label8', 'anchor_k', 'native_GTref8',
                      'source_id', 'teacher_id', 'mask', 'self_flag'):
        try:
            model(**x, **{forbidden:torch.zeros(1)})
        except TypeError:
            pass
        else:
            raise AssertionError(forbidden)
    # Nonzero output weights expose all projected branches for the parity test.
    with torch.no_grad():
        model.fusion[-1].weight.normal_(0, .1)
    flat = {k:(v.repeat_interleave(8, 0) if k == 'target_features'
               else v.reshape(24, *v.shape[2:])) for k,v in x.items()}
    z = model(**x)['support_logit']
    zz = model(**flat)['support_logit']
    assert torch.equal(z.flatten(), zz)
    z.sum().backward()
    g = {k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model(**flat)['support_logit'].sum().backward()
    assert all(torch.equal(g[k], p.grad) for k,p in model.named_parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    zeros = dict(local_features=torch.zeros(4, 2, 384).half(),
                 target_features=torch.zeros(4, 384).half(),
                 context42=torch.zeros(4, 42),
                 validity_fractions=torch.zeros(4, 3).half())
    assert torch.isfinite(model(**zeros)['support_logit']).all()
    assert all(torch.equal(v, untouched[k]) for k,v in x.items())
    # Unknown/nonadmitted targets must not participate in a BCE reduction.
    logits = torch.randn(8, requires_grad=True)
    labels = torch.tensor([1, 0, -1, 1, 0, -1, 1, 0])
    primary = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1], dtype=torch.bool)
    admitted = primary & ((labels == 0) | (labels == 1))
    F.binary_cross_entropy_with_logits(logits[admitted], labels[admitted].float()).backward()
    assert not logits.grad[~admitted].any() and logits.grad[admitted].abs().min() > 0
    report = dict(PASS=True, model=binding(R/'dino_pooled_query_support_head.py'),
                  test=binding(Path(__file__).resolve()), parameter_count=48929,
                  shared_projector=True, projected64_cosine=True,
                  fusion_width=356, random_seed=20261023, last_linear_zero=True,
                  whole_flat_output_bitexact=True, whole_flat_gradient_bitexact=True,
                  no_GT_or_sourceID_forward=True, zero_validity_finite=True,
                  FP16_storage_FP32_head=True, unknown_nonadmitted_gradient_zero=True,
                  source_features_RGB_GT_weights_read=0, optimizer_updates=0,
                  current_reads=0, PnP=0)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(PASS=True, proof=binding(output))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=R/'GENERATED_HEAD_TESTS.json')
    main(ap.parse_args().output)
