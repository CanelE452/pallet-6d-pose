"""Audit actual source positive/no-match/ignore labels without a model or update.

The exact loss function is extracted from model.py by AST, avoiding detector,
source-data and GPU imports. --capture-cache is local-only; the saved small
target snapshot can subsequently be audited with --input using CPU PyTorch.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    code = Path(__file__).resolve().with_name('model.py')
    repo = code.parents[3]
    doc = repo / '_docs/experiments/pallet_observation_refiner_20261009_v1'
    output = Path(args.output).resolve()
    snapshot = doc / 'REVIEW_LOSS_MASK_CHECK.json'
    default_recheck = doc / 'REVIEW_LOSS_MASK_RECHECK.json'
    protected = {code, Path(__file__).resolve()}
    if snapshot.exists():
        protected.add(snapshot.resolve())
    if not args.capture_cache:
        protected.add(Path(args.input).resolve())
    if (output in protected or output.name == 'REVIEW_MANIFEST.json' or
            (output.exists() and output.is_relative_to(repo) and output != default_recheck)):
        print('Refusing to overwrite a loss/source file, input snapshot, manifest, or existing research artifact; use a separate recheck path.', file=sys.stderr)
        return 1
    # Protection precedes imports, loss evaluation and backward execution.
    import torch
    torch.set_num_threads(1)
    tree = ast.parse(code.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'loss')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(code), 'exec'), scope)
    if args.capture_cache:
        import numpy as np
        cache = Path(args.capture_cache)
        order = np.load(cache / 'order.npy', mmap_mode='r')
        ids = order[0]
        arrays = {k: np.asarray(np.load(cache / (k+'.npy'), mmap_mode='r')[ids])
                  for k in ('lo', 'hi', 'weight', 'valid')}
        packet = dict(schema='recorded_source_first_batch_targets_v1',
                      selection='exact first formal batch from already locked order.npy; no score selection',
                      family_indices=ids.tolist(), fields={k: v.tolist() for k, v in arrays.items()},
                      cache_bindings=[dict(path=k+'.npy', sha256=sha(cache/(k+'.npy')))
                                      for k in ('order', 'lo', 'hi', 'weight', 'valid')],
                      real_GT_used=False)
    else:
        packet = json.loads(Path(args.input).read_text())
    fields = packet['fields']
    lo = torch.tensor(fields['lo'], dtype=torch.long)
    hi = torch.tensor(fields['hi'], dtype=torch.long)
    weight = torch.tensor(fields['weight'], dtype=torch.float64)
    valid = torch.tensor(fields['valid'], dtype=torch.bool)
    logits = torch.zeros((*lo.shape, 66), dtype=torch.float64, requires_grad=True)
    value = scope['loss'](logits, lo, hi, weight, valid)
    value.backward()
    grad = logits.grad
    positive, none, ignore = valid & (lo != 65), valid & (lo == 65), ~valid
    assert all(int(mask.sum()) > 0 for mask in (positive, none, ignore))
    image_counts = valid.sum(-1)
    nonempty = int((image_counts > 0).sum())
    expected = torch.zeros_like(logits)
    for i in range(lo.shape[0]):
        if not image_counts[i]:
            continue
        factor = 1.0 / (int(image_counts[i]) * nonempty)
        for j in range(lo.shape[1]):
            if valid[i, j]:
                expected[i, j] = factor / 66
                expected[i, j, int(lo[i, j])] -= factor * (1-float(weight[i, j]))
                expected[i, j, int(hi[i, j])] -= factor * float(weight[i, j])
    checks = dict(actual_ignore_query_gradients_exactly_zero=bool((grad[ignore] == 0).all()),
                  every_positive_query_has_nonzero_gradient=bool((grad[positive].abs().sum(-1) > 0).all()),
                  every_no_match_query_has_nonzero_gradient=bool((grad[none].abs().sum(-1) > 0).all()),
                  independent_softmax_CE_derivative_matches=bool(torch.allclose(grad, expected, atol=1e-15, rtol=1e-12)))
    packet.update(complete=True, passed=all(checks.values()), checks=checks,
                  loss_source=dict(path=str(code.relative_to(code.parents[3])), sha256=sha(code)),
                  counts=dict(images=int(lo.shape[0]), positive=int(positive.sum()),
                              no_match=int(none.sum()), true_ignore=int(ignore.sum())),
                  uniform_logits_loss=float(value),
                  maximum_gradient_difference=float((grad-expected).abs().max()),
                  ignore_gradient_abs_max=float(grad[ignore].abs().max()),
                  definition='Loss-only CPU inspection on constant logits and actual existing source targets; no head/detector inference, fit, training update or seed.',
                  model_head_forwards=0, detector_forwards=0, training_updates=0,
                  loss_forwards=1, CPU_backward_calls=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=packet['passed'], counts=packet['counts'], checks=checks)))
    return 0 if packet['passed'] else 1


if __name__ == '__main__':
    doc = Path(__file__).resolve().parents[3] / '_docs/experiments/pallet_observation_refiner_20261009_v1'
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--capture-cache', help='Existing private learned_cache directory; no feature/image reads')
    group.add_argument('--input', default=str(doc/'REVIEW_LOSS_MASK_CHECK.json'))
    parser.add_argument('--output', default=str(doc/'REVIEW_LOSS_MASK_RECHECK.json'))
    raise SystemExit(run(parser.parse_args()))
