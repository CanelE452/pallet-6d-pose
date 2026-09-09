"""Symmetric two-task PCGrad, summed, on their structural dependency intersection.

The original gradient sum is preserved bit for bit in the absence of conflict.
Task-private coordinates are never projected. A connected zero gradient is not
the same as an absent graph edge. This module performs no model forward or RNG.
"""
from __future__ import annotations

import hashlib
import torch


def parameter_dependencies(loss):
    """Parameter identity set reachable through the actual scalar autograd graph."""
    pending = [loss.grad_fn] if loss.grad_fn is not None else []
    seen, result = set(), set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor) and variable.is_leaf:
            result.add(id(variable))
        pending.extend(n for n, _ in node.next_functions if n is not None)
    return result


def dot_stats(first, second, mask):
    active = [(a, b) for a, b, keep in zip(first, second, mask) if keep]
    if not active:
        return dict(dot=0., norm_a=0., norm_b=0., cosine=None, conflict=False)
    dtype = torch.float64 if active[0][0].dtype == torch.float64 else torch.float32
    dot = torch.stack([(a.to(dtype) * b.to(dtype)).sum() for a, b in active]).sum()
    a2 = torch.stack([a.to(dtype).square().sum() for a, _ in active]).sum()
    b2 = torch.stack([b.to(dtype).square().sum() for _, b in active]).sum()
    values = torch.stack([dot, a2, b2]).detach().cpu().tolist()
    if not all(__import__('math').isfinite(v) for v in values):
        raise FloatingPointError("Nonfinite task gradients")
    dot, a2, b2 = values
    return dict(dot=dot, norm_a=a2 ** .5, norm_b=b2 ** .5,
                cosine=dot / (a2 * b2) ** .5 if a2 > 0 and b2 > 0 else None,
                conflict=dot < 0 and a2 > 0 and b2 > 0)


def project_from_sum(total, auxiliary, main_present, *, enabled=True):
    """Return projected SUM plus statistics, deriving main = total - auxiliary.

    ``main_present`` comes from the main loss's actual graph, not from total.grad.
    Returned private/nonconflicting tensors alias total and remain unchanged.
    """
    if not (len(total) == len(auxiliary) == len(main_present)):
        raise ValueError("Gradient/mask lengths differ")
    mask = [bool(present and t is not None and b is not None)
            for t, b, present in zip(total, auxiliary, main_present)]
    main = [(t - b) if keep else t for t, b, keep in zip(total, auxiliary, mask)]
    stats = dot_stats(main, auxiliary, mask)
    output = list(total)
    if enabled and stats['conflict']:
        # Both projections use the ORIGINAL task gradients (symmetric two-task).
        ca = stats['dot'] / stats['norm_b'] ** 2
        cb = stats['dot'] / stats['norm_a'] ** 2
        for i, keep in enumerate(mask):
            if keep:
                output[i] = total[i] - ca * auxiliary[i] - cb * main[i]
    stats.update(applied=bool(enabled and stats['conflict']), shared_tensors=sum(mask),
                 shared_elements=sum(t.numel() for t, keep in zip(total, mask) if keep))
    return output, stats, mask, main


def mask_digest(names, mask):
    selected = [name for name, keep in zip(names, mask) if keep]
    return selected, hashlib.sha256(('\n'.join(selected) + '\n').encode()).hexdigest()


def group_statistics(names, first, second, mask):
    predicates = {
        'backbone_neck': lambda n: not n.startswith('model.23.'),
        'p4_adapter': lambda n: '.hough.reduce.' in n,
        'hough_parameter_domain': lambda n: '.hough.hough_layers.' in n,
        'semantic_line_head': lambda n: '.hough.line_head.' in n,
        'hough_all': lambda n: '.hough.' in n,
        'shared_all': lambda n: True,
    }
    return {group: dot_stats(first, second, [keep and pred(n) for n, keep in zip(names, mask)])
            for group, pred in predicates.items()}
