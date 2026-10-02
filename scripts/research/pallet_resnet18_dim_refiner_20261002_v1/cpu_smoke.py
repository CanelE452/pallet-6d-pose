"""Discarded synthetic CPU smoke plus read-only source/dimension contract audit."""
from __future__ import annotations

import json

import numpy as np
import torch

from . import common as C
from .data import EXPECTED_PARTITIONS, SourceDimensions, order_digest, paired_order
from .refiner import forward_arm, loss_for_arm, model_for_arm


def synthetic_batch():
    torch.manual_seed(20261002)
    batch = 2
    points = torch.tensor([[[18.0 + k, 20.0 + k] for k in range(9)]]).repeat(batch, 1, 1)
    return dict(
        p3=torch.randn(batch, 2, 8, 8), p4=torch.randn(batch, 3, 4, 4),
        points=points, boxes=torch.tensor([[4.0, 4.0, 59.0, 59.0]]).repeat(batch, 1),
        point_valid=torch.ones(batch, 9, dtype=torch.bool),
        input_shape=torch.tensor([[64, 64]]).repeat(batch, 1),
        gt_points=points + 1, gt_valid=torch.ones(batch, 9, dtype=torch.bool),
        dimension_context=torch.tensor([[0.2, -0.1, 0.3, 0.4, -0.2]]).repeat(batch, 1),
    )


def main():
    torch.set_num_threads(1)
    tiny = dict(c3=2, c4=3, stride3=8, stride4=16, hidden=4, encoded=4,
                stencil_fraction=0.13)
    batch = synthetic_batch()
    arms = {}
    for index, arm in enumerate(C.HEAD_ARMS):
        torch.manual_seed(300 + index)
        model = model_for_arm(arm, tiny).cpu().train()
        output = forward_arm(model, batch, arm, lam=0)
        loss = loss_for_arm(output, batch, arm)
        loss.backward()
        arms[arm] = dict(loss=float(loss), parameters=sum(p.numel() for p in model.parameters()),
                         finite_gradients=all(p.grad is not None and torch.isfinite(p.grad).all()
                                              for p in model.parameters()))
        if not arms[arm]["finite_gradients"]:
            raise RuntimeError(f"Nonfinite or missing gradient in {arm}")
    source = SourceDimensions()
    train_rows = source.rows("train")
    orders = {str(seed): order_digest(paired_order(train_rows, seed)) for seed in C.SEEDS}
    result = dict(
        PASS=True,
        production_namespace_complete=True,
        protocol_seal_performed=False,
        device="cpu",
        GPU_calls=0,
        production_optimizer_updates=0,
        actual_image_reads=0,
        actual_model_forwards=0,
        synthetic_discarded_backward_calls=len(C.HEAD_ARMS),
        source_rows=len(source.records),
        partitions=EXPECTED_PARTITIONS,
        canonical_dimensions_shape=list(source.dimensions.shape),
        normalized_context_shape=list(source.contexts.shape),
        paired_order_sha256=orders,
        arms=arms,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
