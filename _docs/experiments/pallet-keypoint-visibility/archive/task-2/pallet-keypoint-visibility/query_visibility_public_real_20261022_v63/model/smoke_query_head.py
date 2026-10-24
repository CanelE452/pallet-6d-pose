"""CPU smoke: exact center readout, spatial locality, gradients, 2 updates only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from query_head import HeadConfig, QueryHead, parameter_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    cache, label = np.load(args.features), np.load(args.labels)
    indices = label["train"][:4]
    x = torch.from_numpy(cache["x"][indices].reshape(-1, 65, 9, 9)).float()
    y = torch.from_numpy(label["y"][indices].reshape(-1)).float()
    valid = torch.from_numpy(label["valid"][indices].reshape(-1))
    model = QueryHead()
    scalar, dense = model(x, return_dense=True)
    assert scalar.shape == (32,) and dense.shape == (32, 9, 9)
    assert torch.equal(scalar, dense[:, 4, 4])
    assert parameter_count(model) <= 100000
    changed = x.clone()
    changed[:, :, 0, 0] += 100
    center_before, center_after = model(x), model(changed)
    assert torch.equal(center_before, center_after), "remote corner changed local center readout"
    gap = QueryHead(HeadConfig(architecture="gap"))
    gap.load_state_dict(model.state_dict())
    remote_gap_change = float((gap(x) - gap(changed)).abs().max())
    assert remote_gap_change > 0
    probe = x[:2].clone().requires_grad_(True)
    model(probe).sum().backward()
    assert torch.equal(probe.grad[:, :, 0, 0], torch.zeros_like(probe.grad[:, :, 0, 0]))
    assert probe.grad[:, :, 4, 4].abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits[valid], y[valid])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    rgb_head = QueryHead(HeadConfig(rgb_branch=True))
    assert rgb_head(x[:2], torch.rand(2, 3, 33, 33)).shape == (2,)
    report = {"PASS": True, "device": "cpu", "parameters": parameter_count(model),
              "parameters_rgb_variant": parameter_count(rgb_head), "center_scalar_exact": True,
              "outside_center_RF_invariant": True, "center_gradient_nonzero": True,
              "remote_gap_change": remote_gap_change, "optimizer_updates": 2,
              "losses": losses, "smoke_only": True, "checkpoint_saved": False,
              "real_training_queries_used": 0, "eval_queries_used": 0}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
