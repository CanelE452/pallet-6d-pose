# Diagnostic fix log

## FIX-01: DiffPnP3D non-finite observation guard

### Before

A batch with one NaN 2D observation failed before the documented guard:

```text
torch._C._LinAlgError: linalg.eigh: (Batch element 0):
The algorithm failed to converge because the input matrix is ill-conditioned
or has too many repeated eigenvalues
```

The old self-test checked only that the returned scalar was finite on CUDA; it
did not require backward gradients to remain finite and did not expose the CPU
pre-guard exception.

### Minimal change

`Deep_Object_Pose/train/diffpnp3d_loss.py` now:

1. records a per-frame `pred_finite` flag before the GN solve;
2. replaces only NaN/Inf observations with finite zero values for the internal
   batched algebra;
3. includes `pred_finite` in the final valid mask;
4. therefore assigns zero loss and zero gradient to the rejected frame while
   preserving the other frames.

For every finite input, `torch.nan_to_num` is value-identical. No frozen
inference code uses this loss path.

### After

| check | result |
|---|---:|
| mixed batch scalar finite | PASS |
| invalid-frame gradient finite | PASS |
| valid-frame gradient finite | PASS |
| current DiffPnP3D finite-difference relative L2 | `1.014e-9` PASS |
| current oracle reprojection/yaw | PASS |
| new audit pytest | 3/3 PASS |
| existing related regression pytest | 20/20 PASS |
| existing end-to-end DiffPnP3D self-test | PASS |

The detailed values are in `unit_audit.json`.

## Unfixed/blocking findings

- Legacy BPnP backward does not match finite differences: relative L2 error is
  `0.537–0.545` over epsilon `0.001–0.1 px`; legacy `--geo_loss` training is
  blocked.
- Canonical ep57 has no covariance-weighted PnP or covariance head. Local
  moments are detached diagnostics only.
- GT belief generation performs its MSE arithmetic in float64 before the
  float32 accumulator cast. This is a cost/mixed-dtype issue, not a demonstrated
  ep57 correctness failure; it is left unchanged pending a controlled training
  equivalence test.
- `-c/--config`, `config/default.yaml`, the constructor `pretrained` argument,
  and the instantiated torchvision resize transform do not control the
  canonical path as their names/comments imply. They are documented in the
  code audit and left unchanged to avoid silently changing historical runs.
