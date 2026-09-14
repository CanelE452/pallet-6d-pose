"""Locked algebra and full-denominator metric helpers; not a training runner."""
import hashlib
import numpy as np

ARM_WEIGHTS = {
    'T8_FULL': {'target_base': 1.0},
    'T8_QUARTER': {'target_base': .25},
    'REPLAY': {'target_base': .25, 'source_1': .25, 'source_2': .25, 'source_3': .25},
    'T32_COMPUTE': {'target_base': .25, 'target_extra_1': .25,
                    'target_extra_2': .25, 'target_extra_3': .25},
}
MICROBATCH_SIZE = 8
BACKWARD_SCALE = 32
UPDATES = 300


def backward_coefficients(arm):
    return {s: BACKWARD_SCALE * w / MICROBATCH_SIZE for s, w in ARM_WEIGHTS[arm].items()}


def exposure_plan(arm):
    streams = ARM_WEIGHTS[arm]
    real = sum(s.startswith('target_') for s in streams) * MICROBATCH_SIZE * UPDATES
    synthetic = sum(s.startswith('source_') for s in streams) * MICROBATCH_SIZE * UPDATES
    return dict(real=real, synthetic=synthetic, total=real + synthetic, updates=UPDATES)


def stream_seed(seed, step, stream, slot):
    if stream not in set().union(*(set(s) for s in ARM_WEIGHTS.values())):
        raise ValueError(stream)
    value = f'replay-control-v1\n{seed}\n{step}\n{stream}\n{slot}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big')


def pck_counts(gt_xy, supervision, predicted_xy, matched, thresholds=(5., 10., 20.)):
    """Missing/nonfinite predictions score zero; never discard supervised GT.

    Caller must use the frozen highest-score candidate and IoU >= .5 rule.
    This helper neither chooses a candidate using GT nor performs matching.
    """
    gt = np.asarray(gt_xy, dtype=float)
    mask = np.asarray(supervision, dtype=bool)
    if gt.shape != (len(mask), 2) or not np.isfinite(gt[mask]).all():
        raise ValueError('Invalid supervised GT')
    errors = np.full(len(mask), np.inf)
    if matched and predicted_xy is not None:
        pred = np.asarray(predicted_xy, dtype=float)
        if pred.shape != gt.shape:
            raise ValueError('Point-role contract mismatch')
        finite = np.isfinite(pred).all(axis=1)
        errors[finite] = np.linalg.norm(pred[finite] - gt[finite], axis=1)
    return dict(denominator=int(mask.sum()),
                numerators={str(t): int(np.sum(mask & (errors <= t))) for t in thresholds})
