"""Small functional checks; no model, final pose or target annotations."""
from unittest.mock import patch
import cv2
import numpy as np
from . import common as C


def run():
    gray = np.random.default_rng(17).integers(0, 256, (90, 120), dtype=np.uint8)
    q0 = np.array([[20+10*k, 35] for k in range(8)] + [[60, 45]], dtype=np.float64)
    support = np.ones(9, bool)
    native, _ = C.correct(gray, q0, support, 'SUBPIX')
    alone = C.cap_points(q0, native, 120, 90, support)
    _, combined, _ = C.combine(gray, q0, q0, support)
    assert np.array_equal(alone, combined)
    qN = q0.copy(); qN[:8, 0] += .5
    with patch.object(C, 'correct', return_value=(qN.copy(), {'failure': True})):
        qS, final, _ = C.combine(gray, q0, qN, support)
        assert np.array_equal(qS, qN) and np.array_equal(final, qN)
    with patch.object(cv2, 'cornerSubPix', side_effect=cv2.error('synthetic failure')):
        qS, final, diagnostics = C.combine(gray, q0, qN, support)
        assert np.array_equal(qS, qN) and np.array_equal(final, qN)
        assert diagnostics['fallback_counts'] == {'function_error': 8}
    tiny_q0 = q0.copy(); tiny_q0[:8, 1] = 3
    tiny_qN = tiny_q0.copy(); tiny_qN[:8, 0] += .5
    qS, final, diagnostics = C.combine(gray[:8, :], tiny_q0, tiny_qN, support)
    assert diagnostics['fallback_counts'] == {'image_too_small': 8}
    assert np.array_equal(qS, tiny_qN) and np.array_equal(final, tiny_qN)
    # A deliberately overlarge correction proves the bound is around q0.
    far = qN.copy(); far[:8, 0] += 100
    with patch.object(C, 'correct', return_value=(far, {})):
        _, final, _ = C.combine(gray, q0, qN, support)
    assert np.allclose(np.linalg.norm(final[:8]-q0[:8], axis=-1), 1.5, rtol=0, atol=1e-12)
    support[2] = False; q0[3] = [-1, -1]; qN[2:4] = q0[2:4]
    q0[4] = [np.nan, np.nan]; qN[4] = q0[4]
    before0, beforeN, before_support = q0.copy(), qN.copy(), support.copy()
    _, final, _ = C.combine(gray, q0, qN, support)
    assert np.array_equal(final[[2,3,4,8]], q0[[2,3,4,8]], equal_nan=True)
    assert np.array_equal(q0, before0, equal_nan=True)
    assert np.array_equal(qN, beforeN, equal_nan=True)
    assert np.array_equal(support, before_support)
    # Outside and nonfinite returned points retain the N3 start.
    for bad in (np.array([[[-1., 3.]]], np.float32), np.array([[[np.nan, 3.]]], np.float32)):
        with patch.object(cv2, 'cornerSubPix', return_value=bad):
            qS, _, _ = C.combine(gray, q0, qN, support)
        assert np.array_equal(qS, qN, equal_nan=True)
    return dict(status='PASS', qN_equals_q0_matches_SUBPIX=True,
        identity_and_failure_keep_qN_within_cap=True, OpenCV_exception_fallback=True,
        outside_and_nonfinite_output_fallback=True, tiny_image_fallback=True,
        strict_total_cap_from_Base=True, center_missing_unsupported_preserved=True,
        input_arrays_unmodified=True, inference_inputs=['gray','q0','qN','prediction_support'],
        GT_inputs=False, final_F_calls=0, model_calls=0)


if __name__ == '__main__':
    print(C.finite(run()))
