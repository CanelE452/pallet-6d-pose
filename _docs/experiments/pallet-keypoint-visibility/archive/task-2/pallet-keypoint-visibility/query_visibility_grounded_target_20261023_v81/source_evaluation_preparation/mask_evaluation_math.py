"""GT-only saved binary-mask evaluation; no IO or model code."""
import numpy as np


def confusion_CVU(truth, prediction, known_mask):
    y, p, use = np.asarray(truth), np.asarray(prediction), np.asarray(known_mask, bool)
    assert y.shape == p.shape == use.shape
    assert np.isin(y[use], [0, 1]).all() and np.isin(p[use], [0, 1, 2]).all()
    cm = np.zeros((2, 3), np.int64)
    np.add.at(cm, (y[use].astype(int), p[use].astype(int)), 1)
    return cm


def metrics_CVU(cm):
    cm = np.asarray(cm, np.int64)
    assert cm.shape == (2, 3) and (cm >= 0).all()
    cc, cv, cu = cm[0]
    vc, vv, vu = cm[1]
    ratio = lambda n, d: None if d == 0 else float(n / d)
    return {"N": int(cm.sum()), "CM_rows_GT_CV_cols_pred_CVU": cm.tolist(),
            "C_precision_known_CV_conditional": ratio(cc, cc + vc),
            "V_precision_known_CV_conditional": ratio(vv, cv + vv),
            "C_end_to_end_recall_U_is_miss": ratio(cc, cc + cv + cu),
            "V_end_to_end_recall_U_is_miss": ratio(vv, vc + vv + vu),
            "C_accepted_only_recall_diagnostic": ratio(cc, cc + cv),
            "V_accepted_only_recall_diagnostic": ratio(vv, vc + vv),
            "hold_N": int(cu + vu), "hold_rate": ratio(cu + vu, cm.sum()),
            "C_hold_rate": ratio(cu, cm[0].sum()), "V_hold_rate": ratio(vu, cm[1].sum())}


def iou_or_zero(gt, prediction):
    """Missing prediction is always0. Actual both-empty masks yield1."""
    g = np.asarray(gt, bool)
    if prediction is None:
        return 0., 0, int(g.sum())
    p = np.asarray(prediction, bool)
    assert p.shape == g.shape
    inter = int(np.count_nonzero(g & p))
    union = int(np.count_nonzero(g | p))
    return (inter / union if union else 1.), inter, union


def sample_native_mask(mask, query, raw_hw, mask_available):
    """Fixed floor+.5; U=2 for no-mask, nonfinite/sentinel/outside query."""
    q = np.asarray(query, np.float64)
    assert q.shape == (8, 2)
    h, w = raw_hw
    prediction = np.full(8, 2, np.int8)
    for k, (x, y) in enumerate(q):
        if not np.isfinite([x, y]).all() or (x == -1 and y == -1):
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        nx, ny = int(np.floor(x + .5)), int(np.floor(y + .5))
        if not (0 <= nx < w and 0 <= ny < h) or not mask_available:
            continue
        assert mask is not None and np.shape(mask) == (h, w)
        prediction[k] = int(bool(mask[ny, nx]))
    return prediction


def generated_tests():
    cm = confusion_CVU(np.array([0, 0, 0, 1, 1, 1, -1]),
                       np.array([0, 1, 2, 0, 1, 2, 1]), np.array([1, 1, 1, 1, 1, 1, 0], bool))
    m = metrics_CVU(cm)
    assert cm.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert m["C_precision_known_CV_conditional"] == .5
    assert m["V_precision_known_CV_conditional"] == .5
    assert m["C_end_to_end_recall_U_is_miss"] == m["V_end_to_end_recall_U_is_miss"] == 1 / 3
    assert m["hold_rate"] == 1 / 3 and m["C_accepted_only_recall_diagnostic"] == .5
    empty = np.zeros((3, 4), bool)
    assert iou_or_zero(empty, empty)[0] == 1. and iou_or_zero(empty, None)[0] == 0.
    gt = empty.copy(); gt[0, 0] = True
    pred = empty.copy(); pred[0, :2] = True
    assert iou_or_zero(gt, pred) == (.5, 1, 2)
    q = np.array([[0., 0.], [.5, 0.], [3.49, 2.49], [3.5, 2.5],
                  [-.25, 1.], [-1., -1.], [float("nan"), 0.], [4., 1.]])
    p = sample_native_mask(pred, q, (3, 4), True)
    assert p.tolist() == [1, 1, 0, 2, 2, 2, 2, 2]
    assert (sample_native_mask(None, q, (3, 4), False) == 2).all()
    return {"PASS": True, "generated_only": True, "C_V_U_confusion_and_U_miss_denominators": True,
            "known_CV_precision_scope_explicit": True, "IoU_actual_both_empty1_missing0": True,
            "literal_NN_floor_plus_half_boundary_missing_query_cases": True,
            "no_GT_data_RGB_model_current_forward_fit_PnP_reads": True}
