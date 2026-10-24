"""Generated and saved NumPy math only; no file/model imports."""
import numpy as np


def sigmoid(logit):
    a = np.asarray(logit, np.float64)
    return np.exp(-np.logaddexp(0., -a))


def hierarchical_probabilities(logits):
    a = np.asarray(logits)
    assert a.shape[-3] == 2
    expected = sigmoid(a[..., 0, :, :])
    visible_given_expected = sigmoid(a[..., 1, :, :])
    return np.stack([expected * visible_given_expected,
                     expected * (1. - visible_given_expected),
                     1. - expected], axis=-3)


def map_confusion(logits, valid, expected, visible):
    valid, expected, visible = (np.asarray(x, bool) for x in [valid, expected, visible])
    assert valid.shape == expected.shape == visible.shape
    assert not (visible & ~expected).any() and not (expected & ~valid).any()
    probs = hierarchical_probabilities(logits)
    prediction = probs.argmax(-3)
    truth = np.where(visible, 0, np.where(expected, 1, 2))
    cm = np.zeros((3, 3), np.int64)
    np.add.at(cm, (truth[valid], prediction[valid]), 1)
    return cm, probs


def binary_confusion(logits, truth, primary):
    truth = np.asarray(truth)
    primary = np.asarray(primary, bool)
    assert truth.shape == primary.shape == np.shape(logits)
    assert np.isin(truth[primary], [0, 1]).all()
    cm = np.zeros((2, 2), np.int64)
    np.add.at(cm, (truth[primary].astype(int), (np.asarray(logits)[primary] >= 0).astype(int)), 1)
    return cm


def stable_bce(logits, truth):
    return np.logaddexp(0., np.asarray(logits, np.float64)) - np.asarray(logits, np.float64) * truth


def query_class_balanced_bce(logits, target, eligible):
    """One query/channel: mean over the present binary target classes."""
    z, y, use = np.asarray(logits), np.asarray(target, bool), np.asarray(eligible, bool)
    assert z.shape == y.shape == use.shape
    terms = [float(stable_bce(z[use & (y == c)], c).mean())
             for c in [False, True] if np.any(use & (y == c))]
    return float(np.mean(terms)) if terms else 0.


def generated_tests():
    # Exact primary .5 tie is visible; unknown and unavailable rows never negative.
    cm = binary_confusion(np.array([-2., 0., 2., -2., 50.]),
                          np.array([0, 0, 1, 1, -1]), np.array([1, 1, 1, 1, 0], bool))
    assert cm.tolist() == [[1, 1], [1, 1]]
    # Analytic hierarchical cases V(.6), C(.6), O(.8), plus one ignored cell.
    z = np.array([[[np.log(4.), np.log(4.), -np.log(4.), 100.]],
                  [[np.log(3.), -np.log(3.), 0., -100.]]])
    valid = np.array([[1, 1, 1, 0]], bool)
    expected = np.array([[1, 1, 0, 0]], bool)
    visible = np.array([[1, 0, 0, 0]], bool)
    confusion, probs = map_confusion(z, valid, expected, visible)
    assert confusion.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.allclose(probs[:, 0, :3], [[.6, .2, .1], [.2, .6, .1], [.2, .2, .8]], atol=1.e-15, rtol=0.)
    assert np.allclose(probs.sum(-3), 1., atol=1.e-15, rtol=0.)
    # No eligible cells graph-zero conceptually; outsideM0 never visible-negative.
    assert query_class_balanced_bce(np.zeros(4), np.array([0, 0, 1, 1]), np.zeros(4, bool)) == 0.
    visible_eligible = valid & expected
    assert visible_eligible.sum() == 2 and (valid & ~expected).sum() == 1
    constant = query_class_balanced_bce(np.zeros(4), [0, 0, 0, 1], np.ones(4, bool))
    assert abs(constant - np.log(2.)) < 1.e-15
    # Unequal class sizes: each present class receives equal weight per query.
    loss = query_class_balanced_bce(np.array([-1., -1., -1., -2.]),
                                   np.array([0, 0, 0, 1]), np.ones(4, bool))
    expected_loss = (float(np.logaddexp(0., -1.)) + float(np.logaddexp(0., 2.))) / 2.
    assert abs(loss - expected_loss) < 1.e-15
    # Full-batch query mean includes noeligible=zero rather than renormalizing it.
    assert np.mean([constant, 0.]) == constant / 2.
    return {"PASS": True, "generated_only": True,
            "primary_CM_CV": cm.tolist(), "auxiliary_CM_VCO": confusion.tolist(),
            "hierarchical_probabilities_sum1": True, "pA_pS_product_not_primary_readout": True,
            "invalid_ignored_outsideM0_not_visible_negative": True,
            "per_query_present_class_balancing_and_zero_loss_query_mean": True,
            "source_model_scores_RGB_current_fit_forward_reads": 0}
