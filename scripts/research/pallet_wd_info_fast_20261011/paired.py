"""Frame-weighted paired effects with session clusters and exact binary counts."""
import hashlib
import numpy as np
from . import common as C

CONTRASTS = [('G_V_MLP', 'G_MLP'), ('G_V_D_MLP', 'G_V_MLP'),
             ('D_JOINT_DIM_SUBPIX', 'D_CONTROL_SUBPIX'),
             ('G_V_D_MLP', 'OFFICIAL')]
METRICS = ('wd_confusion', 'pose_success')


def bootstrap_effect(before, after, sessions, draws):
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    assert before.shape == after.shape and before.ndim == 2
    names = sorted(set(sessions)); seeds = before.shape[1]
    counts = np.array([sum(s == name for s in sessions) for name in names], dtype=np.int64)
    totals = np.array([(after[np.array(sessions) == name] -
                       before[np.array(sessions) == name]).sum() for name in names], dtype=np.int64)
    full_n = int(totals.sum()); full_d = int(counts.sum() * seeds)
    multiplicity = np.stack([(draws == i).sum(1) for i in range(len(names))], axis=1)
    draw_n = multiplicity @ totals
    draw_d = multiplicity @ (counts * seeds)
    assert np.all(draw_d > 0)
    sample = draw_n / draw_d * 100
    effect = full_n / full_d * 100
    # |draw_n/draw_d-full_n/full_d| >= |full_n/full_d|.
    # Integer cross-products preserve ties for binary seed-averaged outcomes.
    if full_n > 0:
        extreme = (draw_n <= 0) | (draw_n * full_d >= 2 * full_n * draw_d)
    elif full_n < 0:
        extreme = (draw_n >= 0) | (draw_n * full_d <= 2 * full_n * draw_d)
    else:
        extreme = np.ones(len(draws), dtype=bool)
    equal_session = totals / (counts * seeds) * 100
    return dict(effect_pp=effect, CI95=np.quantile(sample, [.025, .975]).tolist(),
                p_raw=float((1 + extreme.sum()) / (1 + len(draws))),
                numerator=full_n, denominator=full_d, frames=len(sessions),
                seeds=seeds, sessions=names, session_equal_effect_pp=float(equal_session.mean()),
                estimand='frame-weighted; seeds within frame; paired session clusters',
                p_definition='centered paired bootstrap, exact integer ties, plus-one',
                uncertainty='exploratory reused DEV; not independent confirmation')


def holm(values):
    assert len(values) == 8
    order = np.argsort(values, kind='stable'); adjusted = np.empty(8)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (8-rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def save_draws(sessions):
    names = sorted(set(sessions))
    assert len(names) == 9
    draw = np.random.default_rng(20260917).integers(0, len(names), size=(10000, len(names)))
    destination = C.DOC / 'BOOTSTRAP_DRAWS.npy'
    assert not destination.exists()
    np.save(destination, draw, allow_pickle=False)
    C.write(C.DOC / 'BOOTSTRAP_LOCK.json', dict(seed=20260917, repetitions=10000,
        sessions=names, sha256=C.sha(destination), array_sha256=hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest(),
        unit='actual allowed session; same frame/all seeds/both candidates stay together',
        existing_13_session_draws_reused=False, reason='New population has 9 actual allowed sessions'))
    return draw
