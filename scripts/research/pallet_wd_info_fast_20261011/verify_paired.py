"""Independent rational check of bootstrap ties and Holm correction."""
from fractions import Fraction
import numpy as np
from .paired import bootstrap_effect, holm
from . import common as C


def main():
    rng = np.random.default_rng(19)
    checks = []
    for seed_count in [1, 3]:
        sessions = ['a'] * 7 + ['b'] * 5 + ['c'] * 4
        before = rng.integers(0, 2, size=(16, seed_count))
        for after in [before.copy(), 1-before, rng.integers(0, 2, size=before.shape)]:
            draws = rng.integers(0, 3, size=(1000, 3))
            actual = bootstrap_effect(before, after, sessions, draws)
            observed = Fraction(int((after-before).sum()), 16*seed_count)
            names = sorted(set(sessions)); extreme = 0
            for draw in draws:
                positions = [i for cluster in draw for i, s in enumerate(sessions)
                             if s == names[int(cluster)]]
                sample = Fraction(int((after[positions]-before[positions]).sum()), len(positions)*seed_count)
                extreme += abs(sample-observed) >= abs(observed)
            expected = (extreme+1)/(len(draws)+1)
            assert actual['p_raw'] == expected
            assert actual['effect_pp'] == float(observed)*100
            checks.append(dict(seeds=seed_count, exact_p=expected, passed=True))
    p = [.02, .01, .6, 1, .001, .11, .002, .05]
    result = holm(p)
    independent = []
    for x in p:
        independent.append(min(1., max((8-rank)*q for rank, q in enumerate(sorted(p)) if q <= x)))
    assert np.array_equal(result, independent)
    C.write(C.DOC/'PAIRED_UNIT_VERIFICATION.json', dict(PASS=True,
        rational_bootstrap_checks=checks, Holm_family=8, Holm_exact=True,
        no_solver_calls=True, no_training=True, verifier=C.binding(__file__)))
    print('PAIRED_RATIONAL_AND_HOLM_PASS', len(checks))


if __name__ == '__main__':
    main()
