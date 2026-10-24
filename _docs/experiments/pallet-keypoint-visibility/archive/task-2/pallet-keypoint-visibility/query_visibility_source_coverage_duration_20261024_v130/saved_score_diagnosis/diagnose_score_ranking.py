"""Source-only saved-score ranking limits. No model or usable threshold output."""
from pathlib import Path
import hashlib
import json
import numpy as np

V = Path(__file__).resolve().parents[1]
OLD = V.parent / 'query_visibility_registered_native_diversity_20261023_v129'


def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def ranking(z, y):
    """All unique-score cuts; ties stay together. U absent in these VAL cohorts."""
    assert len(z) and np.isfinite(z).all() and np.isin(y, [0, 1]).all()
    order = np.argsort(z, kind='stable')
    z, y = z[order], y[order]
    nd, ne = int((y == 0).sum()), int((y == 1).sum())
    assert nd and ne
    d_done = e_done = 0
    pair_wins = 0.0
    best = {'minimum_class_recall': 0.0, 'D_correct': 0, 'E_correct': ne}
    boundaries = np.r_[0, np.flatnonzero(z[1:] != z[:-1]) + 1, len(z)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        dc = int((y[lo:hi] == 0).sum())
        ec = int(hi - lo - dc)
        pair_wins += ec * (d_done + .5 * dc)
        d_done += dc
        e_done += ec
        value = min(d_done / nd, (ne - e_done) / ne)
        if value > best['minimum_class_recall']:
            best = {'minimum_class_recall': float(value),
                    'D_correct': d_done, 'E_correct': ne - e_done}
    return {'D_N': nd, 'E_N': ne, 'AUC_external_E': pair_wins / (nd * ne),
            'posthoc_source_oracle_all_possible_single_cuts': best,
            'any_single_cut_can_reach_both_recalls_80pct':
                best['minimum_class_recall'] >= .8,
            'threshold_value_exported': False}


def main():
    sources = [
        (OLD, 'C20_C21_FIXED1000_SOURCE_CONTROL', 'C20_PUBLIC_REPLAY_CONTROL'),
        (OLD, 'C20_C21_FIXED1000_SOURCE_CONTROL', 'C21_PUBLIC_PLUS_NATIVE'),
        (V, 'C22_C23_FIXED8000_SOURCE_CONTROL', 'C22_PUBLIC_REPLAY_FIXED8000'),
        (V, 'C22_C23_FIXED8000_SOURCE_CONTROL', 'C23_PUBLIC_PLUS_NATIVE_FIXED8000'),
    ]
    result = {'completed': True, 'source_only': True,
              'scope': 'Saved source VAL scores only. Posthoc ranking diagnostic, '
                       'not a selected operating threshold or model eligibility change.',
              'current223_V112_model_fit_PnP': 0, 'results': {}, 'bindings': []}
    for root, run, arm in sources:
        complete = root / 'runs' / run / 'CONTROL_COMPLETE.json'
        assert json.loads(complete.read_text())['completed']
        file = root / 'runs' / run / arm / 'FINAL_ALL61110_SOURCE_SCORES.npz'
        with np.load(file, allow_pickle=False) as f:
            z = f['raw_head_logit_before_U']
            y = f['external_target']
            a = f['query_available']
            split, corpus = f['split'], f['corpus']
        result['bindings'].extend([bind(complete), bind(file)])
        result['results'][arm] = {}
        for name in ('OLD_PUBLIC', 'PUBLIC_AUX', 'NATIVE'):
            mask = (split == 'VALIDATION') & (corpus == name) & (y >= 0)
            assert a[mask].all(), 'Do not silently drop unavailable known VAL inputs.'
            result['results'][arm][name] = ranking(z[mask], y[mask])
    result['code'] = bind(Path(__file__))
    out = V / 'saved_score_diagnosis/SOURCE_THRESHOLD_LIMIT_DIAGNOSTIC.json'
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': bind(out), 'results': result['results']}))


if __name__ == '__main__':
    main()
