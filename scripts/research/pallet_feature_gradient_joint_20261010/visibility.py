"""Post-evaluation human-state strata; labels are never inference arguments."""
from collections import Counter
import numpy as np

from . import common as C

STATES = ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'EXTERNAL_OCCLUDED', 'OUT_OF_FRAME', 'UNKNOWN', 'UNANNOTATED')
RELATIVE = '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'


def distribution(a):
    a = np.asarray(a, float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
        variance=float(a.var(ddof=1)) if len(a) > 1 else None,
        std=float(a.std(ddof=1)) if len(a) > 1 else None,
        median=float(np.median(a)) if len(a) else None,
        P90=float(np.quantile(a, .9)) if len(a) else None,
        max=float(a.max()) if len(a) else None, unit='px', ddof=1)


def main():
    assert C.read(C.DOC / 'COORDINATES_SEAL.json')['status'] == 'PASS'
    assert C.read(C.DOC / 'EXECUTION_LEDGER.json')['status'] in ('ACCURACY_COMPLETE', 'COMPLETE')
    if not (C.DOC / 'VISIBILITY_LABELS.json').exists():
        original = C.read(C.SOURCE / RELATIVE)
        # Publish category/id/index only. Actor names, paths and keyboard events
        # in the original annotation audit are deliberately not copied.
        labels = [dict(id=r['frame_id'], corner=r['corner_id'], category=r['category'])
                  for r in original['rows'] if r['population'] == 'DEV319']
        assert len(labels) == 2499 and len({r['id'] for r in labels}) == 319
        C.write(C.DOC / 'VISIBILITY_LABELS.json', dict(status='EXISTING_HUMAN_STATES_EVALUATION_ONLY',
            source_relative=RELATIVE, source_sha256=C.sha(C.SOURCE / RELATIVE),
            used_in_inference=False, labels=labels))
    labels_file = C.read(C.DOC / 'VISIBILITY_LABELS.json')
    assert labels_file['used_in_inference'] is False
    labels = {(r['id'], r['corner']): r['category'] for r in labels_file['labels']}
    raw = list(C.rows(C.DOC / 'PREDICTIONS.jsonl.gz'))
    lookup = {(r['seed'], r['method'], r['id']): r for r in raw}
    result = {}
    for seed in (1, 2, 3):
        result[str(seed)] = {}
        for method in C.METHODS:
            grouped = {state: [] for state in STATES}
            missing = 0
            for row in (r for r in raw if r['seed'] == seed and r['method'] == method):
                for k, valid in enumerate(row['corner']['canonical_valid']):
                    if not valid:
                        missing += 1
                        continue
                    state = labels.get((row['id'], k), 'UNANNOTATED')
                    assert state in STATES
                    item = dict(id=row['id'], error=row['corner']['canonical_errors'][k],
                                observed=row['canonical_observed'][k])
                    for base in ('BASE', 'N3_DIM_SYM', 'N3_THEN_SUBPIX'):
                        old = lookup[(seed, base, row['id'])]
                        assert old['corner']['canonical_valid'] == row['corner']['canonical_valid']
                        item[base] = old['corner']['canonical_errors'][k]
                    grouped[state].append(item)
            summaries = {}
            for state, values in grouped.items():
                a = np.asarray([v['error'] for v in values], float)
                observed = [v['error'] for v in values if v['observed']]
                summaries[state] = dict(reference_corners=len(values), observed_corners=len(observed),
                    frames=len({v['id'] for v in values}), observed_error_px=distribution(observed),
                    PCK10=float(np.mean(a <= 10)) if len(a) else None,
                    gross20_count=int(np.sum(a > 20)),
                    damage_vs={base: dict(good5_to_bad10=sum(v[base] < 5 and v['error'] > 10 for v in values),
                                         bad20_to_good10=sum(v[base] > 20 and v['error'] <= 10 for v in values))
                               for base in ('BASE', 'N3_DIM_SYM', 'N3_THEN_SUBPIX')})
            assert sum(d['reference_corners'] for d in summaries.values()) == 2499
            result[str(seed)][method] = dict(states=summaries, missing_reference_corner_slots=missing,
                original_corner_slots=319 * 8, reference_corners=2499)
    payload = dict(status='COMPLETE', images=319, human_reference_state_counts=dict(Counter(labels.values())),
        states=STATES, by_seed=result, labels_sha256=C.sha(C.DOC / 'VISIBILITY_LABELS.json'),
        predictions_sha256=C.sha(C.DOC / 'PREDICTIONS.jsonl.gz'),
        inference_used_human_states=False,
        missing_note='53 invalid reference slots are reported separately; no fabricated error/reference',
        semantics='Existing canonical human states after fixed whole-object correspondence. Not external occlusion fraction, independent reference, or causal effect.')
    C.write(C.DOC / 'VISIBILITY_RESULTS.json', payload)
    print('VISIBILITY_COMPLETE', payload['human_reference_state_counts'], flush=True)


if __name__ == '__main__':
    main()
