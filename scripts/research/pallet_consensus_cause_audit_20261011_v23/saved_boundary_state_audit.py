"""Audit saved selection and failure states; perform no numerical pose fit."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
G = Path('/tmp/pallet-whole-u-closure-private-20261011-v21/geometry/GEOMETRY.jsonl.gz')
P = Path('/tmp/pallet-whole-u-closure-private-20261011-v21/evaluation/POSTHOC_ROWS.jsonl.gz')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    bindings = {str(path): digest(path) for path in (G, P, Path(__file__))}
    save('BOUNDARY_STATE_PROTOCOL.json', dict(schema='existing_boundary_state_join_v23',
        inputs=bindings, new_PnP=0, new_projection=0, new_pose_accuracy_evaluations=0,
        physical_ownership_truth='UNKNOWN', thresholds_changed=False))
    posthoc = {r['id']: r for r in read(P)}
    selected, candidate_counts, states, witness, seen = [], collections.Counter(), collections.Counter(), [], set()
    for g in read(G):
        ident = g['id']; assert ident not in seen; seen.add(ident)
        p = posthoc[ident]; s = g['solver']
        assert set(g['selected_corner_ids']) == {b['id'] for b in p['boundary_candidates'] if b['supply_selected']}
        for b in p['boundary_candidates']:
            candidate_counts['saved_candidate_rows'] += 1
            candidate_counts['accepted'] += bool(b['accepted'])
            if not b['supply_selected']:
                continue
            assert b['accepted'] and not b['self_hidden_excluded']
            row = dict(frame_id=ident, **b)
            selected.append(row)
            candidate_counts['selected'] += 1
            candidate_counts['selected_quality:' + b['candidate_proxy_quality']] += 1
            delta = b['delta_vs_N3_px']
            direction = 'UNKNOWN' if delta is None else 'BETTER' if delta < 0 else 'WORSE' if delta > 0 else 'EQUAL'
            candidate_counts['selected_vs_native:' + direction] += 1
        closure = s.get('closure_ledger', {})
        original_empty = s.get('proposal_candidate_count', len(s['proposal_solutions'])) == 0
        closure_valid = closure.get('raw_valid_roots', 0) > 0
        possible_misclassification = original_empty and closure_valid and not s['available']
        states['frames'] += 1
        states['original_four_ID_proposals_empty'] += original_empty
        states['closure_raw_valid_roots_present'] += closure_valid
        states['original_empty_closure_valid_unavailable'] += possible_misclassification
        if possible_misclassification:
            witness.append(dict(id=ident, state=s['state'], reason=s['reason'],
                proposal_candidate_count=s.get('proposal_candidate_count'), closure_ledger=closure))
    assert seen == set(posthoc) and len(seen) == 245
    with (ROOT / 'SELECTED_BOUNDARY_ROWS.jsonl').open('w') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    save('BOUNDARY_STATE_SUMMARY.json', dict(schema='existing_boundary_state_summary_v23',
        complete=True, frames=245, counts=dict(candidate_counts), state_counts=dict(states),
        possible_state_misclassification_witnesses=witness,
        quality='existing annotation reference; physical ownership is UNKNOWN',
        counts_are_saved_rows_not_unique_source_supported_corner_proofs=True,
        new_PnP=0, new_projection=0, new_pose_accuracy_evaluations=0,
        selected_rows_sha256=digest(ROOT / 'SELECTED_BOUNDARY_ROWS.jsonl')))
    assert bindings == {path: digest(Path(path)) for path in bindings}
    save('BOUNDARY_STATE_CHECKS.json', dict(passed=True, all245_unique_ID_join=True,
        all_selected_IDs_match_existing_geometry=True, all_selected_not_H=True,
        inputs_unchanged=True, new_PnP=0, new_pose_accuracy_evaluations=0))
    print(json.dumps(dict(counts=dict(candidate_counts), state_counts=dict(states)), ensure_ascii=False))


if __name__ == '__main__':
    main()
