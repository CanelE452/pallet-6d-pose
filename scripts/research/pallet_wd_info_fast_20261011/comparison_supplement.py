"""Complete specified contrasts from sealed rows; no fits or pose solves.

The original pre-execution SP family remains unchanged. This separate family
aligns the named Stage1 contrast with the user's instruction after execution;
it must not be described as retroactively preregistered.
"""
import csv
import gzip
import json
import time
from collections import Counter
import numpy as np
from . import common as C
from . import paired


PRIMARY = [('G_V_MLP', 'G_MLP'), ('G_V_D_MLP', 'G_V_MLP'),
           ('D_JOINT_DIM', 'D_CONTROL'), ('G_V_D_MLP', 'OFFICIAL')]
C_PAIRS = [(m, 'OFFICIAL') for m in ['TRAIN_PRIOR', 'D_LOGISTIC',
    'G_LOGISTIC', 'V_LOGISTIC', 'V_MLP', 'G_MLP', 'G_V_MLP',
    'G_V_D_MLP', 'V_SHUFFLED_LOGISTIC']]
C_PAIRS += PRIMARY[:2]
D_PAIRS = []
for suffix, n3 in [('', 'N3_DIM_SYM'), ('_SUBPIX', 'N3_THEN_SUBPIX')]:
    D_PAIRS += [(n3, 'OLD_P'+suffix),
        ('D_CONTROL'+suffix, 'OLD_P'+suffix),
        ('D_JOINT_DIM'+suffix, 'OLD_P'+suffix),
        ('D_JOINT_DIM'+suffix, 'D_CONTROL'+suffix),
        ('D_CONTROL'+suffix, n3), ('D_JOINT_DIM'+suffix, n3)]
D_PAIRS += [('D_JOINT_MEAN', 'D_JOINT_DIM'),
            ('D_JOINT_SHUFFLE', 'D_JOINT_DIM')]


def transition(before, after):
    if before is None or after is None:
        return 'UNCLEAR'
    if before and not after:
        return 'RECOVERY'
    if not before and after:
        return 'DAMAGE'
    return 'UNCHANGED'


def run():
    started = time.monotonic()
    C.lock_ready()
    original = {p.name: C.sha(p) for p in [C.DOC/'PAIRED.json',
        C.DOC/'CHANGED_SELECTIONS.csv', C.DOC/'PLAN_LOCK.json']}
    contract = C.read(C.DOC/'INPUT_CONTRACT.json')
    ids = sorted(contract['allowed_ids']); allowed = set(ids)
    frame = {(r['method'], r['id'], r['seed']): r for r in
             C.rows(C.DOC/'FRAME_METRICS.jsonl.gz') if r['backbone'] == 'YOLO'}
    draws = np.load(C.DOC/'BOOTSTRAP_DRAWS.npy', allow_pickle=False)
    drawlock = C.read(C.DOC/'BOOTSTRAP_LOCK.json')
    assert C.sha(C.DOC/'BOOTSTRAP_DRAWS.npy') == drawlock['sha256']
    sessions = [fid.split(':')[0] for fid in ids]
    assert sorted(set(sessions)) == drawlock['sessions']
    effects = []
    for after, before in PRIMARY:
        seeds = [1] if after == 'D_JOINT_DIM' else [1, 2, 3]
        for metric in paired.METRICS:
            aa = [[frame[after, fid, seed][metric] for seed in seeds] for fid in ids]
            bb = [[frame[before, fid, seed][metric] for seed in seeds] for fid in ids]
            effects.append(dict(after=after, before=before, metric=metric,
                **paired.bootstrap_effect(bb, aa, sessions, draws)))
    for row, p in zip(effects, paired.holm([x['p_raw'] for x in effects])):
        row['p_Holm'] = p
    primary_output = dict(
        status='POST_EXECUTION_PROTOCOL_ALIGNMENT_EXPLORATORY', family_size=8,
        deviation='Original PLAN used D after SubPix instead of the explicitly named Stage1 pair. Original family preserved; this requested family was calculated after execution, without fitting or performance selection.',
        original_family=C.binding(C.DOC/'PAIRED.json'),
        raw_rows=C.binding(C.DOC/'FRAME_METRICS.jsonl.gz'),
        draw_binding=C.binding(C.DOC/'BOOTSTRAP_DRAWS.npy'), results=effects)
    primary_path = C.DOC/'REQUESTED_PRIMARY_PAIRED.json'
    if primary_path.exists():
        assert C.read(primary_path) == primary_output
    else:
        C.write(primary_path, primary_output)
    reused = {(r['method'], r['id'], r['seed']): r for r in
              C.rows(C.DOC/'AE_CANDIDATE_METRICS.jsonl.gz') if r['backbone'] == 'YOLO'}
    def candidate_packet(row, kind):
        if kind == 'C':
            return row['candidates']
        if 'candidate_selector_values' in row:
            return row['candidate_selector_values']
        key = (row['method'], row['id'], row['seed'])
        if key in reused:
            return reused[key]['candidate_metrics']
        assert row['method'] == 'OLD_P'
        # OLD_P historical rows retain only the selected pose. Do not invent
        # the other candidate or spend undeclared solves reconstructing it.
        return {name: dict(status='NA_NOT_RETAINED_IN_HISTORICAL_CACHE')
                for name in ('long-face-front', 'short-face-front')}
    source_files = ['C_FRAME_METRICS.jsonl.gz', 'D_REPAIRED_ROWS.jsonl.gz']
    rows = []
    counts = {}
    for kind, filename, pairs in [('C', source_files[0], C_PAIRS),
                                  ('D', source_files[1], D_PAIRS)]:
        lookup = {(r['method'], r['id'], r['seed']): r for r in C.rows(C.DOC/filename)}
        for after, before in pairs:
            seeds = [1, 2, 3] if kind == 'C' else [1]
            counter = Counter()
            for fid in ids:
                for seed in seeds:
                    a, b = lookup[after, fid, seed], lookup[before, fid, seed]
                    x, y = frame[after, fid, seed], frame[before, fid, seed]
                    state = transition(y['axis_wrong'], x['axis_wrong'])
                    strict = transition(1-y['pose_success'], 1-x['pose_success'])
                    avail = transition(1-y['available'], 1-x['available'])
                    counter['axis_'+state] += 1
                    counter['strict_'+strict] += 1
                    counter['availability_'+avail] += 1
                    counter['selection_flips'] += int(y['selected_hypothesis'] != x['selected_hypothesis'])
                    row = dict(source=kind, after=after, before=before, id=fid,
                        seed=seed, session=x['session'], grade=x['grade'],
                        selection_changed=int(y['selected_hypothesis'] != x['selected_hypothesis']),
                        from_hypothesis=y['selected_hypothesis'], to_hypothesis=x['selected_hypothesis'],
                        axis_transition=state, strict_transition=strict,
                        availability_transition=avail, old_available=y['available'],
                        new_available=x['available'], old_success=y['pose_success'],
                        new_success=x['pose_success'], old_axis_wrong=y['axis_wrong'],
                        new_axis_wrong=x['axis_wrong'], old_wd_confusion=y['wd_confusion'],
                        new_wd_confusion=x['wd_confusion'], old_T_cm=y['translation_cm'],
                        new_T_cm=x['translation_cm'], old_R_deg=y['rotation_deg'],
                        new_R_deg=x['rotation_deg'],
                        candidate_semantics=('unchanged final8 candidate metrics' if kind == 'C'
                            else 'new D: initial selector fits with R_cf,t,score; t[2]=centroid Z; unselected final8 metric NA. Historical N3: final8 cached metrics. OLD_P both candidates NA (not retained).'),
                        old_candidates_json=json.dumps(candidate_packet(b, kind), sort_keys=True, separators=(',', ':')),
                        new_candidates_json=json.dumps(candidate_packet(a, kind), sort_keys=True, separators=(',', ':')),
                        corner_delta_px=(a['corner']['frame_mean_px']-b['corner']['frame_mean_px']) if kind == 'D' else 0)
                    rows.append(row)
            counts[after+' - '+before] = dict(counter)
    assert len(rows) == 11*519 + 14*173 == 8131
    assert {r['id'] for r in rows} == allowed
    for arm, flips in [('D_JOINT_MEAN', 2), ('D_JOINT_SHUFFLE', 3)]:
        assert counts[arm+' - D_JOINT_DIM']['selection_flips'] == flips
        assert counts[arm+' - D_JOINT_DIM']['axis_DAMAGE'] == flips
    target = C.DOC/'COMPARISON_TRANSITIONS_COMPLETE.csv'
    assert not target.exists()
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    assert all(C.sha(C.DOC/name) == digest for name, digest in original.items())
    C.write(C.DOC/'COMPARISON_COMPLETION_RECEIPT.json', dict(
        status='DONE_FROM_SEALED_ROWS_ONLY', rows=len(rows), frames=173,
        C_comparisons=11, D_comparisons=14, comparisons=counts,
        primary_results=C.binding(C.DOC/'REQUESTED_PRIMARY_PAIRED.json'),
        complete_transitions=C.binding(target), inputs=[C.binding(C.DOC/name) for name in source_files],
        old_artifacts_unchanged=original, wall_seconds=time.monotonic()-started,
        new_fits=0, new_RGB_forwards=0, new_PnP_calls=0,
        historical_OLD_P_candidate_details='BLOCKED_DATA: only selected historical pose retained; both candidate entries explicitly NA in each OLD_P comparison. No recovered/damaged frame is omitted.',
        initial_wrapper_error='First transition build assumed candidate_selector_values on reused baseline rows; stopped before CSV write, PnP0. Corrected cache-specific adapters; primary JSON unchanged and equality checked.',
        original_transition_column_correction='Original D *_front_depth_m columns are centroid t_z; preserved historical CSV, corrected semantics in new complete CSV.',
        public_fields='Predicted candidate poses/scalars; no RGB or raw reference poses/corners'))
    print('COMPARISON_SUPPLEMENT', len(rows), 'rows; named Stage1 family',
          [(r['metric'], r['effect_pp'], r['CI95'], r['p_raw'], r['p_Holm'])
           for r in effects if r['after'] == 'D_JOINT_DIM'])


if __name__ == '__main__':
    run()
