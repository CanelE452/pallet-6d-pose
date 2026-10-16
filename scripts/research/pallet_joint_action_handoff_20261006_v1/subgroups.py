"""Reaggregate captured A rows with existing DEV319 human grades/condition tags.

No model forward, PnP, fitting, new label or new distance threshold. Run after
the final fresh A evaluation to include all six completed fits.
"""
from __future__ import annotations
import argparse, collections, csv, json, math, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .a_common import ROOT, DOC, read, write, sha
from .a_evaluate import summarize, paired_analysis, finite_json

PRIOR = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1'
STATES = ['DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED', 'OUT_OF_FRAME', 'UNKNOWN']


def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def verify_prior_labels(source_root):
    label_path = PRIOR / 'static/LABEL_PROVENANCE_AUDIT.json'
    visibility_path = PRIOR / 'visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'
    labels = read(label_path)
    visibility = read(visibility_path)
    assert labels['status'] == 'PASS' and all(labels['checks'].values())
    assert labels['classification_semantics_status'] == 'NOT_CONFIRMED'
    verified = []
    for b in labels['sources']:
        p = (source_root if b['path'].startswith('data/') else ROOT) / b['path']
        assert p.stat().st_size == b['bytes'] and sha(p) == b['sha256'], str(p)
        verified.append(binding(p))
    current = [r for r in labels['rows'] if r['population'] == 'DEV319']
    by_id = {r['id']: r for r in current}
    assert len(current) == len(by_id) == 319
    assert dict(collections.Counter(r['severity'] for r in current)) == {'clean': 153, 'moderate': 92, 'severe': 74}
    states = [r for r in visibility['rows'] if r['population'] == 'DEV319']
    by_corner = {(r['frame_id'], r['corner_id']): r for r in states}
    assert len(states) == len(by_corner) == 2499
    assert all(r['category'] in STATES and not r['source_coordinates_replaced'] and not r['independent_reference_claim'] for r in states)
    for r in states:
        label = by_id[r['frame_id']]
        assert r['image_sha256'] == label['image']['sha256']
        assert r['annotation_sha256'] == label['annotation']['sha256']
    workspace = source_root / 'data/evaluation/pallet_eval_v1'
    meta_path = workspace / 'manifests/ALL_AVAILABLE_POSITIVE.csv'
    with meta_path.open() as f:
        metadata = {'data/evaluation/pallet_eval_v1/' + r['image_path']: r for r in csv.DictReader(f)}
    frame_metadata = {}
    for fid, r in by_id.items():
        m = metadata[r['image']['path']]
        assert m['image_sha256'] == r['image']['sha256']
        assert m['session_id'] == r['session']
        assert m['distance_bin'] in ('near', 'mid', 'far', 'unknown')
        frame_metadata[fid] = m
    return by_id, by_corner, frame_metadata, [binding(label_path), binding(visibility_path), binding(meta_path), *verified]


def point_summary(rows, category, by_corner):
    full, observed, frames = [], [], set()
    for r in rows:
        c = r['corner']
        # Current captured DEV319 has eight finite predicted corners per frame.
        # Assert this observation mapping rather than infer it for future data.
        assert len(c['observed_errors']) == (c['corners'] if c['matched'] else 0)
        for i, valid in enumerate(c['canonical_valid']):
            if not valid:
                continue
            state = by_corner[(r['id'], i)]['category']
            if category != 'ALL' and state != category:
                continue
            error = c['canonical_errors'][i]
            assert error is not None and math.isfinite(error)
            full.append(error)
            frames.add(r['id'])
            if c['matched']:
                observed.append(error)
    q = lambda a, quantile: float(np.quantile(a, quantile)) if a else None
    return dict(frames_with_reference_state=len(frames), reference_corners=len(full),
                observed_corners=len(observed), unobserved_reference_corners=len(full)-len(observed),
                conditional_median_px=q(observed, .5), conditional_P90_px=q(observed, .9),
                full_penalty_median_px=q(full, .5), full_penalty_P90_px=q(full, .9),
                full_PCK10_fraction=float(np.mean(np.array(full) <= 10)) if full else None,
                metric_status='MEASURED_DESCRIPTIVE' if full else 'NA_EMPTY_CATEGORY')


def method_family(name):
    return name.rsplit('_seed', 1)[0] if '_seed' in name else name


def flatten_summary(summary):
    c, p = summary['corner'], summary['pose']
    return dict(frames=c['total_frames'], detected=c['detected'], matched=c['matched'],
                reference_corners=c['corners'], observed_corners=c['observed_corners'],
                E_sym=c['E_sym'], corner_median_px=c['matched_pooled_corner8_median_px'],
                corner_P90_px=c['matched_pooled_corner8_P90_px'], full_PCK10_fraction=c['PCK']['10'],
                gross20_fraction=c['gross20'], pose_available=p['available'], pose_failures=p['failures'],
                pose_coverage=p['coverage'], translation_median_cm=p['translation_cm']['median'],
                translation_P90_cm=p['translation_cm']['P90'], rotation_median_deg=p['rotation_deg']['median'],
                rotation_P90_deg=p['rotation_deg']['P90'], ADDsym_median_m=p['ADDsym_m']['median'],
                ADDsym_P90_m=p['ADDsym_m']['P90'], NoOp=summary['NoOp'])


def write_csv(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    args = p.parse_args()
    begin, cpu = time.monotonic(), time.process_time()
    labels, states, metadata, inputs = verify_prior_labels(args.source_root)
    ids = read(DOC / 'results/A_ID_MANIFEST.json')['IDs']['real_evaluation']
    assert len(ids) == 319 and set(ids) == set(labels)
    base_path = DOC / 'results/A_REAL_DEV_BASELINES.json'
    base = read(base_path)
    methods = dict(base['rows'])
    inputs += [binding(base_path), binding(DOC / 'results/A_ID_MANIFEST.json'), binding(Path(__file__))]
    fit_files = []
    for path in sorted((DOC / 'results').glob('A_REAL_DEV_*seed*.json')):
        x = read(path)
        prefix = 'FIT' if x['trained'] else 'FROZEN'
        seed = int(path.stem[-1])
        for key, rows in x['rows'].items():
            methods[f'{prefix}_{key}_seed{seed}'] = rows
        if x['trained']:
            fit_files.append(path.name)
        inputs.append(binding(path))
    expected_corners = {(r['id'], i) for r in methods['RAW'] for i, v in enumerate(r['corner']['canonical_valid']) if v}
    assert expected_corners == set(states)
    assert all([r['id'] for r in rows] == ids for rows in methods.values())
    groups = {'ALL': ids}
    for value in ['clean', 'moderate', 'severe']:
        groups['human_grade:' + value] = [i for i in ids if labels[i]['severity'] == value]
    for value in ['near', 'mid', 'far', 'unknown']:
        groups['existing_distance_tag:' + value] = [i for i in ids if metadata[i]['distance_bin'] == value]
    for value in STATES:
        groups['human_corner_state_any:' + value] = [i for i in ids if any(r['category'] == value for (fid, _), r in states.items() if fid == i)]
    summaries, point_stats, comparisons, frame_export, point_export = {}, {}, {}, [], []
    for name, rows in methods.items():
        summaries[name], point_stats[name] = {}, {}
        for group, members in groups.items():
            member_set = set(members)
            subset = [r for r in rows if r['id'] in member_set]
            if not subset:
                summaries[name][group] = dict(status='NA_EMPTY_GROUP', frames=0)
                continue
            stat = summarize(subset)
            summaries[name][group] = stat
            frame_export.append(dict(method=name, family=method_family(name), aggregation='per_seed_or_single',
                                     group=group, **flatten_summary(stat)))
        for category in [*STATES, 'ALL']:
            stat = point_summary(rows, category, states)
            point_stats[name][category] = stat
            point_export.append(dict(method=name, family=method_family(name), aggregation='per_seed_or_single', category=category, **stat))
    families = collections.defaultdict(list)
    for name in methods:
        families[method_family(name)].append(name)
    family_means, point_family_means = {}, {}
    for family, names in families.items():
        family_means[family] = {}
        if len(names) != 3:
            continue
        for group in groups:
            if not groups[group]:
                continue
            values = [flatten_summary(summaries[n][group]) for n in names]
            stat = {k: float(np.mean([v[k] for v in values])) if all(v[k] is not None for v in values) else None for k in values[0]}
            family_means[family][group] = stat
            frame_export.append(dict(method=family, family=family, aggregation='mean_of_three_seed_statistics_not_prediction_average', group=group, **stat))
        point_family_means[family] = {}
        for category in [*STATES, 'ALL']:
            values = [point_stats[n][category] for n in names]
            stat = {k: float(np.mean([v[k] for v in values])) if all(v[k] is not None for v in values) else None
                    for k in values[0] if k != 'metric_status'}
            stat['metric_status'] = 'MEAN_THREE_SEED_STATISTICS' if stat['reference_corners'] else 'NA_EMPTY_CATEGORY'
            point_family_means[family][category] = stat
            point_export.append(dict(method=family, family=family,
                                     aggregation='mean_of_three_seed_statistics_not_prediction_average', category=category, **stat))
    for name, rows in methods.items():
        if not name.startswith(('FIT', 'FROZEN')):
            continue
        seed = name.rsplit('_seed', 1)[1]
        family = method_family(name)
        references = ['RAW', f'N3_seed{seed}', f'PoseFix_seed{seed}']
        controlled = {'FIT_GEO_J': 'FIT_PERM_J', 'FROZEN_GEO_J': 'FROZEN_GEO_I', 'FROZEN_PERM_J': 'FROZEN_PERM_I'}.get(family)
        if controlled and f'{controlled}_seed{seed}' in methods:
            references.append(f'{controlled}_seed{seed}')
        for ref in references:
            paired = {}
            for group, members in groups.items():
                if not members:
                    continue
                member_set = set(members)
                a = [r for r in rows if r['id'] in member_set]
                b = [r for r in methods[ref] if r['id'] in member_set]
                paired[group] = paired_analysis(a, b)
            comparisons[f'{name}_minus_{ref}'] = paired
    frame_mapping = [dict(id=i, session=labels[i]['session'], severity=labels[i]['severity'],
                          classification_semantics='NOT_CONFIRMED', distance_bin=metadata[i]['distance_bin'],
                          image_sha256=labels[i]['image']['sha256'], label_source=labels[i]['source']) for i in ids]
    out = dict(schema='newly_defined_handoff_A_descriptive_subgroups_v1',
               status='DONE' if len(fit_files) == 6 else 'RUNNING_WAITING_FIXED_FITS',
               population='repeated DEV319 / 13 sessions; no independent confirmation',
               human_grade_counts=dict(collections.Counter(labels[i]['severity'] for i in ids)),
               existing_distance_tag_counts=dict(collections.Counter(metadata[i]['distance_bin'] for i in ids)),
               human_corner_state_counts=dict(collections.Counter(r['category'] for r in states.values())),
               annotation_coordinate_source_counts=dict(collections.Counter(r['label_evidence'].get('annotation_coordinate_source', r['reference_source']) for r in states.values())),
               contract=dict(human_grades='existing user-entered clean/moderate/severe; criterion NOT_CONFIRMED; not measured physical occlusion severity',
                             corner_states='reuse existing explicit human reference-corner visibility; no new labels; model-prediction exposure NOT_CONFIRMED; no independent physical occlusion intervention',
                             distance='reuse existing categorical near/mid/far/unknown workspace tags; no new metre thresholds; not independent range metrology',
                             frame_visibility_groups='any reference corner in category; groups overlap; never sum their frame counts into independent samples',
                             pose='all selected-instance actual F(q) outputs; failures retained; full frame denominator and paired common-success quadrants',
                             point_errors='same approved whole-object 2D branch; canonical GT identity from saved rows; full unmatched penalty plus conditional observed error',
                             seed_aggregation='per-seed rows and arithmetic mean of seed statistics; no point/prediction averaging',
                             inference='no GT/visibility/distance label supplied to scorer; label use is post-hoc descriptive aggregation only',
                             uncertainty='no new CI/significance/selection/verdict from subgroups; overlapping corners are not independent samples',
                             source_coordinate_unknown='unknown annotation coordinate provenance is retained, not promoted by the human visibility label'),
               groups={k:dict(frames=len(v),ids=v) for k,v in groups.items()}, frame_mapping=frame_mapping,
               per_method=summaries, seed_mean_statistics=family_means, per_corner_state=point_stats,
               per_corner_state_seed_mean_statistics=point_family_means,
               paired_comparisons=comparisons, source_bindings=inputs, completed_fit_output_files=fit_files,
               new_model_inference_calls=0, new_PnP_calls=0, new_optimizer_updates=0, new_human_labels=0,
               timing_scope='main aggregation body; excludes Python import/startup and final output serialization',
               seconds_wall=time.monotonic()-begin, seconds_process_cpu=time.process_time()-cpu)
    dest = DOC / 'results'
    output_path = dest / 'A_REAL_DEV_DESCRIPTIVE_SUBGROUPS.json'
    ledger_path = dest / 'A_DESCRIPTIVE_SUBGROUPS_EXECUTIONS.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(schema='newly_defined_A_subgroup_execution_ledger_v1', executions=[])
    if output_path.exists() and not ledger['executions']:
        prior = read(output_path)
        ledger['executions'].append(dict(scope='previous saved descriptive aggregation receipt reused',
            status=prior['status'], methods=len(prior['per_method']),
            seconds_wall=prior['seconds_wall'], seconds_process_cpu=prior['seconds_process_cpu'],
            new_model_inference_calls=0, new_PnP_calls=0, new_optimizer_updates=0,
            code_sha256=None, code_binding_status='NOT_RECORDED_IN_INITIAL_AGGREGATION',
            prior_output_sha256=sha(output_path)))
    write(output_path, finite_json(out))
    write_csv(dest / 'A_REAL_DEV_DESCRIPTIVE_SUBGROUPS.csv', frame_export)
    write_csv(dest / 'A_REAL_DEV_HUMAN_CORNER_STATES.csv', point_export)
    ledger['executions'].append(dict(finished_utc=datetime.now(timezone.utc).isoformat(),
        status=out['status'], methods=len(methods), seconds_wall=out['seconds_wall'],
        seconds_process_cpu=out['seconds_process_cpu'], code_sha256=sha(Path(__file__)),
        output_sha256=sha(output_path), new_model_inference_calls=0, new_PnP_calls=0,
        new_optimizer_updates=0, command=f'python -m scripts.research.pallet_joint_action_handoff_20261006_v1.subgroups --source-root {args.source_root}'))
    write(ledger_path, ledger)
    print(json.dumps(dict(status=out['status'], methods=len(methods), grades=out['human_grade_counts'],
                         distance_tags=out['existing_distance_tag_counts'], corner_states=out['human_corner_state_counts'],
                         new_inference=0, new_PnP=0, seconds_wall=out['seconds_wall']), ensure_ascii=False))


if __name__ == '__main__':
    main()
