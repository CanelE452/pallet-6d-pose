"""Small grade/order wrapper around the existing saved-row statistics."""
from __future__ import annotations

from collections import Counter
import numpy as np

from scripts.research.pallet_n3_subpix_20261008_v1 import statistics as S

EXISTING_ARMS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX')
ARMS = (*EXISTING_ARMS, 'SUBPIX_N3')
GRADES = ('clean', 'moderate', 'severe')
GROUPS = ('ALL', *GRADES)
GRADE_COUNTS = {'clean': 153, 'moderate': 92, 'severe': 74}
COMPARISONS = (('SUBPIX_N3', 'N3_SUBPIX'), ('SUBPIX_N3', 'N3'),
               ('SUBPIX_N3', 'SUBPIX'), ('SUBPIX_N3', 'BASE'),
               ('N3_SUBPIX', 'N3'), ('N3_SUBPIX', 'SUBPIX'), ('N3_SUBPIX', 'BASE'),
               ('N3', 'BASE'), ('SUBPIX', 'BASE'))
METRICS = S.METRICS
STATISTICS = ('mean', 'median', 'P90')


def member_ids(ids, grades):
    assert len(ids) == len(set(ids)) == 319
    assert set(ids) == set(grades)
    counts = Counter(grades[i]['severity'] for i in ids)
    assert dict(counts) == GRADE_COUNTS, dict(counts)
    groups = {'ALL': ids}
    groups.update({g: [i for i in ids if grades[i]['severity'] == g] for g in GRADES})
    assert set().union(*(set(groups[g]) for g in GRADES)) == set(ids)
    assert sum(len(groups[g]) for g in GRADES) == 319
    assert all(not (set(groups[a]) & set(groups[b])) for a in GRADES for b in GRADES if a < b)
    return groups


def marginal_change(after, before):
    result = {}
    for metric in METRICS:
        aa, bb = after['metrics'][metric], before['metrics'][metric]
        result[metric] = {}
        for key in STATISTICS:
            a, b = aa[key], bb[key]
            difference = a - b if a is not None and b is not None else None
            reduction = -difference if difference is not None else None
            result[metric][key] = dict(before=b, after=a, after_minus_before=difference,
                absolute_reduction_before_minus_after=reduction,
                relative_reduction_percent=100. * reduction / b
                if reduction is not None and b != 0 else None,
                before_n=bb['n'], after_n=aa['n'],
                unit=aa['unit'], definition='absolute reduction=before-after; relative=100*(before-after)/before; zero denominator undefined')
    return result


def group_comparison(new, base, members, bootstrap, summaries):
    member_set = set(members)
    aa = [r for r in new if r['id'] in member_set]
    bb = [r for r in base if r['id'] in member_set]
    metrics = {}
    for metric in METRICS:
        # Draw all original13 sessions, then keep original observations of this grade.
        full = S.paired_values(new, base, metric)
        eligible = [v if r['id'] in member_set else np.array([], dtype=np.float64)
                    for r, v in zip(new, full)]
        unit = 'px' if metric == 'corner_px' else 'degree' if metric == 'rotation_deg' else 'cm'
        stat = S.pooled_contrast(eligible, bootstrap, unit)
        stat.update(group_frames=len(members),
                    outside_group_frames=len(new)-len(members),
                    excluded_frames=len(members)-stat['common_eligible_frames'],
                    inference='none; existing grade is used only after prediction',
                    bootstrap_definition='shared original13-session draws; retain only this grade after drawing; original corner/frame pooling')
        if metric == 'corner_px':
            stat['group_reference_corners'] = sum(r['corner'].get('corners', 0) for r in bb)
            stat['excluded_reference_corners'] = stat['group_reference_corners'] - stat['common_eligible_observations']
        metrics[metric] = stat
    av = np.asarray([r['pose']['available'] for r in aa], dtype=bool)
    bv = np.asarray([r['pose']['available'] for r in bb], dtype=bool)
    return dict(statistics=metrics,
                marginal_change=marginal_change(summaries[0], summaries[1]),
                coverage=dict(group_frames=len(members), common_success=int(np.sum(av & bv)),
                    after_failures=int(np.sum(~av)), before_failures=int(np.sum(~bv)),
                    after_only_success=int(np.sum(av & ~bv)), before_only_success=int(np.sum(bv & ~av)),
                    both_failed=int(np.sum(~av & ~bv))),
                canonical_damage=S.damage(bb, aa), pose_quadrants=S.pose_quadrants(aa, bb),
                hypothesis=S.hypothesis_layers(aa, bb),
                signs='paired mean difference is after-before (negative improves); marginal absolute reduction is before-after (positive improves)')


def grade_ranking(comparisons):
    """Separate smallest final error from largest before-after improvement."""
    result = {}
    for name, grouped in comparisons.items():
        result[name] = {}
        for metric in METRICS:
            result[name][metric] = {}
            for statistic in STATISTICS:
                changes = {g: grouped[g]['marginal_change'][metric][statistic] for g in GRADES}
                values = {g: dict(final_error=v['after'], absolute_reduction=v['absolute_reduction_before_minus_after'],
                                  relative_reduction_percent=v['relative_reduction_percent'])
                          for g, v in changes.items()}
                winners = {}
                for key, minimum in (('final_error', True), ('absolute_reduction', False),
                                     ('relative_reduction_percent', False)):
                    finite = {g: v[key] for g, v in values.items() if v[key] is not None}
                    target = (min(finite.values()) if minimum else max(finite.values())) if finite else None
                    winners[key] = [g for g, v in finite.items() if v == target] if target is not None else []
                result[name][metric][statistic] = dict(values=values,
                    smallest_final_error_grades=winners['final_error'],
                    largest_absolute_reduction_grades=winners['absolute_reduction'],
                    largest_relative_reduction_grades=winners['relative_reduction_percent'],
                    clean_final_error_smallest='clean' in winners['final_error'],
                    clean_absolute_reduction_largest='clean' in winners['absolute_reduction'],
                    clean_relative_reduction_largest='clean' in winners['relative_reduction_percent'])
    return result


def reverse_motion(rows):
    """Summarize inference diagnostics without inventing or recomputing N3 limits."""
    if not rows:
        return dict(status='NOT_EXECUTED')
    output = dict(status='DONE', frames=len(rows), stage_displacements={},
                  diagnostic_counts=Counter(), fallback_counts=Counter())
    values = {key: [] for key in ('SUBPIX', 'N3', 'FINAL', 'TOTAL')}
    for r in rows:
        q0 = np.asarray(r['q0'], dtype=np.float64)
        qS = np.asarray(r['qS'], dtype=np.float64)
        qN = np.asarray(r.get('qSN', r.get('qN')), dtype=np.float64)
        final = np.asarray(r['qFinal'], dtype=np.float64)
        support = np.asarray(r['prediction_support'], dtype=bool)
        usable = support[:8] & np.isfinite(q0[:8]).all(-1) & ~np.all(q0[:8] == -1, axis=-1)
        for name, a, b in (('SUBPIX', qS, q0), ('N3', qN, qS), ('FINAL', final, qN), ('TOTAL', final, q0)):
            values[name].extend(np.linalg.norm(a[:8]-b[:8], axis=-1)[usable].tolist())
        correction = r.get('correction', {})
        cap = correction.get('cap_px', .01*np.hypot(*r['raw_hw']))
        before = np.linalg.norm(qN[:8]-q0[:8], axis=-1)
        active = usable & (before > cap)
        output['diagnostic_counts']['final_cap_active_corners'] += int(np.sum(active))
        output['diagnostic_counts']['final_cap_active_frames'] += int(np.any(active))
        diagnostics = correction.get('SUBPIX', correction.get('SUBPIX_diagnostics',
                       correction.get('subpix_diagnostics', correction.get('diagnostics', {}))))
        output['fallback_counts'].update(diagnostics.get('fallback_counts', {}))
        for key, label in (('SUBPIX_cap_active8', 'SUBPIX_input_cap'),
                           ('N3_internal_cap_active8', 'N3_internal_cap'),
                           ('final_cap_active8', 'final_logged_cap')):
            if key in correction:
                mask = np.asarray(correction[key], dtype=bool)
                assert mask.shape == (8,)
                output['diagnostic_counts'][label+'_active_corners'] += int(np.sum(mask))
                output['diagnostic_counts'][label+'_active_frames'] += int(np.any(mask))
                if key == 'final_cap_active8':
                    assert np.array_equal(mask, active), 'Saved final cap flags disagree with q0/qSN'
        used = correction.get('N3_head_used')
        if used is not None:
            output['diagnostic_counts']['N3_head_used_frames'] += int(bool(used))
            output['diagnostic_counts']['N3_head_skipped_frames'] += int(not bool(used))
            if not used:
                assert np.array_equal(qN, qS, equal_nan=True)
        trace = correction.get('input_trace', {})
        if 'output_support' in trace or 'output_support8' in trace:
            output_support = np.asarray(trace.get('output_support', trace.get('output_support8')), dtype=bool)
            assert output_support.shape == (8,)
            output['diagnostic_counts']['N3_output_unsupported_corners'] += int(np.sum(~output_support))
    output['stage_displacements'] = {key: S.distribution(v, 'px') for key, v in values.items()}
    output['diagnostic_counts'] = dict(output['diagnostic_counts'])
    output['fallback_counts'] = dict(output['fallback_counts'])
    output['definition'] = 'SUBPIX=qS-q0; N3=qSN-qS; FINAL=qFinal-qSN; TOTAL=qFinal-q0; actual internal-limit flags reused, not inferred from output distance; head skipping and output support distinct'
    return output


def aggregate(methods, grades):
    arms = tuple(methods)
    assert arms in (EXISTING_ARMS, ARMS), arms
    ids = [r['id'] for r in methods['BASE']]
    sessions = [r['session'] for r in methods['BASE']]
    assert len(set(sessions)) == 13
    for rows in methods.values():
        assert [r['id'] for r in rows] == ids and [r['session'] for r in rows] == sessions
        for row in rows:
            assert row['session'] == grades[row['id']]['session']
    groups = member_ids(ids, grades)
    bootstrap = S.SharedBootstrap(ids, sessions)
    grouped = {}
    for group, members in groups.items():
        member_set = set(members)
        subsets = {name: [r for r in rows if r['id'] in member_set] for name, rows in methods.items()}
        summaries = {name: S.summarize(rows) for name, rows in subsets.items()}
        base = summaries['BASE']
        assert all(v['corner']['reference_corners'] == base['corner']['reference_corners'] and
                   v['corner']['observed_corners'] == base['corner']['observed_corners'] and
                   v['corner']['matched_frames'] == base['corner']['matched_frames']
                   for v in summaries.values())
        grouped[group] = dict(population=dict(frames=len(members),
                sessions=len({r['session'] for r in subsets['BASE']}),
                reference_corners=base['corner']['reference_corners'], observed_corners=base['corner']['observed_corners'],
                matched_frames=base['corner']['matched_frames']), ids=members, summaries=summaries,
            RAW_canonical_damage={name: S.damage(subsets['BASE'], rows) for name, rows in subsets.items()},
            N3_canonical_damage={name: S.damage(subsets['N3'], rows) for name, rows in subsets.items()})
    comparisons = {}
    for after, before in COMPARISONS:
        if after not in methods:
            continue
        comparisons[f'{after}_minus_{before}'] = {
            group: group_comparison(methods[after], methods[before], members, bootstrap,
                        (grouped[group]['summaries'][after], grouped[group]['summaries'][before]))
            for group, members in groups.items()}
    return dict(schema='pallet_subpix_order_20261009_v1', status='DONE', methods=list(arms),
        frames=319, sessions=13, N3_seed=1, grade_counts=GRADE_COUNTS,
        groups=grouped, comparisons=comparisons, grade_effect_ranking=grade_ranking(comparisons),
        bootstrap=bootstrap.meta(), reverse_motion=reverse_motion(methods.get('SUBPIX_N3', [])),
        contract=dict(grades='existing image annotation grade clean/moderate/severe; semantics NOT_CONFIRMED; not measured external-occlusion intensity',
            corner='conditional observed canonical corner pooling; full-reference PCK/gross',
            pose='same actual final F reference errors; successful-frame distributions; full group failures retained separately',
            dispersion='sample variance/SD float64 ddof1; n<2 NA; SD is not CI/standard error/seed variability',
            grade_ranking='descriptive ranks of observed group statistics; no grade-treatment interaction confidence interval or causal claim',
            bootstrap='10000 shared original13-session draws seed20260917; original observation weights conditional on grade; contributing sessions and empty draws retained',
            inference='grade and GT used only after saved predictions; no conditional gate, oracle mix or new annotation',
            causal='distance/view/material/session may differ among grades; no causal occlusion claim or independent confirmation'),
        execution=dict(model_forwards=0, final_F_calls=0, optimizer_updates=0, new_labels=0))
