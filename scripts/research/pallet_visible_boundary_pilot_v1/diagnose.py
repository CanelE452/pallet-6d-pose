"""Post-seal evidence discrimination; reference labels never reach replay.

All comparisons are within a single frame/stage/family. This is explanatory
analysis, not a validated alternative selector or a second observation set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import resource
import time

from .evaluate import sha, read, rows, csvwrite, metric, write
from scripts.research.pallet_selection_score_diagnosis_v1.selection import rank_retained


def run(run_folder, neutral_package, state_private, output):
    started, cpu = time.monotonic(), time.process_time()
    folder, package, state, out = map(Path, (run_folder, neutral_package, state_private, output))
    seal = read(folder/'PREDICTION_SEAL.json')
    for name, value in seal['artifacts'].items():
        if sha(folder/name) != value:
            raise ValueError('Prediction seal mismatch')
    pkg = read(package)
    if sha(package) != seal['neutral_package_sha256']:
        raise ValueError('Neutral input changed')
    frames = rows(package.parent/pkg['frames']['path'])
    candidates = rows(package.parent/pkg['candidates']['path'])
    costs = {(r['id'], r['candidate_id']): r for r in rows(folder/'CANDIDATE_SCORES.jsonl.gz')}
    refs_path = state/'score_v2/FIXED_REFERENCES.json'
    if sha(refs_path) != read(state/'score_v2/SCORE_SEAL.json')['artifacts']['FIXED_REFERENCES.json']:
        raise ValueError('Reference changed')
    refs, frame = read(refs_path)['references'], {f['id']: f for f in frames}
    measurements, pools, candidate_errors = {}, defaultdict(list), []
    for c in candidates:
        f = frame[c['id']]
        m = metric(dict(c['pose'], available=True), refs[f['partition']][c['id']])
        measurements[(c['id'], c['candidate_id'])] = m
        pools[(c['id'], c['stage'], c['family'])].append(c)
        candidate_errors.append(dict(id=c['id'], stage=c['stage'], family=c['family'],
            candidate_id=c['candidate_id'], translation_cm=m['translation_cm'], rotation_deg=m['rotation_deg'],
            success_le5=m['success_le5'], postseal_only=True))

    def cost(c):
        return costs[(c['id'], c['candidate_id'])]

    def good(c):
        return measurements[(c['id'], c['candidate_id'])]['success_le5']

    def best(pool, key):
        if not pool:
            return None
        return rank_retained(pool, {c['candidate_id']: float(key(cost(c))) for c in pool})[0]

    predictions = rows(folder/'PREDICTIONS.jsonl.gz')
    pred = {(p['id'], p['variant']): p for p in predictions}
    discrimination, family_rows, selected_components, final_rows, case_rows = [], [], [], [], []
    for f in frames:
        fid = f['id']
        visible = pred[(fid, 'VISIBLE_SUPPORT')]
        for stage in ('G10', 'G13'):
            for family in ('ORIGINAL_WD', 'SWAPPED_WD'):
                pool = pools[(fid, stage, family)]
                positives, negatives = [c for c in pool if good(c)], [c for c in pool if not good(c)]
                chosen_id = visible['stage_winner_ids'][stage][family]
                chosen = next((c for c in pool if c['candidate_id'] == chosen_id), None)
                ae = lambda v: v['visible_evidence']['robust_mean_A_E']
                combined = lambda v: v['scores']['VISIBLE_SUPPORT']
                ae_best, ae_good, ae_bad = best(pool, ae), best(positives, ae), best(negatives, ae)
                score_good, score_bad = best(positives, combined), best(negatives, combined)
                def val(c, key):
                    return None if c is None else float(key(cost(c)))
                discrimination.append(dict(id=fid, ordinal=f['ordinal'], stage=stage, family=family,
                    retained=len(pool), successful_retained=len(positives), failed_retained=len(negatives),
                    actual_stage_winner_id=chosen_id,
                    actual_stage_winner_success=None if chosen is None else good(chosen),
                    family_selected_by_G10=visible['selected_family'] == family,
                    pure_A_E_best_id=None if ae_best is None else ae_best['candidate_id'],
                    pure_A_E_best_success=None if ae_best is None else good(ae_best),
                    best_success_A_E_candidate=None if ae_good is None else ae_good['candidate_id'],
                    best_failed_A_E_candidate=None if ae_bad is None else ae_bad['candidate_id'],
                    best_success_A_E=val(ae_good, ae), best_failed_A_E=val(ae_bad, ae),
                    A_E_good_minus_bad=None if ae_good is None or ae_bad is None else val(ae_good, ae)-val(ae_bad, ae),
                    best_success_combined_candidate=None if score_good is None else score_good['candidate_id'],
                    best_failed_combined_candidate=None if score_bad is None else score_bad['candidate_id'],
                    best_success_combined=val(score_good, combined), best_failed_combined=val(score_bad, combined),
                    combined_good_minus_bad=None if score_good is None or score_bad is None else val(score_good, combined)-val(score_bad, combined),
                    postseal_diagnostic_only=True))

        for variant in seal['variants']:
            p = pred[(fid, variant)]
            for fam, cid in p['stage_winner_ids']['G10'].items():
                if cid is None:
                    continue
                v, m = costs[(fid, cid)], measurements[(fid, cid)]
                family_rows.append(dict(id=fid, variant=variant, family=fam, candidate_id=cid,
                    score=v['scores'][variant], frozen_family=p['selected_family'] == fam,
                    translation_cm=m['translation_cm'], rotation_deg=m['rotation_deg'],
                    reference_success_after_seal=m['success_le5']))
            family = p['selected_family']
            pool = pools[(fid, 'G13', family)]
            selected = next((c for c in pool if c['candidate_id'] == p['candidate_id']), None)
            positives = [c for c in pool if good(c)]
            comparator = best(positives, lambda v: v['scores'][variant])
            sv, gv = None if selected is None else cost(selected), None if comparator is None else cost(comparator)
            final_rows.append(dict(id=fid, variant=variant, family=family, selected_candidate_id=p['candidate_id'],
                selected_reference_success=False if selected is None else good(selected),
                selected_score=None if sv is None else sv['scores'][variant],
                successful_retained_same_family=len(positives),
                diagnostic_success_candidate_id=None if comparator is None else comparator['candidate_id'],
                diagnostic_success_score=None if gv is None else gv['scores'][variant],
                success_minus_selected_score=None if sv is None or gv is None else gv['scores'][variant]-sv['scores'][variant],
                postseal_oracle_diagnostic_only=True))
            for role, v in [('ACTUAL_SELECTED', sv), ('POSTSEAL_SUCCESS_COMPARATOR', gv)]:
                if v is None:
                    continue
                e, parts = v['visible_evidence'], v['original_weighted_components']
                selected_components.append(dict(id=fid, variant=variant, role=role, candidate_id=v['candidate_id'],
                    family=family, score=v['scores'][variant], C=v['non_outline_cost'], full_BI=parts['full_BI'],
                    partial_normal=parts['partial_normal'], prior=parts['prior'],
                    replacement_outline=v['replacement_outline_costs'][variant], A_E=e['robust_mean_A_E'],
                    raw_distance_sum_px=e['raw_distance_sum_px'], raw_distance_mean_px=e['raw_distance_mean_px'],
                    raw_distance_max_px=e['raw_distance_max_px'], mass_weighted_distance_mean_px=e['mass_weighted_distance_mean_px'],
                    polylines=e['polyline_count'], raw_segments=e['segment_count'], quadrature_samples=e['sample_count'],
                    total_length_px=e['total_length_px'], Z_E=v['visible_mass_Z_E'], Z_H=v['old_observed_mass_Z_H'], r=v['r'],
                    postseal_reference_comparator_not_input=True))

        own = next(r for r in discrimination if r['id'] == fid and r['stage'] == 'G13' and r['family'] == visible['selected_family'])
        final_success = measurements[(fid, visible['candidate_id'])]['success_le5']
        other_good = sum(r['successful_retained'] for r in discrimination if r['id'] == fid and r['stage'] == 'G13' and r['family'] != visible['selected_family'])
        if final_success:
            category = 'ACTUAL_SELECTED_SUCCESS'
        elif own['successful_retained'] == 0:
            category = 'G10_FAMILY_EXCLUDED_SUCCESSFUL_G13_POOL' if other_good else 'NO_SUCCESSFUL_G13_CANDIDATE'
        elif own['pure_A_E_best_success']:
            category = 'A_E_PREFERS_SUCCESS_BUT_C_COMBINATION_LOSES'
        else:
            category = 'A_E_ALSO_PREFERS_FAILURE_WITHIN_SELECTED_FAMILY'
        case_rows.append(dict(id=fid, ordinal=f['ordinal'], selected_family=visible['selected_family'],
            selected_candidate_id=visible['candidate_id'], selected_success=final_success,
            successful_same_family_G13=own['successful_retained'], successful_other_family_G13=other_good,
            pure_A_E_best_same_family_success=own['pure_A_E_best_success'], category=category,
            interpretation='postseal evidence; not proof of physical truth or new-selector validation'))

    out.mkdir(parents=True, exist_ok=False)
    for name, records in [('CANDIDATE_REFERENCE_ERRORS.csv', candidate_errors),
            ('EVIDENCE_DISCRIMINATION.csv', discrimination), ('G10_FAMILY_DIAGNOSTICS.csv', family_rows),
            ('G13_SELECTION_DIAGNOSTICS.csv', final_rows), ('SELECTED_COMPONENTS.csv', selected_components),
            ('CASE_DIAGNOSIS.csv', case_rows)]:
        csvwrite(out/name, records)
    write(out/'DIAGNOSTIC_RECEIPT.json', dict(prediction_seal_sha256=sha(folder/'PREDICTION_SEAL.json'),
        fixed_candidates_scored_after_seal=len(candidates), stage_family_comparisons=len(discrimination),
        references_used_only_after_seal=True, new_selection=False, new_NN=0, new_pose_fit=0,
        artifacts={p.name: sha(p) for p in out.iterdir()}, elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu,
        maximum_resident_set_size_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return case_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('run-folder', 'neutral-package', 'state-private', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    print(run(args.run_folder, args.neutral_package, args.state_private, args.output))
