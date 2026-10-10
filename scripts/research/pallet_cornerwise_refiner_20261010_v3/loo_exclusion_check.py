"""Public, read-only checks of recorded LOO and self-occlusion fit exclusions.

No model, PnP, ground truth or numerical library is loaded. This checks saved
execution evidence, not the correctness of visibility classification or poses.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_cornerwise_refiner_20261010_v3'
PRIMARY = 'N3_CORNERWISE_ROLE'
NO_MASK = 'N3_BASIN_NO_MASK_ROBUST'


def check(raw):
    failures = []
    counts = dict(rows=0, primary_rows=0, corner_records=0, actual_LOO_fits=0,
                  adopted_actual_boundary_coordinates=0, masked_final_rows=0,
                  no_mask_control_rows=0)

    def require(ok, context):
        if not ok:
            failures.append(context)

    def excluded(solver, forbidden, context):
        require(set(solver.get('excluded', [])) == forbidden, context + ': excluded IDs')
        for key in ('used', 'fit_input_ids', 'final_inliers', 'generator_ids'):
            require(not forbidden.intersection(solver.get(key, [])), context + ': ' + key)
        require(solver.get('prior_projection_used_as_observation') is False,
                context + ': prior is not an observation')
        require(solver.get('reprojected_points_reused_as_observations') is False,
                context + ': final projections not observations')

    with gzip.open(raw, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            counts['rows'] += 1
            context = str((row['id'], row['method']))
            H = set(row['hidden_initial'])
            require(row['reprojections_reused_as_observations'] is False,
                    context + ': output projections not refit')
            if row['method'] == NO_MASK:
                counts['no_mask_control_rows'] += 1
                require(not H, context + ': mandatory no-mask control')
                excluded(row['solver'], set(), context)
            else:
                counts['masked_final_rows'] += 1
                require(H == set(row['predicted_initial_N3_hidden']),
                        context + ': actual initial H')
                excluded(row['solver'], H, context)
            require(set(row['reprojected_ids']) == (H if row['new_pose_estimated'] else set()),
                    context + ': final H replaced only after new pose')
            if row['method'] != PRIMARY:
                continue
            counts['primary_rows'] += 1
            summary = row['cornerwise_selection']
            require(set(summary['actual_self_hidden_ids']) == H, context + ': LOO H')
            require(summary['heldout_is_self_occlusion'] is False,
                    context + ': temporary heldout is not a self-occ label')
            require(summary['fully_independent_validation'] is False and
                    summary['initial_prior_includes_heldout_influence'] is True,
                    context + ': shared-prior limitation disclosed')
            records = row['observation_contract']['cornerwise_records']
            calls = 0
            adopted = []
            for record in records:
                counts['corner_records'] += 1
                k = record['id']
                rc = context + ': corner ' + str(k)
                forbidden = H | {k}
                require(set(record['actual_self_hidden_ids']) == H, rc + ': H')
                require(set(record['heldout_fit_requested_excluded_ids']) == forbidden,
                        rc + ': requested H union k')
                require(record['heldout_is_self_occlusion'] is False and
                        record['projected_coordinate_used_as_observation'] is False,
                        rc + ': observation semantics')
                if record['heldout_pose_calls']:
                    calls += record['heldout_pose_calls']
                    counts['actual_LOO_fits'] += 1
                    require(record['heldout_pose_calls'] == 1, rc + ': one solve')
                    excluded(record['loo_solver'], forbidden, rc)
                    require(record['fit_exclusion_verified'] is True,
                            rc + ': recorded fit exclusion')
                if record['accepted']:
                    counts['adopted_actual_boundary_coordinates'] += 1
                    adopted.append(k)
                    require(k not in H, rc + ': H never adopted')
                    require(row['input_points'][k] == record['candidate_xy'],
                            rc + ': actual candidate, not validation projection')
            require(calls == summary['LOO_solve_calls'] <= 8, context + ': solve count')
            require(sorted(adopted) == sorted(summary['selected_boundary_corner_ids']),
                    context + ': adopted IDs')
    expected = dict(rows=980, primary_rows=245, corner_records=447, actual_LOO_fits=423,
                    adopted_actual_boundary_coordinates=140, masked_final_rows=735,
                    no_mask_control_rows=245)
    require(counts == expected, 'preregistered execution counts')
    return dict(passed=not failures, counts=counts, failures=failures,
                raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                recorded_fit_and_generator_exclusion_checked=True,
                shared_initial_prior_influence_not_removed=True,
                independent_visibility_or_pose_accuracy_certification=False,
                actual_model_PnP_GT_training_RGB_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=DOC / 'PREDICTIONS.jsonl.gz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.raw)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
