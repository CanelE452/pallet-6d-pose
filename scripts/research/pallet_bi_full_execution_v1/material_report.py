"""Seal every S3/S4 row and summarize reference-free executed evidence.

This is an archive validator/report generator. It runs no render, matcher,
optimizer, learned forward, or reference-based accuracy calculation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import image_batch as I, material_assets as A


def save_validated(path, values):
    if Path(path).exists():
        if I.canonical(list(I.rows(path))) != I.canonical(values):
            raise ValueError('Completed report rows changed: ' + str(path))
        return
    I.save(path, values)


def report(production, output):
    production, output = map(Path, (production, output))
    output.mkdir(parents=True, exist_ok=True)
    full = I.read(production / 'FULL_RECEIPT.json')
    if not full.get('complete') or full.get('rows') != 1241 or not full.get('all1241_completed'):
        raise ValueError('A complete full1241 producer receipt is required')
    lock = production / 'METHOD_LOCK_RESOURCE_LP4.json'
    if full['method_lock_sha256'] != A.sha(lock):
        raise ValueError('Active method binding differs')
    for name, digest in full['output_artifacts'].items():
        if A.sha(production / name) != digest:
            raise ValueError('An archived producer output differs: ' + name)
    binding = []
    totals = defaultdict(Counter)
    dimension_roles = defaultdict(Counter)
    status_counts = defaultdict(Counter)
    seconds = defaultdict(list)
    row_records = []
    ids = set()
    candidates_by_archive = defaultdict(list)
    for domain, expected in [('real', 238), ('synthetic', 1003)]:
        paths = sorted((production / 'frames' / domain).glob('*/SEAL.json'))
        if len(paths) != expected:
            raise ValueError(f'{domain} has {len(paths)} row seals, expected {expected}')
        for sealpath in paths:
            seal = I.read(sealpath)
            framepath = sealpath.parent / 'FRAME.jsonl.gz'
            if not seal['complete'] or A.sha(framepath) != seal['frame_sha256']:
                raise ValueError('Invalid row archive: ' + str(sealpath))
            row = next(I.rows(framepath))
            ident = row['id']
            if ident != seal['id'] or ident in ids or not row['complete']:
                raise ValueError('Duplicate, changed or incomplete row')
            ids.add(ident)
            binding.append(dict(domain=domain, id=ident,
                row_seal=str(sealpath.relative_to(production)), row_seal_sha256=A.sha(sealpath),
                frame=str(framepath.relative_to(production)), frame_sha256=seal['frame_sha256'],
                consumed_dependency_binding=seal['dependencies']))
            totals[domain]['frames'] += 1
            totals[domain]['physical_initializations'] += len(row['initializations'])
            totals[domain]['exact_unaffected_substage_reuses'] += len(row.get('exact_reused_substages', []))
            totals[domain]['asset/' + row['asset_registration']['proxy_key']] += 1
            status_counts[domain]['row/' + row['status']] += 1
            seconds[domain].append(row['resource']['wall_seconds'])
            observations = row['material_observations']
            if observations is not None:
                common, bi = [observations['policy_packets'][arm] for arm in ('COMMON', 'BI')]
                totals[domain]['COMMON_BI_identical_factor_packets'] += int(common['factor_sha256'] == bi['factor_sha256'])
                totals[domain]['COMMON_BI_different_factor_packets'] += int(common['factor_sha256'] != bi['factor_sha256'])
                for arm in ('COMMON', 'BI'):
                    packet = observations['policy_packets'][arm]
                    totals[domain][arm + '/no_direct_support'] += len(packet['no_direct_support'])
                    totals[domain][arm + '/source_duplicates_removed'] += len(packet['source_duplicates_removed'])
                    sources = [s for factor in packet['factors'] for s in factor['physical_source_ids']]
                    if len(sources) != len(set(sources)):
                        raise ValueError('Raw evidence double counted inside one material packet')
                    for factor in packet['factors']:
                        dimension_roles[domain][f'{arm}/{factor["role"]}/{factor["observation_dimension"]}D'] += 1
                        totals[domain][arm + '/factors'] += 1
                        if 'cue' in factor:
                            totals[domain][arm + '/cue/' + factor['cue']] += 1
                        else:
                            totals[domain][arm + '/junctions'] += 1
                archive = observations['shared_mesh_support_archive']
                totals[domain]['shared_proxy_geometric_edges'] += len(archive)
                totals[domain]['shared_visible_edges'] += sum(r['support_status'] != 'NO_VISIBLE_SEGMENT' for r in archive)
                totals[domain]['shared_image_supported_edges'] += sum(r['support_status'] == 'IMAGE_SUPPORTED' for r in archive)
            record = dict(id=ident, domain=domain, physical_initializations=len(row['initializations']),
                proxy_key=row['asset_registration']['proxy_key'], arms={})
            object_diagonal = float(np.linalg.norm(row['asset_registration']['known_physical_dimensions_xyz_m']))
            for arm in I.ARMS:
                stage, branch = arm.split('_', 1)
                value = row[stage.lower()][branch]
                candidates = value['candidates']
                archived_candidates = candidates or [dict(id=ident, method=arm,
                    candidate_id=arm + '/NO_VALID_INITIAL_POSE', producer_available=False,
                    actual_pose=dict(available=False), producer_status=row['status'])]
                candidates_by_archive[f'EXTRAS_{arm}_{domain}.jsonl.gz'].extend(archived_candidates)
                if stage == 'S4' and row['initializations'] and I.canonical(value['exact_common_initializations']) != I.canonical(row['initializations']):
                    raise ValueError('A S4 arm consumed different physical initializations')
                if any(r.get('update', 0) > 4 for r in candidates) or len(row['initializations']) > 2:
                    raise ValueError('Frozen start/update bound exceeded')
                admitted = [r for r in candidates if r.get('producer_available', r['actual_pose'].get('available', False))]
                actual_changes = [r for r in admitted if r.get('update', 0) > 0 and not any(
                    I.R.same_pose(r['actual_pose'], start['actual_pose'], object_diagonal) for start in row['initializations'])]
                totals[domain][arm + '/visited_endpoints'] += len(candidates)
                totals[domain][arm + '/available_endpoints'] += len(admitted)
                totals[domain][arm + '/unavailable_visited_proposals'] += len(candidates) - len(admitted)
                totals[domain][arm + '/frames_with_pose_changed_from_starts'] += bool(actual_changes)
                for iteration in value['iterations']:
                    status_counts[domain][arm + '/' + iteration['status']] += 1
                    if iteration.get('nfev', 0) > 10:
                        raise ValueError('Frozen residual evaluation limit exceeded')
                record['arms'][arm] = dict(visited_endpoints=len(candidates), available_endpoints=len(admitted),
                    pose_changed_from_starts=bool(actual_changes),
                    preserved_original_unknown_correlation_count=value.get('preserved_original_unknown_correlation_count'))
            row_records.append(record)
    if len(ids) != 1241 or set(full['selected_ids']) != ids:
        raise ValueError('Producer receipt and row archive identities differ')
    for name, expected_rows in candidates_by_archive.items():
        actual_rows = list(I.rows(production / name))
        if I.canonical(actual_rows) != I.canonical(expected_rows):
            raise ValueError('Raw row candidates do not equal sealed producer extras: ' + name)
    save_validated(output / 'ALL_ROW_BINDINGS.jsonl.gz', binding)
    save_validated(output / 'REFERENCE_FREE_ROW_SUMMARY.jsonl.gz', row_records)
    statistics = {domain: dict(mean=float(np.mean(value)), median=float(np.median(value)),
        P95=float(np.quantile(value, .95)), maximum=float(max(value)), sum=float(sum(value)))
        for domain, value in seconds.items()}
    summary = dict(schema='S3_S4_REFERENCE_FREE_EXECUTED_SUMMARY_V1', complete=True, rows=1241,
        counts={k: dict(v) for k, v in totals.items()},
        dimension_by_automatic_role={k: dict(v) for k, v in dimension_roles.items()},
        iteration_status_counts={k: dict(v) for k, v in status_counts.items()},
        completed_scientific_row_wall_seconds=statistics,
        actual_accepted_checkpoint_calls=full['actual_calls'],
        actual_accepted_checkpoint_resources=full['completed_row_resource_totals'],
        physical_owner_accuracy='NOT_APPLICABLE_WITH_REASON_NO_INDEPENDENT_PHYSICAL_LABELS',
        independent_human_confirmations=0, reference_based_accuracy_executed=False,
        pose_changes_are_geometry_differences_not_accuracy_claims=True,
        descriptive_pose_change_tolerance='rotation entries1e-7; translation entries1e-7 times known object diagonal',
        current_real_reference_reads=0, new_render_matcher_optimizer_calls=0)
    I.write(output / 'REFERENCE_FREE_SUMMARY.json', summary)
    information = dict(schema='S3_S4_ACTUAL_INFORMATION_AND_INTERVENTION_TABLE_V1',
        common_access=['original native RGB', 'exact K and native integer pixel coordinates',
            'known native dimensions and original Q', 'registered structured proxy geometry',
            'automatic full and tight SAM masks', 'original D1 typed observations',
            'same at-most-two proper-positive distinct starts'],
        actual_consumption={
            'S3_COMMON': ['proxy edge/surface and N0 zbuffer', 'full grayscale gradient/Canny', 'tight/full mask boundaries',
                'arbitrary object3D point or line supports', 'nonoverlapping original typed observations'],
            'S3_BI': ['same shared supports/cues/masks/visibility as S3_COMMON', 'role-dependent cue adoption/recovery order',
                'arbitrary object3D point or line supports', 'nonoverlapping original typed observations'],
            'S4_PL': ['original native full points and typed line normals', 'automatic full mask cuboid outline',
                'same physical starts; generic Gaussian/uniform-none residuals'],
            'S4_CONTOUR': ['structured proxy visible depth and mask', 'automatic full mask boundary',
                'rerendered object3D-to-observed-boundary normal factors'],
            'S4_IMAGE': ['native whole grayscale observation', 'native whole grayscale structured proxy rendering',
                'current render depth and automatic full mask', 'bidirectional DIS FAST correspondence',
                'actual object3D to observed2D PnPRefineLM updates']},
        S3_interventions={
            '1_cue_adoption': dict(COMMON='minimum common normalized displacement/RMS cost; RGB/tight/full deterministic ties',
                BI='B: tight/full/RGB first supported; I: RGB/tight/full first supported',
                callable='material_support._policy'),
            '2_recovery': dict(COMMON='same all-cue candidate archive ranked jointly',
                BI='same candidate archive with role cue priority', callable='material_support.extract_supports/_policy'),
            '3_dimension': dict(COMMON='same evidence-driven independent-two/one/no-direction rule',
                BI='same rule applied to its actual adopted evidence; role never forces dimension', callable='material_support._policy'),
            '4_uncertainty': dict(COMMON='shared 4+lineRMS^2 conditional variance, propagated junction covariance',
                BI='identical shared rule; role-specific total-error calibration not fitted in S3',
                calibration_parameters=0, callable='material_support._line_candidate/_policy'),
            '5_downstream': 'same old B/I summaries and H1 feature algebra retained in both unions'},
        finite_budget=dict(physical_starts_maximum=2, outer_updates_maximum=4,
            residual_evaluations_per_update_maximum=10, PnPRefineLM_iterations_maximum=10,
            DIS_calls_per_image_update=2),
        physical_model_status='REGISTERED_STRUCTURED_PROXY; neither verified CAD identity nor independent physical ownership',
        hardware='Mesa llvmpipe software EGL, LP_NUM_THREADS=4; zero hardware GPU render jobs',
        current_real_reference_reads=0)
    I.write(output / 'INFORMATION_INTERVENTION_TABLE.json', information)
    receipt = dict(schema='S3_S4_ALL1241_ROW_ARCHIVE_SEAL_V1', complete=True, rows=1241,
        producer_full_receipt_sha256=A.sha(production / 'FULL_RECEIPT.json'),
        active_LP4_method_lock_sha256=A.sha(lock),
        all_row_seal_and_frame_bindings_sha256=A.sha(output / 'ALL_ROW_BINDINGS.jsonl.gz'),
        output_artifacts={name: A.sha(output / name) for name in ('ALL_ROW_BINDINGS.jsonl.gz',
            'REFERENCE_FREE_ROW_SUMMARY.jsonl.gz', 'REFERENCE_FREE_SUMMARY.json', 'INFORMATION_INTERVENTION_TABLE.json')},
        all_producer_extra_hashes_validated=True, all_row_hashes_validated=True,
        all_raw_row_candidate_payloads_equal_sealed_producer_extras=True,
        all_same_S4_initializations_validated=True, all_start_update_limits_validated=True,
        current_real_reference_reads=0, report_source_sha256=A.sha(Path(__file__)))
    I.write(output / 'ROW_ARCHIVE_SEAL.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(I.canonical(report(args.production, args.output)), flush=True)
