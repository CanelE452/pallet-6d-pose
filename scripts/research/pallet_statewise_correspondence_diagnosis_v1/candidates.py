"""Extract the original e G10/G13 archives without scoring or fitting poses.

Outcome rows are not assumed to contain poses. Feasible sample lists are
retained in full, separately from outcome/initialization/selection records.
V15 and V17 later-control proposal unions are never treated as original e
generation banks. All238 IDs must be declared explicitly by the caller.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

VIS = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
B = VIS/'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
G10 = B/'global_outline_all95_eval_v4_10'
D13 = B/'face_partial_BI_outline_eval_v4_13d'
E13 = B/'frozen_wd_partial_refinement_eval_v4_13e'
REST = VIS/'remainder_after_test49_v1/pose_evaluation_v4_14'
D_ARM = 'FACE_AWARE_PARTIAL_BI_OLD_Q'
E_ARM = 'FROZEN_WD_FACE_AWARE_PARTIAL_BI'
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
CONSUMERS = (
    'evaluate_global_outline_pose_v4_10.py',
    'evaluate_face_partial_BI_outline_v4_13d.py',
    'select_frozen_wd_partial_refinement_v4_13e.py',
    'evaluate_remainder_frozen_wd_v4_14.py',
    'statewise_global_outline_solver_v4_10.py',
    'statewise_global_outline_solver_v4_11.py',
    'statewise_partial_BI_outline_solver_v4_13.py',
    'statewise_partial_BI_u_solver_v4_13.py',
    'statewise_evaluation_v4.py',
    'evaluate_fixed_wd_archived_tangent_v4_15.py',
    'evaluate_cached_native_edges_v4_17.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_rows(root, folder, count, bindings):
    protocol_path, seal_path = root/folder/'PROTOCOL.json', root/folder/'PREDICTION_SEAL.json'
    journal = root/folder/'PREDICTIONS.jsonl.gz'
    protocol, seal = read(protocol_path), read(seal_path)
    require(seal['rows'] == count, str(folder)+'/row count')
    require(sha(journal) == seal['predictions_sha256'], str(folder)+'/prediction bytes')
    require(sha(protocol_path) == seal['protocol_sha256'], str(folder)+'/protocol bytes')
    for path in (protocol_path, seal_path, journal):
        bindings[str(path.relative_to(root))] = sha(path)
    expected = protocol['ids']
    require(len(expected) == count and len(set(expected)) == count, str(folder)+'/unique declared IDs')
    with gzip.open(journal, 'rt') as stream:
        seen = []
        for index, line in enumerate(stream):
            row = json.loads(line)
            require(index < count and row['id'] == expected[index], str(folder)+'/exact ordered IDs')
            seen.append(row['id'])
            yield row
        require(seen == expected, str(folder)+'/complete journal')


def pose_record(record, family):
    """Preserve one original R,t pair; physical center/Q are explicit derivations."""
    if record.get('R') is None or record.get('t') is None:
        return None
    R, t = np.asarray(record['R'], float), np.asarray(record['t'], float)
    xyz, Q = np.asarray(family['native_xyz'], float), np.asarray(family['physical_registration_Q'], float)
    require(R.shape == (3, 3) and t.shape == (3,) and xyz.shape == (8, 3) and Q.shape == (3, 3), 'Original pose/model shape')
    require(np.isfinite(R).all() and np.isfinite(t).all() and np.isfinite(xyz).all() and np.isfinite(Q).all(), 'Original pose/model finite')
    return dict(R_cf=record['R'], native_t=record['t'],
                centroid=(R@xyz.mean(0)+t).tolist(), R_physical=(R@Q).tolist(),
                native_model_and_registration_source='same original family record')


def extract_family(family, finished_arm=None):
    solution = family['solution']
    outcomes, feasible = [], []
    counts = Counter()
    for index, record in enumerate(solution.get('candidates', [])):
        pose = pose_record(record, family)
        outcomes.append(dict(original_index=index, original_record=deepcopy(record), pose=pose))
        counts['outcome_rows'] += 1
        counts['outcome_pose_available' if pose is not None else 'outcome_pose_missing'] += 1
        counts['outcome_joint_feasible_flag'] += bool(record.get('joint_feasible'))
        counts['infeasible_outcome_pose_missing'] += pose is None and not bool(record.get('joint_feasible'))
    for index, record in enumerate(solution.get('feasible_pose_samples', [])):
        pose = pose_record(record, family)
        require(pose is not None, 'Feasible sample has no original R,t')
        feasible.append(dict(original_index=index, original_record=deepcopy(record), pose=pose))
    counts['feasible_pose_samples'] = len(feasible)
    counts['accepted'] = bool(solution['accepted'])
    # These are saved optimizer endpoint records. Missing generated candidates
    # cannot be reconstructed from costs/initialization names alone.
    coverage = dict(
        generated_pose_bank_complete=False,
        generated_oracle_status='CANDIDATE_RECORD_INCOMPLETE',
        after_support_pose_bank_complete=False,
        feasible_samples_stored_without_extractor_truncation=True,
        feasibility_predicate_and_dedup_are_original_consumer_defined=True,
        optimizer_internal_iterates_logged=False,
        no_optimizer_outcomes_stored=not bool(outcomes),
        original_solver_schema=solution['schema'],
        missing_outcome_Rt=counts['outcome_pose_missing'],
        reason='Original consider()/fit() stores rejected outcome diagnostics without R,t; outline consumers also omit endpoint R,t before feasible dedup. Initial PnP/seed searches may have additional unrecorded rejected states.')
    return dict(name=family['name'], native_xyz=deepcopy(family['native_xyz']),
        cf_extents=deepcopy(family['cf_extents']), physical_registration_Q=deepcopy(family['physical_registration_Q']),
        coarse_pose=deepcopy(family['coarse_pose']), accepted=bool(solution['accepted']),
        solver_schema=solution['schema'], solver_status=solution.get('status'),
        original_outcomes=outcomes, feasible_samples=feasible,
        selected_solver_pose=pose_record(solution, family),
        published_actual_pose=None if finished_arm is None else deepcopy(finished_arm['actual_pose']),
        published_fallback_audit=None if finished_arm is None else deepcopy(finished_arm.get('fallback_audit')),
        original_objective=solution.get('objective'), original_objective_components=deepcopy(solution.get('objective_components')),
        original_search=deepcopy(solution.get('search')), original_parent_search=deepcopy(solution.get('parent_search_audit')),
        original_rank=deepcopy(solution.get('rank')), original_ambiguity=deepcopy(solution.get('ambiguity')),
        counts=dict(counts), coverage=coverage)


def extract_stage(row):
    require([f['name'] for f in row['families']] == list(FAMILIES), 'Original two-family order differs')
    return dict(selected_family=row['selected_family'], family_ambiguity=deepcopy(row['family_ambiguity']),
        family_selection=deepcopy(row.get('family_selection')),
        families=[extract_family(f, row['arms'].get(f['name'])) for f in row['families']],
        original_row_sha256=canonical(row))


def packet(g10, d, original_e, source):
    ident = g10['id']
    require(ident == d['id'], 'Original G10/G13 frame differs')
    require(g10['image_sha256'] == d['image_sha256'], ident+'/same RGB')
    require(g10['states'] == d['states'] == original_e['states'], ident+'/same original states')
    require(original_e['WD_family'] == g10['selected_family'], ident+'/e frozen G10 family')
    selected = d['arms'][original_e['WD_family']]
    require(selected['actual_pose'] == original_e['actual_pose'], ident+'/exact selected original e numeric pose')
    return dict(id=ident, partition=g10['partition'], session=g10['session'],
        image_sha256=g10['image_sha256'], states=g10['states'],
        stages=dict(G10=extract_stage(g10), G13=extract_stage(d)),
        original_e=dict(frozen_family=original_e['WD_family'], actual_pose=deepcopy(original_e['actual_pose']),
            accepted=original_e['accepted'], fallback_audit=deepcopy(original_e.get('fallback_audit')),
            hierarchy=deepcopy(original_e.get('hierarchical_correspondence_audit')),
            ambiguity=deepcopy(original_e.get('ambiguity_audit'))),
        original_source=source,
        all_generated_candidate_oracle_available=False,
        original_feasible_bank_only_not_V17_later_proposal_union=True,
        GT_reference_or_success_inputs=False)


def extract_existing(*, source_root, ordered_ids, output):
    started = time.monotonic()
    root, output = Path(source_root), Path(output)
    require(len(ordered_ids) == 238 and len(set(ordered_ids)) == 238, 'Explicit unique full238 ordered IDs required')
    require(not output.exists(), 'Preserve previous extraction attempts')
    bindings, originals = {}, {}
    for row in checked_rows(root, G10, 95, bindings):
        originals[row['id']] = row
    e95 = {r['id']: r for r in checked_rows(root, E13, 95, bindings)}
    output.mkdir(parents=True)
    target = output/'ORIGINAL_E_CANDIDATE_BANK.jsonl.gz'
    aggregate, schemas = {}, {'G10': Counter(), 'G13': Counter()}
    missing_rows = {'G10': set(), 'G13': set()}
    seen, cache_reuse = [], Counter()
    with gzip.open(target, 'xt', encoding='utf-8') as stream:
        def emit(value):
            seen.append(value['id'])
            for stage in ('G10', 'G13'):
                key = stage+'_'+value['partition']
                c = aggregate.setdefault(key, Counter())
                c['rows'] += 1
                for family in value['stages'][stage]['families']:
                    c['family_solutions'] += 1
                    c.update(family['counts'])
                    schemas[stage][family['solver_schema']] += 1
                    if family['counts'].get('outcome_pose_missing', 0):
                        missing_rows[stage].add(value['id'])
            stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
        for row in checked_rows(root, D13, 95, bindings):
            ident = row['id']
            require(ident in originals and ident in e95, ident+'/original95 joins')
            d = row['arms'][D_ARM]
            require(row['arms']['SAME_OLD_G10_NUMERIC'] == originals[ident], ident+'/immutable G10 copy')
            cache_reuse['exact_G10_cache_reused'] += bool(row['exact_whole_old_numeric_cache_reuse'])
            emit(packet(originals[ident], d, e95[ident]['arms'][E_ARM],
                        dict(G10=str(G10/'PREDICTIONS.jsonl.gz'), G13=str(D13/'PREDICTIONS.jsonl.gz'),
                             E0=str(E13/'PREDICTIONS.jsonl.gz'),
                             exact_D_G10_cache_reuse=row['exact_whole_old_numeric_cache_reuse'])))
        for row in checked_rows(root, REST, 143, bindings):
            d = row['partial_global_outline']
            cache_reuse['exact_G10_cache_reused'] += bool(d['exact_whole_old_numeric_cache_reuse'])
            emit(packet(row['old_global_outline'], d['arms'][D_ARM], row['arms'][E_ARM],
                        dict(G10=str(REST/'PREDICTIONS.jsonl.gz')+'#old_global_outline',
                             G13=str(REST/'PREDICTIONS.jsonl.gz')+'#partial_global_outline',
                             E0=str(REST/'PREDICTIONS.jsonl.gz')+'#arms.'+E_ARM,
                             exact_D_G10_cache_reuse=d['exact_whole_old_numeric_cache_reuse'])))
    require(seen == list(ordered_ids), 'Exact explicitly declared full238 order differs')
    require(Counter(k.split('_', 1)[1] for k in aggregate) == Counter(CAL=2, TEST=2, REMAINDER=2), 'Partition coverage')
    consumers = {}
    for name in CONSUMERS:
        path = Path('scripts/research/pallet_visibility_curriculum_20261013_v1')/name
        consumers[str(path)] = sha(root/path)
    coverage = dict(schema='ORIGINAL_E_G10_G13_CANDIDATE_COVERAGE_V1', rows=238,
        original_G10_and_G13_family_solutions_each=476, aggregate={k: dict(v) for k, v in aggregate.items()},
        schemas={k: dict(v) for k, v in schemas.items()},
        rows_with_missing_outcome_Rt={k: sorted(v) for k, v in missing_rows.items()},
        exact_D_G10_cache_reuse_rows=cache_reuse['exact_G10_cache_reused'],
        generated_and_after_support_oracle='CANDIDATE_RECORD_INCOMPLETE',
        feasible_oracle_scope='All retained distinct original feasible endpoint samples, with original family and stage. Rejected generated poses and optimizer iterates are not covered.',
        mandatory_replay_for_complete_generated_pruning_diagnosis=True,
        V15_later_start_bank_not_used=True, V17_later_proposal_union_not_used=True,
        new_model_PnP_optimizer_or_reference_calls=0)
    write(output/'CANDIDATE_COVERAGE.json', coverage)
    write(output/'CANDIDATE_MANIFEST.json', dict(schema='ORIGINAL_E_CANDIDATE_EXTRACTION_MANIFEST_V1',
        ordered_ids=seen, input_sha256=bindings, original_consumer_sha256=consumers,
        source_sha256=sha(Path(__file__)), artifacts={p.name: sha(p) for p in sorted(output.iterdir())},
        world_complete_oracle_claimed=False, new_fit_or_reference_calls=0,
        original_artifacts_modified=False, wall_seconds=time.monotonic()-started))
    return coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--ids-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    ids = read(args.ids_file)
    require(isinstance(ids, list), 'IDs file must be an explicit full238 JSON list')
    result = extract_existing(source_root=args.source_root, ordered_ids=ids, output=args.output)
    print(json.dumps(dict(rows=result['rows'], generated_oracle=result['generated_and_after_support_oracle'])))
