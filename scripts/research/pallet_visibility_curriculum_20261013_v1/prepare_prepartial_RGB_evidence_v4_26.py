"""CAL46 metadata preparation for the new prepartial RGB source policy.

No references, RGB arrays, likelihood objectives or optimizers are accessed.
The current v25 input CAL prefix and separately sealed fresh grazing17c CAL
observations are joined exactly. Only the existing O-abstention metadata
transform explains the observation-vs-runtime domain difference. Original
geometry/Q, state gates, q/noise/reliability, U, partial archives, actual chains
and support/fit priors stay fixed. Runtime active groups are sealed before fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil

from . import common as C
from . import prepartial_RGB_point_preservation_v4_26 as AUTH
from . import o_localization_abstention_v4_19 as O_ABSTENTION


B = C.DOC / 'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OBS = B / 'grazing_native_edge_observations_v4_18b'
TASKS = B / 'coarse_prior_containment_eval_v4_25'
OUTPUT = B / 'prepartial_RGB_evidence_v4_26_checks'
DECLARATION = B / 'prepartial_RGB_preservation_plan_v4_26/PREDECLARATION.json'
DECLARED_ARMS = ('SAME_FROZEN_E', 'PREPARTIAL_RGB_NATIVE', 'PREPARTIAL_RGB_NATIVE_E_FAMILY',
    'PREPARTIAL_RGB_TRANSLATION_NATIVE', 'PREPARTIAL_RGB_TRANSLATION_NATIVE_E_FAMILY',
    'PREPARTIAL_RGB_TRANSLATION_CONTAINMENT', 'PREPARTIAL_RGB_TRANSLATION_CONTAINMENT_E_FAMILY')


def _verify(mapping):
    for relative, digest in mapping.items():
        if C.sha(C.ROOT / relative) != digest:
            raise ValueError('Bound bytes changed: ' + relative)


def _cal_task_prefix(path, ids):
    """Decode only the fixed CAL46 prefix, never request the following row."""
    iterator = iter(C.rows(path))
    for expected in ids:
        task = next(iterator, None)
        if task is None or task.get('partition') != 'CAL' or task.get('id') != expected:
            raise ValueError('Exact ordered CAL-only input prefix required')
        yield task


def _validate_row_bindings(task, row):
    if task.get('partition') != 'CAL' or row.get('partition') != 'CAL':
        raise ValueError('CAL only before observation fields')
    if task['id'] != row['id']:
        raise ValueError('Same ordered CAL identity required')
    spec, geometry = task['spec'], row['initial_geometry']
    states, hidden = task['states'], task['initial_self_hidden']
    if len(states) != 8 or len(hidden) != 8 or any(bool(hidden[k]) != (states[k] == 'S') for k in range(8)):
        raise ValueError('Exact permanent initial S gate required')
    # Gate original points and partial records before equality/copy/fields.
    for records in (spec['visible'], row['visible_measurements']):
        for measurement in records:
            k = measurement['native_corner']
            if hidden[k] or states[k] not in ('B', 'I') or measurement.get('state', states[k]) != states[k]:
                raise ValueError('Only initial-nonS BI visible records before point fields')
    for boundary in task['boundary_merged']:
        k = boundary['native_corner']
        if hidden[k] or states[k] not in ('B', 'I') or boundary.get('state') != states[k]:
            raise ValueError('Only initial-nonS BI partial records before archived fields')
    family = task['families']['ORIGINAL_WD']
    if (row['states'] != states or spec['states'] != states
            or row['visible_measurements'] != spec['visible']
            or row['u_regions'] != task['u_regions'] or row['image_sha256'] != task['image_sha256']
            or geometry['initial_self_hidden'] != hidden or geometry['native_xyz'] != family['native_xyz']
            or geometry['physical_registration_Q'] != family['physical_registration_Q']
            or geometry['K'] != spec['K'] or geometry['image_hw'] != spec['image_hw']
            or geometry['initial_projection_xy'] != task['complete_BI_evidence']['initial_projection_xy']
            or row['chain_observation'] != task['new_chain_observation']
            or row['junction_proposals'] != task['complete_BI_evidence']['junction_proposals']):
        raise ValueError('Same frozen BI/state/S/U/geometry/Q/K/image/actual-chain metadata required')
    # These runtime domains already use v19 canvas-only O. No new range or
    # parameter is chosen here, and the original OBS domains remain archived.
    domains = O_ABSTENTION.abstain_external_o_localization(row['domains'], states, spec['image_hw'],
        initial_self_hidden=hidden)
    if domains['domains'] != spec['domains']:
        raise ValueError('Only the exact unchanged v19 O-domain adaptation is allowed')
    if domains['records'] != task['O_localization_abstention']['records']:
        raise ValueError('Original O domain provenance changed')
    return spec, family, geometry


def prepare_one(task, row):
    """Authenticate and plan once without evaluating a pose or error residual."""
    spec, family, geometry = _validate_row_bindings(task, row)
    cfg = deepcopy(spec['config'])
    cfg['prior_weight'] = 0.
    original = AUTH.N.prepare_native_edge_problem(family['native_xyz'], spec['K'], spec['states'],
        spec['visible'], spec['domains'], spec['image_hw'], family['coarse_pose'], config=cfg,
        boundary_observations=task['boundary_merged'], u_regions=task['u_regions'],
        edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
    proxy = AUTH.prepare_prepartial_native_problem(original, visible_measurements=spec['visible'],
                                                    initial_self_hidden=task['initial_self_hidden'])
    bundle = AUTH.authenticate_complete_bi_image_chain_groups(proxy, row['chain_observation'],
        junction_proposals=row['junction_proposals'], visible_measurements=spec['visible'],
        initial_self_hidden=task['initial_self_hidden'],
        image_binding=dict(sha256=task['image_sha256'], image_hw=spec['image_hw']),
        initial_projection_xy=geometry['initial_projection_xy'], normal_variance_component_px2=0.,
        angular_variance_component_sine2=0., noise_provenance=dict(fit_partition='CAL',
            observation_family=AUTH.NOISE_OBSERVATION_FAMILY, mode='ZERO_COMPONENT_PREPARATION_NOT_FITTED_NOISE'),
        calibration_preparation=True)
    plan = AUTH.plan_complete_bi_replacement(proxy, bundle, overlap_policy=AUTH.PROTOTYPE_OVERLAP_POLICY)
    local = Counter()
    sources = Counter()
    for corner in plan['corner_plans']:
        mode = ('FULL_Q_RETAINED' if not corner['point_cost_revoked'] else
                'ARCHIVED_COMPLETE_TANGENT' if 'tangent_kind' in corner else 'FINITE_TYPED_LINES_ONLY')
        local[mode] += 1
        sources[corner['original_measurement_archive']['original_measurement_source']] += 1
    if (sum(local.values()) != len(proxy.complete)
            or plan['ledger']['original_complete_q_factors_once'] != local['FULL_Q_RETAINED']
            or plan['ledger']['archived_complete_tangent_factors_once'] != local['ARCHIVED_COMPLETE_TANGENT']
            or len(plan['active_groups']) != len({tuple(g['physical_native_edge']) for g in plan['active_groups']})
            or plan['restored_prepartial_complete_ids'] != list(proxy.restored_prepartial_complete_ids)):
        raise ValueError('Exact restored-q/tangent/unique-edge runtime ledger required')
    prepared = dict(id=task['id'], partition='CAL', session=task['session'], initial_geometry=geometry,
        authenticated_group_bundle={**bundle, 'groups':plan['active_groups']},
        authentication_inventory_groups=bundle['groups'], replacement_plan=plan,
        fit_population='ONLY v26 prepartial-preserving runtime active_groups; full authentication inventory separate',
        preparation_only_no_reference_decode=True)
    metadata = dict(id=task['id'], session=task['session'], complete_modes=dict(local),
        original_complete_count=len(original.complete), complete_count_after_prepartial_restore=len(proxy.complete),
        restored_prepartial_complete_ids=list(proxy.restored_prepartial_complete_ids),
        restored_source_counts=proxy.prepartial_preservation_audit['restored_source_counts'],
        prepartial_preservation_audit=proxy.prepartial_preservation_audit,
        authenticated_groups=len(bundle['groups']), active_groups=len(plan['active_groups']), ledger=plan['ledger'],
        relation_max_error_px=max((r['maximum_relation_error_px'] for r in bundle['actual_lattice_gradient_subpixel_relation']), default=0.),
        corner_plans=plan['corner_plans'], rejections=plan['main_replacement_rejections'],
        preparation_bundle_not_used_for_pose_objective=True)
    totals = Counter(original_complete=len(original.complete), complete_after_restore=len(proxy.complete),
        restored_prepartial_complete=len(proxy.restored_prepartial_complete_ids),
        groups=len(bundle['groups']), alternatives=sum(len(g['components']) for g in bundle['groups']),
        active_groups=len(plan['active_groups']), active_alternatives=sum(len(g['components']) for g in plan['active_groups']),
        distinct_supports=sum(len(g['support_prior_ledger']) for g in bundle['groups']),
        active_distinct_supports=sum(len(g['support_prior_ledger']) for g in plan['active_groups']),
        relation_checks=len(bundle['actual_lattice_gradient_subpixel_relation']))
    return prepared, metadata, local, sources, totals


def main(output=OUTPUT, *, declaration=DECLARATION):
    output, declaration = Path(output).resolve(), Path(declaration).resolve()
    if output.exists():
        raise FileExistsError('Preserve every prepartial metadata preparation')
    declared = C.read(declaration)
    if declared['arms'] != list(DECLARED_ARMS):
        raise ValueError('Fixed root seven-arm declaration required before preparation')
    obs_seal, protocol, task_seal = (C.read(path) for path in
        (OBS/'CAL_OBSERVATION_SEAL.json', TASKS/'PROTOCOL.json', TASKS/'INPUT_SEAL.json'))
    if (obs_seal['rows'] != 46 or C.sha(OBS/'CAL_OBSERVATIONS.jsonl.gz') != obs_seal['observations_sha256']
            or task_seal['rows'] != 238 or C.sha(TASKS/'INPUTS.jsonl.gz') != task_seal['input_sha256']
            or C.sha(TASKS/'PROTOCOL.json') != task_seal['protocol_sha256']):
        raise ValueError('Sealed fresh CAL46 observations and exact runtime238 inputs required')
    ids = obs_seal['ordered_ids']
    if len(ids) != 46 or len(set(ids)) != 46 or ids != protocol['ids'][:46]:
        raise ValueError('Original unique CAL46 prefix required')
    source = dict(protocol['source_sha256'])
    for module in (AUTH, AUTH.SOURCE, AUTH.SOURCE.OLD, AUTH.N, AUTH.M, AUTH.M.E,
                   AUTH.OBS, AUTH.N.J, AUTH.N.P, AUTH.N.T, O_ABSTENTION, C):
        path = Path(module.__file__).resolve()
        source[str(path.relative_to(C.ROOT))] = C.sha(path)
    source[str(Path(__file__).resolve().relative_to(C.ROOT))] = C.sha(__file__)
    inputs = {str(path.relative_to(C.ROOT)):C.sha(path) for path in
        (OBS/'CAL_OBSERVATION_SEAL.json', OBS/'CAL_OBSERVATIONS.jsonl.gz', OBS/'PROTOCOL.json',
         TASKS/'PROTOCOL.json', TASKS/'INPUT_SEAL.json', TASKS/'INPUTS.jsonl.gz', declaration)}
    for mapping in (source, inputs, protocol['protected_user_sha256']):
        _verify(mapping)
    output.mkdir(parents=True)
    C.write(output/'PREPARATION_SEAL.json', dict(schema='CAL46_PREPARTIAL_RGB_ZERO_COMPONENT_METADATA_PREFLIGHT_V26',
        ids=ids, source_sha256=source, input_sha256=inputs, protected_user_sha256=protocol['protected_user_sha256'],
        before_first_CAL_observation_decode=True, CAL_reference_labels_used=False,
        TEST_and_REMAINDER_rows_decoded=False, root_declaration_binding=dict(path=str(declaration.relative_to(C.ROOT)),sha256=C.sha(declaration))))
    for relative, digest in source.items():
        target = output/'code_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative, target)
        if C.sha(target) != digest:
            raise ValueError('Source snapshot mismatch')
    observations = list(C.rows(OBS/'CAL_OBSERVATIONS.jsonl.gz'))
    if [row['id'] for row in observations] != ids or any(row['partition'] != 'CAL' for row in observations):
        raise ValueError('Exact CAL-only observation archive required')
    prepared, records, errors = [], [], []
    modes, sources, totals, reasons = Counter(), Counter(), Counter(), Counter()
    for task, observation in zip(_cal_task_prefix(TASKS/'INPUTS.jsonl.gz', ids), observations):
        try:
            item, metadata, local, source_counts, counts = prepare_one(task, observation)
            prepared.append(item); records.append(metadata)
            modes.update(local); sources.update(source_counts); totals.update(counts)
            for rejected in metadata['rejections']:
                reasons[rejected['reason']] += 1
        except Exception as error:
            errors.append(dict(id=task['id'], exception=type(error).__name__, reason=str(error)))
    if len(records) + len(errors) != 46:
        raise ValueError('Every original CAL row must be accounted for')
    for mapping in (source, inputs, protocol['protected_user_sha256']):
        _verify(mapping)
    C.save_rows(output/'CAL_PREPARED_GROUPS.jsonl.gz', prepared)
    report = dict(schema='CAL46_PREPARTIAL_RGB_METADATA_PREFLIGHT_V26', rows=46,
        successful_metadata_rows=len(records), error_count=len(errors), errors=errors,
        totals=dict(totals), complete_modes=dict(modes), complete_measurement_sources=dict(sources),
        rejection_reasons=dict(reasons), records=records, CAL46_inventory_not_CAL44_fitted_model=True,
        maximum_lattice_relation_error_px=max((r['relation_max_error_px'] for r in records), default=0.),
        prepared_groups_sha256=C.sha(output/'CAL_PREPARED_GROUPS.jsonl.gz'),
        source_policy=AUTH.SOURCE_POLICY, observation_generation=AUTH.NOISE_OBSERVATION_FAMILY,
        support_fit_variant_ids_and_normalized_priors_sealed_before_any_reference_decode=True,
        CAL_fit_applicability='ONLY restored-prepartial source-policy runtime active groups',
        root_declaration_sha256=C.sha(declaration), runtime_source_gate_has_no_GT_ID_error_input=True,
        new_CAL_fit=False, label_decodes=0, image_queries=0, new_line_fits=0, pose_objective_calls=0,
        pose_optimizations=0, TEST_REMAINDER_rows_decoded=0, tangent_gamma_fitted=False,
        old_sources_or_models_changed=False, no_information_independence_or_coverage_claim=True)
    C.write(output/'CAL_PREFLIGHT_V1.json', report)
    C.write(output/'CAL_PREFLIGHT_SEAL_V1.json', dict(report_sha256=C.sha(output/'CAL_PREFLIGHT_V1.json'),
        preparation_sha256=C.sha(output/'PREPARATION_SEAL.json'), prepared_groups_sha256=C.sha(output/'CAL_PREPARED_GROUPS.jsonl.gz'),
        source_sha256=source, input_sha256=inputs, protected_user_sha256=protocol['protected_user_sha256']))
    print(dict(rows=46, successful=len(records), errors=len(errors), totals=dict(totals),
               modes=dict(modes), label_decodes=0, fitted=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--declaration', type=Path, default=DECLARATION)
    args = parser.parse_args()
    main(args.output, declaration=args.declaration)
