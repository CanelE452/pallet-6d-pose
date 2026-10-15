"""CAL46-only source-preserving metadata preflight; no labels/images/fits."""
import argparse
from collections import Counter
from copy import deepcopy
from itertools import islice
from pathlib import Path
import shutil

from . import common as C
from . import complete_bi_image_chain_evidence_v4_20c as SERIAL
from . import source_preserving_complete_bi_evidence_v4_24 as V

B = C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OBS = B/'grazing_native_edge_observations_v4_18b'
TASKS = B/'grazing_native_candidates_v4_18'
OUTPUT = B/'source_preserving_bi_evidence_v4_24_checks'


def _verify(mapping):
    for relative, digest in mapping.items():
        if C.sha(C.ROOT/relative) != digest:
            raise ValueError('Bound bytes changed: '+relative)


def main(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():raise FileExistsError('Append-only preflight evidence')
    seal = C.read(OBS/'CAL_OBSERVATION_SEAL.json'); protocol = C.read(TASKS/'PROTOCOL.json')
    if seal['rows'] != 46 or C.sha(OBS/'CAL_OBSERVATIONS.jsonl.gz') != seal['observations_sha256']:
        raise ValueError('Sealed CAL46 actual observations required')
    ids = seal['ordered_ids']
    if ids != protocol['ids'][:46]:raise ValueError('Same original CAL46 prefix required')
    source = {str(Path(m.__file__).resolve().relative_to(C.ROOT)): C.sha(m.__file__)
        for m in (V, V.OLD, SERIAL, SERIAL.TYPED, V.BROAD, V.BROAD.OLD, V.N, V.M, V.M.E, V.OBS, V.N.J, V.N.P, V.N.T)}
    source[str(Path(__file__).resolve().relative_to(C.ROOT))] = C.sha(__file__)
    inputs = {str(p.relative_to(C.ROOT)): C.sha(p)for p in
        (OBS/'CAL_OBSERVATION_SEAL.json', OBS/'CAL_OBSERVATIONS.jsonl.gz', OBS/'PROTOCOL.json',
         TASKS/'PROTOCOL.json', TASKS/'INPUT_SEAL.json', TASKS/'INPUTS.jsonl.gz')}
    _verify(source); _verify(inputs); _verify(protocol['protected_user_sha256'])
    output.mkdir(parents=True)
    C.write(output/'PREPARATION_SEAL.json', dict(schema='CAL46_COMPLETE_BI_ZERO_NOISE_METADATA_PREFLIGHT_V24_SOURCE_PRESERVING',
        ids=ids, source_sha256=source, input_sha256=inputs,
        protected_user_sha256=protocol['protected_user_sha256'], before_first_CAL_observation_decode=True,
        CAL_reference_labels_used=False, TEST_and_REMAINDER_rows_decoded=False))
    for relative, digest in source.items():
        dest = output/'code_snapshot'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative, dest)
        if C.sha(dest) != digest:raise ValueError('Snapshot mismatch')
    observations = list(C.rows(OBS/'CAL_OBSERVATIONS.jsonl.gz'))
    if [r['id']for r in observations] != ids:raise ValueError('CAL observation IDs changed')
    records = []; errors = []; prepared = []; modes = Counter(); reasons = Counter(); sources = Counter(); totals = Counter()
    for i, task in enumerate(islice(C.rows(TASKS/'INPUTS.jsonl.gz'), 46)):
        row = observations[i]
        if task['id'] != ids[i] or task['partition'] != 'CAL' or row['partition'] != 'CAL':
            raise ValueError('Never decode TEST/remaining observations for this preflight')
        spec = task['spec']; family = task['families']['ORIGINAL_WD']; geometry = row['initial_geometry']
        if (row['states'] != spec['states']or row['visible_measurements'] != spec['visible']
                or row['domains'] != spec['domains']or row['u_regions'] != task['u_regions']
                or row['image_sha256'] != task['image_sha256']
                or geometry['initial_self_hidden'] != task['initial_self_hidden']
                or geometry['native_xyz'] != family['native_xyz']
                or geometry['physical_registration_Q'] != family['physical_registration_Q']
                or geometry['K'] != spec['K']or geometry['image_hw'] != spec['image_hw']):
            raise ValueError('Same frozen state/visible/domain/image/S metadata required')
        cfg = deepcopy(spec['config']); cfg['prior_weight'] = 0.
        try:
            base = V.N.prepare_native_edge_problem(family['native_xyz'], spec['K'], spec['states'],
                spec['visible'], spec['domains'], spec['image_hw'], family['coarse_pose'],
                config=cfg, boundary_observations=task['boundary_merged'], u_regions=task['u_regions'],
                edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
            bundle = V.authenticate_complete_bi_image_chain_groups(base, row['chain_observation'],
                junction_proposals=row['junction_proposals'], visible_measurements=spec['visible'],
                initial_self_hidden=task['initial_self_hidden'],
                image_binding=dict(sha256=task['image_sha256'], image_hw=spec['image_hw']),
                initial_projection_xy=geometry['initial_projection_xy'],
                normal_variance_component_px2=0., angular_variance_component_sine2=0.,
                noise_provenance=dict(fit_partition='CAL', observation_family=V.NOISE_OBSERVATION_FAMILY,
                    mode='ZERO_COMPONENT_PREPARATION_NOT_FITTED_NOISE'), calibration_preparation=True)
            plan = V.plan_complete_bi_replacement(base, bundle, overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)
            local = Counter()
            for p in plan['corner_plans']:
                mode = 'FULL_Q_RETAINED'if not p['point_cost_revoked']else 'ARCHIVED_COMPLETE_TANGENT'if 'tangent_kind'in p else 'FINITE_TYPED_LINES_ONLY'
                local[mode] += 1
                sources[p['original_measurement_archive']['original_measurement_source']] += 1
            assert sum(local.values()) == len(base.complete)
            assert len(plan['active_groups']) == len({tuple(g['physical_native_edge'])for g in plan['active_groups']})
            assert plan['ledger']['original_complete_q_factors_once'] == local['FULL_Q_RETAINED']
            assert plan['ledger']['archived_complete_tangent_factors_once'] == local['ARCHIVED_COMPLETE_TANGENT']
            for r in plan['main_replacement_rejections']:reasons[r['reason']] += 1
            for g in bundle['groups']:
                assert abs(sum(g['component_weights'])-1.) < 1e-12
                for support in g['support_prior_ledger']:
                    assert abs(sum(c['mixture_component_prior'] for c in g['components'] if c['support_id']==support['support_id'])-support['total_prior']) < 1e-12
            prepared.append(dict(id=task['id'], partition='CAL', session=task['session'],
                initial_geometry=geometry,
                authenticated_group_bundle={**bundle, 'groups':plan['active_groups']},
                authentication_inventory_groups=bundle['groups'], replacement_plan=plan,
                fit_population='ONLY exact-source-preserving active_groups; unchanged authentication inventory kept separately',
                preparation_only_no_reference_decode=True))
            modes.update(local); totals.update(complete=len(base.complete), groups=len(bundle['groups']),
                alternatives=sum(len(g['components'])for g in bundle['groups']),
                active_groups=len(plan['active_groups']),
                active_alternatives=sum(len(g['components'])for g in plan['active_groups']),
                distinct_supports=sum(len(g['support_prior_ledger'])for g in bundle['groups']),
                active_distinct_supports=sum(len(g['support_prior_ledger'])for g in plan['active_groups']),
                multiple_variant_supports=sum(x['fit_variants']>1 for g in bundle['groups']for x in g['support_prior_ledger']),
                relation_checks=len(bundle['actual_lattice_gradient_subpixel_relation']))
            records.append(dict(id=task['id'], session=task['session'], complete_modes=dict(local),
                original_complete_count=len(base.complete), authenticated_groups=len(bundle['groups']),
                active_groups=len(plan['active_groups']), ledger=plan['ledger'],
                relation_max_error_px=max((r['maximum_relation_error_px']for r in bundle['actual_lattice_gradient_subpixel_relation']), default=0.),
                corner_plans=plan['corner_plans'], rejections=plan['main_replacement_rejections'],
                preparation_bundle_not_used_for_pose_objective=True))
        except Exception as error:
            errors.append(dict(id=task['id'], exception=type(error).__name__, reason=str(error)))
    if len(records)+len(errors) != 46:raise ValueError('Every CAL row must be accounted for')
    _verify(source); _verify(inputs); _verify(protocol['protected_user_sha256'])
    C.save_rows(output/'CAL_PREPARED_GROUPS.jsonl.gz', prepared)
    report = dict(schema='CAL46_ZERO_NOISE_COMPLETE_BI_METADATA_PREFLIGHT_V24_SOURCE_PRESERVING', rows=46,
        successful_metadata_rows=len(records), errors=errors, error_count=len(errors), totals=dict(totals),
        complete_modes=dict(modes), complete_measurement_sources=dict(sources), rejection_reasons=dict(reasons),
        maximum_lattice_relation_error_px=max((r['relation_max_error_px']for r in records), default=0.),
        records=records, CAL46_inventory_not_CAL44_fitted_model=True, new_CAL_fit=False,
        prepared_groups_sha256=C.sha(output/'CAL_PREPARED_GROUPS.jsonl.gz'),
        support_fit_variant_ids_and_normalized_priors_sealed_before_any_reference_decode=True,
        CAL_fit_applicability='source-preserving fallback-owner-or-oldpartial active groups only',
        source_policy=V.SOURCE_POLICY, runtime_source_gate_has_no_GT_ID_or_error_input=True,
        label_decodes=0, image_queries=0, new_line_fits=0, pose_objective_calls=0, pose_optimizations=0,
        old_sources_or_noise_models_changed=False, no_information_independence_or_coverage_claim=True)
    C.write(output/'CAL_PREFLIGHT_V1.json', report)
    C.write(output/'CAL_PREFLIGHT_SEAL_V1.json', dict(report_sha256=C.sha(output/'CAL_PREFLIGHT_V1.json'),
        preparation_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        prepared_groups_sha256=C.sha(output/'CAL_PREPARED_GROUPS.jsonl.gz'), source_sha256=source,
        input_sha256=inputs, protected_user_sha256=protocol['protected_user_sha256']))
    print(dict(rows=46, success=len(records), errors=len(errors), totals=dict(totals), modes=dict(modes),
        error_inventory=errors), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    main(parser.parse_args().output)
