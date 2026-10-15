"""One-shot fixed-three-CAL V31 metadata smoke, never a new rank or fit.

Declaration hashes opaque raw spools and already sealed V31 graphs. Execution
joins exactly one raw row and one saved selected graph at a time. It rebuilds
exact chain refs and mask-only visibility metadata, then prepares both source
policies with zero prefitted CAL components. No RGB/NMS/TLS/rank/labels/pose
objective/optimization or primary row is invoked or decoded.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import numpy as np

from . import prepare_geometry_guided_native_evidence_v4_31 as M
from . import run_observed_anchor_native_ray_CAL_v4_31 as SELECTED
from . import rerank_observed_bi_edge_population_v4_30 as REFS

C, B, G, AUTH = M.C, M.B, M.G, M.AUTH
CASES = SELECTED.CASES
PLAN = B / 'geometry_guided_native_metadata_smoke_plan_v4_31/PREDECLARATION.json'
OUTPUT = B / 'geometry_guided_native_metadata_smoke_v4_31'


def _union(target, values):
    for path, digest in values.items():
        if path in target and target[path] != digest:
            raise ValueError('Immutable ancestry conflict: ' + path)
        target[path] = digest


def _sources():
    modules = (M, M.PARENT, AUTH, AUTH.PARENT, G, G.PARENT, G.V30, G.BASELINE,
        G.N, G.V, G.J, G.P, G.EDGE, G.ASSOCIATION, G.ASSOCIATION.F,
        SELECTED, SELECTED.F.P, SELECTED.F.P.VIS, SELECTED.F.G, REFS, C, M.PREFIX)
    paths = [Path(module.__file__).resolve() for module in modules]
    paths.extend((Path(__file__).resolve(), Path(__file__).with_name('test_smoke_geometry_guided_native_metadata_v4_31.py').resolve(),
                  Path(M.__file__).with_name('test_prepare_geometry_guided_native_evidence_v4_31.py').resolve()))
    return {str(path.relative_to(C.ROOT)): C.sha(path) for path in paths}


def declare():
    if PLAN.exists():
        raise FileExistsError('Preserve every fixed-three-CAL metadata declaration')
    previous = C.read(SELECTED.PLAN)
    previous_seal = C.read(SELECTED.PLAN.parent / 'PLAN_SEAL.json')
    selected_seal = C.read(SELECTED.OUTPUT / 'RERANK_SEAL.json')
    if (C.sha(SELECTED.PLAN) != previous_seal['plan_sha256']
            or selected_seal['plan_sha256'] != C.sha(SELECTED.PLAN)
            or selected_seal['selection_profile'] != G.SELECTION_PROFILE):
        raise ValueError('Exact sealed V31 fixed-three graph population required')
    SELECTED._fixed_cases(previous['cases'])
    task_protocol = C.read(M.TASKS / 'PROTOCOL.json')
    task_seal = C.read(M.TASKS / 'INPUT_SEAL.json')
    if C.sha(M.TASKS / 'INPUTS.jsonl.gz') != task_seal['input_sha256']:
        raise ValueError('Exact immutable physics input bytes required')
    sources = dict(previous['source_sha256'])
    _union(sources, _sources())
    inputs = dict(previous['input_sha256'])
    for path in (SELECTED.PLAN, SELECTED.PLAN.parent / 'PLAN_SEAL.json',
                 SELECTED.OUTPUT / 'RERANK_SEAL.json', M.TASKS / 'PROTOCOL.json',
                 M.TASKS / 'INPUT_SEAL.json', M.TASKS / 'INPUTS.jsonl.gz'):
        _union(inputs, {str(path.relative_to(C.ROOT)): C.sha(path)})
    cases = []
    for number, case in enumerate(previous['cases']):
        path = SELECTED.OUTPUT / 'graphs' / ('case' + str(number) + '.jsonl.gz')
        digest = selected_seal['assets'][str(path.relative_to(SELECTED.OUTPUT))]
        if C.sha(path) != digest:
            raise ValueError('Exact saved V31 selected graph bytes required')
        _union(inputs, {str(path.relative_to(C.ROOT)): digest})
        cases.append({**case, 'selected_graph_path': str(path.relative_to(C.ROOT)),
                      'selected_graph_sha256': digest})
    protected = dict(previous['protected_user_sha256'])
    _union(protected, task_protocol['protected_user_sha256'])
    for mapping in (sources, inputs, protected):
        M.PREFIX._verify(mapping)
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN, dict(schema='FIXED_THREE_CAL_V31_ZERO_COMPONENT_METADATA_SMOKE_PLAN',
        UTC=datetime.now(timezone.utc).isoformat(), cases=cases,
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        selection_profile=G.SELECTION_PROFILE, observation_family=G.OBSERVATION_FAMILY,
        noise_population=G.POPULATION, raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
        task_CAL_prefix_ids=task_protocol['ids'][:46],
        source_policies=list(G.SOURCE_POLICIES),
        zero_prefit_components_only=True, fitted_model_or_noise_validity_claim=False,
        actual_cached_row_mask_or_graph_decode_before_declaration=0,
        actual_RGB_NMS_rank_TLS_labels_posefit_primary_or_metrics=0,
        memory=dict(workers=1,live_large_raw_rows=1,live_saved_graphs=1,
                    bounded_small_prepared_outputs=3,engineering_not_peak_RAM_guarantee=True),
        no_new_reference_or_case_selection=True,
        separate_actual_GO_and_G31_A31_M31_essential_PASS_required=True))
    for relative, digest in sources.items():
        target = PLAN.parent / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Exact source snapshot required')
    C.write(PLAN.parent / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN),
        source_sha256=sources, input_sha256=inputs, actual_row_mask_graph_or_label_decode=0))
    return dict(plan_sha256=C.sha(PLAN), source_count=len(sources), input_count=len(inputs))


def compose_selected_row(row, result, visibility):
    """Build a new metadata view; raw source/ref/graph coefficient arrays stay put."""
    if (row['partition'] != 'CAL' or row['id'] not in CASES
            or result.get('selection_profile') != G.SELECTION_PROFILE):
        raise ValueError('Only fixed CAL3 and exact V31 saved graph profile')
    old = row['observation']
    if old['observation_generation'] != G.RAW_OBSERVATION_GENERATION:
        raise ValueError('Original raw V29 measurement generation required')
    graph = result['graph']
    refs = REFS.rebuild_selected_references(graph, old['chain_observation'],
        row['states'], row['initial_geometry']['initial_self_hidden'])
    observation = {**old, 'joint_graph': graph, 'selected_chain_refs': refs,
        'edge_visibility': visibility, 'selection_profile': G.SELECTION_PROFILE,
        'raw_observation_generation': G.RAW_OBSERVATION_GENERATION,
        'association_recipe': deepcopy(G.ASSOCIATION.RECIPE),
        'association_configuration': deepcopy(old['chain_observation']['configuration']),
        'association_ledger': result['node_association_ledger'],
        'parent_legacy_selected_physical_edges': old.get('selected_physical_edges'),
        'selected_physical_edges': len(refs),
        'old_V29_or_V30_noise_validated_for_this_selection_population': False}
    return {**row, 'selection_profile': G.SELECTION_PROFILE, 'observation': observation,
        'summary': {**row.get('summary', {}), 'selected_physical_edges': len(refs),
                    'zero_selected_edges': not refs}}


def run(*, root_CAL_metadata_smoke_GO=False):
    if root_CAL_metadata_smoke_GO is not True:
        raise PermissionError('Root fixed CAL3 metadata GO before any IO')
    if OUTPUT.exists():
        raise FileExistsError('Preserve every metadata smoke attempt')
    plan = C.read(PLAN)
    seal = C.read(PLAN.parent / 'PLAN_SEAL.json')
    if C.sha(PLAN) != seal['plan_sha256'] or tuple(c['id'] for c in plan['cases']) != CASES:
        raise ValueError('Exact predeclared fixed-three CAL source plan required')
    for mapping in (plan['source_sha256'], plan['input_sha256'], plan['protected_user_sha256']):
        M.PREFIX._verify(mapping)
    OUTPUT.mkdir(parents=True)
    records, bindings = [], []
    tasks = iter(M.PREFIX._cal_task_prefix(M.TASKS / 'INPUTS.jsonl.gz', plan['task_CAL_prefix_ids']))
    pending = {case['id']: case for case in plan['cases']}
    prepared_paths = []
    try:
        for task in tasks:
            if task['id'] not in pending:
                continue
            case = pending.pop(task['id'])
            row = SELECTED.OBS._read_single(C.ROOT / case['spool_relative'], case)
            saved = iter(C.rows(C.ROOT / case['selected_graph_path']))
            result = next(saved, None)
            if result is None or next(saved, None) is not None:
                raise ValueError('One exact sealed V31 saved graph required')
            before = SELECTED.R._canonical_sha(SELECTED._immutable_observation_view(row))
            graph_before = SELECTED.R._canonical_sha(result)
            maskpath = C.ROOT / row['mask_binding']['path']
            if C.sha(maskpath) != row['mask_binding']['sha256']:
                raise ValueError('Original cached target mask bytes required')
            with np.load(maskpath, allow_pickle=False) as data:
                mask = data['pallet_mask_clean'].copy()
            if SELECTED.F.P.VIS.mask_pixel_sha256(mask) != row['observation']['target_mask_pixel_sha256']:
                raise ValueError('Original target mask pixels required')
            context = SELECTED.F.P.VIS.prepare_edge_visibility(mask, row['initial_geometry'],
                row['states'], row['visible_measurements'],
                image_binding=dict(sha256=row['image_sha256'], image_hw=row['initial_geometry']['image_hw']),
                occluder_mask=None, occluder_binding=None)
            selected = compose_selected_row(row, result, [])
            visibility = []
            for ref in selected['observation']['selected_chain_refs']:
                ray = row['observation']['chain_observation']['edge_records'][ref['ray_index']]
                chain = ray['chains'][ref['chain_index']]
                actual = np.asarray(chain['observed_points'])[np.asarray(chain['inlier_mask'], bool)]
                visibility.append(context.classify_expected_edge(ref['native_corner'], ref['native_neighbor'],
                    observed_chain_points=actual, mask_context_kind=ray.get('mask_context_kind')))
            selected['observation']['edge_visibility'] = visibility
            prepared, metadata = M.prepare_one(task, selected)
            if (SELECTED.R._canonical_sha(SELECTED._immutable_observation_view(row)) != before
                    or SELECTED.R._canonical_sha(result) != graph_before):
                raise ValueError('Original raw q/state/guide/chain or stored V31 graph mutated')
            if C.sha(maskpath) != row['mask_binding']['sha256']:
                raise ValueError('Original target mask changed during smoke')
            destination = OUTPUT / ('prepared_case_' + str(CASES.index(task['id'])) + '.jsonl.gz')
            C.save_rows(destination, [prepared]); prepared_paths.append(destination)
            records.append(dict(**metadata, original_spool_sha256=case['sha256'],
                selected_graph_sha256=case['selected_graph_sha256'], selected_line_count=len(visibility),
                original_raw_q_state_guide_chain_canonical_sha256=before,
                saved_graph_canonical_sha256=graph_before, before_after_immutable=True))
            bindings.append(dict(path=str(maskpath.relative_to(C.ROOT)), sha256=row['mask_binding']['sha256']))
            print(dict(id=task['id'], profiles=metadata['profiles'], metadata_only=True), flush=True)
            del row, result, mask, context, selected, prepared
            if not pending:
                break
        if pending or {record['id'] for record in records} != set(CASES):
            raise ValueError('All fixed CAL3 including zero-selection results required')
    except BaseException as error:
        C.write(OUTPUT / 'FAILURE_RECEIPT.json', dict(exception=type(error).__name__, reason=str(error),
            completed_ids=[record['id'] for record in records], plan_sha256=C.sha(PLAN),
            source_sha256=plan['source_sha256'], actual_rank_RGB_labels_posefit=0))
        raise
    records.sort(key=lambda record: CASES.index(record['id']))
    C.write(OUTPUT / 'RESULT.json', dict(rows=3, records=records, source_policies=list(G.SOURCE_POLICIES),
        selection_profile=G.SELECTION_PROFILE, observation_family=G.OBSERVATION_FAMILY,
        noise_population=G.POPULATION, raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
        CAL_numeric_labels_references_or_primary_rows=0, actual_rank_RGB_NMS_TLS_posefit=0,
        zero_prefit_components_only=True, fitted_noise_or_accuracy_validation_claim=False))
    for mapping in (plan['source_sha256'], plan['input_sha256'], plan['protected_user_sha256']):
        M.PREFIX._verify(mapping)
    C.write(OUTPUT / 'SMOKE_SEAL.json', dict(plan_sha256=C.sha(PLAN), rows=3,
        result_sha256=C.sha(OUTPUT / 'RESULT.json'), source_sha256=plan['source_sha256'],
        input_sha256=plan['input_sha256'], mask_bindings=bindings,
        assets={str(path.relative_to(OUTPUT)): C.sha(path) for path in [OUTPUT/'RESULT.json', *prepared_paths]},
        protected_user_sha256=plan['protected_user_sha256'], actual_labels_rank_RGB_posefit=0))
    return dict(rows=3, smoke_seal_sha256=C.sha(OUTPUT / 'SMOKE_SEAL.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('declare', 'run'))
    parser.add_argument('--root-CAL-metadata-smoke-GO', action='store_true')
    args = parser.parse_args()
    print(declare() if args.stage == 'declare' else run(root_CAL_metadata_smoke_GO=args.root_CAL_metadata_smoke_GO))
