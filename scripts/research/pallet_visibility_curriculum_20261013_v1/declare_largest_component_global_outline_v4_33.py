"""Metadata-only declaration of one fixed LCC8 global-outline control.

No mask, positive packet, prediction, score, reference or model values are
decoded here. The frozen R2 metadata anchors the original union control and
all 238 source-packet identities. Final operator/controller bytes and their
independent reviews are supplied by the caller, after they are frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
from pathlib import Path

from . import common as C
from . import largest_component_global_outline_v4_33 as LCC


D = C.ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1'
B = D / 'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
S = Path(__file__).resolve().parent
OUTPUT = B / 'largest_component_global_outline_method_plan_v4_33'
PLAN = OUTPUT / 'PREDECLARATION.json'
PARENT = B / 'clean_fresh_BI_pose_eval_v4_32_r2'
CENTRAL32 = B / 'initial_only_observed_ray_method_plan_v4_32/PREDECLARATION.json'
MODEL32 = B / 'CAL_geometry_guided_edge_noise_v4_32/MODEL_SEAL.json'
OBS32 = B / 'initial_only_observed_ray_observations_v4_32/OBSERVATION_SEAL.json'
OPERATOR_NAME = 'observe_largest_component_global_outline_CAL_v4_33'
CONTROLLER_NAME = 'evaluate_largest_component_global_outline_poses_v4_33'
PARTITIONS = dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192, severe=0)
PARENT_ARMS = ('SAME_FROZEN_E', 'CLEAN_FRESH_BI_ALL_BI_NATIVE',
    'CLEAN_FRESH_BI_ALL_BI_CONTAINMENT', 'CLEAN_FRESH_BI_SOURCE_NATIVE',
    'CLEAN_FRESH_BI_SOURCE_CONTAINMENT')
PINNED = {
    CENTRAL32: 'c95598253d42f06ace0083d30b5fb46adbb64191fa59ee5b76ce348ef2be10ef',
    PARENT / 'PROTOCOL.json': 'fc60bb62ff728b1642c94c8f8c891a7ebbefd9e221be0f212960b2420dfc5df7',
    PARENT / 'INPUT_SEAL.json': 'ccd255dbfb8ce8d359f16ed487b7dbbb026963b52f0e93393b7955f674564447',
    PARENT / 'PREDICTION_SEAL.json': 'eff80efdb1cb9c17f7eb13df50b953467f82bbe3415c8385b9aa146456948b48',
    PARENT / 'SCORE_SEAL.json': '7bcc9bce3e8b14a87021955c513e3061dcb561cc309fabd61964e06efbcc6c80',
    MODEL32: '4b7d20df18f62269360a873e530ee29f7535c9d3d4ec4c6b6f72e49b7be48ab7',
    OBS32: '9d6b68bc50137fdc6356b70091aa81c783756a3ffec1f39408480812e186e280',
}
LCC_SHA256 = '112e9035d522c4487a816e3612218d44a5fe53470c75a9da4371ec700a276c46'
FIXED_REVIEWS = (
    (B / 'largest_component_global_outline_v4_33_checks/ROOT_SOURCE_REVIEW_V1.json',
     '3385d557b0746679ffcef21e745eba69429ba6c5bb769645a81750796de67e3e'),
    (B / 'largest_component_global_outline_peer_review_v4_33/SOURCE_REVIEW_V1.json',
     '1ab74c614354b1e7d7dd37d1d8503422ec5aa61ad13ab3253124e4dc1a7bf0c1'),
)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _relative(path):
    return str(Path(path).resolve().relative_to(C.ROOT))


def _binding(path):
    return dict(path=_relative(path), sha256=C.sha(path))


def _union(*maps):
    result = {}
    for mapping in maps:
        if not isinstance(mapping, dict):
            raise ValueError('Explicit path-to-SHA map required')
        for path, digest in mapping.items():
            if not isinstance(path, str) or not _sha(digest):
                raise ValueError('Exact path-to-SHA binding required')
            if path in result and result[path] != digest:
                raise ValueError('Conflicting immutable binding: ' + path)
            result[path] = digest
    return result


def _verify(mapping):
    for relative, digest in _union(mapping).items():
        path = C.ROOT / relative
        if not path.resolve().is_relative_to(C.ROOT) or C.sha(path) != digest:
            raise ValueError('Frozen bytes changed: ' + relative)


def _validate_binding(binding):
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
            or not isinstance(binding['path'], str) or not _sha(binding['sha256'])
            or Path(binding['path']).is_absolute() or '..' in Path(binding['path']).parts):
        raise ValueError('Explicit repository-relative final path/SHA required before IO')
    return binding


def _source_bindings(value):
    result = {}
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and _sha(value.get('sha256')):
            result = _union(result, {value['path']: value['sha256']})
        for key, child in value.items():
            if key == 'source_sha256' and isinstance(child, dict):
                result = _union(result, child)
            result = _union(result, _source_bindings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result = _union(result, _source_bindings(child))
    return result


def _peer(binding, source_paths):
    _validate_binding(binding)
    path = C.ROOT / binding['path']
    if C.sha(path) != binding['sha256']:
        raise ValueError('Peer SHA must match before metadata decode')
    review = C.read(path)
    verdict = next((review[k] for k in ('status', 'verdict', 'essential_review',
        'essential_source_review') if k in review), None)
    if (not isinstance(verdict, str) or not (verdict.startswith('PASS') or verdict.startswith('ESSENTIAL_PASS'))
            or any(bool(review.get(k, 0)) for k in ('blockers', 'essential_blockers',
                'essential_blocker_count', 'failure_count', 'failures', 'material_findings'))):
        raise ValueError('Final independent essential PASS without blockers required')
    sources = _source_bindings(review)
    for source in source_paths:
        if sources.get(_relative(source)) != C.sha(source):
            raise ValueError('Peer must bind exact final critical source: ' + Path(source).name)
    return path


def _components(operator_binding, operator_peer_binding, controller_binding, controller_peer_binding):
    sources, peers = {}, []
    modules = []
    for name, binding, review in ((OPERATOR_NAME, operator_binding, operator_peer_binding),
            (CONTROLLER_NAME, controller_binding, controller_peer_binding)):
        expected = S / (name + '.py')
        if binding['path'] != _relative(expected) or C.sha(expected) != binding['sha256']:
            raise ValueError('Final component exact module path/SHA required before import')
        test = S / ('test_' + name + '.py')
        peer = _peer(review, (expected, test))
        module = importlib.import_module(__package__ + '.' + name)
        if Path(module.__file__).resolve() != expected or C.sha(module.__file__) != binding['sha256']:
            raise ValueError('Imported component is not the reviewed frozen source')
        modules.append(module)
        peers.append(peer)
        sources = _union(sources, {_relative(p): C.sha(p) for p in (expected, test)})
    operator, controller = modules
    if operator.SELECTION_PROFILE != LCC.SELECTION_PROFILE or controller.SELECTION_PROFILE != LCC.SELECTION_PROFILE:
        raise ValueError('One uniform LCC8 selection profile required')
    arms = tuple(controller.ARMS)
    if len(arms) != 5 or len(set(arms)) != 5 or arms[0] != 'SAME_FROZEN_E':
        raise ValueError('Comparison e plus four uniform new arms required')
    return arms, sources, peers


def _validate_parent_metadata(protocol, input_seal, prediction_seal, observation_seal, central):
    ids = protocol['ordered_ids']
    records = protocol['records']
    if (len(ids) != 238 or len(set(ids)) != 238 or len(records) != 238
            or [record['id'] for record in records] != ids
            or set(protocol['partitions']) != {'CAL', 'TEST', 'REMAINDER'}
            or protocol['partitions'] != dict(CAL=46, TEST=49, REMAINDER=143)
            or Counter(record['partition'] for record in records) != Counter(CAL=46, TEST=49, REMAINDER=143)
            or [record['partition'] for record in records] != ['CAL']*46+['TEST']*49+['REMAINDER']*143
            or any(set(record) != {'id', 'partition', 'ready', 'positive_packet_sha256'}
                or type(record['ready']) is not bool or not _sha(record['positive_packet_sha256']) for record in records)):
        raise ValueError('Fixed ordered238/CAL46/TEST49/REMAINDER143 records required, including unavailable/zero rows')
    if (tuple(protocol['arms']) != PARENT_ARMS
            or central['ordered_ids'] != ids or central['partitions'] != PARTITIONS
            or len(protocol['protected_user_sha256']) != 6
            or len(protocol['original_CAL_file_sha256']) != 53
            or protocol['parent_e_comparison_separate_from_numeric_inputs'] is not True
            or protocol['old_pose_bank_start_or_fallback_inputs'] != 0):
        raise ValueError('Frozen clean R2 population/protection/CAL53 and e separation required')
    for seal in (input_seal, prediction_seal):
        if (seal['rows'] != 238 or seal['ordered_ids'] != ids
                or tuple(seal['arms']) != PARENT_ARMS or seal['arm_count'] != 5
                or seal['protocol_sha256'] != C.sha(PARENT / 'PROTOCOL.json')):
            raise ValueError('Parent positive/prediction all238x5 seal mismatch')
    if (prediction_seal['input_seal_sha256'] != C.sha(PARENT / 'INPUT_SEAL.json')
            or prediction_seal['all_declared_arms_complete'] is not True
            or observation_seal['rows'] != 238 or observation_seal['ordered_ids'] != ids
            or protocol['central_plan_sha256'] != PINNED[CENTRAL32]
            or protocol['new_model_seal_sha256'] != PINNED[MODEL32]):
        raise ValueError('Exact central32/model32/OBS32/positive parent ancestry required')
    return ids, records


def _parent_metadata():
    # Read only pinned metadata. Inherited input-map targets (including fixed
    # references and raw journals) are NOT opened by this declaration helper.
    for path, digest in PINNED.items():
        if C.sha(path) != digest:
            raise ValueError('Frozen parent metadata pin changed: ' + path.name)
    protocol = C.read(PARENT / 'PROTOCOL.json')
    input_seal = C.read(PARENT / 'INPUT_SEAL.json')
    prediction_seal = C.read(PARENT / 'PREDICTION_SEAL.json')
    observation_seal, central = C.read(OBS32), C.read(CENTRAL32)
    ids, records = _validate_parent_metadata(protocol, input_seal, prediction_seal, observation_seal, central)
    payloads = _parent_payload_bindings(protocol, input_seal, prediction_seal)
    # Opaque byte hashes only; neither journal is decoded or exposed to the
    # declaration's numerical path (which does not exist).
    _verify(payloads)
    return protocol, ids, records, payloads


def _parent_payload_bindings(protocol, input_seal, prediction_seal):
    positive, comparison = input_seal.get('input_sha256'), input_seal.get('e_comparison_sha256')
    if (not _sha(positive) or not _sha(comparison)
            or comparison != protocol.get('e_comparison_sha256')
            or comparison != prediction_seal.get('separate_e_comparison_sha256')):
        raise ValueError('Exact independently sealed positive INPUTS and parent-only E comparison required')
    return {_relative(PARENT / 'INPUTS.jsonl.gz'): positive,
        _relative(PARENT / 'E_COMPARISON.jsonl.gz'): comparison}


def _make_plan(protocol, ids, records, arms, sources, inputs, operator_binding,
        operator_peer_binding, controller_binding, controller_peer_binding, parent_payload_sha256=None):
    return dict(schema='UNIFORM_LCC8_OUTLINE_ONLY_METHOD_PLAN_V33',
        UTC=datetime.now(timezone.utc).isoformat(), selection_profile=LCC.SELECTION_PROFILE,
        original_union_profile=LCC.BASELINE_PROFILE, arms=list(arms), original_union_arms=list(PARENT_ARMS),
        ordered_ids=ids, partitions=PARTITIONS, mask_source_packet_records=records,
        mask_source_record_scope='Opaque R2 id/partition/positive-packet SHA only; operator joins target_mask_binding and original union exactness, not decoded here.',
        source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=protocol['protected_user_sha256'],
        original_CAL_file_sha256=protocol['original_CAL_file_sha256'],
        CAL_immutable_file_sha256=protocol['original_CAL_file_sha256'],
        parent_metadata_pins={_relative(p): h for p, h in PINNED.items()},
        opaque_parent_payload_sha256=parent_payload_sha256,
        operator_source_binding=operator_binding, operator_peer_binding=operator_peer_binding,
        controller_source_binding=controller_binding, controller_peer_binding=controller_peer_binding,
        recipe=LCC.RECIPE, changed_numerical_input='regularizers.outline_observation ONLY: original union -> largest8-connected component then frozenV13 sampler',
        unchanged=['native point/line population32', 'line/tangent CAL32 noise and coefficients',
            'fresh BI SQPNP and six-diverse plus own-coarse start policy',
            'bounded O/S priors', 'C25 numerical function/weight1', 'all original hardguards/readouts'],
        new_noise_fit_or_distillation=False, numeric_C_or_noise_or_line_population_or_guard_changes=False,
        original_union_control='Frozen V32 R2 all238x5 and scores retained unchanged; no per-row choice between union and LCC.',
        native_arms_outline_independent_exact_parity_required=True,
        S_image_raw_2D_or_region_queries=0, S_model_projection_and_existing_bounded_prior_only=True,
        U_precise_point_or_local_queries=0, U_approximate_regions_only=True,
        e_comparison_parent_only=True, numeric_e_prior_seed_start_candidate_familylock_or_fallback=0,
        GT_error_frameID_partition_or_material_routing=False,
        empty_mask_or_unsupported_or_invalid_coarse='Retain every row and original fallback/no-pose semantics; no other-method fallback.',
        CAL_inventory=46, compatible_CAL_inventory=44, CAL_not_combined_with_primary=True,
        CAL_first='Seal all46 mask observations/union exactness before REST192; no threshold/coefficient search from CAL or primary.',
        primary_inventory=192, original_TEST_inventory=49, severe_inventory=0,
        user_goal='All192 translation<=5cm and original49 preserved; joint5cm/5deg also reported on all rows.',
        component_limitations='Largest pixel component need not be pallet; disconnected true fragments may be discarded. No semantic occluder or visibility proof.',
        mask_provider_or_NN_or_local_query_changes=0,
        repeated_CAL_TEST_research_not_new_holdout=True, default='SAME_FROZEN_E; no adoption declared',
        inherited_input_map_target_bytes_rehashed_here=False,
        inherited_input_map_scope='Carried by exact frozen R2 PROTOCOL SHA; no reference/raw-journal/packet numeric decode here. Operational stages must verify their actual supplied inputs.',
        current_source_protected_and_CAL53_bytes_verified=True,
        actual_mask_packet_prediction_score_reference_or_model_value_decodes_in_declare=0,
        actual_PnP_optimizer_new_RGB_or_NN_calls_in_declare=0)


def declare(*, root_metadata_GO=False, operator_binding=None, operator_peer_binding=None,
        controller_binding=None, controller_peer_binding=None):
    if root_metadata_GO is not True:
        raise PermissionError('Explicit ROOT metadata declaration GO required before any IO')
    for binding in (operator_binding, operator_peer_binding, controller_binding, controller_peer_binding):
        _validate_binding(binding)
    if OUTPUT.exists():
        raise FileExistsError('Preserve every V33 declaration attempt')
    arms, new_sources, peers = _components(operator_binding, operator_peer_binding,
        controller_binding, controller_peer_binding)
    if C.sha(LCC.__file__) != LCC_SHA256:
        raise ValueError('Exact reviewed LCC8 helper required')
    for path, digest in FIXED_REVIEWS:
        peers.append(_peer(dict(path=_relative(path), sha256=digest), (Path(LCC.__file__),)))
    protocol, ids, records, payloads = _parent_metadata()
    own_paths = (Path(__file__).resolve(), Path(__file__).with_name('test_declare_largest_component_global_outline_v4_33.py'),
        Path(C.__file__).resolve(), Path(LCC.__file__).resolve(),
        Path(LCC.__file__).with_name('test_largest_component_global_outline_v4_33.py'))
    sources = _union(protocol['source_sha256'], new_sources,
        {_relative(p): C.sha(p) for p in own_paths})
    inputs = _union(protocol['input_sha256'], payloads,
        {_relative(p): C.sha(p) for p in (*PINNED, *peers)})
    protected = protocol['protected_user_sha256']
    for mapping in (sources, protected, protocol['original_CAL_file_sha256']):
        _verify(mapping)
    plan = _make_plan(protocol, ids, records, arms, sources, inputs, operator_binding,
        operator_peer_binding, controller_binding, controller_peer_binding, payloads)
    OUTPUT.mkdir(parents=True)
    C.write(PLAN, plan)
    for relative, digest in sources.items():
        target = OUTPUT / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Exact source snapshot required')
    for mapping in (sources, protected, protocol['original_CAL_file_sha256']):
        _verify(mapping)
    C.write(OUTPUT / 'PLAN_SEAL.json', dict(schema='SEALED_UNIFORM_LCC8_METHOD_PLAN_V33',
        plan_sha256=C.sha(PLAN), source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=protected, original_CAL_file_sha256=protocol['original_CAL_file_sha256'],
        ordered_ids=ids, partitions=PARTITIONS, before_actual_new_mask_or_pose_execution=True,
        actual_numeric_decodes=0))
    return dict(plan_sha256=C.sha(PLAN), rows=238, partitions=PARTITIONS, arms=list(arms),
        actual_new_mask_or_pose_execution=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-metadata-GO', action='store_true')
    for name in ('operator', 'operator-peer', 'controller', 'controller-peer'):
        parser.add_argument('--' + name, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    def binding(name):
        return dict(path=getattr(args, name), sha256=getattr(args, name + '_sha256'))
    print(declare(root_metadata_GO=args.root_metadata_GO,
        operator_binding=binding('operator'), operator_peer_binding=binding('operator_peer'),
        controller_binding=binding('controller'), controller_peer_binding=binding('controller_peer')), flush=True)
