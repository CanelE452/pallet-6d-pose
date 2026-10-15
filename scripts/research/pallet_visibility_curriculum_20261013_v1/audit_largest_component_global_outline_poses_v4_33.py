"""Independent selected-pose audit adapter for the LCC8 outline-only control.

Numerical native/guard/O/S/C replay is the exact reviewed A32.audit_row. This
module adds separate original-packet/one-field/outline/native-e parity checks
and V33 metadata gates; it never invokes the controller, planner or optimizer.
Pixel-component replay belongs to the separately sealed observation audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from . import audit_clean_fresh_BI_poses_v4_32 as A32


C = A32.C
B = A32.B
OUTPUT = B / 'largest_component_global_outline_pose_eval_v4_33'
CENTRAL = B / 'largest_component_global_outline_method_plan_v4_33/PREDECLARATION.json'
CENTRAL_SHA256 = '44e8b7835be40246997d76603e96a8b28ec80826fc16afd3be72d57478d290de'
PARENT = A32.OUTPUT
OUTLINES = B / 'largest_component_global_outline_observations_v4_33'
PROFILE = 'LCC8_MAX_AREA_LOWEST_LABEL_FROZEN_V13_OUTLINE_V33'
BASELINE_PROFILE = 'ORIGINAL_UNION_FROZEN_V13_OUTLINE'
MODEL = A32.MODEL
MODEL_SHA256 = '4b7d20df18f62269360a873e530ee29f7535c9d3d4ec4c6b6f72e49b7be48ab7'
PARENT_PREDICTION_SHA256 = 'eff80efdb1cb9c17f7eb13df50b953467f82bbe3415c8385b9aa146456948b48'
CONTROLLER = Path(__file__).with_name('evaluate_largest_component_global_outline_poses_v4_33.py')
CONTROLLER_SHA256 = 'e997f2d1c33924989d8f5de599943717df9d0fc06001a2e053a1dc44747b83ca'
A32_SHA256 = 'a8ad3335c5750d961c8845769cbaa1cf433d1bc3e3bdaabb7a414155bf7b1203'
WORKER_SHA256 = 'fa9ebe1d327972c585ee5f1f750a50f341151ec421c0e8ad5124a9a2e6b38187'
SOLVER_SHA256 = 'cf1256956ec0da7784c98d0b2a860af4105a1a1ba557b0f5fcef85407f0c3237'
ARMS = A32.ARMS
NATIVE_ARMS = tuple(names[0] for names in A32.POLICY_ARMS.values())
PARTITIONS = A32.PARTITIONS
POPULATION_INVENTORY = dict(PARTITIONS, PRIMARY=192, severe=0)

# No private clone, monkeypatch or new consumer objective is used.
audit_row = A32.audit_row
digest = A32.digest


def _relative(path):
    return str(Path(path).resolve().relative_to(C.ROOT))


def _check_transition(original, changed, outline, record):
    """Independent one-field replacement and source-record check; no fitting."""
    checks = []
    def check(value, reason):
        checks.append(dict(passed=bool(value), reason=reason))
    if set(original) != A32.REPLAY.WORKER_KEYS or set(changed) != A32.REPLAY.WORKER_KEYS:
        raise ValueError('Exact original seven-key positive packets required')
    identifiers = original['identifiers']
    check(identifiers == changed['identifiers'] and outline['id'] == identifiers['id']
          and outline['partition'] == identifiers['partition'], 'same original/current/outline identity')
    check(record['parent_positive_packet_sha256'] == digest(original)
          and record['positive_packet_sha256'] == digest(changed)
          and record['outline_row_sha256'] == digest(outline), 'parent/current/outline exact canonical record hashes')
    check(outline['positive_packet_sha256'] == digest(original)
          and outline['selection_profile'] == PROFILE and outline['baseline_profile'] == BASELINE_PROFILE
          and outline['union_exact_match'] is True, 'uniform saved outline bound to original positive source')
    for key in A32.REPLAY.WORKER_KEYS - {'regularizers'}:
        check(digest(original[key]) == digest(changed[key]), 'unchanged positive field '+key)
    if original['source_contract']['ready'] is False:
        check(digest(original) == digest(changed) and record['changed_field'] is None,
              'invalid coarse packet retained exactly without any regularizer payload')
        return checks
    old, new = original['regularizers'], changed['regularizers']
    check(set(old) == set(new), 'original regularizer keys unchanged')
    for key in set(old) - {'outline_observation'}:
        check(digest(old[key]) == digest(new.get(key)), 'unchanged regularizer '+key)
    check(record['changed_field'] == 'regularizers.outline_observation', 'sole declared replacement path')
    union = {key:value for key,value in outline['original_union'].items() if key != 'selection_profile'}
    current = new['outline_observation']; previous = old['outline_observation']
    check(digest(previous) == digest(union) and digest(current) == digest(outline['largest_component']),
          'original union and new LCC observation exact, no controller replacement helper')
    check(current['selection_profile'] == PROFILE
          and current['image_hw'] == previous['image_hw'] == original['physics']['image_hw']
          and current['image_binding'] == previous['image_binding']
          and current['configuration'] == previous['configuration'] == new['outline_configuration']
          and outline['target_mask_binding']['sha256'] == previous['image_binding']['mask_file_sha256'],
          'same cached mask/image/canvas/configuration, only global observation changed')
    audit = current['foreground_component_selection']
    check(audit['selection_profile'] == PROFILE and audit['connectivity'] == 8 and audit['mask_threshold'] == .5
          and audit['holes_filled'] is False and audit['morphology_used'] is False
          and audit['original_mask_modified'] is False, 'fixed uniform LCC8 rule without new masks')
    return checks


def checked_rows(packets, comparisons, predictions, records, indices, ids,
                 original_packets, outline_rows, parent_predictions, outline_entries):
    """One streaming pass; canonical zero/no-pose rows are never filtered."""
    joined = A32.joined_rows(packets, comparisons, predictions, records, indices, ids)
    count = 0
    for triple, original, outline, parent, indexed in zip_longest(
            joined, original_packets, outline_rows, parent_predictions, outline_entries):
        if any(value is None for value in (triple, original, outline, parent, indexed)):
            raise ValueError('All238 original/current/outline/parent streams required')
        packet, comparison, row = triple
        if (count >= len(ids) or original['identifiers']['id'] != ids[count]
                or outline['id'] != ids[count] or parent['id'] != ids[count] or indexed['id'] != ids[count]
                or indexed['partition'] != row['partition'] or indexed['canonical_row_sha256'] != digest(outline)
                or parent['partition'] != row['partition']):
            raise ValueError('Exact original/LCC/current ordered238 metadata join required')
        extra = _check_transition(original, packet, outline, records[count])
        for arm in (*NATIVE_ARMS, 'SAME_FROZEN_E'):
            extra.append(dict(passed=digest(parent['arms'][arm]) == digest(row['arms'][arm]),
                reason='entire parent native/e arm canonical exact '+arm))
        count += 1
        yield packet, comparison, row, extra
    if count != 238:
        raise ValueError('All238 rows, including no-pose/unsupported, must remain')


def _source_map():
    return A32._verify(A32._source_map(), {_relative(path):C.sha(path) for path in (
        Path(__file__), Path(__file__).with_name('test_audit_largest_component_global_outline_poses_v4_33.py'),
        CONTROLLER)})


def _metadata(output, expected_prediction_seal_sha256, expected_central_plan_sha256,
              expected_model_seal_sha256, expected_outline_seal_sha256):
    if (expected_central_plan_sha256 != CENTRAL_SHA256 or expected_model_seal_sha256 != MODEL_SHA256
            or C.sha(CENTRAL) != CENTRAL_SHA256 or C.sha(MODEL/'MODEL_SEAL.json') != MODEL_SHA256
            or C.sha(output/'PREDICTION_SEAL.json') != expected_prediction_seal_sha256
            or C.sha(OUTLINES/'OUTLINE_OBSERVATION_SEAL.json') != expected_outline_seal_sha256
            or C.sha(CONTROLLER) != CONTROLLER_SHA256 or C.sha(A32.__file__) != A32_SHA256
            or C.sha(A32.HELPER) != A32.HELPER_SHA256):
        raise ValueError('Exact current V33/unchanged V32 source and caller-pinned seals required')
    protocol = C.read(output/'PROTOCOL.json'); inputs = C.read(output/'INPUT_SEAL.json')
    pred = C.read(output/'PREDICTION_SEAL.json'); central = C.read(CENTRAL)
    ids = protocol['ordered_ids']
    if (len(ids) != 238 or len(set(ids)) != 238 or ids != central['ordered_ids']
            or central['partitions'] != POPULATION_INVENTORY or central['selection_profile'] != PROFILE
            or central['arms'] != list(ARMS) or protocol['arms'] != list(ARMS)
            or protocol['schema'] != 'SEALED_OUTLINE_ONLY_POSITIVE_PNP_INPUTS_V33'
            or protocol['partitions'] != PARTITIONS or protocol['population_inventory'] != POPULATION_INVENTORY
            or protocol['central_plan_sha256'] != CENTRAL_SHA256
            or protocol['native_model_seal_sha256'] != MODEL_SHA256
            or protocol['outline_seal_sha256'] != expected_outline_seal_sha256
            or protocol['outline_selection_profile'] != PROFILE
            or protocol['native_observation_family'] != A32.REPLAY.FAMILY
            or protocol['native_selection_profile'] != A32.REPLAY.PROFILE
            or protocol['native_noise_population'] != A32.REPLAY.POPULATION
            or protocol['native_CAL32_model_relabelled'] is not False
            or protocol['only_positive_packet_field_replaced'] != 'regularizers.outline_observation'
            or protocol['worker_keys'] != sorted(A32.REPLAY.WORKER_KEYS)
            or protocol['e_comparison_parent_only'] is not True):
        raise ValueError('Same original native32/positive7/fivearm population; only C observation33 required')
    for seal in (inputs, pred):
        if (seal['rows'] != 238 or seal['ordered_ids'] != ids or seal['arms'] != list(ARMS)
                or seal['arm_count'] != 5 or seal['outline_selection_profile'] != PROFILE):
            raise ValueError('Complete238x5 V33 positive and prediction seals required')
    if (pred['all_declared_arms_complete'] is not True or pred['before_fixed_reference_scoring'] is not True
            or pred['no_population_drop'] is not True or pred['native_CAL32_model_relabelled'] is not False
            or inputs['e_comparison_sha256'] != protocol['e_comparison_sha256']):
        raise ValueError('No-drop/pre-score/exact separate E stream required')
    bindings = [('PROTOCOL.json',inputs['protocol_sha256']),('INPUTS.jsonl.gz',inputs['input_sha256']),
        ('E_COMPARISON.jsonl.gz',inputs['e_comparison_sha256']),('PROTOCOL.json',pred['protocol_sha256']),
        ('INPUT_SEAL.json',pred['input_seal_sha256']),('PREDICTIONS.jsonl.gz',pred['predictions_sha256']),
        ('E_COMPARISON.jsonl.gz',pred['separate_e_comparison_sha256']),
        ('PREDICTION_EXECUTION_PROTOCOL.json',pred['execution_protocol_sha256']),
        ('PREDICTION_ROW_INDEX.json',pred['row_index_sha256']),
        ('NATIVE_AND_E_PARITY_V1.json',pred['native_and_e_parity_sha256'])]
    for name,digest_ in bindings:
        if C.sha(output/name) != digest_:
            raise ValueError('Sealed V33 link changed: '+name)
    parity = C.read(output/'NATIVE_AND_E_PARITY_V1.json')
    if (parity['rows'] != 238 or parity['native_arm_payload_equalities'] != 476
            or parity['e_payload_equalities'] != 238 or parity['failure_count'] != 0 or parity['numeric_tolerance'] != 0.):
        raise ValueError('Full native/e exact parity receipt required, then independently replayed below')
    execution = C.read(output/'PREDICTION_EXECUTION_PROTOCOL.json')
    if (execution['controller_source_sha256'] != CONTROLLER_SHA256
            or execution['worker_source_sha256'] != WORKER_SHA256
            or execution['solver_source_sha256'] != SOLVER_SHA256
            or execution['outline_selection_profile'] != PROFILE
            or execution['positive_worker_keys'] != sorted(A32.REPLAY.WORKER_KEYS)
            or not 1 <= execution['workers'] <= 3 or execution['maximum_pending'] != execution['workers']
            or execution['e_numeric_worker_inputs'] != 0 or execution['reference_or_score_inputs'] != 0):
        raise ValueError('Original worker/solver, bounded3 and parent-only E execution required')
    plan = protocol['execution_plan']
    if C.sha(C.ROOT/plan['path']) != plan['sha256']:
        raise ValueError('Separate V33 execution plan changed')
    outline_seal = C.read(OUTLINES/'OUTLINE_OBSERVATION_SEAL.json')
    outline_protocol = C.read(OUTLINES/'FINAL_PROTOCOL.json')
    outline_entries = C.read(OUTLINES/'OUTLINE_ROW_INDEX.json')['entries']
    if (outline_seal['rows'] != 238 or outline_seal['ordered_ids'] != ids
            or outline_seal['partitions'] != PARTITIONS or outline_seal['selection_profile'] != PROFILE
            or outline_seal['canonical_validation'] != dict(rows=238,canonical_row_digests_verified=238,missing_duplicate_or_extra_rows=0)
            or outline_seal['protocol_sha256'] != C.sha(OUTLINES/'FINAL_PROTOCOL.json')
            or outline_seal['observations_sha256'] != C.sha(OUTLINES/'OUTLINE_OBSERVATIONS.jsonl.gz')
            or outline_seal['row_index_sha256'] != C.sha(OUTLINES/'OUTLINE_ROW_INDEX.json')
            or outline_protocol['central_plan_sha256'] != CENTRAL_SHA256
            or outline_protocol['ordered_ids'] != ids or outline_protocol['selection_profile'] != PROFILE
            or [entry['id'] for entry in outline_entries] != ids):
        raise ValueError('Exact full canonical LCC238 observation ancestry required')
    for part,count,selected,entries in (('CAL',46,ids[:46],outline_entries[:46]),('REST',192,ids[46:],outline_entries[46:])):
        path = OUTLINES/(part+'_OBSERVATION_SEAL.json'); seal = C.read(path)
        index_path = OUTLINES/(part+'_ROW_INDEX.json')
        if (seal['rows'] != count or seal['ordered_ids'] != selected
                or seal['selection_profile'] != PROFILE or seal['union_exact_match_rows'] != count
                or seal['row_index_sha256'] != C.sha(index_path)
                or seal['observations_sha256'] != C.sha(OUTLINES/(part+'_OUTLINE_OBSERVATIONS.jsonl.gz'))
                or outline_seal['partition_seals'][part] != C.sha(path)
                or C.read(index_path)['entries'] != entries):
            raise ValueError('Original CAL46 and REST192 partition closure required')
    maps = [protocol[key] for key in ('source_sha256','input_sha256','protected_user_sha256',
        'original_CAL_file_sha256','immutable_LCC_CAL_file_sha256')]
    A32._verify(*maps)
    if len(maps[2]) != 6 or len(maps[3]) != 53 or len(maps[4]) != 53:
        raise ValueError('Protected6/originalCAL53/newLCCCAL53 required')
    if C.sha(PARENT/'PREDICTION_SEAL.json') != PARENT_PREDICTION_SHA256:
        raise ValueError('Original union R2 prediction seal changed')
    parent_seal = C.read(PARENT/'PREDICTION_SEAL.json')
    if (parent_seal['rows'] != 238 or parent_seal['ordered_ids'] != ids
            or parent_seal['arms'] != list(ARMS)
            or C.sha(PARENT/'PREDICTIONS.jsonl.gz') != parent_seal['predictions_sha256']
            or C.sha(PARENT/'INPUTS.jsonl.gz') != C.read(PARENT/'INPUT_SEAL.json')['input_sha256']
            or C.sha(PARENT/'E_COMPARISON.jsonl.gz') != inputs['e_comparison_sha256']):
        raise ValueError('Original positive/union prediction/E ancestry required')
    records = protocol['records']; indices = C.read(output/'PREDICTION_ROW_INDEX.json')['entries']
    if [record['id'] for record in records] != ids or [index['id'] for index in indices] != ids:
        raise ValueError('All ordered positive/current worker-index entries required')
    return dict(protocol=protocol, ids=ids, maps=maps, bindings=bindings,
        records=records, indices=indices, outline_entries=outline_entries)


def main(*, root_independent_audit_GO=False, expected_prediction_seal_sha256=None,
         expected_central_plan_sha256=None, expected_model_seal_sha256=None,
         expected_outline_seal_sha256=None, output=OUTPUT):
    if root_independent_audit_GO is not True:
        raise PermissionError('Separate ROOT all238 independent V33 audit GO before IO')
    if not all(A32._sha(value) for value in (expected_prediction_seal_sha256,
            expected_central_plan_sha256,expected_model_seal_sha256,expected_outline_seal_sha256)):
        raise ValueError('Explicit current prediction/central/model/outline SHA before IO')
    output = Path(output).resolve(); dest = output/'independent_root_audit_v1'
    if dest.exists():
        raise FileExistsError('Preserve every independent V33 audit attempt')
    metadata = _metadata(output,expected_prediction_seal_sha256,expected_central_plan_sha256,
        expected_model_seal_sha256,expected_outline_seal_sha256)
    models,_ = A32._load_models(MODEL)
    sources = _source_map(); ids = metadata['ids']; dest.mkdir()
    C.write(dest/'PLAN.json',dict(schema='INDEPENDENT_LCC8_OUTLINE_ONLY_SELECTED_AUDIT_PLAN_V33',
        UTC=datetime.now(timezone.utc).isoformat(),source_sha256=sources,rows=238,arms=list(ARMS),
        central_plan_sha256=CENTRAL_SHA256,model_seal_sha256=MODEL_SHA256,
        prediction_seal_sha256=expected_prediction_seal_sha256,outline_seal_sha256=expected_outline_seal_sha256,
        original_R2_prediction_seal_sha256=PARENT_PREDICTION_SHA256,
        numeric_packet_prediction_or_outline_decodes_before_plan=0,
        native_O_S_C_replay='exact A32.audit_row: positive independent replay/V29 scalar and frozen O/S/C25',
        scope='selected-pose arithmetic/guards plus source-packet one-field change and full native/e canonical parity; no freshbank regeneration, pixels or semantic/accuracy certification'))
    failures=[]; components=[]; count=checks=accepted=fallback=no_pose=0; partitions=Counter()
    try:
        rows = checked_rows(C.rows(output/'INPUTS.jsonl.gz'),C.rows(output/'E_COMPARISON.jsonl.gz'),
            C.rows(output/'PREDICTIONS.jsonl.gz'),metadata['records'],metadata['indices'],ids,
            C.rows(PARENT/'INPUTS.jsonl.gz'),C.rows(OUTLINES/'OUTLINE_OBSERVATIONS.jsonl.gz'),
            C.rows(PARENT/'PREDICTIONS.jsonl.gz'),metadata['outline_entries'])
        for packet,comparison,row,extra in rows:
            result = audit_row(packet,comparison,row,models=models)
            for item in [*extra,*result['checks']]:
                checks+=1
                if not item['passed']:failures.append(dict(id=ids[count],**item))
            components.extend(result['components']); accepted+=result['accepted']; fallback+=result['fallback']
            no_pose+=int(result['no_pose']); partitions[packet['identifiers']['partition']]+=1; count+=1
            if count%20==0:print(dict(rows=count,checks=checks,failures=len(failures)),flush=True)
    except BaseException as error:
        C.write(dest/'AUDIT_FAILURE_V1.json',dict(rows_completed=count,checks=checks,failures=failures,
            exception=type(error).__name__,reason=str(error),pending_ids=ids[count:],attempt_preserved=True,
            missing_or_mismatched_rows_not_dropped=True,complete_zero_failure_audit_seal_created=False))
        raise
    if count!=238 or partitions!=Counter(PARTITIONS):
        raise ValueError('Every CAL46/TEST49/REMAINDER143 row required; severe0')
    A32._verify(*metadata['maps'],sources)
    for name,expected in metadata['bindings']:
        if C.sha(output/name)!=expected:raise ValueError('Sealed V33 bytes changed during audit: '+name)
    C.save_rows(dest/'SELECTED_POSE_COMPONENTS.jsonl.gz',components)
    prediction=dict(path=_relative(output/'PREDICTION_SEAL.json'),sha256=expected_prediction_seal_sha256)
    report=dict(schema='INDEPENDENT_LCC8_OUTLINE_ONLY_POSITIVE_SELECTED_AUDIT_V33',rows=count,checks=checks,
        failure_count=len(failures),failures=failures,partition_counts=dict(partitions),accepted_arms=accepted,
        fallback_arms=fallback,no_pose_rows=no_pose,maximum_absolute_scalar_difference=max(
            (item['absolute_total_difference'] for item in components),default=0.),prediction_seal=prediction,
        source_sha256=sources,input_sha256=metadata['protocol']['input_sha256'],
        protected_user_sha256=metadata['protocol']['protected_user_sha256'],
        original_CAL_file_sha256=metadata['protocol']['original_CAL_file_sha256'],
        immutable_LCC_CAL_file_sha256=metadata['protocol']['immutable_LCC_CAL_file_sha256'],
        central_plan_sha256=CENTRAL_SHA256,model_seal_sha256=MODEL_SHA256,
        outline_selection_profile=PROFILE,native_observation_family=A32.REPLAY.FAMILY,
        native_noise_population=A32.REPLAY.POPULATION,native_selection_profile=A32.REPLAY.PROFILE,
        outline_seal_sha256=expected_outline_seal_sha256,native_and_e_payload_equalities_checked=714,
        original_positive_packet_one_field_transitions_checked=238,
        new_consumer_planner_objective_or_PnP_optimizer_calls=0,reference_RGB_or_NN_calls=0,
        pixel_component_replay_performed_here=False,full_unseen_seed_bank_or_start_rotations_regenerated=False,
        selected_pose_arithmetic_only=True,semantic_mask_identity_or_accuracy_certified=False,default_adoption=False)
    C.write(dest/'AUDIT.json',report)
    C.write(dest/'AUDIT_SEAL.json',dict(receipt_sha256=C.sha(dest/'AUDIT.json'),
        components_sha256=C.sha(dest/'SELECTED_POSE_COMPONENTS.jsonl.gz'),plan_sha256=C.sha(dest/'PLAN.json'),
        source_sha256=sources,prediction_seal=prediction))
    print(dict(rows=count,checks=checks,failures=len(failures)),flush=True)
    if failures:raise SystemExit(1)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-independent-audit-GO',action='store_true')
    for name in ('prediction','central-plan','model','outline'):
        parser.add_argument('--'+name+'-seal-sha256' if name!='central-plan' else '--central-plan-sha256')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    main(root_independent_audit_GO=args.root_independent_audit_GO,
        expected_prediction_seal_sha256=args.prediction_seal_sha256,
        expected_central_plan_sha256=args.central_plan_sha256,expected_model_seal_sha256=args.model_seal_sha256,
        expected_outline_seal_sha256=args.outline_seal_sha256,output=args.output)
