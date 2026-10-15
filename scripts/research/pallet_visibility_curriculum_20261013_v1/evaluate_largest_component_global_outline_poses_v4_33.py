"""LCC8 containment control over the sealed positive V32 R2 packets.

Only regularizers.outline_observation changes. Native evidence, its CAL32
noise, source policies, guards, q archives, fresh SQPnP and continuous solver
remain V32. The five arm names retain their frozen solver identities; this
separate output and its outline_selection_profile identify the V33 control.
All operational stages require a new explicit ROOT GO. Importing is IO-free.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
import shutil
import sys

from . import evaluate_clean_fresh_BI_poses_v4_32_r2 as M
from . import largest_component_global_outline_v4_33 as LCC

C = M.C
B = M.B
OUTPUT = B/'largest_component_global_outline_pose_eval_v4_33'
PLAN_OUTPUT = B/'largest_component_global_outline_pose_eval_plan_v4_33'
DECLARATION = B/'largest_component_global_outline_method_plan_v4_33/PREDECLARATION.json'
OUTLINES = B/'largest_component_global_outline_observations_v4_33'
PARENT = M.OUTPUT
ARMS = M.ARMS
NATIVE_ARMS = tuple(names[0] for names in M.POLICY_ARMS.values())
PARTITIONS = M.PARTITIONS
POPULATION_INVENTORY = dict(PARTITIONS, PRIMARY=192, severe=0)
OUTLINE_SELECTION_PROFILE = LCC.SELECTION_PROFILE
SELECTION_PROFILE = OUTLINE_SELECTION_PROFILE  # Central declaration API; native profile stays V32.
PARENT_CONTROLLER_SHA256 = 'fa9ebe1d327972c585ee5f1f750a50f341151ec421c0e8ad5124a9a2e6b38187'
LCC_SHA256 = '112e9035d522c4487a816e3612218d44a5fe53470c75a9da4371ec700a276c46'

# Exact direct reuse. No private namespace or frozen module-global mutation.
solve_worker_packet = M.solve_worker_packet
_bounded_predict = M._bounded_predict
_merged_rows = M._merged_rows


def _sha(value):
    return M._sha(value)


def _digest(value):
    return M.G._digest(C.finite(value))


def _relative(path):
    return str(Path(path).resolve().relative_to(C.ROOT))


def _binding(path):
    return dict(path=_relative(path), sha256=C.sha(path))


def _union(*maps):
    return M.OBS._union(*maps)


def _verify(*maps):
    M._verify(*maps)


def _source_map():
    paths = [Path(__file__), Path(__file__).with_name('test_evaluate_largest_component_global_outline_poses_v4_33.py'),
             Path(M.__file__), Path(LCC.__file__), Path(LCC.FROZEN_OUTLINE.__file__),
             Path(LCC.FROZEN_CONTAINMENT.__file__)]
    return _union(M._source_map(), {_relative(path): C.sha(path) for path in paths})


def _pins():
    for module, expected in ((M, PARENT_CONTROLLER_SHA256), (LCC, LCC_SHA256),
                             (M.SOLVER, M.SOLVER_SHA256)):
        if C.sha(module.__file__) != expected:
            raise ValueError('Frozen positive solver or LCC source changed')


def _central(expected):
    if not _sha(expected):
        raise ValueError('Explicit central33 lowercase SHA before IO')
    if C.sha(DECLARATION) != expected:
        raise ValueError('Exact immutable central33 declaration required')
    value = C.read(DECLARATION)
    if (value['selection_profile'] != OUTLINE_SELECTION_PROFILE
            or value['ordered_ids'] != list(dict.fromkeys(value['ordered_ids']))
            or len(value['ordered_ids']) != 238 or value['partitions'] != POPULATION_INVENTORY
            or value['arms'] != list(ARMS)):
        raise ValueError('Uniform ordered238/five-arm LCC33 declaration required')
    _verify(*(value[key] for key in ('source_sha256', 'input_sha256', 'protected_user_sha256')))
    if len(value['protected_user_sha256']) != 6:
        raise ValueError('Original protected six bindings required')
    for path, digest in _source_map().items():
        if value['source_sha256'].get(path) != digest:
            raise ValueError('Central33 must pin every current controller/helper source')
    _pins()
    return value


def _parent_metadata(central):
    """Verify frozen packets/models by metadata and bytes, no old task decode."""
    protocol = M._load_prepared(PARENT)
    seal = C.read(PARENT/'PREDICTION_SEAL.json')
    ids = protocol['ordered_ids']
    if (ids != central['ordered_ids'] or len(ids) != 238
            or Counter(record['partition'] for record in protocol['records']) != Counter(PARTITIONS)
            or [record['id'] for record in protocol['records']] != ids
            or seal['rows'] != 238 or seal['ordered_ids'] != ids
            or seal['arms'] != list(ARMS) or seal['arm_count'] != 5
            or seal['all_declared_arms_complete'] is not True
            or seal['before_fixed_reference_scoring'] is not True
            or seal['no_population_drop'] is not True):
        raise ValueError('Completed original R2 ordered238/five-arm inputs and predictions required')
    bindings = (('PREDICTIONS.jsonl.gz', 'predictions_sha256'),
                ('PROTOCOL.json', 'protocol_sha256'), ('INPUT_SEAL.json', 'input_seal_sha256'),
                ('PREDICTION_EXECUTION_PROTOCOL.json', 'execution_protocol_sha256'),
                ('PREDICTION_ROW_INDEX.json', 'row_index_sha256'),
                ('E_COMPARISON.jsonl.gz', 'separate_e_comparison_sha256'))
    for name, key in bindings:
        if C.sha(PARENT/name) != seal[key]:
            raise ValueError('Frozen parent R2 seal/stream binding changed')
    for name in ('PROTOCOL.json', 'INPUT_SEAL.json', 'INPUTS.jsonl.gz',
                 'E_COMPARISON.jsonl.gz', 'PREDICTION_SEAL.json'):
        if central['input_sha256'].get(_relative(PARENT/name)) != C.sha(PARENT/name):
            raise ValueError('Central33 must bind the original sealed positive/e files')
    if C.sha(M.CAL.OUTPUT/'MODEL_SEAL.json') != protocol['new_model_seal_sha256']:
        raise ValueError('Original fitted CAL32 noise remains unchanged, no new population')
    return protocol


def _outline_metadata(expected, central, parent):
    """Full LCC CAL46 then REST192 closure is mandatory before preparation."""
    if not _sha(expected) or C.sha(OUTLINES/'OUTLINE_OBSERVATION_SEAL.json') != expected:
        raise ValueError('Caller-pinned completed all238 LCC outline seal required')
    final = C.read(OUTLINES/'FINAL_PROTOCOL.json')
    seal = C.read(OUTLINES/'OUTLINE_OBSERVATION_SEAL.json')
    entries = C.read(OUTLINES/'OUTLINE_ROW_INDEX.json')['entries']
    ids = central['ordered_ids']
    if (final['central_plan_sha256'] != C.sha(DECLARATION)
            or final['selection_profile'] != OUTLINE_SELECTION_PROFILE
            or final['ordered_ids'] != ids or seal['ordered_ids'] != ids
            or seal['rows'] != 238 or seal['partitions'] != PARTITIONS
            or seal['selection_profile'] != OUTLINE_SELECTION_PROFILE
            or seal['protocol_sha256'] != C.sha(OUTLINES/'FINAL_PROTOCOL.json')
            or seal['observations_sha256'] != C.sha(OUTLINES/'OUTLINE_OBSERVATIONS.jsonl.gz')
            or seal['row_index_sha256'] != C.sha(OUTLINES/'OUTLINE_ROW_INDEX.json')
            or seal['canonical_validation'] != dict(rows=238, canonical_row_digests_verified=238,
                missing_duplicate_or_extra_rows=0)
            or [entry['id'] for entry in entries] != ids
            or Counter(entry['partition'] for entry in entries) != Counter(PARTITIONS)):
        raise ValueError('Exact canonical all238 LCC observations with full denominators required')
    for part, count, selected in (('CAL', 46, ids[:46]), ('REST', 192, ids[46:])):
        partition = C.read(OUTLINES/(part+'_OBSERVATION_SEAL.json'))
        index_path = OUTLINES/(part+'_ROW_INDEX.json')
        index = C.read(index_path)['entries']
        if (partition['rows'] != count or partition['ordered_ids'] != selected
                or partition['selection_profile'] != OUTLINE_SELECTION_PROFILE
                or partition['row_index_sha256'] != C.sha(index_path)
                or partition['observations_sha256'] != C.sha(OUTLINES/(part+'_OUTLINE_OBSERVATIONS.jsonl.gz'))
                or partition['union_exact_match_rows'] != count
                or seal['partition_seals'][part] != C.sha(OUTLINES/(part+'_OBSERVATION_SEAL.json'))):
            raise ValueError('Both complete immutable LCC partitions and exact union controls required')
        expected_entries = entries[:46] if part == 'CAL' else entries[46:]
        if index != expected_entries:
            raise ValueError('Full outline index must exactly concatenate CAL46 and REST192')
    if (final['original_CAL_file_sha256'] != parent['original_CAL_file_sha256']
            or len(final['original_CAL_file_sha256']) != 53
            or len(final['immutable_LCC_CAL_file_sha256']) != 53):
        raise ValueError('Original CAL53 and separate new LCC CAL53 must stay immutable')
    if (any(final['source_sha256'].get(path) != digest for path, digest in central['source_sha256'].items())
            or final['protected_user_sha256'] != central['protected_user_sha256']):
        raise ValueError('Completed LCC protocol must inherit exact central source/protected closure')
    _verify(*(final[key] for key in ('source_sha256', 'input_sha256', 'protected_user_sha256',
        'original_CAL_file_sha256', 'immutable_LCC_CAL_file_sha256')))
    return final, entries


def _stage_inputs(expected_central, expected_outlines):
    central = _central(expected_central)
    parent = _parent_metadata(central)
    outlines, entries = _outline_metadata(expected_outlines, central, parent)
    maps = {key: _union(*(value[key] for value in (central, parent, outlines)))
            for key in ('source_sha256', 'input_sha256', 'protected_user_sha256')}
    maps['source_sha256'] = _union(maps['source_sha256'], _source_map())
    paths = [DECLARATION, *(PARENT/name for name in ('PROTOCOL.json', 'INPUT_SEAL.json',
        'INPUTS.jsonl.gz', 'E_COMPARISON.jsonl.gz', 'PREDICTION_SEAL.json', 'PREDICTIONS.jsonl.gz')),
        *(OUTLINES/name for name in ('FINAL_PROTOCOL.json', 'OUTLINE_OBSERVATION_SEAL.json',
        'OUTLINE_ROW_INDEX.json', 'OUTLINE_OBSERVATIONS.jsonl.gz', 'CAL_OBSERVATION_SEAL.json',
        'REST_OBSERVATION_SEAL.json', 'CAL_ROW_INDEX.json', 'REST_ROW_INDEX.json',
        'CAL_OUTLINE_OBSERVATIONS.jsonl.gz', 'REST_OUTLINE_OBSERVATIONS.jsonl.gz'))]
    maps['input_sha256'] = _union(maps['input_sha256'], {_relative(path): C.sha(path) for path in paths})
    maps['original_CAL_file_sha256'] = outlines['original_CAL_file_sha256']
    maps['immutable_LCC_CAL_file_sha256'] = outlines['immutable_LCC_CAL_file_sha256']
    _verify(*maps.values())
    return dict(maps=maps, parent=parent, entries=entries, ordered_ids=central['ordered_ids'])


def _peer(binding):
    from . import declare_initial_only_observed_ray_v4_32 as D
    path = D._peer(binding, (LCC, sys.modules[__name__]))
    sources = D._source_bindings(C.read(path))
    test = Path(__file__).with_name('test_evaluate_largest_component_global_outline_poses_v4_33.py')
    if sources.get(_relative(test)) != C.sha(test):
        raise ValueError('Controller peer must also bind the exact toy test source')
    return path


def plan(*, root_metadata_GO=False, expected_central_plan_sha256=None,
         expected_outline_seal_sha256=None, controller_peer_binding=None, output=PLAN_OUTPUT):
    if root_metadata_GO is not True:
        raise PermissionError('Explicit ROOT V33 metadata plan GO before IO')
    if not _sha(expected_central_plan_sha256) or not _sha(expected_outline_seal_sha256) or controller_peer_binding is None:
        raise ValueError('Explicit central/outline/peer bindings before IO')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve every V33 execution plan')
    peer = _peer(controller_peer_binding)
    stage = _stage_inputs(expected_central_plan_sha256, expected_outline_seal_sha256)
    value = dict(stage['maps'])
    value['input_sha256'] = _union(value['input_sha256'], {_relative(peer): C.sha(peer)})
    value.update(schema='LCC8_OUTLINE_ONLY_POSITIVE_PNP_EXECUTION_PLAN_V33',
        UTC=datetime.now(timezone.utc).isoformat(), central_plan_sha256=expected_central_plan_sha256,
        outline_seal_sha256=expected_outline_seal_sha256, controller_peer_binding=controller_peer_binding,
        ordered_ids=stage['ordered_ids'], partitions=PARTITIONS, population_inventory=POPULATION_INVENTORY, arms=list(ARMS),
        outline_selection_profile=OUTLINE_SELECTION_PROFILE,
        native_observation_family=M.G.OBSERVATION_FAMILY, native_selection_profile=M.G.SELECTION_PROFILE,
        native_noise_population=M.G.POPULATION, native_CAL32_model_relabelled=False,
        native_model_seal_sha256=stage['parent']['new_model_seal_sha256'],
        only_positive_packet_field_replaced='regularizers.outline_observation',
        original_parent_positive_input=_binding(PARENT/'INPUT_SEAL.json'),
        original_parent_prediction=_binding(PARENT/'PREDICTION_SEAL.json'),
        e_comparison_copied_opaque_and_parent_only=True, worker_keys=sorted(M.PROJECT.WORKER_KEYS),
        maximum_workers_and_pending=3, native_two_arm_exact_invariant_required=True,
        same_frozen_solver_and_fresh_SQPNP=True, new_PnP_or_optimizer_in_plan=0,
        all238_retained=True, original_TEST49_preservation_must_be_reported_after_scoring=True,
        no_actual_accuracy_or_adoption_claim=True, default='SAME_FROZEN_E unchanged')
    output.mkdir(parents=True)
    C.write(output/'PREDECLARATION.json', value)
    for relative, digest in value['source_sha256'].items():
        target = output/'code_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Current V33 execution source snapshot changed')
    C.write(output/'PLAN_SEAL.json', dict(plan_sha256=C.sha(output/'PREDECLARATION.json'),
        source_sha256=value['source_sha256']))
    return _binding(output/'PREDECLARATION.json')


def _load_plan(folder):
    folder = Path(folder)
    value = C.read(folder/'PREDECLARATION.json')
    if (C.read(folder/'PLAN_SEAL.json')['plan_sha256'] != C.sha(folder/'PREDECLARATION.json')
            or value['outline_selection_profile'] != OUTLINE_SELECTION_PROFILE
            or value['arms'] != list(ARMS) or value['maximum_workers_and_pending'] != 3):
        raise ValueError('Exact separately sealed LCC33 five-arm plan required')
    _verify(*(value[key] for key in ('source_sha256', 'input_sha256', 'protected_user_sha256',
        'original_CAL_file_sha256', 'immutable_LCC_CAL_file_sha256')))
    if C.sha(DECLARATION) != value['central_plan_sha256']:
        raise ValueError('Central33 remains immutable')
    return value


def replace_outline_observation(packet, row):
    """Pure one-field replacement; no masks, old tasks or new model are inputs."""
    if not isinstance(packet, dict) or set(packet) != M.PROJECT.WORKER_KEYS or packet['schema'] != M.PROJECT.WORKER_SCHEMA:
        raise ValueError('Only original exact seven-key positive V32 packets allowed')
    identifiers = packet['identifiers']
    if (row['id'] != identifiers['id'] or row['partition'] != identifiers['partition']
            or row['selection_profile'] != OUTLINE_SELECTION_PROFILE
            or row['baseline_profile'] != LCC.BASELINE_PROFILE
            or row['positive_packet_sha256'] != _digest(packet)
            or row['union_exact_match'] is not True):
        raise ValueError('Exact original positive packet and uniform union/LCC row required')
    if not packet['source_contract']['ready']:
        return packet  # Preserve invalid coarse row; no regularizer/core payload reads.
    regularizers = packet['regularizers']
    if set(regularizers) != M.PROJECT.REGULARIZER_KEYS:
        raise ValueError('Original positive regularizer fields only')
    old = regularizers['outline_observation']
    union = {key: value for key, value in row['original_union'].items() if key != 'selection_profile'}
    new = row['largest_component']
    if (row['original_union']['selection_profile'] != LCC.BASELINE_PROFILE
            or _digest(union) != _digest(old)
            or new['selection_profile'] != OUTLINE_SELECTION_PROFILE
            or new['image_binding'] != old['image_binding']
            or new['image_hw'] != old['image_hw'] or old['image_hw'] != packet['physics']['image_hw']
            or new['configuration'] != old['configuration']
            or new['configuration'] != regularizers['outline_configuration']
            or new['image_binding']['image_sha256'] != identifiers['image_sha256']
            or row['target_mask_binding']['sha256'] != old['image_binding']['mask_file_sha256']):
        raise ValueError('Only outline observation may change; original union/image/mask/config exact')
    audit = new['foreground_component_selection']
    if (audit['selection_profile'] != OUTLINE_SELECTION_PROFILE or audit['connectivity'] != 8
            or audit['mask_threshold'] != .5 or audit['holes_filled'] is not False
            or audit['morphology_used'] is not False or audit['original_mask_modified'] is not False):
        raise ValueError('One frozen LCC8 observation recipe required')
    changed = dict(packet)
    changed['regularizers'] = dict(regularizers, outline_observation=deepcopy(new))
    return changed


def _prepared_rows(packets, outlines, parent_records, outline_entries, ids, records):
    seen = []
    for packet, row, original, entry in zip_longest(packets, outlines, parent_records, outline_entries):
        if any(value is None for value in (packet, row, original, entry)):
            raise ValueError('Exact all238 input/outline/index counts required')
        index = len(seen)
        if (index >= len(ids) or packet['identifiers']['id'] != ids[index]
                or original['id'] != ids[index] or entry['id'] != ids[index]
                or original['partition'] != packet['identifiers']['partition']
                or entry['partition'] != packet['identifiers']['partition']
                or original['positive_packet_sha256'] != _digest(packet)
                or entry['canonical_row_sha256'] != _digest(row)):
            raise ValueError('Exact canonical positive/outline ordered join required')
        changed = replace_outline_observation(packet, row)
        records.append(dict(id=ids[index], partition=original['partition'],
            ready=packet['source_contract']['ready'], positive_packet_sha256=_digest(changed),
            parent_positive_packet_sha256=original['positive_packet_sha256'],
            outline_row_sha256=entry['canonical_row_sha256'],
            changed_field='regularizers.outline_observation' if packet['source_contract']['ready'] else None))
        seen.append(ids[index])
        yield changed
    if seen != ids or Counter(record['partition'] for record in records) != Counter(PARTITIONS):
        raise ValueError('All238 including invalid/unsupported rows must be retained')


def prepare(*, root_metadata_GO=False, plan_folder=PLAN_OUTPUT, output=OUTPUT):
    if root_metadata_GO is not True:
        raise PermissionError('Explicit ROOT V33 positive preparation GO before IO')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve every V33 preparation attempt')
    declared = _load_plan(plan_folder)
    stage = _stage_inputs(declared['central_plan_sha256'], declared['outline_seal_sha256'])
    peer = declared['controller_peer_binding']
    stage['maps']['input_sha256'] = _union(stage['maps']['input_sha256'], {peer['path']: peer['sha256']})
    for key, mapping in stage['maps'].items():
        if mapping != declared[key]:
            raise ValueError('Current V33 execution maps differ from frozen plan')
    output.mkdir(parents=True)
    records = []
    try:
        M.OBS.save_rows_large(output/'INPUTS.jsonl.gz', _prepared_rows(C.rows(PARENT/'INPUTS.jsonl.gz'),
            C.rows(OUTLINES/'OUTLINE_OBSERVATIONS.jsonl.gz'), stage['parent']['records'], stage['entries'],
            stage['ordered_ids'], records))
        shutil.copyfile(PARENT/'E_COMPARISON.jsonl.gz', output/'E_COMPARISON.jsonl.gz')
        if C.sha(output/'E_COMPARISON.jsonl.gz') != C.sha(PARENT/'E_COMPARISON.jsonl.gz'):
            raise ValueError('Parent-only e stream must be copied byte exactly')
        _verify(*(declared[key] for key in stage['maps']))
        protocol = dict(declared, schema='SEALED_OUTLINE_ONLY_POSITIVE_PNP_INPUTS_V33',
            execution_plan=_binding(Path(plan_folder)/'PREDECLARATION.json'), records=records,
            e_comparison_sha256=C.sha(output/'E_COMPARISON.jsonl.gz'), e_comparison_parent_only=True)
        C.write(output/'PROTOCOL.json', protocol)
        C.write(output/'INPUT_SEAL.json', dict(rows=238, ordered_ids=stage['ordered_ids'],
            arms=list(ARMS), arm_count=5, protocol_sha256=C.sha(output/'PROTOCOL.json'),
            input_sha256=C.sha(output/'INPUTS.jsonl.gz'), e_comparison_sha256=protocol['e_comparison_sha256'],
            outline_selection_profile=OUTLINE_SELECTION_PROFILE, before_fresh_PnP_or_prediction=True))
    except BaseException as error:
        C.write(output/'PREPARATION_FAILURE_V1.json', dict(exception=type(error).__name__, reason=str(error),
            prepared_rows=len(records), attempt_preserved=True, source_or_inputs_silently_modified=False))
        raise
    return _binding(output/'INPUT_SEAL.json')


def _load_prepared(output):
    protocol = M._load_prepared(output)
    if (protocol['schema'] != 'SEALED_OUTLINE_ONLY_POSITIVE_PNP_INPUTS_V33'
            or protocol['outline_selection_profile'] != OUTLINE_SELECTION_PROFILE
            or protocol['native_observation_family'] != M.G.OBSERVATION_FAMILY
            or protocol['native_selection_profile'] != M.G.SELECTION_PROFILE
            or protocol['native_noise_population'] != M.G.POPULATION
            or protocol['native_CAL32_model_relabelled'] is not False
            or C.sha(DECLARATION) != protocol['central_plan_sha256']):
        raise ValueError('V33 outline control must retain the original native32 population')
    _verify(protocol['immutable_LCC_CAL_file_sha256'])
    _pins()
    return protocol


def verify_native_and_e_parity(previous, current, ids):
    """Exact native arm payloads, including poses, costs, guards and starts.

    Row-level elapsed time is outside each arm and deliberately not compared.
    C-arm diagnostics and results are expected to differ and are not compared.
    """
    seen = []
    for old, new in zip_longest(previous, current):
        index = len(seen)
        if (old is None or new is None or index >= len(ids) or old['id'] != ids[index]
                or new['id'] != ids[index] or old['partition'] != new['partition']):
            raise ValueError('Exact238 parent/current prediction row join required')
        for arm in (*NATIVE_ARMS, 'SAME_FROZEN_E'):
            if _digest(old['arms'][arm]) != _digest(new['arms'][arm]):
                raise ValueError('Original native/e invariant failed: '+ids[index]+'/'+arm)
        seen.append(ids[index])
    if seen != ids:
        raise ValueError('Native/e invariant must cover every retained row')
    return dict(rows=len(ids), native_arm_count=2, native_arm_payload_equalities=len(ids)*2,
        e_payload_equalities=len(ids), failure_count=0, numeric_tolerance=0.,
        compared='entire native arm payloads including actual_pose/N/O/S/guards/fresh starts',
        excluded='row-level elapsed time and the two C arm payloads',
        no_reference_or_accuracy_evaluation=True)


def predict(*, root_prediction_GO=False, workers=3, output=OUTPUT):
    if root_prediction_GO is not True:
        raise PermissionError('Explicit ROOT V33 full prediction GO before IO/PnP')
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 3:
        raise ValueError('Bounded worker/pending budget is 1..3')
    output = Path(output)
    protocol = _load_prepared(output)
    execution = output/'PREDICTION_EXECUTION_PROTOCOL.json'
    if execution.exists() or (output/'PREDICTIONS.jsonl.gz').exists():
        raise FileExistsError('Preserve every V33 prediction attempt')
    (output/'prediction_spool').mkdir()
    (output/'frames').mkdir()
    C.write(execution, dict(workers=workers, maximum_pending=workers, controller_source_sha256=C.sha(__file__),
        worker_source_sha256=PARENT_CONTROLLER_SHA256, solver_source_sha256=M.SOLVER_SHA256,
        outline_selection_profile=OUTLINE_SELECTION_PROFILE, positive_worker_keys=sorted(M.PROJECT.WORKER_KEYS),
        UTC=datetime.now(timezone.utc).isoformat(), e_numeric_worker_inputs=0, reference_or_score_inputs=0))
    try:
        ids = protocol['ordered_ids']
        entries = _bounded_predict(C.rows(output/'INPUTS.jsonl.gz'), ids, output, workers)
        C.write(output/'PREDICTION_ROW_INDEX.json', dict(entries=[entries[identifier] for identifier in ids]))
        M.OBS.save_rows_large(output/'PREDICTIONS.jsonl.gz', _merged_rows(output, ids, entries,
            C.rows(output/'E_COMPARISON.jsonl.gz')))
        parity = verify_native_and_e_parity(C.rows(PARENT/'PREDICTIONS.jsonl.gz'),
            C.rows(output/'PREDICTIONS.jsonl.gz'), ids)
        C.write(output/'NATIVE_AND_E_PARITY_V1.json', parity)
        _load_prepared(output)
        C.write(output/'PREDICTION_SEAL.json', dict(rows=238, ordered_ids=ids, arms=list(ARMS), arm_count=5,
            all_declared_arms_complete=True, predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),
            protocol_sha256=C.sha(output/'PROTOCOL.json'), input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),
            execution_protocol_sha256=C.sha(execution), row_index_sha256=C.sha(output/'PREDICTION_ROW_INDEX.json'),
            separate_e_comparison_sha256=C.sha(output/'E_COMPARISON.jsonl.gz'),
            native_and_e_parity_sha256=C.sha(output/'NATIVE_AND_E_PARITY_V1.json'),
            outline_selection_profile=OUTLINE_SELECTION_PROFILE, native_CAL32_model_relabelled=False,
            before_fixed_reference_scoring=True, no_population_drop=True, default='SAME_FROZEN_E unchanged'))
    except BaseException as error:
        C.write(output/'CONTROLLER_PREDICTION_FAILURE_V1.json', dict(exception=type(error).__name__,
            reason=str(error), attempt_preserved=True, silently_restarted=False))
        raise
    return _binding(output/'PREDICTION_SEAL.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'prepare', 'predict'))
    parser.add_argument('--root-metadata-GO', action='store_true')
    parser.add_argument('--root-prediction-GO', action='store_true')
    parser.add_argument('--central-plan-sha256')
    parser.add_argument('--outline-seal-sha256')
    parser.add_argument('--controller-peer')
    parser.add_argument('--controller-peer-sha256')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.stage == 'plan':
        print(plan(root_metadata_GO=args.root_metadata_GO, expected_central_plan_sha256=args.central_plan_sha256,
            expected_outline_seal_sha256=args.outline_seal_sha256,
            controller_peer_binding=dict(path=args.controller_peer, sha256=args.controller_peer_sha256)))
    elif args.stage == 'prepare':
        print(prepare(root_metadata_GO=args.root_metadata_GO))
    else:
        print(predict(root_prediction_GO=args.root_prediction_GO, workers=args.workers))
