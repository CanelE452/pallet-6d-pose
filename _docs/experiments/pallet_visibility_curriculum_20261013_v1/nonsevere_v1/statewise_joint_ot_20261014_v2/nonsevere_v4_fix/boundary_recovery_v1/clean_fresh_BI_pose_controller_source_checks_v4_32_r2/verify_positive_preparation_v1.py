"""Read only prepared positive packets and separately sealed e comparisons.

No source task, raw observation, image, reference, PnP or optimizer is opened
or called. Current ancestry files are hashed as opaque bytes by the frozen
controller verifier; the numeric packet stream is inspected structurally.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from scripts.research.pallet_visibility_curriculum_20261013_v1 import evaluate_clean_fresh_BI_poses_v4_32_r2 as M


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for text in stream:
            yield json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    receipt = M.OUTPUT/'OWNER_POSITIVE_PREPARATION_RECEIPT_V1.json'
    seal_path = M.OUTPUT/'OWNER_POSITIVE_PREPARATION_SEAL_V1.json'
    if receipt.exists() or seal_path.exists():
        raise FileExistsError('Preserve owner verification attempts')
    protocol = M._load_prepared(M.OUTPUT)
    input_seal = M.C.read(M.OUTPUT/'INPUT_SEAL.json')
    identifiers = protocol['ordered_ids']
    assert len(identifiers) == len(set(identifiers)) == 238
    assert protocol['central_plan_sha256'] == 'c95598253d42f06ace0083d30b5fb46adbb64191fa59ee5b76ce348ef2be10ef'
    assert protocol['new_model_seal_sha256'] == '4b7d20df18f62269360a873e530ee29f7535c9d3d4ec4c6b6f72e49b7be48ab7'
    assert len(protocol['protected_user_sha256']) == 6 and len(protocol['original_CAL_file_sha256']) == 53
    assert input_seal['before_fresh_PnP_or_prediction'] is True
    assert protocol['parent_e_comparison_separate_from_numeric_inputs'] is True
    assert protocol['old_pose_bank_start_or_fallback_inputs'] == 0
    assert protocol['e_comparison_parent_only'] is True
    assert input_seal['arms'] == list(M.ARMS) and input_seal['arm_count'] == 5
    records = protocol['records']
    assert len(records) == 238 and [record['id'] for record in records] == identifiers
    ready = Counter();groups = Counter();seen=[];partitions=Counter()
    forbidden = {'task','baseline','proposals','initializations','pose_guides','e','observation','guide_bank'}
    for index, packet in enumerate(rows(M.OUTPUT/'INPUTS.jsonl.gz')):
        assert index < 238
        record = records[index]
        assert set(packet) == M.PROJECT.WORKER_KEYS
        assert packet['schema'] == M.PROJECT.WORKER_SCHEMA
        assert set(packet['clean_input']) == M.PROJECT.CLEAN_INPUT_KEYS
        assert not forbidden.intersection(packet) and not forbidden.intersection(packet['clean_input'])
        assert packet['identifiers'] == packet['clean_input']['identifiers']
        assert packet['identifiers']['id'] == record['id'] == identifiers[index]
        assert packet['identifiers']['partition'] == record['partition']
        assert digest(packet) == record['positive_packet_sha256']
        is_ready = packet['source_contract']['ready']
        assert isinstance(is_ready,bool) and record['ready'] is is_ready
        ready[str(is_ready)] += 1
        if is_ready:
            assert set(packet['physics']) == M.PROJECT.PHYSICS_KEYS
            assert set(packet['regularizers']) == M.PROJECT.REGULARIZER_KEYS
            assert set(packet['typed_bundles']) == set(M.G.SOURCE_POLICIES)
            for policy, families in packet['typed_bundles'].items():
                assert set(families) == set(M.FAMILIES)
                for family, bundle in families.items():
                    assert bundle['observation_family'] == M.G.OBSERVATION_FAMILY
                    assert bundle['population'] == M.G.POPULATION
                    assert bundle['selection_profile'] == M.G.SELECTION_PROFILE
                    groups[policy+'/'+family] += len(bundle['groups'])
        else:
            assert packet['physics'] is packet['typed_bundles'] is packet['regularizers'] is None
        seen.append(record['id']);partitions[record['partition']] += 1
    assert seen == identifiers and dict(partitions) == M.PARTITIONS
    comparisons=[];comparison_partitions=Counter()
    for index, comparison in enumerate(rows(M.OUTPUT/'E_COMPARISON.jsonl.gz')):
        assert index < 238 and comparison['id'] == identifiers[index]
        assert comparison['partition'] == records[index]['partition']
        assert comparison['legacy_e_pose_semantics']['numeric_worker_exposure'] is False
        assert comparison['legacy_e_pose_semantics']['original_native_xyz_zero_mean_verified'] is True
        comparisons.append(comparison['id']);comparison_partitions[comparison['partition']] += 1
    assert comparisons == identifiers and comparison_partitions == partitions
    cal_owner = M.C.read(M.OBS.OUTPUT/'POPULATION_OWNER_COMPLETION_RECEIPT_V32_V2.json')
    assert protocol['original_CAL_file_sha256'] == cal_owner['original_CAL_file_sha256']
    failed = M.B/'clean_fresh_BI_pose_eval_v4_32'
    failure_receipt_path = failed/'OWNER_PREPARATION_FAILURE_RECEIPT_V1.json'
    assert M.C.sha(failure_receipt_path) == 'bf1ffc0507fbab7115adedc263b4025a0172f7715c996de597c3b6325bdcd598'
    failure = M.C.read(failure_receipt_path)
    for asset in failure['attempts']:
        assert M.C.sha(M.C.ROOT/asset['path']) == asset['sha256']
    assert M.C.sha(M.C.ROOT/failure['source']['path']) == failure['source']['sha256']
    assert M.C.sha(M.B/'clean_fresh_BI_pose_eval_plan_v4_32/PREDECLARATION.json') == failure['plan_sha256']
    assert not (M.OUTPUT/'PREDICTION_EXECUTION_PROTOCOL.json').exists()
    assert not (M.OUTPUT/'PREDICTIONS.jsonl.gz').exists()
    assets = ['INPUT_SEAL.json','PROTOCOL.json','INPUTS.jsonl.gz','E_COMPARISON.jsonl.gz']
    data = dict(schema='CLEAN_FRESH_BI_R2_OWNER_POSITIVE_PACKET_CLOSURE_V1',
        UTC=datetime.now(timezone.utc).isoformat(),rows=238,partitions=dict(partitions),severe=0,
        positive_packets_exact_seven_keys=238,canonical_positive_packet_digests_matching_records=238,
        separate_e_comparison_rows=238,e_comparison_numeric_worker_exposure=0,
        readiness_counts=dict(ready),typed_group_counts_by_policy_family=dict(groups),
        current_source_files=len(protocol['source_sha256']),current_input_files=len(protocol['input_sha256']),
        current_protected6_exact=True,current_CAL53_exact=True,current_model_and_OBS_ancestry_exact=True,
        source_sha256=protocol['source_sha256'],protected_user_sha256=protocol['protected_user_sha256'],
        original_CAL_file_sha256=protocol['original_CAL_file_sha256'],
        assets={name:M._binding(M.OUTPUT/name) for name in assets},
        execution_plan=protocol['execution_plan'],
        verifier_source=M._binding(Path(__file__).resolve()),
        fresh_PnP_optimizer_or_pose_estimate_calls=0,source_task_raw_OBS_RGB_reference_or_score_decodes=0,
        prerequisite_scope='Positive packet/schema/current-byte readiness only; no full238 numerical or accuracy claim.',
        prediction_not_started_before_owner_closure=True,previous_failed_generation_preserved=True)
    write_new(receipt,data)
    write_new(seal_path,dict(receipt_sha256=M.C.sha(receipt),verifier_source_sha256=M.C.sha(__file__),
        input_seal_sha256=M.C.sha(M.OUTPUT/'INPUT_SEAL.json'),all_declared_pre_prediction_conditions_pass=True))
    print(json.dumps(dict(receipt=M._binding(receipt),seal=M._binding(seal_path),rows=238,ready=dict(ready)),ensure_ascii=False))


if __name__ == '__main__':
    main()
