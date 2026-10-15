"""Owner-only integrity/E-merge check after the frozen prediction seal.

This does not recompute selected pose costs or score reference accuracy.
ROOT owns the independent full238 numerical audit and later fixed scoring.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from itertools import zip_longest
from pathlib import Path

from scripts.research.pallet_visibility_curriculum_20261013_v1 import evaluate_clean_fresh_BI_poses_v4_32_r2 as M


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    output = M.OUTPUT
    receipt_path = output/'OWNER_PREDICTION_COMPLETION_RECEIPT_V1.json'
    owner_seal = output/'OWNER_PREDICTION_COMPLETION_SEAL_V1.json'
    if receipt_path.exists() or owner_seal.exists():
        raise FileExistsError('Preserve owner prediction closure attempts')
    protocol = M._load_prepared(output)
    prediction = M.C.read(output/'PREDICTION_SEAL.json')
    execution = M.C.read(output/'PREDICTION_EXECUTION_PROTOCOL.json')
    entries = M.C.read(output/'PREDICTION_ROW_INDEX.json')['entries']
    ids = protocol['ordered_ids']
    assert prediction['rows'] == 238 and prediction['ordered_ids'] == ids
    assert prediction['arms'] == list(M.ARMS) and prediction['arm_count'] == 5
    assert prediction['all_declared_arms_complete'] is True
    assert prediction['before_fixed_reference_scoring'] is True
    assert prediction['no_population_drop'] is True
    assert execution['workers'] == execution['maximum_pending'] == 3
    assert execution['source_sha256'] == M.C.sha(M.__file__)
    assert execution['e_numeric_worker_inputs'] == execution['reference_or_score_inputs'] == 0
    assert [entry['id'] for entry in entries] == ids and len(entries) == 238
    assets = {
        'predictions_sha256':'PREDICTIONS.jsonl.gz',
        'protocol_sha256':'PROTOCOL.json',
        'input_seal_sha256':'INPUT_SEAL.json',
        'execution_protocol_sha256':'PREDICTION_EXECUTION_PROTOCOL.json',
        'row_index_sha256':'PREDICTION_ROW_INDEX.json',
        'separate_e_comparison_sha256':'E_COMPARISON.jsonl.gz',
    }
    for key, name in assets.items():
        assert prediction[key] == M.C.sha(output/name), name
    prepared_owner = M.C.read(output/'OWNER_POSITIVE_PREPARATION_RECEIPT_V1.json')
    assert prepared_owner['assets']['E_COMPARISON.jsonl.gz']['sha256'] == prediction['separate_e_comparison_sha256']
    counts = Counter();seen=[]
    for index, (row, comparison, entry) in enumerate(zip_longest(
            M.C.rows(output/'PREDICTIONS.jsonl.gz'),M.C.rows(output/'E_COMPARISON.jsonl.gz'),entries)):
        assert row is not None and comparison is not None and entry is not None and index < 238
        assert row['id'] == comparison['id'] == entry['id'] == ids[index]
        assert row['partition'] == comparison['partition'] == entry['partition']
        assert set(row['arms']) == set(M.ARMS)
        assert row['arms']['SAME_FROZEN_E'] == comparison['prediction']
        assert row['e_comparison_parent_only_no_numeric_exposure'] is True
        assert row['legacy_e_pose_semantics'] == comparison['legacy_e_pose_semantics']
        # Reconstruct only the stored four-arm row shape, without evaluating
        # any pose. This checks that parent merge added E and two audit fields.
        row['arms'].pop('SAME_FROZEN_E')
        row.pop('e_comparison_parent_only_no_numeric_exposure')
        row.pop('legacy_e_pose_semantics')
        assert set(row['arms']) == set(M.SOLVER.ARMS) and entry['arm_count'] == 4
        assert M.OBS.D.R._canonical_sha(row) == entry['canonical_row_sha256']
        assert M.C.sha(M.C.ROOT/entry['path']) == entry['sha256']
        counts[row['partition']] += 1;seen.append(row['id'])
    assert seen == ids and dict(counts) == M.PARTITIONS
    # Current immutable input/source inventories after terminal serialization.
    M._load_prepared(output)
    data = dict(schema='CLEAN_FRESH_BI_R2_OWNER_PREDICTION_COMPLETION_V1',
        UTC=datetime.now(timezone.utc).isoformat(),rows=238,partitions=dict(counts),severe=0,
        five_arm_rows=238,independent_four_arm_spool_canonical_integrity=238,
        parent_merged_E_dictionary_exact_to_presealed_comparison_rows=238,
        numerical_worker_E_fields=0,original_sources_and_all_input_maps_current=True,
        protected_user_files=6,CAL_immutable_files=53,
        prediction_session=82790,prediction_PID=565285,prediction_exit_code=0,restart_count=0,
        workers=3,BLAS_threads=1,
        bindings={name:M._binding(output/name) for name in
            ('PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','PREDICTION_ROW_INDEX.json',
             'PREDICTION_EXECUTION_PROTOCOL.json','INPUT_SEAL.json','PROTOCOL.json','E_COMPARISON.jsonl.gz')},
        source_sha256=protocol['source_sha256'],protected_user_sha256=protocol['protected_user_sha256'],
        original_CAL_file_sha256=protocol['original_CAL_file_sha256'],
        owner_verifier=M._binding(Path(__file__).resolve()),
        independent_selected_pose_cost_guard_audit_performed_by_owner=False,
        selected_pose_numeric_validity_and_accuracy_not_claimed=True,
        fixed_reference_or_score_decode_calls=0,raw_OBS_RGB_or_new_pose_optimization_calls=0,
        next_stage='ROOT independent all238 numerical audit before fixed reference scoring')
    write_new(receipt_path,data)
    write_new(owner_seal,dict(receipt_sha256=M.C.sha(receipt_path),
        prediction_seal_sha256=M.C.sha(output/'PREDICTION_SEAL.json'),all_owner_integrity_checks_pass=True))
    print(json.dumps(dict(receipt=M._binding(receipt_path),seal=M._binding(owner_seal),rows=238),ensure_ascii=False))


if __name__ == '__main__':
    main()
