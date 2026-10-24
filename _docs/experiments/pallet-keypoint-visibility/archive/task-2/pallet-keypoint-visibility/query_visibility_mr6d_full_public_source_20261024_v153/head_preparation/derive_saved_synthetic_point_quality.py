"""New separate point-quality labels from exact, saved synthetic GT only.

No features/checkpoints/RGB/private data; original visibility labels stay intact.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np

H = Path(__file__).resolve().parent
CONTRACT = H / 'QUALITY_LABEL_CONTRACT_BEFORE_SOURCE_ERROR_AUDIT.json'
CONTRACT_SHA = '4b059459b6805f661cf6f0bc721c102f60b952f547e2b47758a2e7ce8b5bfebe'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def main():
    assert sha(CONTRACT) == CONTRACT_SHA
    contract = json.loads(CONTRACT.read_text())
    out = H / 'saved_synthetic_point_quality_v1'
    out.mkdir(exist_ok=False)
    totals, allrows = {}, []
    for family, binding, expected in zip(('OLD6144', 'NEW20480'), contract['source_GT'], (6144, 20480)):
        assert sha(binding['path']) == binding['sha256']
        seen, records, counters = set(), [], defaultdict(Counter)
        with Path(binding['path']).open() as file:
            for line in file:
                row = json.loads(line)
                if row['convention'] != 'CF_PRIMARY_DECLARED':
                    continue
                assert row['query_id'] not in seen
                seen.add(row['query_id'])
                q, ref = row['query_xy'], row['reference_xy']
                error = row['error_px']
                finite = q is not None and ref is not None and error is not None
                finite = finite and np.isfinite(np.asarray(q, float)).all() and np.isfinite(np.asarray(ref, float)).all() and math.isfinite(error)
                if finite:
                    recalculated = float(np.linalg.norm(np.asarray(q, float) - np.asarray(ref, float)))
                    assert math.isclose(recalculated, error, rel_tol=1e-12, abs_tol=1e-10)
                available = bool(row['nearest_inbounds'])
                label, reason = None, None
                if not available:
                    reason = 'INPUT_UNAVAILABLE'
                elif not finite:
                    reason = 'NO_FINITE_SAME_NATIVE_REFERENCE'
                elif row['correspondence_boundary']:
                    reason = 'CORRESPONDENCE_BOUNDARY'
                elif row['teacher_state'] not in ('D', 'E'):
                    reason = 'ROLE_SELF_OOF_OR_UNKNOWN_OUTSIDE_EXTERNAL_TASK'
                elif error <= 3:
                    label, reason = 1, 'EXACT_SYNTHETIC_SAME_NATIVE_ERROR_LE3'
                elif error > 20:
                    label, reason = 0, 'EXACT_SYNTHETIC_SAME_NATIVE_ERROR_GT20'
                else:
                    reason = 'INTERMEDIATE_ERROR_UNLABELED_QUALITY_ONLY'
                record = dict(family=family, global_index=len(records), query_id=row['query_id'],
                    native_slot=row['native_slot'], split=row['split'], variant=row['variant'],
                    original_query_xy=q, original_roi_side=row['roi_side'], reference_xy_exact_synthetic=ref,
                    original_role_teacher_state_UNCHANGED=row['teacher_state'],
                    correspondence_boundary=row['correspondence_boundary'],
                    nearest_input_available=available, exact_synthetic_error_px=error,
                    original_position_band_UNCHANGED=row['position_band'],
                    point_quality_label=label, point_quality_reason=reason,
                    role_visibility_training_mask_UNCHANGED=True, coordinates_corrected=False,
                    actual_real_pixel_gold=False, source_only=True)
                records.append(record)
                for key in ('ALL', row['split'], row['split'] + '::' + row['teacher_state'],
                            row['split'] + '::' + row['factors']['aspect'],
                            row['split'] + '::' + row['factors']['material']):
                    counters[key]['accurate1' if label == 1 else 'inaccurate0' if label == 0 else 'unlabeled'] += 1
                counters['UNLABELED_REASONS'][reason] += label is None
        assert len(records) == expected
        path = out / f'{family}_ALL_ROWS_POINT_QUALITY_SEPARATE.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records))
        totals[family] = dict(source_GT=binding, rows=expected, labels=bind(path),
            counts={k: dict(v) for k, v in counters.items()}, all_rows_preserved=True)
        allrows.extend(records)
    summary = dict(completed=True, contract=bind(CONTRACT), code=bind(__file__), source_only=True,
        total_rows=len(allrows), source_families=totals,
        all_counts=dict(Counter('accurate1' if r['point_quality_label'] == 1 else 'inaccurate0' if r['point_quality_label'] == 0 else 'unlabeled' for r in allrows)),
        all_original_queries_L_order_roles_and_visibility_masks_preserved=True,
        middle_error_rows_retained_no_visibility_loss_filter=True,
        real_MR6D_coordinate_accuracy_labels_created=0,
        source_feature_checkpoint_RGB_or_model_body_reads_or_calls=0,
        private_current223_PnP_or_training_calls=0)
    path = out / 'RESULTS.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(bind(path)))


if __name__ == '__main__':
    main()
