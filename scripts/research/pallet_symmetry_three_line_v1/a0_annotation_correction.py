"""Separate authoritative-target rescore; retain all historical A0 artifacts."""
import csv, math
import numpy as np
import env as E
from audit_math import symmetry_error
from a0_square import stats


def main():
    view_path = E.RAW / 'A/square_annotation_target_view.json'
    view = E.read(view_path)
    targets = {r['id']: r for r in view['records']}
    membership = E.read(E.RAW / 'A/square_membership.json')['val']['records']
    perms = np.array(E.read(E.DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations'])
    arms = ['F0', 'F1', 'F2', 'clean_label/stage1_indexed', 'clean_label/stage2_c4']
    summaries, rows = {}, []
    for arm in arms:
        pred_path = E.RAW / f"A/A0/{arm.replace('/', '_')}_predictions.json"
        payload = E.read(pred_path)
        assert len(payload['records']) == len(membership)
        arm_rows, pooled = [], []
        for pred_record, member in zip(payload['records'], membership):
            assert pred_record['id'] == member['id']
            target = np.array(targets[member['id']]['yolo_target'])[0, 5:].reshape(9, 3)
            image_hw = np.array(member['raw_hw']) + 200
            gt = target[:, :2] * image_hw[::-1] - 100
            valid = target[:, 2] > 0
            prediction = (np.array(pred_record['keypoints'][0])[:, :2] - 100
                          if pred_record['keypoints'] else np.full((9, 2), np.nan))
            diagonal = math.hypot(*member['raw_hw'])
            score = symmetry_error(prediction, gt, valid, perms, diagonal)
            assert score['evaluable']
            row = dict(arm=arm, id=member['id'], E_sym=score['equivalent_mean']/diagonal,
                       E_fixed=score['fixed_mean']/diagonal, sym_mean_px=score['equivalent_mean'],
                       branch=score['branch'], annotated_corners=score['n_corners'],
                       source_target_changed=targets[member['id']]['changed_from_prepared'],
                       mask_noninvariant=bool(np.any(valid[perms] != valid)))
            arm_rows.append(row)
            pooled.extend(score['errors'][:8][score['gt_mask'][:8]].tolist())
        rows.extend(arm_rows)
        summaries[arm] = dict(frames=len(arm_rows),
            primary_E_sym=stats([r['E_sym'] for r in arm_rows]),
            frame_corner_mean_px=stats([r['sym_mean_px'] for r in arm_rows]),
            pooled_corner_px=stats(pooled),
            mask_noninvariant_frames=sum(r['mask_noninvariant'] for r in arm_rows),
            prediction_binding=E.bound(pred_path))
    changed = [r for r in view['records'] if r['changed_from_prepared']]
    E.freeze(E.DOC / 'A/ANNOTATION_TARGET_AMENDMENT.json', dict(
        target_view=E.bound(view_path), input_records=len(view['records']), changed_records=changed,
        source='authoritative objects[0].keypoint_annotations; existing preparation pure functions',
        reason='Historical prepared labels marked padded absent-point sentinels as visible.',
        original_sources_modified=False, original_A0_preserved=True,
        future_INDEXED_and_EQUIV_must_share_same_view=True,
        fixes_whole_tuple_and_resulting_bbox=True, fits_on_old_or_new_view=0,
        checkpoint_selection_changed=False, new_inference_calls=0,
        note='All five checkpoints now rescored against one corrected clean target view. Historical v4/v6 scores remain separately available; target changes are not model improvements.'))
    E.write(E.DOC / 'A/A0_ANNOTATION_CORRECTED.json', dict(
        status='COMPLETE_TARGET_CORRECTION_ONLY', arms=summaries,
        reused_prediction_rescores=len(rows), actual_new_inferences=0,
        main_training_updates=0, original_historical_results_preserved=True,
        target_binding=E.bound(view_path), not_evidence_of_training_improvement=True))
    with (E.DOC / 'A/annotation_corrected_per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    print('A0 corrected annotation rescores', len(rows), 'changed targets', len(changed), flush=True)


if __name__ == '__main__':
    main()
