"""Compare original and common-anchor crops of the SAME saved BEFORE image.

Original q and L change together; this does not isolate translation vs size.
For C51 the SAM box can also change. No new model/pixel sampling is performed.
"""
from pathlib import Path
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def binding(path):
    path = Path(path).resolve()
    assert path.is_relative_to(PROJECT)
    assert not any(s in str(path).lower() for s in ('_v112', '_v177', '_v178', '_v169', 'current223', 'synthetic_val'))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(b):
    assert binding(b['path']) == b
    return Path(b['path'])


def load(b, keys):
    with np.load(checked(b), allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


def run():
    cp = HERE / 'frozen_common_query_heads_v1/COMPLETE.json'
    frozen = json.loads(cp.read_text())
    assert frozen['PASS'] and frozen['predictions_sealed_before_target_read']
    raw = json.loads(checked(frozen['common_input_complete']).read_text())
    rr = [json.loads(s) for s in checked(raw['metadata']).read_text().splitlines() if s]
    new_y = load(raw['targets'], ['support_target'])['support_target']
    old_meta = json.loads(checked(json.loads(checked(frozen['protocol']).read_text())['original_source_cache_metadata']).read_text())
    old_domain = old_meta['domains']['SYNTH_TRAIN4096']
    old_x = load(old_domain['inputs'], ['row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available'])
    old_y = load(old_domain['targets'], ['support_target'])['support_target']
    fit = json.loads(checked(frozen['frozen_V181_fit']).read_text())
    mapping = {int(g): i for i, g in enumerate(old_x['original_global_index'])}
    out = HERE / 'saved_crop_geometry_sensitivity_v1'
    assert not out.exists()
    out.mkdir()
    all_rows = []
    for arm, model in frozen['arms'].items():
        new = load(model['outputs'], ['raw_support_logits', 'support_prediction', 'original_query_available'])
        old_binding = fit['arms'][arm]['outputs']['SYNTH_TRAIN4096']
        old = load(old_binding, ['raw_support_logits', 'support_prediction', 'original_global_index', 'row_id'])
        assert np.array_equal(old['original_global_index'], old_x['original_global_index']) and np.array_equal(old['row_id'], old_x['row_id'])
        for i, r in enumerate(rr):
            if r['variant'] != 'BEFORE':
                continue
            j = mapping[r['source_image_original_global_index']]
            assert str(old_x['row_id'][j]) == r['source_image_original_query_id']
            common_available = bool(new['original_query_available'][i])
            original_available = bool(old_x['original_query_available'][j])
            q_old, q_new = r['image_original_query_xy'], r['query_xy']
            q_shift = None if q_old is None or q_new is None else float(np.linalg.norm(np.array(q_new) - np.array(q_old)))
            delta_l = None if r['image_original_L'] is None or r['L'] is None else r['L'] - r['image_original_L']
            both_known = bool((new_y[i] >= 0).all() and (old_y[j] >= 0).all())
            same_truth = both_known and np.array_equal(new_y[i], old_y[j])
            valid = common_available and original_available and same_truth
            all_rows.append(dict(arm=arm, base_index=r['base_index'], native_slot=r['native_slot'], diagnostic_row_index=i,
                source_image_original_global_index=r['source_image_original_global_index'], anchor_original_global_index=r['anchor_original_global_index'],
                same_saved_BEFORE_RGB_binding=r['source_RGB_binding'], original_q=q_old, common_q=q_new,
                q_shift_image_px=q_shift, original_L=r['image_original_L'], common_L=r['L'], delta_L_image_px=delta_l,
                original_available=original_available, common_available=common_available, both_support_truth_known_and_same=same_truth,
                original_truth=old_y[j].tolist(), common_truth=new_y[i].tolist(),
                original_prediction=old['support_prediction'][j].tolist(), common_prediction=new['support_prediction'][i].tolist(),
                original_correct=bool(np.array_equal(old['support_prediction'][j], old_y[j])) if valid else None,
                common_correct=bool(np.array_equal(new['support_prediction'][i], new_y[i])) if valid else None))
    summary = {}
    for arm in frozen['arms']:
        available_same_truth = [r for r in all_rows if r['arm'] == arm and r['original_correct'] is not None]
        lost = [r for r in available_same_truth if r['original_correct'] and not r['common_correct']]
        summary[arm] = dict(BEFORE_rows=1024, same_known_truth_and_both_available=len(available_same_truth),
            original_correct=sum(r['original_correct'] for r in available_same_truth),
            common_correct=sum(r['common_correct'] for r in available_same_truth),
            original_correct_became_wrong=len(lost),
            lost_with_q_shift_le1px=sum(r['q_shift_image_px'] is not None and r['q_shift_image_px'] <= 1 for r in lost),
            lost_with_q_shift_le3px=sum(r['q_shift_image_px'] is not None and r['q_shift_image_px'] <= 3 for r in lost),
            lost_cases=lost)
    lp = out / 'ALL2048_BEFORE_CROP_GEOMETRY_COMPARISONS.jsonl'
    lp.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in all_rows))
    result = dict(completed=True, PASS=True, code=binding(__file__), frozen_predictions=binding(cp), original_source_inputs=old_domain['inputs'], original_source_targets=old_domain['targets'], common_inputs=raw['inputs'], common_targets=raw['targets'],
        arms=summary, rows=binding(lp),
        interpretation='The same saved BEFORE image is sampled at original BEFORE q/L and original COVER q/L. Known support truth is unchanged for reported losses. C50 has no SAM channels: only crop geometry changes. C51 also changes the SAM prompt box/mask; effects are not separated. This is source TRAIN sensitivity, not heldout or real transfer evidence.',
        no_new_pixels_forward_fit_or_model_threshold_selection=True, fully_solved=False)
    (out / 'COMPLETE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({a: {k: v for k, v in z.items() if k != 'lost_cases'} for a, z in summary.items()}))


if __name__ == '__main__':
    run()
