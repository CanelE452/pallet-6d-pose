"""GT-only label join for the immutable selected5120 input pack.

No feature/RGB/model/solver access. Input selection and q/L stay unchanged.
Primary validity is the EXISTING input pack validity, including nearest-pixel
bounds. Pure continuous validity is a separate diagnostic, not extra training.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / 'source_contract'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def verify(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']


def main():
    destination = HERE / 'SYNTH5120_GT_ONLY.npz'
    binding_path = HERE / 'GT_LABEL_BINDING.json'
    assert not destination.exists() and not binding_path.exists()
    selection_path = HERE / 'SOURCE_ROW_SELECTION_LOCK.json'
    lineage_path = HERE / 'SYNTH5120_INPUT_PACK/INDEX_LINEAGE_INPUT_ONLY.jsonl'
    input_seal_path = HERE / 'SYNTH5120_INPUT_PACK/PACK_SEAL.json'
    gt_seal_path = CONTRACT / 'BUILD_V2/TYPED_SOURCE_PACK_SEAL.json'
    source_gt_path = CONTRACT / 'BUILD_V2/TYPED_SOURCE_PACK_GT_ONLY.npz'
    coords_path = CONTRACT / 'TYPED_SOURCE_INPUT_COORDS_ONLY.npz'
    selection = json.loads(selection_path.read_text())
    input_seal = json.loads(input_seal_path.read_text())
    gt_seal = json.loads(gt_seal_path.read_text())
    assert input_seal['completed'] and input_seal['PASS']
    assert gt_seal['completed'] and gt_seal['PASS']
    verify(input_seal['lineage']); verify(gt_seal['GT_pack'])
    verify(gt_seal['input_coords_unchanged'])
    assert Path(input_seal['lineage']['path']).resolve() == lineage_path.resolve()
    assert Path(gt_seal['GT_pack']['path']).resolve() == source_gt_path.resolve()
    assert Path(gt_seal['input_coords_unchanged']['path']).resolve() == coords_path.resolve()
    records = selection['selected']
    lineage = [json.loads(x) for x in lineage_path.read_text().splitlines()]
    assert len(records) == len(lineage) == 5120
    index = np.asarray([r['cache_row'] for r in records], np.int32)
    assert len(set(index.tolist())) == len(index)
    with np.load(source_gt_path, allow_pickle=False) as g:
        ids = g['sample_id'][index]
        source = g['source'][index]
        partition = g['partition'][index]
        proxy = g['reference_proxy_state8'][index]
        matched = g['detector_matched_to_source_target'][index]
        reference = g['reference_projected8'][index]
        assert np.all(g['certified_semantic_state8'][index] == -1)
        assert np.all(g['target_pixel_support8'][index] == -1)
    with np.load(coords_path, allow_pickle=False) as c:
        query = c['initial_query9_raw'][index, :8]
        detected = c['detected'][index]
        continuous = c['query_native_valid9'][index, :8]
        nearest_with_continuous = c['query_nearest_valid9'][index, :8]
        raw_hw = c['raw_hw'][index]
        original_point_valid = c['initial_point_valid9'][index, :8]
    pack_valid = np.asarray([r['query_valid8'] for r in lineage], bool)
    side = np.asarray([r['raw_L'] for r in lineage], np.float64)
    assert (np.isfinite(side) & (side > 0)).all()
    for i, (s, r) in enumerate(zip(records, lineage)):
        assert r['frame_index'] == i and r['cache_row'] == index[i]
        assert s['id'] == r['id'] == ids[i]
        assert s['source'] == r['source'] == source[i]
        assert s['partition'] == r['partition'] == partition[i]
        assert bool(r['detected']) == bool(detected[i])
        np.testing.assert_array_equal(np.asarray(r['raw_q9_audit_only'])[:8], query[i])
    finite = np.isfinite(query).all(2) & original_point_valid & detected[:, None]
    h = raw_hw[:, 0, None]; w = raw_hw[:, 1, None]
    pure_continuous = finite & (query[:, :, 0] >= 0) & (query[:, :, 0] < w)
    pure_continuous &= (query[:, :, 1] >= 0) & (query[:, :, 1] < h)
    pixel = np.floor(np.nan_to_num(query, nan=-1e9) + .5)
    nearest_pixel = finite & (pixel[:, :, 0] >= 0) & (pixel[:, :, 0] < w)
    nearest_pixel &= (pixel[:, :, 1] >= 0) & (pixel[:, :, 1] < h)
    np.testing.assert_array_equal(pure_continuous, continuous)
    np.testing.assert_array_equal(pure_continuous & nearest_pixel, pack_valid)
    np.testing.assert_array_equal(pack_valid, nearest_with_continuous)
    des = np.isin(proxy, [0, 1, 2])
    primary = matched[:, None] & pack_valid & des
    continuous_diagnostic = matched[:, None] & pure_continuous & des
    assert not (primary & ((proxy >= 3) | ~matched[:, None])).any()
    # Vector points from the unchanged initializer query TO the GT reference.
    offset = (reference - query) / side[:, None, None]
    # Same physical local square; half-open right/bottom boundary, no loss veto.
    in_square = np.isfinite(offset).all(2) & (offset >= -.5).all(2) & (offset < .5).all(2)
    summary = {}
    for part, expected in [('train', 29353), ('heldout', 7358)]:
        rows = partition == part
        assert int(primary[rows].sum()) == expected
        summary[part] = {
            'frames': int(rows.sum()), 'slots': int(rows.sum() * 8),
            'matched_frames': int(matched[rows].sum()),
            'detected_frames': int(detected[rows].sum()),
            'pack_query_valid_slots': int(pack_valid[rows].sum()),
            'pure_continuous_query_valid_slots': int(pure_continuous[rows].sum()),
            'primary_DES_loss_slots': int(primary[rows].sum()),
            'continuous_DES_diagnostic_slots': int(continuous_diagnostic[rows].sum()),
            'continuous_only_DES_diagnostic_extra': int((continuous_diagnostic[rows] & ~primary[rows]).sum()),
            'primary_DES_by_state': {str(k): int(((proxy[rows] == k) & primary[rows]).sum()) for k in range(3)},
            'local_square_reference_inside_slots': int(in_square[rows].sum()),
            'primary_DES_reference_outside_local_square': int((primary[rows] & ~in_square[rows]).sum()),
        }
    np.savez_compressed(destination,
        frame_index=np.arange(5120, dtype=np.int32), cache_row=index,
        sample_id=ids, source=source, partition=partition,
        proxy_state8=proxy.astype(np.int8), matched=matched, detected=detected,
        query_valid=pack_valid, pure_continuous_query_valid=pure_continuous,
        nearest_pixel_valid=nearest_pixel, primary_DES_loss_mask=primary,
        continuous_DES_diagnostic_loss_mask=continuous_diagnostic,
        gt_projected8=reference,
        query_to_reference_offset_over_L8=offset,
        reference_in_local_square8=in_square,
        target_pixel_presence8=np.full((5120, 8), -1, np.int8),
        human_semantic_visibility8=np.full((5120, 8), -1, np.int8))
    result = {
        'schema': 'V74_SELECTED_SYNTH5120_GT_ONLY_BINDING_V1',
        'completed': True, 'PASS': True,
        'output': bound(destination), 'code': bound(Path(__file__)),
        'inputs': [bound(x) for x in [selection_path, lineage_path, input_seal_path,
                    gt_seal_path, source_gt_path, coords_path]],
        'ordering': 'Exact immutable selection/inputlineage frame_index0..5119; cache_row and sample_id both checked',
        'class_codes': {'0': 'D_CONVEX_UNBLOCKED_RAY_PROXY', '1': 'E_EXTERNAL_RAY_GE_HALF_PROXY',
                        '2': 'S_CONVEX_SELF_HIDDEN_PROXY', '3': 'F_REFERENCE_OUTSIDE_PROXY',
                        '4': 'U_MISSING_SOURCE_EVIDENCE'},
        'primary_DES_loss_mask': 'matched & pack_query_valid & (proxy_state8 in {0,1,2})',
        'pack_query_valid_actual_definition': 'finite originalinitialpoint & detected & 0<=rawq<W/H & floor(rawq+.5) pixelinbounds',
        'validity_wording_correction': 'Root initially wrote continuous validity while requesting29353/7358. Literalcontinuous adds20TRAIN+3heldout DESslots. Root fixed primary to existing sealedpack query_valid before this materialization; extra23 remain diagnostic only.',
        'F_U_unmatched_or_invalid_primary_loss': 0,
        'offset_definition_GT_only': '(gt_projected8 - original_saved_query8) / unchanged inputlineage raw_L',
        'reference_in_local_square_GT_only': 'Both offsetcoordinates >=-.5 and <.5; right/bottom half-open. Not a primary-loss filter.',
        'local_square_additional_filter_applied': False,
        'human_semantic_visibility8': '-1 UNKNOWN for all40960 slots; not copied from ray/convex proxy',
        'target_pixel_presence8': '-1 UNKNOWN for all40960 slots; separate from reference proxy',
        'forbidden_model_inputs': 'This entire GT_ONLY file, including referencecoords/offset/availability, is supervision/diagnostic only.',
        'counts': summary,
        'input_original_query_L_order_changed': False,
        'source_permutation_or_18exception_rows_changed': False,
        'old_source_or_BUILD_V2_artifacts_changed': False,
        'new_forward': 0, 'new_fit': 0, 'RGB_or_features_read': 0,
        'current_reads': 0, 'PnP': 0,
    }
    binding_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output': result['output'], 'counts': summary, 'binding_sha256': sha(binding_path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
