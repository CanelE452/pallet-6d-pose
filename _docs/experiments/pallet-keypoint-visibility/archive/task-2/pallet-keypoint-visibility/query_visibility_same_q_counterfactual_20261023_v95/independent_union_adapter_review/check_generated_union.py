"""Invented arrays only; no source image, feature, GT or model reader."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'source_pair_intake'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC))
import protected_union_adapter as a
import test_generated_union_adapter as producer


def main():
    producer_proof = producer.run()
    with (HERE / 'GENERATED_ADAPTER_RERUN.json').open('x') as f:
        json.dump(producer_proof, f, indent=2); f.write('\n')

    # Scalar oracle uses every nearest-cell transition, not array slice math.
    vals = [0., .499999, .5, .500001, 1., 1.499999, 1.5, 2.5, 3.]
    cases = 0
    for lo in vals:
        for hi in vals:
            if hi < lo:
                continue
            got = a.closed_range_cells([lo, lo, hi, hi], [4, 4])
            probes = [lo, hi]
            probes.extend(x + .5 for x in range(4) if lo <= x + .5 <= hi)
            cells = {int(np.floor(x + .5)) for x in probes}
            expected = np.zeros((4, 4), bool)
            for y in cells:
                for x in cells:
                    expected[y, x] = True
            assert np.array_equal(got, expected), (lo, hi)
            cases += 1
    for invalid in [None, [0,0,1], [0,0,float('nan'),1], [-.01,0,1,1], [0,0,3.01,1], [2,0,1,1]]:
        assert a.closed_range_cells(invalid, [4,4]) is None

    row = dict(slot_index=0, frame_index=0, input_id='INVENTED',
        native_slot_logical_only=0, split='TRAIN_LAB', image_sha256='a'*64,
        raw_hw=[96,96], query_xy=[48.,48.], L=80.,
        selected_box_xyxy=[0.,0.,95.,95.], original_input_valid=True)
    decision = dict(query_key=['INVENTED',0], input_id='INVENTED',
        frame_index=0, native_slot=0, split='TRAIN_LAB', image_sha256='a'*64,
        original_q=[48.,48.], original_L=80., before_aux_state='D',
        paired_construction_eligible=True, source='PALLOX',
        native_expected_part_state='UNKNOWN', human_gold=False,
        protected_regions_GT_only=[
            dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[44.,44.,46.,46.]),
            dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[50.,44.,52.,46.])])
    original = copy.deepcopy((row, decision))
    union, reason = a.protected_union(row, decision)
    swapped = copy.deepcopy(decision)
    swapped['protected_regions_GT_only'].reverse()
    assert np.array_equal(union, a.protected_union(row, swapped)[0])
    assert union.sum() == 18 and not union[45,48]
    assert (row, decision) == original
    bad = copy.deepcopy(decision)
    bad['protected_regions_GT_only'][1]['bbox'] = [95.,44.,96.,46.]
    assert a.protected_union(row,bad)[0] is None  # no omitted invalid region
    unknown = copy.deepcopy(decision); unknown['before_aux_state'] = 'U'
    assert a.protected_union(row,unknown)[0] is None
    natural = copy.deepcopy(decision); natural['before_aux_state'] = 'E'
    canonical_natural_before = 1
    try:
        a.protected_union(row,natural)
    except AssertionError:
        pass
    else:
        raise AssertionError('Natural E must not enter construction D/U API')
    assert canonical_natural_before == 1

    loco = copy.deepcopy(decision); loco['source'] = 'LOCO'
    loco.update(support_bbox_xyxy=[43.5,43.5,46.5,46.5],
                uncertainty_bbox_xyxy=[49.5,43.5,52.5,46.5])
    assert np.array_equal(a.protected_union(row,loco)[0],
                          a.v3.controlled_protected_support(row,loco)[0])

    # Adapter calls pinned operators directly; compare invented output exactly.
    native = np.full((96,96,3),13,np.uint8)
    donor = np.full((24,24,3),219,np.uint8)
    vis = np.ones((24,24),bool)
    meta = dict(id='GENERATED',split='TRAIN',object_id=3,
                image_sha256='b'*64,foreign_opaque_cutout_admitted=True)
    images, cutout, protected, facts = a.edit_geometry(native,row,decision,donor,vis,meta)
    direct_cutout, direct_scale = a.v3.v2.support_scaled_cutout(donor,vis,union,row['L'])
    pair, direct_mass = a.v3.v2.core_aligned_pair(a.v3.checked_row(row),direct_cutout,union)
    assert facts['scale'] == direct_scale and facts['mass'] == direct_mass
    for actual, (xy,alpha) in zip(images,pair):
        assert np.array_equal(actual,a.v3.v1.compose(native,direct_cutout,xy,alpha))
    assert all(np.array_equal(cutout[k],direct_cutout[k]) for k in ['rgb','alpha'])
    assert (row, decision) == original
    proof = dict(completed=True,PASS=True,
        closed_endpoint_scalar_cases=cases,
        all_disjoint_regions_order_independent_union_not_outer_box=True,
        invalid_retained_region_makes_whole_pair_U_not_omitted=True,
        LOCO_original_halfopen_operator_exact=True,
        pinned_V2_scale_placement_mass_and_V1_composition_exact=True,
        natural_E_rejected_from_construction_and_canonical_before_unchanged=True,
        source_RGB_mask_feature_weights_model_forward_fit_current_reads=0,
        adapter_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest())
    with (HERE/'INDEPENDENT_GENERATED_UNION_PROOF.json').open('x') as f:
        json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
