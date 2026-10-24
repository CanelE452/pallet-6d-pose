from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import protected_union_adapter as a


def run():
    row=dict(slot_index=0,frame_index=0,input_id='INVENTED',native_slot_logical_only=0,split='TRAIN_LAB',
             image_sha256='a'*64,raw_hw=[96,96],query_xy=[48.,48.],L=80.,selected_box_xyxy=[0.,0.,95.,95.],original_input_valid=True)
    d=dict(query_key=['INVENTED',0],input_id='INVENTED',frame_index=0,native_slot=0,split='TRAIN_LAB',image_sha256='a'*64,
           original_q=[48.,48.],original_L=80.,before_aux_state='D',paired_construction_eligible=True,source='PALLOX',
           native_expected_part_state='UNKNOWN',human_gold=False,null_unlocated_limit_count=2,
           protected_regions_GT_only=[dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[44.,44.,46.,46.]),dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[50.,44.,52.,46.])])
    original=copy.deepcopy(d);mask,reason=a.protected_union(row,d)
    assert mask.sum()==18 and mask[44,44] and mask[46,46] and mask[46,52] and not mask[45,48]
    assert d==original
    point=a.closed_range_cells([.5,.5,.5,.5],[4,4]);assert point.sum()==1 and point[1,1]
    native=np.zeros((96,96,3),np.uint8);donor=np.full((24,24,3),180,np.uint8);vis=np.ones((24,24),bool)
    meta=dict(id='GENERATED',split='TRAIN',object_id=3,image_sha256='b'*64,foreign_opaque_cutout_admitted=True)
    images,cutout,protected,facts=a.edit_geometry(native,row,d,donor,vis,meta)
    assert images is not None and facts['pair_ready'] and facts['protected_all_alpha1_cover'] and facts['protected_all_alpha0_sham']
    assert facts['mass_recheck']['PASS'] and np.array_equal(images[1][protected],native[protected])
    assert np.all(images[0][protected]==180)
    assert d==original
    return dict(completed=True,PASS=True,two_disjoint_boxes_union_not_outer_bbox=True,inclusive_endpoint_and_half_tie=True,
                unchanged_original_coordinates=True,full_union_cover_sham_certified=True,unchanged_V2_scale_placement_mass=True,
                actual_source_pixels_models_P3_fit_current_PnP=0,
                adapter={'path':a.__file__,'sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()})


if __name__=='__main__':
    out=Path(__file__).with_name('GENERATED_UNION_ADAPTER_TESTS.json');x=run()
    with out.open('x')as f:json.dump(x,f,indent=2);f.write('\n')
    print(json.dumps(x))
