"""Independent small generated checks; no source data or backbone loading."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'actual_pair_builder_preparation/build_changed_inputs.py'
spec=importlib.util.spec_from_file_location('independent_v95_driver',SRC)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

def main():
    _,math=d.helpers();rgb=np.zeros((80,100,3),np.uint8)
    row=dict(slot_index=0,frame_index=0,input_id='GENERATED',native_slot_logical_only=0,
        split='TRAIN',image_sha256='a'*64,raw_hw=[80,100],query_xy=[50.,40.],L=32.,
        selected_box_xyxy=[0,0,99,79],original_input_valid=True)
    crop,valid=math.crop(rgb,row['query_xy'],row['L'],96)
    base=dict(p3_roi=np.ones((65,9,9),np.float16),rgb96=crop,crop_valid96=valid)
    changed=rgb.copy();changed[0,0]=219;calls=[]
    def extract(image,q,L):
        calls.append((d.array_sha(image),q.copy(),L))
        assert d.array_sha(image)==d.array_sha(changed)
        p=np.ones((65,9,9),np.float16);p[:64]*=int(image.sum())
        return p,{'generated':True}
    occurrence=dict(full_rgb_changed=True,full_rgb_decoded_sha256=d.array_sha(changed))
    result,isnew,facts=d.materialize_one(row,occurrence,base,d.array_sha(rgb),lambda _:changed,extract,math)
    assert isnew and len(calls)==1 and facts['new_feature_forward']==1
    assert np.array_equal(result['rgb96'],base['rgb96']) and not np.array_equal(result['p3_roi'],base['p3_roi'])
    def forbidden(*args):raise AssertionError('Unchanged image must not load or extract')
    unchanged=dict(full_rgb_changed=False,full_rgb_decoded_sha256=d.array_sha(rgb))
    got,isnew,facts=d.materialize_one(row,unchanged,base,d.array_sha(rgb),forbidden,forbidden,math)
    assert got is base and not isnew and facts['new_feature_forward']==0
    bad=copy.deepcopy(occurrence);bad['full_rgb_decoded_sha256']='0'*64
    try:d.materialize_one(row,bad,base,d.array_sha(rgb),lambda _:changed,extract,math)
    except AssertionError:pass
    else:raise AssertionError('Decoded image SHA mismatch accepted')
    assert len(calls)==1

    rr=[];ll=[];oo=[];gg=[]
    for i in range(960):
        r=dict(row,slot_index=i,input_id=f'GENERATED_{i}')
        label=0 if i<143 else 1 if i<163 else -1
        rr.append(r);ll.append(dict(slot_index=i,input_id=r['input_id'],native_slot_logical_only=0,
            query_xy=r['query_xy'],L=r['L'],image_sha256=r['image_sha256'],
            before_input_refs={k:{} for k in d.FIELDS},decoded_RGB_sha256=d.array_sha(rgb)))
        gg.append(dict(base_row_index=i,original_natural_label_GT_only=label,before_aux_state='D' if label==0 else 'U'))
        for j,v in enumerate(d.VARIANTS):oo.append(dict(occurrence_index=3*i+j,slot_index=i,variant=v,
            occurrence_id=f"{r['input_id']}::slot0::{v}",full_rgb_changed=False,full_rgb_decoded_sha256=d.array_sha(rgb)))
    original=copy.deepcopy((rr,ll,oo,gg))
    summary=d.validate_metadata(rr,ll,oo,gg)
    assert summary['natural_before_counts']=={0:143,1:20,-1:797} and summary['controlled_D_eligible']==143
    assert summary['occurrences']==2880 and (rr,ll,oo,gg)==original
    proof=dict(completed=True,PASS=True,code=d.bind(SRC),all960_rows_2880_occurrences_preserved=True,
        natural_before_D143_E20_U797_immutable=True,changed_fullRGB_even_localcrop_identical_extracts_new_P3=True,
        full_decoded_SHA_checked_before_extractor=True,exact_unchanged_RGB_reuses_bound_triplet_without_forward=True,
        unchanged_q_L_native_slot_order=True,actual_source_arrays_RGB_GT_weights_model_current_fit_PnP=0)
    d.write(HERE/'INDEPENDENT_GENERATED_DRIVER_PROOF.json',proof);print(json.dumps(proof))

if __name__=='__main__':main()
