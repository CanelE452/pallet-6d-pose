"""Invented arrays/metadata only; no actual source image or mask reader."""
from pathlib import Path
import copy,json
import numpy as np
import query_anchored_pair_kernel as k

HERE=Path(__file__).resolve().parent
def run():
    row=dict(slot_index=0,frame_index=0,input_id='INVENTED',native_slot_logical_only=0,split='TRAIN_LAB',image_sha256='a'*64,
        raw_hw=[96,96],query_xy=[48.,48.],L=80.,selected_box_xyxy=[0,0,95,95],original_input_valid=True)
    target=dict(base_row_index=0,original_q=[48.,48.],original_L=80.,query_anchored_label=0,matched_certified_group='ONE_CERTIFIED_GROUP',source_family='GENERATED',
        central4_native_stencil=[list(p)for p in k.central_stencil([48.,48.],80.)],original_regions=[
            dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[44.,44.,50.,50.]),
            dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[55.,44.,57.,46.])])
    frozen=copy.deepcopy((row,target));protected,reason=k.protected_mask(row,target)
    assert protected[44,44]and protected[50,50]and protected[46,57]and not protected[45,52]
    assert all(protected[y,x]for x,y in k.central_stencil([48.,48.],80.))
    native=np.full((96,96,3),13,np.uint8);donor=np.full((24,24,3),219,np.uint8);selected=np.ones((24,24),bool);other=np.zeros_like(selected)
    meta=dict(id='GENERATED',split='TRAIN',scene_id=0,object_id=3,image_sha256='b'*64,foreign_opaque_cutout_admitted=True)
    pair=k.prepare_rank(native,row,target,donor,selected,other,True,meta)
    assert pair['facts']['pair_ready']and pair['after_labels']==(1,0)and pair['facts']['mass_recheck']['PASS']
    assert np.all(pair['images'][0][protected]==219)and np.all(pair['images'][1][protected]==13)
    assert (row,target)==frozen
    incomplete=k.prepare_rank(native,row,target,donor,selected,other,False,meta)
    assert incomplete['after_labels']==(-1,-1)and not incomplete['facts']['pair_ready']
    assert all(np.array_equal(x,native)for x in incomplete['images'])
    big=copy.deepcopy(target);big['original_regions']=[dict(kind='LOCO_HALFOPEN_SUPPORT',bbox=[8.,8.,88.,88.])]
    failed=k.prepare_rank(native,row,big,donor,selected,other,True,meta)
    assert failed['after_labels']==(-1,-1)and not failed['facts']['pair_ready']and failed['facts']['protected_pixels']==6561
    assert failed['facts']['protected_area_over_L_squared']==6561/6400 and big['original_regions'][0]['bbox']==[8.,8.,88.,88.]
    naturalE=copy.deepcopy(target);naturalE['query_anchored_label']=1
    assert k.prepare_rank(native,row,naturalE,donor,selected,other,True,meta)['after_labels']==(-1,-1)

    donors=[dict(id=f'D{i:03d}',split='TRAIN'if i%2==0 else'VALIDATION',scene_id=0 if i%2==0 else 8,
        object_id=3 if i%2==0 else 4,image_sha256=f'{i:064x}',foreign_opaque_cutout_admitted=False)for i in range(50)]
    got=k.fixed_donors(row,donors);assert len(got)==len({d['id']for d in got})==16
    assert [d['id']for d in got]==[d['id']for d in k.fixed_donors(row,list(reversed(donors)))]
    assert all(d['scene_id']<=7 and d['object_id']%2==1 for d in got)
    val=dict(row,split='VAL_HALL');assert all(d['scene_id']>=8 and d['object_id']%2==0 for d in k.fixed_donors(val,donors))
    assert all(d['foreign_opaque_cutout_admitted']is False for d in got)  # rank is not a success/opacity filter
    rr=[dict(row,slot_index=i,input_id=f'INVENTED_{i}')for i in range(960)]
    tt=[dict(target,base_row_index=i,query_anchored_label=0 if i<61 else 1 if i<65 else -1)for i in range(960)]
    plan=k.occurrence_plan(rr,tt);assert len(plan)==31680
    for i in range(960):
        part=plan[33*i:33*i+33];assert part[0]['variant']=='BEFORE'
        assert [p['donor_rank']for p in part[1:]]==[j for j in range(16)for _ in range(2)]
        assert all(p['base_row_index']==i for p in part)
    return dict(completed=True,PASS=True,all960x33_31680_occurrences=True,fixed16_distinct_minSHA_order_invariant=True,
        TRAIN_VAL_scene_object_family_disjoint=True,no_rank_quality_or_success_filter=True,
        all_matched_group_original_disjoint_boxes_kept=True,exact_central4_native_stencil_protected=True,
        cover_all1_sham_all0_same_RGBA_fixed_mass=True,two_sided_failedU=True,large_LOCO_style_region_not_shrunk=True,
        protected_fraction_reported_on_failure=True,natural_E_before_only_no_control=True,
        original_q_L_unchanged=True,actual_source_RGB_mask_features_weights_models_fit_current_PnP=0)

if __name__=='__main__':
    r=run()
    with(HERE/'GENERATED_QUERY_ANCHORED_PAIR_TESTS.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))
