"""Integrated V3 tests: invented pixels/ledger and extractor stubs only."""
from pathlib import Path
import copy
import hashlib
import json

import numpy as np

import integrated_same_q_v3 as v3

ROOT=Path(__file__).resolve().parent
P=ROOT.parents[2]
CROP=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
CROP_SHA='802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66'


def bind(p):
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}


def decision(row,state='D',eligible=True):
    return dict(query_key=[row['input_id'],row['native_slot_logical_only']],input_id=row['input_id'],
                frame_index=row['frame_index'],native_slot=row['native_slot_logical_only'],split=row['split'],
                image_sha256=row['image_sha256'],original_q=row['query_xy'],original_L=row['L'],
                before_aux_state=state,support_bbox_xyxy=[1.5,29.5,29.5,32.5]if state=='D'else None,
                uncertainty_bbox_xyxy=[1.5,29.5,29.5,32.5]if state=='D'else None,
                paired_construction_eligible=bool(eligible and state=='D'),
                native_expected_part_state='UNKNOWN',human_gold=False,training_admitted=False)


def run():
    crop,crop_ast=v3.v1.frozen_rgb96_operator(CROP,CROP_SHA)
    row=dict(slot_index=0,frame_index=0,input_id='INVENTED',native_slot_logical_only=0,
             split='TRAIN_LAB',image_sha256='a'*64,raw_hw=[96,96],query_xy=[32.,32.],L=64.,
             selected_box_xyxy=[0.,0.,70.,70.],original_input_valid=True)
    old_row=copy.deepcopy(row);rootdecision=decision(row)
    target=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3)
    donor=np.arange(32*60*3,dtype=np.uint8).reshape(32,60,3)
    vis=np.zeros((32,60),bool);vis[4:26,4:56]=True
    dm=dict(id='ONE_ASSIGNED_DONOR',split='TRAIN',object_id=3,image_sha256='b'*64,foreign_opaque_cutout_admitted=True)
    before=np.zeros((65,9,9),np.float16);before[64]=1
    feature_calls=[]
    def generated_extractor(rgb,q,L):
        feature_calls.append((rgb.copy(),q.copy(),L))
        result=np.full((65,9,9),float(np.mean(rgb)),np.float32);result[64]=1
        return result
    material=v3.materialize_row(target,row,rootdecision,donor,vis,dm,crop,before,generated_extractor)
    assert material['pair_ready']and row==old_row and len(feature_calls)==2
    assert [x['gt_only']['candidate_aux_hidden_label']for x in material['occurrences']]==[0,1,0]
    cert=material['gt_edit_certificate_only'];assert cert['donor_count']==1 and cert['same_donor_RGBA_transform']
    assert cert['mass_recheck']['PASS']and cert['mass_recheck']['native_global_alpha_cells']==cert['mass_recheck']['native_in_crop_alpha_cells']
    for o in material['occurrences']:
        assert o['input_only']['query_xy']==row['query_xy']and o['input_only']['L']==row['L']
        assert o['input_only']['split']=='TRAIN_LAB'and o['input_only']['selected_box_xyxy']==row['selected_box_xyxy']
        assert set(o['head_inputs_only'])=={'p3_roi','local_rgb96','local_valid96'}
        assert o['gt_only']['native_expected_part_state']=='UNKNOWN'and not o['gt_only']['training_admitted']
        expected,valid=crop(o['full_rgb'],np.array(row['query_xy']),row['L'],96)
        assert np.array_equal(expected,o['head_inputs_only']['local_rgb96'])and np.array_equal(valid,o['head_inputs_only']['local_valid96'])
    assert np.array_equal(material['occurrences'][0]['head_inputs_only']['p3_roi'],before)
    for call,occ in zip(feature_calls,material['occurrences'][1:]):
        assert np.array_equal(call[0],occ['full_rgb'])and np.array_equal(call[1],row['query_xy'])and call[2]==row['L']
    assert not np.array_equal(material['occurrences'][1]['full_rgb'],target)

    # Geometry/mass failure cannot drop a variant or relabel root D-before.
    huge=decision(row);huge['support_bbox_xyxy']=huge['uncertainty_bbox_xyxy']=[1.,1.,64.,64.]
    bad_donor=dict(dm,object_id=2)
    unusable=decision(row);unusable['support_bbox_xyxy']=None
    for d,metadata in [(huge,dm),(rootdecision,bad_donor),(unusable,dm)]:
        result=v3.prepare_occurrences(target,row,d,donor,vis,metadata,crop)
        assert len(result['occurrences'])==3 and not result['pair_ready']
        assert [x['gt_only']['candidate_aux_hidden_label']for x in result['occurrences']]==[0,-1,-1]
        assert all(np.array_equal(x['full_rgb'],target)for x in result['occurrences'])
    original_mass=v3.v2.mass_certificate
    try:
        v3.v2.mass_certificate=lambda *args:(False,{'PASS':False,'reason':'GENERATED_FORCED_MASS_FAILURE'})
        result=v3.prepare_occurrences(target,row,rootdecision,donor,vis,dm,crop)
        assert not result['pair_ready']and [x['gt_only']['candidate_aux_hidden_label']for x in result['occurrences']]==[0,-1,-1]
    finally:
        v3.v2.mass_certificate=original_mass
    unknown=v3.prepare_occurrences(target,row,decision(row,'U'),None,None,None,crop)
    assert all(x['gt_only']['candidate_aux_hidden_label']==-1 for x in unknown['occurrences'])
    assert all(np.array_equal(x['full_rgb'],target)for x in unknown['occurrences'])
    # Native pixel-edge bbox conversion at exact half-integer boundaries.
    cells=v3.pixel_edge_box_cells([.5,.5,1.5,1.5],[4,4]);expected=np.zeros((4,4),bool);expected[1,1]=True
    assert np.array_equal(cells,expected)
    assert v3.pixel_edge_box_cells([-.6,0,1,1],[4,4])is None
    # Absent q only gets zero placeholders; finite centre-invalid keeps old pixels.
    missing=dict(row,query_xy=None,L=None,original_input_valid=False)
    n_before=len(feature_calls)
    m=v3.materialize_row(target,missing,decision(missing,'U'),None,None,None,crop,before,generated_extractor)
    assert len(feature_calls)==n_before and all(not x['rgb96'].any()for x in m['occurrences'])
    finite=dict(row,query_xy=[-.1,32.],original_input_valid=False)
    f=v3.materialize_row(target,finite,decision(finite,'U'),None,None,None,crop,before,generated_extractor)
    oldcrop,oldvalid=crop(target,np.array(finite['query_xy']),finite['L'],96)
    assert oldcrop.any()and oldvalid.any()
    assert all(np.array_equal(x['rgb96'],oldcrop)and np.array_equal(x['crop_valid96'],oldvalid)for x in f['occurrences'])
    # Whole cohort is preserved, even when most decisions are U or q is absent.
    rows=[];ledger=[]
    for i in range(160):
        r=dict(row,slot_index=i,frame_index=i//8,input_id=f'INVENTED_FRAME_{i//8}',native_slot_logical_only=i%8,
               split='TRAIN'if i<96 else'VALIDATION',original_input_valid=i<122)
        rows.append(r);ledger.append(decision(r,'D'if i in [0,1,2,96,97,98]else'U'))
    plan=v3.all_row_occurrence_plan(rows,ledger)
    assert len(plan)==480 and sum(x['root_before_aux_state_GT_ONLY']=='D'for x in plan)==18
    assert [x['slot_index']for x in plan[::3]]==list(range(160))
    bad=copy.deepcopy(ledger);bad[0]['original_q']=[33.,32.]
    try:v3.all_row_occurrence_plan(rows,bad)
    except AssertionError:pass
    else:raise AssertionError('Changed original q accepted')
    return dict(completed=True,PASS=True,code=bind(ROOT/'integrated_same_q_v3.py'),generated_code=bind(Path(__file__)),
       V1_bound_unchanged=bind(v3.v1.__file__ and Path(v3.v1.__file__)),V2_bound_unchanged=bind(v3.V2_CODE),crop_AST_sha256=crop_ast,
       integrated_V2_scale_and_mass_pair_success=True,integrated_mass_failure_both_afterU=True,
       one_donor_three_variants_all_rows_preserved=True,after_full_RGB_P3_callback_same_q_L_no_reselection=True,
       GT_ledger_not_forward_and_only_three_head_tensors=True,root_D_before_not_relabelled_by_failed_edit=True,
       input_flag_operator_support_and_U_separate=True,native_pixel_edge_halfopen_cells_checked=True,
       original_split_q_L_slot_candidate_preserved=True,GT_nearest_or_best_query_selection=False,
       actual_source_annotation_pixel_array_checkpoint_model_current_reads=0,actual_forward_fit_PnP=0,
       source_training_admission=False)


if __name__=='__main__':
    result=run();out=ROOT/'GENERATED_INTEGRATED_V3_TESTS.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
