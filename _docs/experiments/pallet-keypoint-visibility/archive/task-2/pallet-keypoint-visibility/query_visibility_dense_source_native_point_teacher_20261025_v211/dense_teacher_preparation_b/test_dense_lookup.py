"""Invented metadata only; no GT/config/array/header/model reads."""
from pathlib import Path
import copy,hashlib,json
import lookup_dense_source_points as M

def run():
    checks=[]
    def ok(n,v):assert v,n;checks.append(n)
    def bad(n,f):
        try:f()
        except (AssertionError,KeyError,ValueError,TypeError):checks.append(n);return
        raise AssertionError(n)
    records=[dict(CF_index=k,canonical_index=(k+2)%8,native_canonical_index=(k+2)%8,target_witness_states=['D'],combined_witness_states=['E'if k==2 else'S'],
        state='U',target_alone_region_state='D',projected_reference_xy=[20.25,30.75])for k in range(8)]
    selected=dict(scene_index=1,base_index=0,variant='COVER',native_rows=[dict(native_slot=k,declared_CF_to_canonical=(k+2)%8)for k in range(8)])
    scene=dict(scene_index=1,base_index=0,variant='COVER',records_CF=records,permCF_to_canonical=[(k+2)%8 for k in range(8)])
    doc=dict(rows=[dict(scene_index=0,base_index=0,variant='BEFORE'),scene])
    rr,reason=M.select_scene(doc,selected)
    ok('Ownscene_exact_variant_scene_native_perm',reason is None and rr[2]['CF_index']==2 and rr[2]['canonical_index']==4)
    result=M.extract(rr[2]);ok('Point_target_combined_region_distinct',result['point0_target_alone_state']=='D'and result['point0_combined_state']=='E'and result['region_state_context_only']=='U')
    absent=copy.deepcopy(doc);del absent['rows'][1]['records_CF']
    ok('Missing_records_UNKNOWN_no_nearest_fallback',M.select_scene(absent,selected)[0]is None)
    absent['rows'][1]['nearest_pixel_diagnostic_CF']=records
    ok('Nearest_diagnostic_not_point_authority',M.select_scene(absent,selected)[0]is None)
    conflict=copy.deepcopy(doc);conflict['rows'][1]['records_CF'][0]['canonical_index']=7
    bad('Canonical_conflict_rejected',lambda:M.select_scene(conflict,selected))
    duplicate=copy.deepcopy(doc);duplicate['rows'].append(scene)
    bad('Ambiguous_ownscene_rejected',lambda:M.select_scene(duplicate,selected))
    ok('Missing_state_reference_remain_UNKNOWN',M.extract({})['point0_combined_state']=='UNKNOWN'and M.extract({})['saved_projected_reference_xy']is None)
    loc=M.locality([20.,30.],96,False,[20.,30.],[100,100])
    ok('Finite_q_cannot_promote_input_U',loc['raw96_reference_index_xy']==[47.5,47.5]and loc['original_input_U_preserved'])
    edge=M.locality([0.,0.],96,True,[48.,0.],[100,100])
    ok('Samplecenters_and_halfopen_canvas_boundary_distinct',edge['raw96_reference_index_xy'][0]==95.5 and not edge['inside_crop_canvas_extent'])
    ok('Missing_q_does_not_recenter',M.locality(None,None,False,[20.,30.],[100,100])['raw96_reference_index_xy']is None)
    smap={}
    for si,var in [(0,'BEFORE'),(1,'COVER')]:
        native=[dict(original_global_index=si*8+k,original_occurrence_id=f'scene{si}:native{k}',original_query_xy=[10.+si+k,20.],original_L=96.,original_input_flags={'query_available':True},declared_CF_to_canonical=k)for k in range(8)]
        smap[si]=dict(scene_index=si,base_index=0,variant=var,original_input_metadata8=native,source_RGB_binding={'path':'inert','sha256':'a'*64})
    k=2;anchor=smap[0]['original_input_metadata8'][k];own=smap[1]['original_input_metadata8'][k]
    row=dict(row_id='invented_BEFORE_ANCHOR_COVER',original_global_index=61999,native_slot=k,source_image_original_global_index=10,anchor_original_global_index=2,
        scene_index=1,base_index=0,variant='COVER',anchor_scene_index=0,query_xy=anchor['original_query_xy'],L=anchor['original_L'],original_query_available=True,
        source_image_original_query_id=own['original_occurrence_id'],anchor_original_query_id=anchor['original_occurrence_id'],original_flags_ref=[0,k],image_flags_ref=[1,k],family='BEFORE_ANCHOR4',state_name='NOMINAL',state_index=0)
    unchanged=copy.deepcopy(row);mapping=M.map_input(row,smap)
    ok('BEFORE_anchor_COVER_image_gets_COVER_teacher',mapping['source_teacher_key']==[1,k]and mapping['anchor_scene_index']==0 and mapping['actual_image_variant']=='COVER'and row==unchanged)
    wrong=copy.deepcopy(row);wrong['query_xy']=[999.,999.]
    bad('Original_anchor_q_drift_rejected',lambda:M.map_input(wrong,smap))
    missing=copy.deepcopy(row);del missing['source_image_original_global_index']
    ok('Missing_image_identity_retained_UNKNOWN_no_cache_alias',M.map_input(missing,smap)['source_teacher_key']is None)
    selfrow=copy.deepcopy(row);selfrow.update(family='SELF_SCENE15',anchor_original_global_index=10,anchor_scene_index=1,query_xy=own['original_query_xy'],L=own['original_L'],anchor_original_query_id=own['original_occurrence_id'],original_flags_ref=[1,k])
    ok('SELF_nominal_ownscene_q_and_image_match',M.map_input(selfrow,smap)['source_teacher_key']==[1,k])
    bad('Original_GT_blocked_before_actual_gate',lambda:M.raw(M.P/'not_opened/SCENES_GT_ONLY.json'))
    bad('Binary_path_rejected_before_stat',lambda:M.raw(M.P/'not_opened/input.npz'))
    old_get,old_bind=M.get,M.bind
    try:
        M.bind=lambda p:{};M.get=lambda b:dict(approved_SOURCE_TRAIN_dense_saved_point_lookup=False,new_training_admission=False)
        bad('False_root_entry_rejected_before_GT',lambda:M.run(Path('/not_opened.json')))
    finally:M.get,M.bind=old_get,old_bind
    assert not M.GT_ALLOWED
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_GT_config_arrays_RGB_weights_headers_read=0)

if __name__=='__main__':
    result=run();path=Path(__file__).parent/'INVENTED_CHECKS.json'
    with path.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,checks=result['count'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
