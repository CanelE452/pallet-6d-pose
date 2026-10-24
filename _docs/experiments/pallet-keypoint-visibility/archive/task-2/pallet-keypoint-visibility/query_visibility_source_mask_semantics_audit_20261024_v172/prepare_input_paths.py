"""V172 metadata/code/stat inventory only. No NPZ/image open, header read or hash."""
from pathlib import Path
import hashlib,json

V=Path(__file__).resolve().parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_SOURCE_MASK_OR_DEPTH_BODIES.json'
V129=P/'query_visibility_registered_native_diversity_20261023_v129'
PROTO=V129/'render_preparation/CORPUS_PROTOCOL_BEFORE_PIXELS.json'
def sha_text(p):
    p=Path(p);assert p.suffix in ('.json','.jsonl','.py')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def bind_text(p):return {'path':str(Path(p).resolve()),'sha256':sha_text(p)}
def checked_text(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v169','_v112','/private_evaluation/'))
    assert p.suffix in ('.json','.jsonl','.py') and sha_text(p)==b['sha256'];return p
def read(p):
    assert Path(p).suffix=='.json';return json.loads(Path(p).read_text())
def lines(p):
    assert Path(p).suffix=='.jsonl';return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def stat_inherited(b,kind):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v169','_v112','/private_evaluation/'))
    return dict(kind=kind,inherited_binding=b,path_exists=p.is_file(),file_bytes_stat=p.stat().st_size if p.is_file() else None,
                array_header_or_body_opened=False,array_or_image_SHA_reverified=False,status='PATH_PRESENT_SCHEMA_FROM_CODE_NOT_ARRAY_INSPECTED' if p.is_file() else 'UNKNOWN_PATH_ABSENT')
def main():
    scope=read(SCOPE);assert scope['fixed_rows']==[5353,5586,5609]
    selection=read(checked_text(scope['source_selection']));protocol=read(PROTO)
    codepaths=[protocol[k] for k in ('code','core_code','raster_code','geometry_code','reviewed_renderer_helper')]
    for b in codepaths:checked_text(b)
    # Schema is declared by static producer code, not inferred by peeking at NPZ headers.
    schema={
      'RGB_uint8':{'dtype':'uint8','shape':[512,512,3]},
      'camera_depth':{'dtype':'float64','shape':[512,512],'background':'NaN'},
      'part_id':{'dtype':'int32','shape':[512,512],'semantics':'target component IDs0..num_target_parts-1; foreign999; background-1; consult ambiguity for ties'},
      'global_triangle_id':{'dtype':'int32','shape':[512,512],'background':-1},
      'ambiguity':{'dtype':'bool','shape':[512,512]},
      'tie_count':{'dtype':'uint16','shape':[512,512]},
      'native8_CF_xy':{'dtype':'float64','shape':[8,2]},
      'native8_canonical_xy':{'dtype':'float64','shape':[8,2]},
      'permCF_to_canonical':{'dtype':'int32','shape':[8]},
      'state8_CF':{'dtype':'Unicode1','shape':[8]},
      'camera_K':{'dtype':'float64','shape':[3,3]},'camera_R':{'dtype':'float64','shape':[3,3]},'camera_t':{'dtype':'float64','shape':[3]},
      'witness_xy_CF':{'dtype':'float64','shape':[8,37,2]},
      'witness_nearest_xy_CF':{'dtype':'int32','shape':[8,37,2]},
      'witness_nearest_in_image_CF':{'dtype':'bool','shape':[8,37]},
      'witness_nearest_owner_CF':{'dtype':'int32','shape':[8,37]},
      'witness_nearest_triangle_CF':{'dtype':'int32','shape':[8,37]},
      'witness_nearest_depth_CF':{'dtype':'float64','shape':[8,37]},
      'witness_nearest_ambiguity_CF':{'dtype':'bool','shape':[8,37]},
      'witness_continuous_owner_CF':{'dtype':'int32','shape':[8,37]},
      'witness_continuous_depth_CF':{'dtype':'float64','shape':[8,37]},
      'witness_continuous_U_CF':{'dtype':'bool','shape':[8,37]}}
    meta_bindings=[bind_text(SCOPE),scope['source_selection'],bind_text(PROTO),*codepaths]
    # V132 original q/L metadata is already bound by V171 selection; same3 scenes/all8 slots only.
    qb=next(b for b in selection['bindings'] if b['path'].endswith('ORIGINAL_NATIVE_QUERY6144_INPUT_ONLY.jsonl'))
    queries=lines(checked_text(qb));meta_bindings.append(qb)
    rows=[]
    for witness in selection['unique_witnesses']:
        i=witness['global_index'];base=witness['base_index'];variant=witness['variant'];scene=witness['scene_index']
        directory=V129/'actual_corpus_v1/bases'/witness['base_id'];bcpath=directory/'BASE_COMPLETE.json';bc=read(bcpath)
        assert bc['PASS'] and bc['base_index']==base and bc['target_layers_before']==bc['target_layers_after']
        gt=read(checked_text(bc['GT_scenes']))['rows'];inp=read(checked_text(bc['input_scenes']))['rows']
        selected=next(x for x in gt if x['variant']==variant);before=next(x for x in gt if x['variant']=='BEFORE');inputrow=next(x for x in inp if x['variant']==variant)
        assert selected['scene_index']==scene and inputrow['raw_RGB_PNG']==witness['raw_RGB_PNG_binding_inherited']
        configbinding=bc['config']['file'];cfg=read(checked_text(configbinding))['rows'][base]
        assert cfg['base_index']==base and cfg['native8_CF_xy']==selected['native8_CF_xy'] and cfg['permCF_to_canonical']==selected['permCF_to_canonical']
        q8=sorted([q for q in queries if q['scene_index']==scene],key=lambda q:q['native_slot']);assert [q['native_slot'] for q in q8]==list(range(8))
        qselected=q8[witness['native_slot']];assert qselected['query_xy']==witness['query_xy'] and qselected['roi_side']==witness['L']
        projected_canonical=[None]*8
        for slot,canonical in enumerate(selected['permCF_to_canonical']):projected_canonical[canonical]=selected['native8_CF_xy'][slot]
        recordkeys=('CF_index','canonical_index','name','canonical_name','primary_part_id','incident_part_ids','projected_reference_xy','state','target_alone_region_state','geometric_S','front_face_axes','active_witness_indices','projected_active_witness_bbox_xyxy','projected_all3face_patch_bbox_xyxy','projected_support_scope')
        records=[{k:r[k] for k in recordkeys if k in r} for r in selected['records_CF']]
        assert len(records)==8 and selected['native8_CF_xy'][witness['native_slot']]==witness['source_GT_metadata']['reference_xy']
        rows.append(dict(global_index=i,query_id=witness['query_id'],base_index=base,base_id=witness['base_id'],scene_index=scene,variant=variant,
            original_qL=dict(native_slot=witness['native_slot'],query_xy=witness['query_xy'],L=witness['L']),all8_original_input_queries=q8,
            original_metadata=dict(base_complete=bind_text(bcpath),GT_scenes=bc['GT_scenes'],input_scenes=bc['input_scenes'],base_config=configbinding,base_config_row=base),
            selected_combined_scene_buffer=stat_inherited(selected['raw_buffers'],'all-target-components plus fixed foreign variant resolved raster'),
            saved_target_only_all_parts_buffer=stat_inherited(before['raw_buffers'],'BEFORE target-only all-pallet-parts resolved raster; foreign=None in producer'),
            scene_RGB=stat_inherited(inputrow['raw_RGB_PNG'],'original scene PNG; not opened in V172 preparation'),
            geometry=stat_inherited(selected['geometry'],'scaled registered target geometry; not opened'),foreign_geometry=stat_inherited(selected['foreign_geometry'],'fixed per-base cover/near/far foreign geometry; not opened'),
            projected_native_metadata=dict(native8_CF_xy=selected['native8_CF_xy'],permCF_to_canonical=selected['permCF_to_canonical'],native8_canonical_xy_exact_reindexing=projected_canonical,native8_canonical_xyz_from_config=cfg['native8_canonical_xyz'],camera=selected['camera'],records_CF_metadata=records),
            target_only_provenance=dict(BEFORE_variant_scene=before['scene_index'],renderer_code=protocol['code'],target_layers_digest_before=bc['target_layers_before'],target_layers_digest_after=bc['target_layers_after'],same_static_target_camera_geometry=True),
            selected_part_alone_mask_or_per_part_depth_stack=dict(status='NOT_SAVED_BY_ORIGINAL_PRODUCER',path=None,reason=protocol['storage'],resolved_BEFORE_is_not_selected_part_alone_or_hidden_surface_stack=True),
            actual_measurement_authorized=False,no_semantic_or_owner_measurements_performed=True))
        meta_bindings.extend([bind_text(bcpath),bc['GT_scenes'],bc['input_scenes'],configbinding])
    for b in meta_bindings:checked_text(b)
    out=dict(completed=True,PASS=True,phase='V172',status='METADATA_CODE_PATH_READINESS_COMPLETE_NO_ARRAY_INSPECTION',scope=bind_text(SCOPE),code=bind_text(__file__),source_selection=scope['source_selection'],fixed_rows=scope['fixed_rows'],fixed_scenes=scope['fixed_scenes'],
        scenes=rows,declared_scene_buffer_schema=schema,schema_authority='Exact np.savez declarations in bound V129 renderer + V93 raster/V128 nearest helper; NPZ headers and bodies have NOT been read',
        provenance_metadata_and_code_bindings=meta_bindings,
        readiness_summary=dict(selected_combined_buffers_present=sum(r['selected_combined_scene_buffer']['path_exists'] for r in rows),saved_target_only_resolved_buffers_present=sum(r['saved_target_only_all_parts_buffer']['path_exists'] for r in rows),per_part_or_selected_alone_pixel_stacks_saved=0,projected_native_metadata_all8_scenes=3),
        interpretation='Original source region-teacher semantic states, continuous ray witnesses, nearest raster ownership and literal q support are distinct. This preparation only locates original evidence; no owner/native visibility/gold inference or correction.',
        limitations=['BEFORE is the whole target pallet resolved raster, not a selected-part-alone mask and not the unsaved per-part depth stack.','Missing pixel layers remain NOT_SAVED; no new render or inferred replacement is created.','All numeric shape/dtype schemas are static declarations pending a separately frozen actual measurement protocol.'],
        array_image_geometry_depth_mask_bodies_or_headers_opened=0,array_image_geometry_depth_mask_hashes_computed=0,new_RGB_generation_or_rendering=0,new_source_label_qL_correction_or_training_admission=0,new_threshold_or_model_selection=False,
        V112_private_current_V169_model_weight_feature_access=0,actual_measurements=0,actual_array_entry_authorized=False,selected_model=None,fully_solved=False)
    with (V/'INPUT_PATH_READINESS.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(readiness=bind_text(V/'INPUT_PATH_READINESS.json'),summary=out['readiness_summary'])),flush=True)
if __name__=='__main__':main()
