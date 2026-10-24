#!/usr/bin/env python3
"""Metadata-only fixed12 census. Does not stat/hash/open any referenced payload."""
import collections
import csv
import hashlib
import importlib.util
import io
import json
import os

H=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location('reader',H+'/read_fixed12_selected_GT_metadata.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
P='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility'
V216=P+'/query_visibility_public_train_part_query_support_contract_20261025_v216'
V153=P+'/query_visibility_mr6d_full_public_source_20261024_v153'
V154=P+'/query_visibility_mr6d_source_supervision_bridge_20261024_v154'
V159=P+'/query_visibility_native_part_support_feasibility_20261024_v159'
SCOPE=m.D+'/ROOT_PROSPECTIVE_FIXED12_PUBLIC_GT_CORRESPONDENCE_AUDIT_SCOPE.json'
SCOPE_SHA='b396012313065b36781a927dc51222c8d6acf64d0fd19186f8a80d3e6ea1a4c4'
PRIOR={
 'selection':(V153+'/full_source_v1/FULL_SOURCE_SELECTION_BEFORE_NEW_PAYLOADS.json','9e1ca9bfb9422a8dd27f83ca2ec7ed6d411e31e9532f742eba0e5d7619cbc71f'),
 'download':(V153+'/full_source_v1/FULL_SOURCE_DOWNLOAD_COMPLETE.json','61d1e05c9d6159a289a46d9aa1592c5bd07e0495349815c429a1cc9c4e690cc9'),
 'intake':(V153+'/full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json','a6aef4a2a3903d3689ca32bac007482e3e173f24b3242578609225eca3cc4fd6'),
 'intake_code':(V153+'/full_intake_preparation/intake_full_source.py','f2f564e30ef60428a13952d5d5bb8a907681d8b179546dd4dc7f73e24aa87920'),
 'prior_instance_role':(V154+'/ROOT_SUPERVISION_SEAL.json','d6915b0abcfb8c7058bf4cc2d83b73492f57edd88e4862e55684712a310c4b34'),
 'nominal_part_definition':(V159+'/native_definition_audit/DEFINITION_AND_FEASIBILITY.json','76c0a83888f3c93445a8c216ceb4c6af069a939af97451fefab34bdae38d4ddc'),
}

def save_csv(name,rows):
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    raw=buf.getvalue().encode()
    with os.fdopen(m.opened(H+'/'+name,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(raw)
    return m.bind(H+'/'+name,raw)

def main():
    source_bindings={};docs={}
    raw=m.checked({'path':SCOPE,'sha256':SCOPE_SHA});scope=json.loads(raw);source_bindings['scope']=m.bind(SCOPE,raw)
    for i,b in enumerate(scope['preparation_inputs']):
        raw=m.checked(b);source_bindings['declared_'+str(i)]=m.bind(b['path'],raw);docs[b['path']]=json.loads(raw)
    for k,(p,s) in PRIOR.items():
        raw=m.checked({'path':p,'sha256':s});source_bindings[k]=m.bind(p,raw);docs[k]=json.loads(raw) if p.endswith('.json') else raw.decode()
    assert len(PRIOR)==6<=16
    original=docs[V216+'/joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json']['rows96']
    split=docs[V216+'/component_split_preparation_b/INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json']
    split_rows={r['row_id']:r for r in split['rows96']}
    assert len(original)==len(split_rows)==96
    selected_by_key={f['source_id']:f for f in docs['selection']['frames']}
    downloaded={r['relative_path']:r for r in docs['download']['files']}
    gt_meta={b['path'].rsplit('/',2)[-2]:b for b in docs['selection']['GT_metadata_bindings']}
    payloads={};rows=[];frames=[]
    def payload(kind,relative):
        d=downloaded[relative]
        assert d['completed'] and d['source_content_ID_verified']
        record={'kind':kind,'relative_source_path':relative,'path':d['local_path'],'sha256':d['sha256'],'size_bytes':d['bytes'],'sha_status':'RECORDED_RAW_SHA256_FROM_HISTORICAL_DOWNLOAD_RECEIPT','current_payload_existence_or_SHA_reverified':False,'basis':source_bindings['download']}
        if record['path'] in payloads:assert payloads[record['path']]==record
        payloads[record['path']]=record
        return record
    model=payload('NOMINAL_OBJ1_CAD','models/obj_000001.ply')
    for position in range(12):
        rr=[r for r in original if r['input']['position']==position]
        assert len(rr)==8 and [r['input']['native_slot'] for r in rr]==list(range(8))
        i=rr[0]['input'];f=selected_by_key[i['frame_key']]
        assert f['frame_index']==i['original_frame_index'] and f['official_split']=='val' and f['scene'] in ['000000','000002','000005','000006']
        scene=f['scene'];g=gt_meta[scene]
        for kind,base in [('SCENE_GT_JSON','scene_gt.json'),('SCENE_GT_INFO_JSON','scene_gt_info.json'),('SCENE_CAMERA_JSON','scene_camera.json')]:
            path=g['path'].rsplit('/',1)[0]+'/'+base
            rec={'kind':kind,'scene':scene,'path':path,'sha256':g['sha256'] if kind=='SCENE_GT_JSON' else None,'size_bytes':None,'sha_status':'RECORDED_RAW_SHA256_FROM_SOURCE_SELECTION' if kind=='SCENE_GT_JSON' else 'PRIOR_RAW_SHA_UNKNOWN_EXACT_CODE_LITERAL_PATH','current_payload_existence_or_SHA_reverified':False,'basis':source_bindings['selection'] if kind=='SCENE_GT_JSON' else source_bindings['intake_code']}
            if path in payloads:assert payloads[path]==rec
            payloads[path]=rec
        full={r['gt_instance_index']:payload('FULL_INSTANCE_MASK_PNG',r['path']) for r in f['source_files']['full_masks']}
        visible={r['gt_instance_index']:payload('VISIBLE_INSTANCE_MASK_PNG',r['path']) for r in f['source_files']['visible_masks']}
        candidates=f['obj1_gt_instance_indices']
        assert sorted(full)==sorted(visible)==candidates
        depth=payload('DEPTH_PNG',f['source_files']['depth'])
        masks={str(n):{'full_mask':full[n],'visible_mask':visible[n]} for n in candidates}
        frame={'frame_position':position,'frame_key':i['frame_key'],'original_frame_index':i['original_frame_index'],'scene':scene,'json_frame_key':f['frame_key'],'frame_filename_stem':f['frame_filename_stem'],'original_role':'TRAIN','internal_source_role':i['internal_source_role'],'component_id':i['component_id'],'source_group_key':i['source_group_key'],'obj1_candidate_gt_list_indices':candidates,'selected_gt_instance_index':None,'selected_instance_correspondence':'UNKNOWN_NO_GT_BEST_OR_FIRST_INSTANCE_FALLBACK','scene_GT_path':g['path'],'scene_GT_info_path':g['path'].rsplit('/',1)[0]+'/scene_gt_info.json','scene_camera_path':g['path'].rsplit('/',1)[0]+'/scene_camera.json','candidate_masks':masks,'depth':depth,'nominal_CAD':model,'training_admission':False}
        frames.append(frame)
        for r in rr:
            x=r['input'];s=split_rows[x['row_id']]
            for key in ['position','frame_key','original_frame_index','component_id','source_group_key','native_slot','row_id','original_global_index','query_xy','L','original_query_available','original_flags','internal_source_role']:
                assert x[key]==s[key],('Original input mismatch',key,x['row_id'])
            assert x['role']==s['role']=='TRAIN'
            rows.append({'frame_position':position,'frame_key':x['frame_key'],'original_frame_index':x['original_frame_index'],'component_id':x['component_id'],'source_group_key':x['source_group_key'],'original_role':'TRAIN','internal_source_role':x['internal_source_role'],'row_id':x['row_id'],'original_global_index':x['original_global_index'],'native_slot':x['native_slot'],'query_xy':x['query_xy'],'L':x['L'],'original_query_available':x['original_query_available'],'original_flags':x['original_flags'],'source_json_frame_key':f['frame_key'],'scene':scene,'obj1_candidate_gt_list_indices':candidates,'selected_gt_instance_index':None,'native_to_fixed_nominal_mapping':'dynamic[k]=fixed[perm[k]]; no new permutation or projection executed','training_admission':False})
    m.validate_all8_inputs(rows)
    missing=[r for r in payloads.values() if r['sha256'] is None]
    assert len(missing)==8
    counts={'frames':len(frames),'native_rows':len(rows),'inputU':sum(not r['original_query_available'] for r in rows),'role_rows':dict(collections.Counter(r['internal_source_role'] for r in rows)),'role_frames':dict(collections.Counter(f['internal_source_role'] for f in frames)),'prefix_frames':dict(collections.Counter(f['scene'] for f in frames)),'candidate_instances_total':sum(len(f['obj1_candidate_gt_list_indices']) for f in frames),'payload_kinds':dict(collections.Counter(r['kind'] for r in payloads.values())),'unique_payload_paths':len(payloads),'known_raw_SHA256':len(payloads)-len(missing),'prior_SHA_UNKNOWN':len(missing),'selected_instance_UNKNOWN_frames':12}
    assert counts['role_rows']=={'SOURCE_FIT':64,'SOURCE_DEV':32} and counts['inputU']==11
    artifacts={}
    artifacts['frames']=m.write_exclusive(H+'/FIXED12_TRAIN_FRAME_AND_INSTANCE_CANDIDATE_METADATA.json',{'completed':True,'frames12':frames,'counts':counts,'source_bindings':source_bindings,'actual_GT_payload_access':0})
    artifacts['rows']=m.write_exclusive(H+'/ALL96_ORIGINAL_INPUTS_AND_GT_POINTERS.json',{'completed':True,'rows96':rows,'all8_original_qL_U_and_roles_preserved':True,'GT_reference_or_labels_copied_into_input':False})
    artifacts['whitelist']=m.write_exclusive(H+'/EXACT_PUBLIC_GT_PAYLOAD_WHITELIST.json',{'completed':True,'metadata_only':True,'payloads':sorted(payloads.values(),key=lambda r:(r['kind'],r['path'])),'missing_prior_SHA_bindings':missing,'all_known_SHAs_from_historical_text_receipts_only':True,'actual_payload_FS_calls':0,'no_RGB_or_source_NPZ_or_foreign_donor_paths':True,'runtime_JSON_reader_only_three_scene_JSON_kinds':True,'mask_depth_CAD_requires_separate_bounded_actual_reader_or_entry':True})
    csvrows=[]
    for f in frames:
        csvrows.append({'frame_position':f['frame_position'],'frame_key':f['frame_key'],'original_frame_index':f['original_frame_index'],'outer_role':'TRAIN','internal_source_role':f['internal_source_role'],'component_id':f['component_id'],'scene_gt_path':f['scene_GT_path'],'scene_gt_SHA256':payloads[f['scene_GT_path']]['sha256'],'scene_gt_info_path':f['scene_GT_info_path'],'scene_gt_info_SHA256':'UNKNOWN','scene_camera_path':f['scene_camera_path'],'scene_camera_SHA256':'UNKNOWN','obj1_candidate_GT_list_indices':json.dumps(f['obj1_candidate_gt_list_indices']),'selected_GT_instance_index':'UNKNOWN','full_mask_paths':json.dumps([v['full_mask']['path'] for v in f['candidate_masks'].values()]),'visible_mask_paths':json.dumps([v['visible_mask']['path'] for v in f['candidate_masks'].values()]),'depth_path':f['depth']['path'],'depth_SHA256':f['depth']['sha256'],'model_path':model['path'],'model_SHA256':model['sha256']})
    artifacts['frame_CSV']=save_csv('FIXED12_TRAIN_GT_BINDING_READINESS.csv',csvrows)
    artifacts['missing_SHA_CSV']=save_csv('EXACT8_PRIOR_SHA_UNKNOWN.csv',[{'kind':r['kind'],'scene':r['scene'],'path':r['path'],'prior_SHA256':'UNKNOWN','future_action':'ROOT_EXPLICIT_FIRST_OPAQUE_SHA_INTAKE_BEFORE_SELECTED_VALUES_NO_AUTO_DOWNLOAD'} for r in missing])
    nominal=docs['nominal_part_definition']
    artifacts['method_contract']=m.write_exclusive(H+'/INSTANCE_NATIVE_AND_EVIDENCE_CONTRACT.json',{'completed':True,'prior_V154_instance_role_source':source_bindings['prior_instance_role'],'prior_V154_method':'Historical actual-query entity+role correspondence, quality unknown and wrong-entity-never-E; no new historical weak-label bodies opened.','prior_V159_nominal_definition':source_bindings['nominal_part_definition'],'exact_coordinate_and_order_contract':nominal['exact_coordinate_and_order_contract'],'authority_separation':nominal['authority_separation'],'current_selected_instance_correspondence':'UNKNOWN; enumerate all obj1 GT-list candidates, no first/best/nearest fallback','native_part_correspondence':'nominal reference evidence only; explicit future same-instance/same-part support needed','separate_evidence_axes':['GT-list instance identity','nominal native-to-fixed part convention','full silhouette mask support','visible-surface mask support','valid measured depth and overlap/registration','query-local support/owner','native point/part visibility candidate with UNKNOWN'],'not_automatic_E':['mask absence','query OFF/background','another object at q','missing/invalid depth','nominal point outside or drift','uncertain part/instance'],'projection_policy':'SOURCE nominal pose projection may only run in a separately frozen actual protocol; no projection now, original production q/L/native/U are not moved; no PnP/coordinate optimization/GT-best permutation.','no_perfect_mesh_corner_blanket_gate':True,'future_DEV_all_training_branches_excluded':True,'historical_appearance_or_warmstart_exposure_not_freshblind':True,'training_admission':False})
    artifacts['metadata_completion']=m.write_exclusive(H+'/METADATA_PREPARATION_COMPLETE.json',{'completed':True,'PASS':True,'scope':source_bindings['scope'],'source_bindings':source_bindings,'declared_V216_metadata_files_read':4,'additional_prior_texts_read':len(PRIOR),'additional_prior_cap':16,'counts':counts,'artifacts':artifacts.copy(),'payload_stat_hash_read_decode':0,'new_labels_training_projection_or_model':0,'all_selected_instance_and_native_admission_UNKNOWN_false':True})
    print(json.dumps({'completed':True,'PASS':True,'counts':counts,'metadata_COMPLETE':artifacts['metadata_completion'],'additional_prior_texts':len(PRIOR),'actual_payload_FS_calls':0},ensure_ascii=False))

if __name__=='__main__':main()
