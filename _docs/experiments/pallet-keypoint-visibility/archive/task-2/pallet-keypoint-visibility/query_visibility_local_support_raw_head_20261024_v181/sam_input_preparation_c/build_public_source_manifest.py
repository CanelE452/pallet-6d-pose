"""V181 metadata-only original public185/1480 SAM input transport; no NPZ/RGB IO."""
from pathlib import Path
from collections import Counter
import json,hashlib,math
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
SRC=P/'query_visibility_mr6d_full_public_source_20261024_v153'
FACTS=SRC/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
QUERIES=SRC/'actual_full230_initializer_v1/ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'
FRAMES=SRC/'initializer_feature_preparation/FEATURE_FRAMES230_INPUT_ONLY.json'
RGB=SRC/'full_source_v1/RGB230_INPUT_ONLY_MANIFEST.json'
CROPS=SRC/'full230_native_DINO_v1/CROP_PACKETS230_INPUT_ONLY.json'
DONE=SRC/'full230_native_DINO_v1/FEATURE_COMPLETE.json'
PRODUCER=SRC/'initializer_feature_preparation/encode_full230.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def save(p,a):
 with Path(p).open('x') as f:json.dump(a,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_rows(p,a):
 with Path(p).open('x') as f:
  for r in a:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def stat_only(binding):
 p=Path(binding['path']);s=p.stat();return dict(exists=p.is_file(),bytes=s.st_size,body_header_or_hash_read=False)
def main():
 assert bind(SCOPE)['sha256']=='8cd21d5eb4caf9663927859ead0cbcbd823e701fd4a2164a66aa765e046bfcdc'
 scope=read(SCOPE);assert scope['fitting_data']['public_original'].startswith('Existing SOURCE_DEVELOPMENT185')
 complete=read(DONE);frame_doc=read(FRAMES);rgbs=read(RGB)['rows'];cropdoc=read(CROPS)
 assert complete['completed'] and complete['PASS'] and complete['inputfacts']==bind(FACTS) and complete['inputs']==bind(FRAMES) and complete['crop_manifest']==bind(CROPS) and frame_doc['queries']==bind(QUERIES)
 assert bind(RGB)['sha256']=='437ceec1608d5bec77f1d567c6aa912ce0b2ab91eddf2e12078874704d1b5141'
 facts=rows(FACTS);queries=rows(QUERIES);assert len(facts)==len(queries)==1840
 frame_rows=[r for r in frame_doc['frames'] if r['split']=='SOURCE_DEVELOPMENT'];assert len(frame_rows)==185
 cropmap={r['frame_index']:r for r in cropdoc['sheets']};rgbmap={r['frame_index']:r for r in rgbs}
 output=[];frame_output=[];paths=[];native_count=Counter();sequences=Counter();out_of_frame=0;no_intersection=0
 for pos,f in enumerate(frame_rows):
  fi=f['frame_index'];assert fi not in range(73,118);image=rgbmap[fi];crop=cropmap[fi];rr=facts[fi*8:fi*8+8];qq=queries[fi*8:fi*8+8]
  assert f['raw_hw']==[720,1280] and len(rr)==len(qq)==8 and [r['native_slot'] for r in rr]==list(range(8))
  assert image['split']==f['split']=='SOURCE_DEVELOPMENT' and image['canonical_RGB_NPY']==f['canonical_RGB_NPY'] and image['decoded_RGB_sha256']==f['decoded_RGB_sha256']
  q8=[q['query_xy'] for q in qq];L8=[q['L'] for q in qq]
  assert all(q is not None and len(q)==2 and all(math.isfinite(x) for x in q) and q!=[-1.,-1.] for q in q8)
  assert all(L is not None and math.isfinite(L) and L>0 for L in L8)
  box=[min(q[0] for q in q8),min(q[1] for q in q8),max(q[0] for q in q8),max(q[1] for q in q8)]
  assert box[0]<box[2] and box[1]<box[3] and all(r['context_box_xyxy']==box for r in rr)
  wholly=0<=box[0] and box[2]<1280 and 0<=box[1] and box[3]<720
  outside=box[2]<0 or box[3]<0 or box[0]>=1280 or box[1]>=720
  out_of_frame+=not wholly;no_intersection+=outside
  original_image=dict(path=image['image'],sha256=image['image_sha256'])
  source_image=dict(binding=f['canonical_RGB_NPY'],format='NPY_RGB_UINT8_HWC',key=None,raw_hw=f['raw_hw'],decoded_RGB_sha256=f['decoded_RGB_sha256'])
  fr=dict(sam_input_occurrence_id=f'PUBLIC_FRAME_{fi:04d}',family='PUBLIC_ORIGINAL',development_position=pos,original_frame_index=fi,frame_id=f['id'],source_id=f['source_id'],split=f['split'],image=source_image,original_RGB=original_image,original_RGB_decoder=image['decoder'],original_initial_points8=q8,original_L8=L8,original_q8_context_box_xyxy=box,box_wholly_in_original_continuous_domain=wholly,box_has_no_raw_intersection=outside,q8_prompt_valid=True,saved_rgb96_packet=crop['crop_packet'],original_global_indices=[r['global_index'] for r in rr],original_query_ids=[q['query_id'] for q in qq],training_admission=False,selected_instance_identity_certified=False)
  frame_output.append(fr);sequences[f['source_id'].split(':')[0]]+=1
  paths.append(dict(original_frame_index=fi,canonical_RGB=stat_only(f['canonical_RGB_NPY']),original_RGB=stat_only(original_image),saved_rgb96_packet=stat_only(crop['crop_packet'])))
  for k,(r,q) in enumerate(zip(rr,qq)):
   gi=fi*8+k;assert r['global_index']==gi and r['frame_index']==q['frame_index']==fi and r['native_slot']==q['native_slot']==k
   for name in ('frame_id','source_id','split','query_xy','L','query_id'):assert r[name]==q[name]
   assert q['finite_non_sentinel'] and not q['no_detection'] and r['native_continuous_valid']==q['native_continuous_valid'] and r['nearest_valid']==q['query_nearest_pixel_valid']
   flags={key:r[key] for key in ('feature_constructed','native_continuous_valid','nearest_valid','local_central4_available','context_available','Q_inside')}
   flags['query_available']=r['native_continuous_valid']
   query_flags={key:q[key] for key in ('no_detection','finite_non_sentinel','query_nearest_pixel_valid','query_RGB96_centre_domain_valid','native_grid_any_valid')}
   output.append(dict(input_occurrence_id=f'PUBLIC_ORIGINAL:{gi}',sam_input_occurrence_id=fr['sam_input_occurrence_id'],family='PUBLIC_ORIGINAL',original_global_index=gi,original_occurrence_id=q['query_id'],original_frame_index=fi,frame_id=r['frame_id'],source_id=r['source_id'],split=r['split'],native_slot=k,original_query_xy=r['query_xy'],original_L=r['L'],original_query_available=r['native_continuous_valid'],original_input_flags=flags,original_query_flags=query_flags,image=source_image,original_initial_points8=q8,original_q8_context_box_xyxy=box,saved_rgb96_packet=crop['crop_packet'],saved_crop_native_row=k,saved_rgb96_sha256=r['RGB96_HWC_sha256'],saved_valid96_sha256=r['valid96_sha256'],original_input_facts=r,original_query_metadata=q,initializer_association_status=q['association_status'],training_admission=False,selected_instance_identity_certified=False))
   native_count[k]+=1
 assert len(output)==1480 and len({r['original_global_index'] for r in output})==1480 and all(v==185 for v in native_count.values())
 available=sum(r['original_query_available'] for r in output);assert available==1286 and all(r['split']=='SOURCE_DEVELOPMENT' for r in output)
 op=H/'PUBLIC1480_ORIGINAL_INPUTS_METADATA_ONLY.jsonl';fp=H/'PUBLIC185_FRAMES_METADATA_ONLY.jsonl';save_rows(op,output);save_rows(fp,frame_output)
 bindings={key:bind(path) for key,path in dict(scope=SCOPE,inputfacts=FACTS,queries=QUERIES,frame_inputs=FRAMES,RGB_manifest=RGB,crop_manifest=CROPS,feature_complete=DONE,producer_text=PRODUCER).items()}
 review=dict(completed=True,PASS=True,code=bind(__file__),bindings=bindings,slots_manifest=bind(op),frames_manifest=bind(fp),counts=dict(frames=185,rows=1480,available=available,inputU=1480-available,no_detection=0,missing_qL=0,feature_constructed=sum(r['original_input_flags']['feature_constructed'] for r in output),nearest_valid=sum(r['original_input_flags']['nearest_valid'] for r in output),center4_valid=sum(r['original_input_flags']['local_central4_available'] for r in output),context_available=sum(r['original_input_flags']['context_available'] for r in output),available_but_nearest_invalid=sum(r['original_query_available'] and not r['original_input_flags']['nearest_valid'] for r in output),available_but_center4_invalid=sum(r['original_query_available'] and not r['original_input_flags']['local_central4_available'] for r in output),q8boxes_valid=185,q8boxes_not_wholly_in_raw_continuous_domain=out_of_frame,q8boxes_no_intersection=no_intersection),sequence_frames=dict(sequences),native_slots={str(k):v for k,v in native_count.items()},raw_hw=[720,1280],availability_authority='encode_full230.py:181 query_available=qdata.continuous_valid unchanged; nearest/center4/context are preserved flags and are not new gates.',prompt_authority='Exact min/max of all original initial q8, identical to frozen context_box_xyxy; no clip/expand/GT replacement, including full OOF frame185.',canonical_RGB_authority='Existing canonical RGB NPY uses pinned CV2 JPEG decode; actual npy shape/dtype and decoded SHA must be verified after root entry, not inferred from PIL JPEG re-decoding.',identity_authority='Original initializer association is UNASSIGNED_BY_INPUT_PRODUCER. No GT/weak owner/native/part label supplies a SAM point, box, identity certificate or selection rule.',pathstat_only=paths,source_features_RGB_crop_or_other_NPZ_body_header_or_hash_reads=0,held360_manifest_rows=0,held_label_score_RGB_or_feature_reads=0,owner253_or_part_witness_labels_read=0,private_or_V112_access=0,new_model_forward_training_PnP_or_qL_changes=0,actual_SAM_forward_authorized=False,training_admission=False,selected_model=None)
 save(H/'PUBLIC_INPUT_READINESS.json',review)
 print(json.dumps(dict(completed=True,readiness=bind(H/'PUBLIC_INPUT_READINESS.json'),frames_manifest=bind(fp),slots_manifest=bind(op),counts=review['counts'])),flush=True)
if __name__=='__main__':main()
