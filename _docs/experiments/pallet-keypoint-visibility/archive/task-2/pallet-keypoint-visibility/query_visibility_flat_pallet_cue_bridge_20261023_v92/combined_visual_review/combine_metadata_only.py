from pathlib import Path
import json,hashlib,collections,datetime
R=Path(__file__).resolve().parent;V=R.parent
bind=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def load(p):return json.loads(Path(p).read_text())
rootpath=V/'public_visual_review_ROOT/ACTUAL_FIRST10_SOURCE_QUERY_VISUAL_OBSERVATIONS.json';bpath=V/'public_visual_review_B/FIXED10_FULLFRAME_AND80_QUERY_CROP_OBSERVATIONS.json'
a=load(rootpath);b=load(bpath);as_=load(rootpath.parent/'SEAL.json');bs=load(bpath.parent/'VISUAL_OBSERVATION_SEAL.json')
assert as_['completed'] and bs['completed'] and as_['observation']['sha256']==bind(rootpath)['sha256'] and bs['report']['sha256']==bind(bpath)['sha256']
meta=[json.loads(z) for z in (V/'rgb96_review/ROWS_INPUT_ONLY.jsonl').read_text().splitlines()];assert len(meta)==160
metaby={(z['frame_index'],z['native_slot']):z for z in meta};assert len(metaby)==160
manifest=load(V/'rgb96_review/RGB_PROVENANCE_AND_FIGURES.json');figby={z['frame_index']:z for z in manifest['frames']}
assert len(a['frames'])==10 and [z['frame_index'] for z in a['frames']]==list(range(10))
assert len(b['frames'])==10 and [z['frame_index'] for z in b['frames']]==list(range(10,20))
frames=[];allrows=[]
for author,obs,sourcepath in [('ROOT',a,rootpath),('B',b,bpath)]:
 slots=obs['slots'] if author=='ROOT' else [s for f in obs['frames'] for s in f['slots']]
 bykey={(s['frame_index'],s['native_slot']):s for s in slots};assert len(bykey)==80
 for f in obs['frames']:
  fi=f['frame_index'];fm=figby[fi];assert f['figure']['path']==fm['figure']['path'] and f['figure']['sha256']==fm['figure']['sha256']
  viewed=f.get('actually_viewed',f.get('actually_viewed_fullframe'));assert viewed
  if author=='B':assert f['actually_viewed_all8_crop_panels']
  target=f['target_instance'] if author=='ROOT' else f['target_instance_visual_association']
  frames.append({'frame_index':fi,'input_id':metaby[(fi,0)]['input_id'],'split':metaby[(fi,0)]['split'],'reviewer':author,'target_instance_visual_association':target,'fullframe_observation':f.get('observation_KO',f.get('fullframe_observation')),'original_figure_binding':f['figure'],'original_observation_binding':bind(sourcepath),'actually_viewed_reported':True,'no_new_image_view_by_combiner':True})
  for k in range(8):
   s=bykey[(fi,k)];m=metaby[(fi,k)];assert s['input_id']==m['input_id'] and s['query_center_valid']==m['query_center_valid'];assert s.get('original_input_status',s.get('input_status'))==m['input_status']
   assert s['expected_native_part_identity']=='UNKNOWN' and s['semantic_D_E_S']=='UNKNOWN' and not s['usable_training_gold']
   if 'query_xy'in s:assert s['query_xy']==m['query_xy'] and s['L']==m['L']
   cat=s['query_local_evidence'];assert (cat=='NO_QUERY')==(not m['query_exists']);assert (cat=='OOB')==(m['query_exists'] and not m['query_center_valid'])
   allrows.append({'row_index':fi*8+k,'frame_index':fi,'input_id':m['input_id'],'native_slot':k,'source_split':m['split'],'image_sha256':m['image_sha256'],'query_xy':m['query_xy'],'L':m['L'],'query_exists':m['query_exists'],'query_center_valid':m['query_center_valid'],'continuous_valid':m['continuous_valid'],'nearest_valid':m['nearest_valid'],'crop_valid_fraction':m['crop_valid_fraction'],'input_status':m['input_status'],'reviewer':author,'target_instance_visual_association':target,'query_local_evidence':cat,'observation':s.get('observation_KO',s.get('observation')),'expected_native_part_identity':'UNKNOWN','semantic_D_E_S':'UNKNOWN','usable_training_gold':False,'new_training_label':False,'original_observation_path':str(sourcepath),'original_observation_sha256':bind(sourcepath)['sha256'],'original_key':{'frame_index':fi,'native_slot':k}})
allrows.sort(key=lambda z:z['row_index']);frames.sort(key=lambda z:z['frame_index']);assert [z['row_index'] for z in allrows]==list(range(160))
counts={'frames':len(frames),'slots':len(allrows),'query_center_valid':sum(s['query_center_valid'] for s in allrows),'factual_input_U':sum(not s['query_center_valid'] for s in allrows),'target_instance_visual_association':dict(collections.Counter(z['target_instance_visual_association'] for z in frames)),'query_local_evidence':dict(collections.Counter(z['query_local_evidence'] for z in allrows)),'native_expected_part_UNKNOWN':160,'semantic_D_E_S_UNKNOWN':160,'new_training_labels':0}
assert counts['target_instance_visual_association']=={'NO_DETECTION':2,'MULTIPLE_OR_AMBIGUOUS':10,'CLEAR_SINGLE_TARGET':4,'NO_CLEAR_TARGET':4}
assert counts['query_local_evidence']=={'NO_QUERY':16,'OOB':22,'VISIBLE_DISTINCT_CORNER_EDGE_FOOT':34,'AMBIGUOUS':39,'TARGET_MATERIAL_ONLY':4,'FOREIGN_SURFACE_OR_BACKGROUND':45}
assert counts['query_center_valid']==122 and counts['factual_input_U']==38
out=R/'ALL160_WEAK_VISUAL_OBSERVATIONS.jsonl';assert not out.exists();out.write_text(''.join(json.dumps(z,ensure_ascii=False,allow_nan=False)+'\n' for z in allrows))
x={'completed':True,'schema':'V92_DISJOINT_REVIEWER_OBSERVATION_METADATA_JOIN','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':bind(Path(__file__)),'root_contract':bind(V/'SOURCE_QUERY_VISUAL_OBSERVATION_CONTRACT_BEFORE_REVIEW.json'),'sources':{'ROOT_observation':bind(rootpath),'ROOT_seal':bind(rootpath.parent/'SEAL.json'),'B_observation':bind(bpath),'B_seal':bind(bpath.parent/'VISUAL_OBSERVATION_SEAL.json'),'input_only_metadata':bind(V/'rgb96_review/ROWS_INPUT_ONLY.jsonl'),'figure_manifest':bind(V/'rgb96_review/RGB_PROVENANCE_AND_FIGURES.json')},'counts':counts,'join_checks':{'all160_unique_and_in_original_frame_slot_order':True,'all_ID_inputstatus_valid_exact':True,'all_B_qL_exact_original_public_metadata':True,'ROOT_qL_added_from_original_input_metadata_not_new_predictions':True,'figure_bindings_equal_existing_render_manifest':True,'all_unknown_native_and_semantic_states_preserved':True},'row_table':bind(out),'frames':frames,'scope_ko':['ROOT는00..09, B는10..19를 실제 관찰한 서로 겹치지 않는 배정입니다. 이 파일은 두 관찰을 원래160행 순서로 묶은 metadata join일 뿐, 동일 사례에 대한 두 reviewer 일치율이나 독립 human consensus를 의미하지 않습니다.','관찰자의 visual target/cue 분류와 native8 semantic D/E/S 정답은 별개입니다. 전160행 native 기대part와 D/E/S는 UNKNOWN, traininggold0입니다.','34개 local distinct-cue 관찰은 무조건적 D 학습 정답이 아닙니다. material/foreign/background 및 query부재/OOB를 E/S로 자동 바꾸지 않았습니다.','현재223/모델/feature/GT를 읽거나 threshold/checkpoint를 고르지 않았습니다. 기존 개별 관찰은 모든 byte를 보존했습니다.'],'actions':{'metadata_only':True,'new_RGB_pixel_or_array_reads':0,'new_GT_model_feature_weight_reads':0,'new_forward_fit_PnP':0,'current223_reads':0,'new_training_labels':0,'original_artifact_modified':0}}
p=R/'COMBINED20_SOURCE_QUERY_VISUAL_REVIEW.json';assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q=R/'SEAL.json';assert not q.exists();q.write_text(json.dumps({'completed':True,'PASS_metadata_join':True,'report':bind(p),'all160rows':bind(out),'code':bind(Path(__file__)),'individual_reviews_unchanged':True,'no_consensus_or_gold_claim':True,'counts':counts},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'report':bind(p),'rows':bind(out),'seal':bind(q),'counts':counts},ensure_ascii=False))
