"""Actual source BEFORE/AFTER RGB judgments by A; batch04 only."""
from pathlib import Path
import datetime,collections,json,hashlib
import annotation_io as io
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
V='APPARENT_SAME_CUE_VISIBLE_AI';C='EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI';M='MISSING_BEFORE_REFERENCE_U'
scope=ROOT/'SOURCE_CUE_CORPUS_SCOPE_BEFORE_NEW_RGB.json';fmt=ROOT/'ANNOTATION_RECORD_FORMAT_BEFORE_LABELS.json'
assert all((io.pair(i)/'BEFORE_SEAL.json').exists() for i in range(5))
assert io.read(R/'ALL_BEFORE_CONTACTS_VIEWED_BEFORE_AFTER.json')['any_batch04_after_RGB_opened_or_decoded'] is False
notes=[
 'AFTER 별도 cardboard가 화면 왼쪽 아래 전면에 들어와 가까운 broad wall의 고정 lowest-wide-wall-left cue uncertainty 전체를 가립니다. 두 upper wall cue와 lowest-wide-wall-right는 남습니다. Base 두 슬롯은 BEFORE부터 image 아래로 잘렸으므로 그대로 null/U입니다.',
 'BEFORE 오른쪽 목재 앞 낮은 부분을 가리던 cardboard가 AFTER에는 중앙 왼쪽으로 옮겨집니다. 목재 upper-left cue는 양쪽에서 관찰됩니다. AFTER에서 더 드러난 낮은 목재 구조를 before reference로 후향 추가하지 않았습니다.',
 'BEFORE의 큰 foreground cardboard가 AFTER에는 장면 왼쪽으로 옮겨져 목재 front wall/base가 더 보입니다. 고정 upper-left 목재 junction은 유지되며 나머지5slot은 BEFORE reference가 없으므로 U를 보존합니다.',
 'AFTER cardboard 일부가 화면 아래 왼쪽에 들어오지만 BEFORE에서 정한 두 upper wall contour junction은 계속 관찰됩니다. Lowest board/base의 네 슬롯은 BEFORE부터 image아래로 잘렸고 그대로 null/U입니다.',
 'BEFORE 오른쪽 목재 앞의 cardboard가 AFTER에는 안전기둥/실험대 근처 왼쪽으로 옮겨집니다. 고정 upper-left 목재 junction은 양쪽에서 보입니다. 나머지 BEFORE-null 슬롯을 AFTER에서 드러난 구조에 맞춰 채우지 않았습니다.'
]
manifest=[];bc=collections.Counter();ac=collections.Counter();perpair=[]
for i in range(5):
 d=io.pair(i);before=io.read(d/'BEFORE_CUES.json');meta=io.read(d/'PAIR_METADATA_INPUT_ONLY.json');out=[]
 assert [x['part_slot'] for x in before['slots']]==io.POLICY['slots']
 for k,row in enumerate(before['slots']):
  bc[row['before_state']]+=1
  if row['reference_xy'] is None:
   state=M;covered=None;blocker=None
   evidence='BEFORE에서 '+row['before_state']+'였고 reference/uncertainty가 null이므로 MISSING_BEFORE_REFERENCE_U를 유지합니다. AFTER에서 추가 구조가 드러나도 새 좌표나 D/E를 만들지 않습니다. BEFORE 근거: '+row['evidence']
  elif i==0 and k==2:
   state=C;covered=True;blocker='Separate opaque brown cardboard carton in front-left of the wooden wall'
   evidence='BEFORE의 '+row['part_slot']+' 목재 junction/part-line이 AFTER에서 사라지고 고정 uncertainty rectangle 전체가 distinct cardboard의 갈색 면 안에 놓입니다. Fullscene에서 cardboard의 윗면과 옆면이 목재 앞에 놓여 인접 목재 윤곽을 끊는 관계를 직접 보았습니다. Mask/depth나 단순 target-mask0이 아닌 RGB foreign-object 증거입니다.'
  else:
   state=V;covered=False;blocker=None
   evidence='BEFORE에 고정한 '+row['part_slot']+'의 목재 외곽/인접 part-line이 AFTER에서도 같은 uncertainty region에 관찰됩니다. 새로 놓이거나 이동한 cardboard는 이 region을 덮지 않습니다. Exact centerpixel material visibility나 정밀3D 대응을 주장하지 않습니다.'
  out.append(dict(part_slot=row['part_slot'],reference_xy=row['reference_xy'],uncertainty_bbox_xyxy_half_open=row['uncertainty_bbox_xyxy_half_open'],after_state=state,evidence=evidence,foreign_blocker=blocker,coverage_entire_region=covered,registration_note='Fullscene과 같은 고정 crop에서 남은 목재/배경 윤곽의 큰 이동은 관찰되지 않았습니다. 작은 camera/target shift, blur/노출차이는 배제하지 못하며 pixel-registration을 인증하지 않습니다. 동일 supplied T를 정렬 증거로 쓰거나 fit/warp/relocalization을 하지 않았습니다.',native8_semantic_id=None,native8_status='UNKNOWN',human_gold=False,official_gold=False,training_admitted=False,reference_state_is_AI_weak_evidence=True,RGB_cover_is_not_alpha_mask_certificate=True));ac[state]+=1
 obj=dict(completed=True,pair_id=before['pair_id'],source_split=before['source_split'],source_after_RGB=dict(meta['after_RGB'],raw_hw=[480,640]),before_annotation=io.bind(d/'BEFORE_CUES.json'),before_seal=io.bind(d/'BEFORE_SEAL.json'),slots=out,full_scene_observation=notes[i],all6_slots_retained=True,coordinates_and_uncertainty_exact_from_before=True,actual_before_after_fullscenes_and_all6_paired_display_rows_directly_viewed=True,gross_target_or_camera_motion_observed=False,pixel_registration_certified=False,small_motion_or_capture_change_not_excluded=True,numerical_registration_or_relocalization_calls=0,source_family_limit='Same onephysical woodenbin; TRAIN_LAB/VAL_HALL preserved. Not human gold/native8/independentobject generalization.',primary_after_judgments_quarantined_from_secondary_auditors=True,other_primary_after_judgments_read=False,mask_depth_sourceq_model_current_reads=0,new_model_forward_fit_composites_PnP=0,training_admitted=False,root_scope=io.bind(scope),format=io.bind(fmt),own_policy=io.bind(R/'ANNOTATOR_POLICY_BEFORE_NEW_RGB.json'),display=io.bind(d/'AFTER_DISPLAY_RECEIPT.json'),writer=io.bind(Path(__file__)),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 io.write(d/'AFTER_CUE_STATES.json',obj)
 io.write(d/'AFTER_VISUAL_QA.json',dict(completed=True,PASS=True,full_after_unmarked_RGB_directly_viewed=True,all6_display_rows_directly_viewed=True,before_uncertainty_never_adjusted=True,all_unknown_before_preserved_null_and_missing_before_U=True,all_foreign_cover_assertions_have_actual_fullscene_cardboard_evidence=True,subpixel_registration_not_certified=True,secondary_after_judgments_unread=True,new_model_current_fit_PnP=0))
 ev=[d/'AFTER_CUE_STATES.json',d/'AFTER_VISUAL_QA.json',d/'AFTER_DISPLAY_RECEIPT.json',d/'PAIRED_FULL_SCENE.png',d/'AFTER_ALL6_CONTACT.png']+[d/f'AFTER_PAIR_{k:02d}.png' for k in range(6)]
 io.write(d/'AFTER_SEAL.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pair_id=before['pair_id'],source_split=before['source_split'],after_annotation=io.bind(d/'AFTER_CUE_STATES.json'),before_annotation=io.bind(d/'BEFORE_CUES.json'),before_seal=io.bind(d/'BEFORE_SEAL.json'),source_after_RGB=meta['after_RGB'],root_policy=io.bind(scope),reader_policy=io.bind(R/'ANNOTATOR_POLICY_BEFORE_NEW_RGB.json'),files=[io.bind(p) for p in ev],actual_viewed=True,source_all6_retained=True,no_reference_changes_or_registration=True,primary_after_quarantined=True,mask_depth_sourceq_otherreview_model_current_reads=0,new_forward_fit_PnP=0))
 manifest.append(dict(pair_id=before['pair_id'],before_annotation=io.bind(d/'BEFORE_CUES.json'),before_seal=io.bind(d/'BEFORE_SEAL.json'),after_annotation=io.bind(d/'AFTER_CUE_STATES.json'),after_seal=io.bind(d/'AFTER_SEAL.json')))
 perpair.append(dict(pair_id=before['pair_id'],source_split=before['source_split'],before_visible_N=before['known_visible_cues'],before_unknown_N=before['unknown_cues'],after_counts=dict(collections.Counter(x['after_state'] for x in out))))
# Prior sealed batches remain immutable; inspect bindings only, no judgment review.
prior=[]
for old,expected in [(R.parent/'BATCH_SEAL.json','3622deeef038595719f9f0a94bae53948d127353c99cf16f9f817cc0b79d2a76'),(R.parent/'batch02/BATCH_SEAL.json','ce4c7fa5c8e29f0f2d6f98c64ebc10f88f8f85cd42fe7245483ac9924c29256c'),(R.parent/'batch03/BATCH_SEAL.json','af2562a6315b9a8c447d6dcfd23152bf7bed725ba81c9b350b0ba614fca82bdd')]:
 oldseal=io.read(old);assert io.bind(old)['sha256']==expected
 for binding in oldseal['files']:assert io.bind(Path(binding['path']))['sha256']==binding['sha256'],binding['path']
 prior.append(dict(seal=io.bind(old),bound_files_rehashed_N=len(oldseal['files'])))
io.write(R/'PRIOR_BATCHES_IMMUTABILITY_CHECK.json',dict(completed=True,PASS=True,prior_seals=prior,all_original_bound_bytes_unchanged=True,old_judgment_contents_not_reviewed=True))
io.write(R/'BATCH_MANIFEST.json',dict(completed=True,batch='04',annotator='a',pairs=manifest,root_scope=io.bind(scope),assignment=io.bind(ROOT/'ANNOTATOR_A_BATCH04_BEFORE_NEW_RGB.json'),format=io.bind(fmt),all5assigned_no_swaps_or_skips=True,all6slots_per_pair=True,primary_after_states_quarantined=True))
io.write(R/'BATCH_REPORT_KO.json',dict(completed=True,batch='04',annotator='a',finished_pair_N=5,slot_rows_N=30,pairs=perpair,before_counts=dict(bc),after_counts=dict(ac),all5_before_sealed_before_any_batch_after_RGB_read=True,actual_before_and_after_full5plus5_rawframes_directly_viewed=True,actual_before_contacts5_and_after_all6_contacts5_directly_viewed=True,unknown_not_converted_to_external_or_self=True,after_revealed_parts_not_added_as_new_before_refs=True,coordinates_uncertainty_unchanged=True,native8UNKNOWN_all30=True,no_training_admission=True,no_model_fit_current_PnP=True,primary_after_states_remain_quarantined=True,prior_batches_bytes_unchanged=True,notes_KO='이 batch는 오른쪽/아래로 잘린 목재 및 BEFORE부터 cardboard가 가린 장면을 포함합니다. 21개 missing-before U를 그대로 보존했으며 어려운 pair를 교체하지 않았습니다. 6개 view-relative physicalpart 슬롯은 native8 GT가 아닙니다.'))
# Canonical join and every pair-before/after identity check before batch seal.
assignment=io.read(ROOT/'ANNOTATOR_A_BATCH04_BEFORE_NEW_RGB.json')
assert [x['pair_id'] for x in manifest]==[x['pair_id'] for x in assignment['rows']]
for row in manifest:
 b=io.read(Path(row['before_annotation']['path']));a=io.read(Path(row['after_annotation']['path']))
 assert len(b['slots'])==len(a['slots'])==6 and b['pair_id']==a['pair_id']==row['pair_id'] and b['source_split']==a['source_split']
 for x,y in zip(b['slots'],a['slots']):
  for key in ('part_slot','reference_xy','uncertainty_bbox_xyxy_half_open','native8_semantic_id'):assert x[key]==y[key]
  if x['reference_xy'] is None:assert y['after_state']==M
  assert not x['training_admitted'] and not y['training_admitted']
 for p in ('before_annotation','before_seal','after_annotation','after_seal'):assert io.bind(Path(row[p]['path']))['sha256']==row[p]['sha256']
io.write(R/'METADATA_JOIN_AND_IMMUTABILITY_QA.json',dict(completed=True,PASS=True,all5assignment_exact_order=True,all30slots_exact_order=True,all30_reference_and_uncertainty_exact_before_after=True,all21_null_before_preserved_as_missing_before_U=True,all_pair_seal_bindings_exact=True,prior_batch_original_bound_bytes_unchanged=True,no_model_mask_depth_q_current_PnP=True,format=io.bind(fmt)))
files=sorted(p for p in R.rglob('*') if p.is_file() and '__pycache__' not in str(p))
io.write(R/'BATCH_SEAL.json',dict(completed=True,PASS=True,batch='04',annotator='a',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest=io.bind(R/'BATCH_MANIFEST.json'),report=io.bind(R/'BATCH_REPORT_KO.json'),metadata_QA=io.bind(R/'METADATA_JOIN_AND_IMMUTABILITY_QA.json'),files=[io.bind(p) for p in files],all5assigned_pairs_completed=True,stop_for_root_QC=True,primary_after_quarantine=True,no_raw_RGB_copies_only_review_crops_contacts=True,mask_depth_sourceq_model_current_reads=0,new_model_forward_fit_composites_PnP=0))
print(json.dumps(dict(manifest=io.bind(R/'BATCH_MANIFEST.json'),seal=io.bind(R/'BATCH_SEAL.json'),before_counts=dict(bc),after_counts=dict(ac))))
