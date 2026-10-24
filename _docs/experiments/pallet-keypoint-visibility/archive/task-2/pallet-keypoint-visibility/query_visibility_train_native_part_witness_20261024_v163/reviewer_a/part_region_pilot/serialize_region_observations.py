from pathlib import Path
import json,hashlib,datetime,collections
R=Path(__file__).resolve().parent;V=R.parents[1];E=V/'evidence_recovered_v3'
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
scope=V/'ROOT_POST_CONSENSUS_PART_REGION_PILOT_SCOPE.json';s=json.loads(scope.read_text());assert bind(scope)['sha256']=='3022758ca92ffb180d53665fa225c52aab4b802931bceff6362d1ddc547dea95'
f=E/'BLIND_CROSS40_ALL_REVIEWERS.json';meta={x['case_id']:x for x in json.loads(f.read_text())['cases']}
manual={}
for line in (R/'manual_regions.tsv').read_text().splitlines():
 k,status,box,reason=line.split('\t');assert k not in manual;manual[k]=(status,None if box=='-' else list(map(int,box.split(','))),reason)
assert list(manual)==s['cases_fixed_before_new_region_observations'];assert len(manual)==22
rows=[];images=[]
for k in s['cases_fixed_before_new_region_observations']:
 c=meta[k];status,box,reason=manual[k];off=c['nominal_reference_diagnostic_raw_xyxy'];full=None
 if status=='SUPPORTED':
  assert box and 0<=box[0]<box[2]<=224 and 0<=box[1]<box[3]<=224
  full=[box[0]+off[0],box[1]+off[1],box[2]+off[0],box[3]+off[1]]
  h,w=c['raw_hw'];assert 0<=full[0]<full[2]<=w and 0<=full[1]<full[3]<=h
 else:assert status=='UNKNOWN' and box is None
 viewed=[c['images'][n] for n in ['card','nominal_ref_diagnostic']]
 for z in viewed:assert bind(Path(z['path']))==z
 images.extend(viewed)
 row={key:c[key] for key in ['case_id','global_index','query_id','frame_index','source_id','split','native_slot','native_name','query_xy','L','selected_gt_instance_index','nominal_expected_part']}
 row.update(region_status=status,diagnostic_bbox_xyxy=box,original_fullRGB_bbox_xyxy=full,diagnostic_raw_xyxy_offset=off,bbox_convention='integer half-open [x0,y0,x1,y1], diagnostic224 coordinates; full=diagnostic+[raw_x0,raw_y0,raw_x0,raw_y0]',region_semantics='coarse visible expected deck/runner MATERIAL-END neighborhood; axis-aligned bbox, not exact material mask, whole component segmentation or physical-corner coordinate',reason_KO=reason,ambiguity_KO='수동의 거친 영역이며 사각형 경계에 배경 또는 접합 경계가 포함될 수 있다. 박스 안 모든 픽셀의 재료 소유권이나 정확한 코너를 보증하지 않는다. 추후 개입에는 별도 프로토콜과 픽셀 지지 검사가 필요하다.' if box else '이전 가시성 관찰은 유지하지만 독립된 러너 재료 영역의 경계를 보증할 수 없어 UNKNOWN으로 남긴다. 지지블록이나 상판을 대용하지 않았다.',actually_viewed={'manifest':bind(f),'images':viewed,'tool':'tools.view_image','viewed':True},post_consensus_assistant_weak_only=True,new_blind_vote=False,human_or_official_gold=False,new_native_target=-1,new_quality_target=-1,training_admitted=False,corrected_query_or_native_corner_xy_created=False,original_blind_observations_modified=False,original_q_L_native_split_selectedinstance_unchanged=True)
 rows.append(row)
fout=R/'ALL22_COARSE_PART_REGION_OBSERVATIONS.jsonl';fout.write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in rows))
counts=dict(collections.Counter(z['region_status'] for z in rows));receipt={'schema':'V163_POST_CONSENSUS_ALL22_COARSE_MATERIAL_END_REGION_PILOT_COMPLETE','completed':True,'PASS':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':bind(scope),'blinded_geometry_input':bind(f),'manual_observations':bind(R/'manual_regions.tsv'),'serialization_code':bind(Path(__file__)),'output':bind(fout),'rows':22,'all22_fixed_order_exact':True,'region_status_counts':counts,'actually_viewed_images':images,'actual_image_count':44,'prior_A_review_unchanged':bind(V/'reviewer_a/review/REVIEW_COMPLETE.json'),'prior_cross_seal_unchanged':bind(V/'reviewer_a/review/CROSS40_SEALED.json'),'post_consensus_not_fifth_blind_vote':True,'native_quality_targets_all_minus1':True,'training_admitted':False,'source_RGB_reread_beyond_saved_card_and_diagnostic':False,'new_images_rendered':False,'masks_oldlabels_owner_models_other_reviews_read':False,'all_UNKNOWN_retained':True,'bbox_is_not_pixel_mask_or_precise_corner':True,'future_intervention_requires_separate_protocol':True,'selected_model':None,'adopted':False,'fully_solved':False}
r=R/'COMPLETE.json';r.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':bind(fout),'complete':bind(r),'region_status_counts':counts},indent=2))
