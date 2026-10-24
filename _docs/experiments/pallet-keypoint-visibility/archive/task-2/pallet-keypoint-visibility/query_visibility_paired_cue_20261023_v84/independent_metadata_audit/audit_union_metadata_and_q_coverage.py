"""Metadata-only union parity and analytic fixed-source crop containment."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,csv
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_paired_cue_20261023_v84';R=V/'independent_metadata_audit'
OLD=P/'query_visibility_controlled_real_20261022_v66';Q=OLD/'public_pallet/PAIRED_QUERIES'
def rd(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def check(x):assert bind(x['path'])==x,x['path']
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def geometry(xy,box,q,L,hw):
 xy,box,q=np.array(xy,float),np.array(box,float),np.array(q,float)
 assert box.shape==(4,) and (box[2:]>box[:2]).all()
 crop=np.r_[q-L/2,q+L/2]
 h,w=hw;validcrop=np.r_[np.maximum(crop[:2],[0,0]),np.minimum(crop[2:],[w,h])]
 lo=np.maximum(box[:2],validcrop[:2]);hi=np.minimum(box[2:],validcrop[2:]);inter=float(np.maximum(hi-lo,0).prod())
 area=float((box[2:]-box[:2]).prod());full=bool((box[:2]>=validcrop[:2]).all()and(box[2:]<=validcrop[2:]).all())
 state='FULL'if full else'PARTIAL'if inter>0 else'MISS'
 center=bool((xy>=validcrop[:2]).all()and(xy<validcrop[2:]).all())
 return {'crop_xyxy_half_open':crop.tolist(),'image_clipped_valid_crop_xyxy':validcrop.tolist(),'uncertainty_coverage':state,'uncertainty_area_fraction':inter/area,'reference_center_in_crop':center}
# Generated set-boundary tests: equal endpoints subset; touch-only has no area;
# center on upper bound is outside, and native image clipping can block FULL.
assert geometry([5,5],[0,0,10,10],[5,5],10,[20,20])['uncertainty_coverage']=='FULL'
assert geometry([10,5],[10,0,11,10],[5,5],10,[20,20])['uncertainty_coverage']=='MISS'
assert geometry([10,5],[9,0,11,10],[5,5],10,[20,20])['reference_center_in_crop']is False
assert geometry([0,0],[-1,-1,1,1],[0,0],4,[20,20])['uncertainty_coverage']=='PARTIAL'
u_path=V/'ROOT_BEFORE_REFERENCE_UNION_LOCK.json';u=rd(u_path)
assert bind(u_path)['sha256']=='ae6e71a2a85733bcb7ed7081e3317a04d0a612d3e69fd27d51753d7d01e4dd4a'
for b in u['bindings']:check(b)
a_path=V/'reviewer_a/BEFORE_CUE_PROPOSALS.json';b_path=V/'reviewer_b/BEFORE_ONLY_CUE_PROPOSALS.json'
a,b=rd(a_path),rd(b_path);md=rd(V/'FIXED_ONE_PAIR_METADATA_BEFORE_RGB.json')
expected_before=md['raw_files']['before']['image'];assert a['source_before_RGB']==expected_before
assert all(b['source_before_only'][k]==expected_before[k]for k in ['path','sha256'])
assert b['source_before_only']['raw_hw']==a['raw_hw']==[480,640]
originals={x['cue_id']:('A',x)for x in a['proposals']};originals.update({x['proposal_id']:('B',x)for x in b['proposals']})
assert len(originals)==8 and len(u['proposals'])==u['proposals_N']==8
assert len({x['union_id']for x in u['proposals']})==8
parity=[]
for x in u['proposals']:
 ident=x['union_id'];reviewer,origin=originals[ident]
 assert x['reviewer']==reviewer and x['original_proposal']==origin
 xy=origin['native_xy']if reviewer=='A'else origin['reference_xy']
 box=origin['uncertainty_bbox_xyxy_half_open']if reviewer=='A'else origin['uncertainty_bbox_xyxy']
 desc=origin['image_part_name_KO']if reviewer=='A'else origin['cue_name']
 assert x['reference_xy_unchanged']==xy and x['uncertainty_bbox_xyxy_half_open_unchanged']==box and x['cue_description_unchanged']==desc
 assert x['native8_status']=='UNKNOWN'and origin['native8_semantic_id']is None and origin['training_admitted']is False
 parity.append({'union_id':ident,'reviewer':reviewer,'entire_original_proposal_dict_exact':True,'xy_box_description_exact':True})
assert u['possible_duplicate_groups_before_only']==[['V84_A_CUE_00','B01'],['V84_A_CUE_01','B02']]
links=u['possible_duplicate_groups_before_only'];assert len({z for t in links for z in t})==4
assert u['qualitative_distinct_part_cue_regions_N']==8-sum(len(z)-1 for z in links)==6
sa=rd(V/'reviewer_a/BEFORE_REVIEW_SEAL.json');sb=rd(V/'reviewer_b/BEFORE_REVIEW_SEAL.json')
assert sa['completed']and sa['PASS']and sb['completed']and sb['before_only']
assert sa['proposals']=={k:bind(a_path)[k]for k in sa['proposals']}
assert sb['proposals']=={k:bind(b_path)[k]for k in sb['proposals']}
# Parse only the two input-only lines of this exact frozen pair, not GT labels.
source_input=Q/'INPUTS_RGB_ONLY.jsonl';selected=[];selected_line_numbers=[]
with source_input.open()as f:
 for n,line in enumerate(f,1):
  if '"group_id": "PALLOXPAIR:0.55m_lab:0000_0001"'in line:
   selected.append(json.loads(line));selected_line_numbers.append(n)
assert len(selected)==2 and [x['id']for x in selected]==['PALLOXPAIR:0.55m_lab:0000_0001:A','PALLOXPAIR:0.55m_lab:0000_0001:B']
x,y=selected
for k in ['query','query8','side','box','raw_hw','group_id','split','source_split','source']:
 assert x[k]==y[k],k
assert x['query']==x['query8']and len(x['query'])==8
assert x['image']==expected_before['path']and x['image_sha256']==expected_before['sha256']
assert y['image']==md['raw_files']['after']['image']['path']and y['image_sha256']==md['raw_files']['after']['image']['sha256']
assert x['split']=='TRAIN'and x['source_split']=='TRAIN_LAB'and x['source']=='PALLOX'
side=float(x['side']);q=np.array(x['query'],float)
assert side==float(np.clip(.25*max(np.ptp(q[:,0]),np.ptp(q[:,1])),16,max(x['raw_hw'])))
qlock=rd(Q/'QUERY_INPUT_LOCK.json');check({**qlock['recipe'],'bytes':Path(qlock['recipe']['path']).stat().st_size})
recipe=rd(qlock['recipe']['path']);assert recipe['seed']==20261022
assert 'SAMteacher' in recipe['teacher_prompt_box']['source']
rows=[];percue=[]
for proposal in u['proposals']:
 ident=proposal['union_id'];cube=[]
 for slot,point in enumerate(x['query']):
  record={'cue_id':ident,'saved_source_query_slot':slot,**geometry(proposal['reference_xy_unchanged'],proposal['uncertainty_bbox_xyxy_half_open_unchanged'],point,side,x['raw_hw'])}
  rows.append(record);cube.append(record)
 counts=Counter(t['uncertainty_coverage']for t in cube)
 percue.append({'cue_id':ident,'all8_crop_states':[t['uncertainty_coverage']for t in cube],
  'counts_FULL_PARTIAL_MISS':[counts['FULL'],counts['PARTIAL'],counts['MISS']],
  'any_FULL':counts['FULL']>0,'any_intersection':counts['FULL']+counts['PARTIAL']>0,
  'any_center_in_crop':any(t['reference_center_in_crop']for t in cube),
  'availability_summary_only_not_GT_best_query_selection':True})
with (R/'FIXED_SOURCE_Q8_CUE_COVERAGE.csv').open('x')as f:
 cols=['cue_id','saved_source_query_slot','uncertainty_coverage','uncertainty_area_fraction','reference_center_in_crop','crop_xyxy_half_open','image_clipped_valid_crop_xyxy'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
summary={'original_proposals':8,'qualitative_distinct_links_only':6,'source_geometries_per_view':8,'cue_by_crop_combinations_one_geometry_set':64,
 'combinations_FULL_PARTIAL_MISS':dict(Counter(t['uncertainty_coverage']for t in rows)),
 'proposals_any_FULL':sum(t['any_FULL']for t in percue),'proposals_any_intersection':sum(t['any_intersection']for t in percue),
 'proposals_any_center_in_crop':sum(t['any_center_in_crop']for t in percue),
 'A_B_metadata_geometry_exact_and_all_coverage_identical':True,'independent_source_RGB_frames':2,'independent_physical_pallet_instances':1}
report={'schema':'V84_UNION_METADATA_PARITY_AND_SAVED_SOURCE_INPUT_COVERAGE','completed':True,'PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
 'union_sha256':bind(u_path)['sha256'],'union_original_proposals_preserved':parity,
 'all8_full_original_dict_and_source_before_bindings_exact':True,'linked_duplicate_groups_only':links,
 'duplicates_not_averaged_or_geometry_moved':True,'qualitative_distinct_regions_arithmetic':6,
 'qualitative_identity_not_independently_RGB_certified_by_this_metadata_audit':True,
 'before_seal_binding_integrity':{'A':bind(V/'reviewer_a/BEFORE_REVIEW_SEAL.json'),'B':bind(V/'reviewer_b/BEFORE_REVIEW_SEAL.json')},
 'original_input_lineage_KO':'V66 source-guided q8이다. SAMteacher A/B union bbox를 jitter한 roughbox, conditional V/E evidence로 선택한 source anchors2Ecore/2commonV/2Vboundary/2boxcorners에 동일±3pxjitter. 실제 frozen detector/native8 initializer가 아니다. 슬롯0..7은 sampler 위치일 뿐 native8 의미가 없다. source input file은 GTfree 필드이지만 생성분포/box는 teacher-guided였다.',
 'selected_input_line_numbers':selected_line_numbers,'source_input_rows_this_pair_only':selected,
 'q8_L_box_rawHW_A_B_exact':True,'L_from_unchanged_q8_recomputed_exact':True,
 'coverage_formula':'float64 continuous half-open square [q-L/2,q+L/2) intersect native image [0,W)x[0,H). FULL iff whole frozen half-open uncertainty box subset; PARTIAL iff positive-area intersection but notFULL; elseMISS. Center membership separately half-open. No epsilon, changed radius, GT snap or crop/q movement. Not model receptive-field or semantic visibility proof.',
 'summary':summary,'per_original_cue_all8':percue,
 'honest_auxiliary_test_KO':[
  '기존두RGB의 동일q/L/box를 그대로 사용한 paired source availability/part-cue evidence를 검사할 수 있다. 모든8×8조합을 남기며 cue에가장가까운q를GT로고르지않는다.',
  'FULL/PARTIAL/MISS는 입력에서 frozenreference uncertainty가 포함되는 기하조건이며 가려짐E나D label이 아니다. 특히 MISS를E로 만들 수 없다.',
  'after evidence가 별도로동결돼야 같은 fixedreference의 visible/external/partial/U를 논할 수 있다. 이감사는after사진·after판단을읽지않았다.',
  '후속source-onlyaux test를 하더라도 teacher-guided sourceq 조건과 crop포함별분모를 명시해야 하며 actual detector/native8 deployment test나 인간DESgold라고할수없다.',
  '일부crop에여러candidate physicalrefs가있으면 q+RGB가어느것을질의하는지 추가reference-selector계약이필요하다. GTcueID/좌표를모델에주거나 nearestq로nativeID를만들면안된다.'
 ],
 'generated_boundary_tests_PASS':True,'new_labels_admitted':0,'source_RGB_after_reviews_mask_depth_ML_features_weights_current_newforward_fit_PnP_reads':0,
 'old_SEAL_and_report_modified':False,'bindings':[bind(u_path),bind(a_path),bind(b_path),bind(V/'FIXED_ONE_PAIR_METADATA_BEFORE_RGB.json'),bind(source_input),bind(Q/'QUERY_INPUT_LOCK.json'),bind(OLD/'PAIRED_QUERY_RECIPE.json'),bind(Q/'QUERY_RECIPE_INTERPRETATION.json'),bind(Path(__file__))]}
o=R/'UNION_METADATA_PARITY.json';save(o,report)
s=R/'UNION_METADATA_PARITY_SEAL.json';save(s,{'completed':True,'PASS':True,'files':[bind(o),bind(R/'FIXED_SOURCE_Q8_CUE_COVERAGE.csv'),bind(Path(__file__))],'old_SEAL_or_report_modified':False,'new_RGB_after_mask_depth_model_current_forward_fit_PnP':0})
print(json.dumps({'audit':bind(o),'seal':bind(s),'summary':summary,'percue':percue},ensure_ascii=False))
