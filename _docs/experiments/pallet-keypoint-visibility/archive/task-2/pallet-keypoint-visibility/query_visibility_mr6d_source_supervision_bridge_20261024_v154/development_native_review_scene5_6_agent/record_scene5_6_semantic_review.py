"""Record actual source-structure visual review, independent of actual R0 q.
Manual per-frame eight-role proposals; no masks/proxy/model/nearest association.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V153=P/'query_visibility_mr6d_full_public_source_20261024_v153'
POLICY=V/'ROOT_POLICY_BEFORE_SOURCE_BRIDGE_REVIEW.json'
REFSEAL=V153/'native_reference_preparation/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json'
CONTACT=V153/'native_reference_preparation/source_reference_contact_v1/ALL230_SOURCE_REFERENCE_CONTACT_INDEX.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
# These explicit tuples were decided from all nine relevant source contacts and
# the full-size/raw anchor views recorded below. O is OOF; no data-driven rules.
MANUAL={
'000005':{
0:'DDDDUEUU',1:'DDDDUEUU',2:'DDDDUEUU',3:'DDDDUEUU',4:'DDDDUEUU',
5:'DDDDEEUU',6:'DDDDEEUU',7:'DDDDEEUU',8:'DDDDEEUU',9:'DDDDEEUU',10:'DDDDEEUU',11:'DDDDEEUU',
12:'DUUDDEUD',13:'DUUDDEUD',14:'DUUDDEUD',15:'DUUDDEUD',
16:'DDDDDEUD',17:'ODDODEUD',18:'ODDODEUD',19:'ODDODEUD',20:'ODDOOEUO',
21:'DDDDDEUD',22:'DDDDDEUD',23:'DDDDDEUD',24:'DDDDDEUD',25:'DDDDDEUD',26:'DDDDDEUD',
27:'DUUDUEUU',28:'DDDDUEUU',29:'DDDDDEUU',30:'DDDDDEUU',31:'DDDDDEUU',32:'DDDDDEUU'},
'000006':{
0:'DDDODEUU',1:'DDDODEUU',2:'DDDUDEUU',3:'DDDDDEUU',4:'DDDDDEUU',5:'DDDUDEUU',
6:'DDDODEUD',7:'DDDUDEUD',8:'DDDODEUD',9:'DDDODEUD',10:'DDDUDEUD',11:'DDDODEUD',12:'DDDODEUD',13:'DDDDDEUD',
14:'DDDDEDUU',15:'DDODEDUU',16:'DDODEDUU',17:'DOODEDUU',18:'DOODEDUU',19:'DOODEDUU',20:'DOOUEDUU',
21:'OOOOEDUU',22:'OOOOEDUU',23:'OOOOEDUU',24:'OOOOEDUU',25:'OOOOEDUU',26:'OOOOEDUU',27:'OOOOEDUU',
28:'ODUOOEUO',29:'ODUOOEUO',30:'ODUOOEUO'}}
assert set(MANUAL['000005'])==set(range(33))and set(MANUAL['000006'])==set(range(31))
assert all(len(s)==8 and set(s)<=set('DESOU')for d in MANUAL.values()for s in d.values())
FULL_ANCHORS={166,172,177,178,182,183,186,193,197,198,199,205,208,212,213,216,219,226,227,229}
RAW_ANCHORS={166,172,178,182,198}

policy=read(POLICY);projection=read(REFSEAL);assert projection['completed']and projection['PASS']and policy['source_reference_seal']==bind(REFSEAL)
refbinding=projection['references'];originalsha=sha(ck(refbinding));original=[json.loads(s)for s in Path(refbinding['path']).read_text().splitlines()if s]
refs=[r for r in original if r['scene']in MANUAL];assert len(refs)==512
index=read(CONTACT);frames=[r for r in index['frames']if r['scene']in MANUAL];assert len(frames)==64 and Counter(r['scene']for r in frames)=={'000005':33,'000006':31}
byframe={r['frame_index']:r for r in frames};wanted=set(byframe);pages=[r for r in index['pages']if set(r['frame_indices'])&wanted];assert len(pages)==9
assert {i for p in pages for i in p['frame_indices']if i in wanted}==wanted
contactbyframe={i:p['contact']for p in pages for i in p['frame_indices']if i in wanted}
viewrows=[]
for p in pages:
 ck(p['contact']);viewrows.append(dict(kind='EXISTING_GT_SOURCE_CONTACT_ACTUALLY_VIEWED',file=p['contact'],assigned_frame_indices=[i for i in p['frame_indices']if i in wanted]))
for i in sorted(FULL_ANCHORS):
 f=byframe[i];ck(f['all_instance_overlay']);viewrows.append(dict(kind='FULL_SIZE_GT_NATIVE_OVERLAY_ACTUALLY_VIEWED',frame_index=i,source_id=f['source_id'],file=f['all_instance_overlay']))
for i in sorted(RAW_ANCHORS):
 f=byframe[i];ck(f['raw_RGB']);viewrows.append(dict(kind='ORIGINAL_RAW_RGB_ACTUALLY_VIEWED',frame_index=i,source_id=f['source_id'],file=f['raw_RGB']))
viewpath=H/'ACTUAL_VISUAL_VIEW_RECORD.json'
write(viewpath,dict(completed=True,UTC=datetime.now(timezone.utc).isoformat(),all64_source_contact_cells_actually_viewed=True,contact_pages=9,full_size_overlay_images_actually_viewed=20,unoverlaid_raw_RGB_images_actually_viewed=5,views=viewrows,actual_R0_query_crop_candidate_or_visibility_score_reads_during_this_annotation=0,
 independence='Labels use only provided GT native roles and visible whole-source structure. This agent previously audited initializer inputs in other phases; this is not a claim of lifetime blindness to old source query boards. No actual query, selected box, prediction, confidence, feature or head score was read for this annotation task.'))

def evidence(r,state):
 scene,key,slot=r['scene'],int(r['frame_key']),r['native_slot']
 if state=='OOF':return('OUTSIDE_IMAGE','HIGH_CONTEXT_CLIPPED_ROLE','해당 팔레트의 연속된 외곽을 원본에서 따라보면 이 역할의 끝이 영상 위/아래 경계로 잘린다. 명목 투영은 경계 안내일 뿐 정확 pixel gold가 아니며, OOF를 외부가림 E로 바꾸지 않는다.')
 if state=='D':
  if scene=='000005':
   desc='세워진 GT3 팔레트의 노출된 외곽 상단/측면 끝과 이어지는 목재 구조를 확인했다.'
   if slot in (4,7)and key>=29:desc='작은 앞쪽 목재 패널과 분리되어 드러난 세워진 팔레트의 하단 왼쪽 외곽 끝을 확인했다.'
   elif 16<=key<=26 and slot in (1,2):desc='회색 적재함 오른쪽으로 나온 실제 팔레트의 하단 오른쪽 목재 끝/연결 구조를 확인했다. 명목 점이 배경에 조금 어긋나도 pixel 색으로 D/E를 바꾸지 않았다.'
  else:
   desc='평판 GT0 팔레트의 상판/받침 외곽 모서리와 인접한 같은 팔레트 목재의 연결을 원본에서 확인했다.'
   if slot in (4,7)and key<=13:desc='흰 포장블록 바깥으로 남은 팔레트의 왼쪽 뒤 목재 외곽/측면 끝을 확인했다. 흰 블록의 단순 pixel 점유와 구별했다.'
   elif slot in (1,5):desc='회색 바구니와 흰 포장블록 오른쪽 바깥으로 드러난 팔레트의 외곽 상단 목재 끝을 확인했다.'
  return('NONE_EXPOSED_STRUCTURAL_ROLE','MEDIUM_ROLE_NOT_EXACT_PIXEL',desc+' 원 reference는 cm급 불확실성이 있는 역할 안내이며 보이는 단일 query pixel이나 3px 좌표 정답을 뜻하지 않는다.')
 if state=='E':
  if scene=='000005':
   if key>=27 and slot==5:occ='GRAY_BOX';desc='팔레트의 오른쪽 하단 외곽 역할은 별도 회색 적재함 뒤로 이어져 직접 드러나지 않는다.'
   elif slot==4:occ='FOREGROUND_WOOD_PANEL';desc='팔레트의 왼쪽 하단 역할을 이어 보면 앞쪽 별도 목재 패널이 그 구조를 가린다.'
   else:occ='FOREGROUND_WOOD_CARGO';desc='팔레트의 하단 외곽 역할을 전체 목재 연결에서 따라 보면 별도 앞쪽 목재 상자/패널 뒤로 들어간다.'
  else:occ='GRAY_CRATE_AND_WHITE_PACKAGED_BLOCK';desc='뒤쪽 상단 역할은 같은 팔레트의 노출 목재 외곽에서 이어지지만, 별도 흰 포장블록과 그 위 회색 바구니에 가려 직접 드러나지 않는다.'
  return(occ,'MEDIUM_CONTEXT_EXTERNAL_BLOCKAGE',desc+' 외부 물체의 pixel 또는 mask0 하나로 E를 자동 부여한 것이 아니며, 명목 좌표를 이동하거나 보정하지 않았다.')
 if scene=='000005'and key in (12,13,14,15,27)and slot in (1,2):return('NOMINAL_BOUNDARY_AND_OCCLUDER_AMBIGUITY','UNRESOLVED','실제 팔레트 끝이 목재 상자/회색 적재함 경계 가까이 일부 드러나지만 명목 corner와의 cm급 차이 때문에 이 역할의 노출/가림을 확정하기 어렵다. U로 보존한다.')
 if scene=='000006'and((key in (2,5,7,10)and slot==3)or(key==20 and slot==3)):return('IMAGE_BOUNDARY_UNCERTAINTY','UNRESOLVED','하부 모서리가 영상 경계에 매우 가까워 실제 끝의 노출 여부와 cm급 reference 오차를 분리하기 어렵다. 좌표 in/out 계산만으로 D나 OOF를 결정하지 않고 U로 남긴다.')
 return('SELF_FOREIGN_OR_THIN_EDGE_UNRESOLVED','UNRESOLVED','뒤쪽 하부/얇은 두께 방향 역할은 팔레트 자체 목재와 외부 적재물의 차폐 또는 명목 경계 차이를 원본만으로 분명히 구분하기 어렵다. proxy나 mask로 S/E를 자동 부여하지 않고 U로 보존한다.')

records=[];perframe=[]
for r in refs:
 f=byframe[r['frame_index']];key=int(r['frame_key']);slot=r['native_slot'];state=MANUAL[r['scene']][key][slot];state='OOF'if state=='O'else state
 occ,conf,reason=evidence(r,state)
 assert r['split']=='SOURCE_DEVELOPMENT'and r['semantic_state']=='U_UNREVIEWED'
 record=dict(r,proposal_state=state,semantic_state_original_preserved=r['semantic_state'],assistant_weak_proposal=True,actual_visual_review=True,
  proposal_confidence=conf,occluder_or_uncertainty=occ,review_reason_KO=reason,source_entity_review='GT3_UPRIGHT_EURO_PALLET'if r['scene']=='000005'else'GT0_FLAT_EURO_PALLET',
  human_gold=False,train_ready=False,coordinate_supervision_ready=False,exact_reference_pixel_gold=False,source_native_visibility_candidate_only=state in ('D','E'),
  actual_query_entity_correspondence='NOT_READ_NOT_ASSUMED',actual_query_availability='NOT_READ',input_or_head_score_used_for_label=False,mask_used_to_autofill_label=False,convex_self_proxy_used_to_autofill_label=False,
  native_order_or_provided_pose_or_reference_changed=False,evidence_contact=contactbyframe[r['frame_index']],evidence_full_native_overlay_available=f['all_instance_overlay'],full_native_overlay_actually_viewed=r['frame_index']in FULL_ANCHORS,raw_RGB_available=f['raw_RGB'],raw_RGB_unoverlaid_actually_viewed=r['frame_index']in RAW_ANCHORS,root_independent_adjudication_pending=True,actual_query_association_pending=True)
 records.append(record)
for f in frames:
 rr=[r for r in records if r['frame_index']==f['frame_index']];assert len(rr)==8 and[r['native_slot']for r in rr]==list(range(8))
 perframe.append(dict(frame_index=f['frame_index'],source_id=f['source_id'],scene=f['scene'],frame_key=f['frame_key'],split=f['split'],GT_instance_indices=f['obj1_gt_instance_indices'],native8_proposal_states=[r['proposal_state']for r in rr],counts=dict(Counter(r['proposal_state']for r in rr)),source_contact_actually_viewed=True,full_overlay_actually_viewed=f['frame_index']in FULL_ANCHORS,raw_RGB_unoverlaid_actually_viewed=f['frame_index']in RAW_ANCHORS,human_gold=False,train_ready=False))
assert len(records)==512 and len(perframe)==64 and len({(r['frame_index'],r['gt_instance_index'],r['native_slot'])for r in records})==512
for old,new in zip(refs,records):assert {k:new[k]for k in old}==old
assert sha(refbinding['path'])==originalsha
allpath=H/'SOURCE_DEVELOPMENT512_NATIVE_ROLE_PROPOSALS_AGENT.jsonl'
with allpath.open('x')as f:
 for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
framepath=H/'DEVELOPMENT64_PER_FRAME_REVIEW.json';write(framepath,dict(completed=True,frames=perframe))
counts=dict(Counter(r['proposal_state']for r in records));sealpath=H/'SEMANTIC_REVIEW_COMPLETE.json'
write(sealpath,dict(completed=True,PASS=True,UTC=datetime.now(timezone.utc).isoformat(),code=bind(__file__),root_policy=bind(POLICY),source_reference_seal=bind(REFSEAL),source_references=refbinding,source_contact_index=bind(CONTACT),roles=bind(allpath),per_frame=bind(framepath),actual_views=bind(viewpath),frames=64,GT_instance_observations=64,native_roles=512,scenes={'000005':33,'000006':31},counts=counts,DE_role_proposals=sum(r['proposal_state']in ('D','E')for r in records),all512_original_native_roles_and_reference_fields_retained=True,original_reference_file_unchanged=True,source_validation45_proposals_read_or_modified=False,
 actual_query_candidate_crop_feature_visibility_score_weight_private_V112_reads_during_annotation=0,models_fit_PnP_or_coordinate_correction_calls=0,human_gold=False,train_ready=False,source_fit_admitted=False,root_independent_adjudication_pending=True,actual_query_association_pending=True,
 interpretation=['These are weak provided-GT-native role proposals, not actual query/crop examples. No actual selected entity/native correspondence is assumed.','No S is forced where own-structure versus foreign blockage is unclear; those lower/thin-edge roles remain U.','Scene5 upright orientation does not change the provided anatomical/native ordering. Roles are not relabelled by image top/bottom.','D includes exposed expected structural support with nominal reference uncertainty; it is not exact coordinate supervision.','O/OOF was explicitly visually reviewed, not inferred as E or produced by a mask.','All assigned source contacts and recorded full/raw anchors were actually viewed. Human gold and train-ready remain false.']))
print(json.dumps(dict(completed=True,PASS=True,seal=bind(sealpath),counts=counts,frames=64,roles=512),ensure_ascii=False))
