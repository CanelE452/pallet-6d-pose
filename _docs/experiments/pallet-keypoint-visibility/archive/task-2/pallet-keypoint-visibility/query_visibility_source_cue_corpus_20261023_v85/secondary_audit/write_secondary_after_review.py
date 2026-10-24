from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib
R=Path(__file__).resolve().parent;V=R.parent
rd=lambda p:json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
packet=rd(V/'SECONDARY5_REVIEW_PACKET_WITHOUT_PRIMARY_AFTER.json');before=rd(R/'SECONDARY_BEFORE_SUPPORT.json');bmap={(x['pair_id'],x['part_slot']):x for x in before['records']}
assert rd(R/'SECONDARY_BEFORE_SUPPORT_SEAL.json')['no_after_read_before_this_seal']
VIS='APPARENT_SAME_CUE_VISIBLE_AI';EXT='EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI';UNC='PARTIAL_OR_UNCLEAR_U';MISS='MISSING_BEFORE_REFERENCE_U'
# No primary-after file, seal, contact or batch result is opened by this writer.
after={
'1.92m_lab:0000_0001':[
(VIS,'앞wall 왼쪽 위의 동일rim/세로끝 junction과 주변목재가 그대로 드러난다. 새cardboard는 화면왼쪽아래에만 일부들어와 이 frozenregion을 덮지 않는다.',None,False),
(VIS,'앞wall 오른쪽 topcorner의 목재윗선과 끝면이 같은 위치범위에서 보인다. foreground cardboard가 여기에 오지 않는다.',None,False),None,None,None,None],
'0.55m_lab:0016_0017':[
(VIS,'앞wall 왼쪽 상단이 cardboard위로 드러나며 동일topedge/outeredge junction이 관찰된다. cardboard윗선은 이 cue보다 충분히 아래다.',None,False),
(VIS,'오른쪽 topcorner와 두incidentwoodline가 그대로 보인다.',None,False),
(EXT,'별개의 큰 갈색 cardboard 앞면이 이전 lowestwideboard-left junction의 전체 frozen8px범위를 덮는다. 전후 rawcrop에서 목재lip이 사라지고 종이면이 같은자리보다넓게 차지한다.','foreground cardboard box; opaque paper front face',True),
(VIS,'오른쪽 lowestwideboard 끝/lip은 box바깥에 있고 동일목재경계가 보인다.',None,False),
(EXT,'왼쪽 base-runner 끝의 원래위치 전체범위가 foregroundcardboard앞면 안에 들어간다. box외곽이 referencebox에서멀어 fullcover가 분명하다.','foreground cardboard box; opaque side/front face',True),
(VIS,'오른쪽 base runner의 얇은 윗선/끝 contour가 노출되어 있다. cardboard는왼쪽에있다.',None,False)],
'0.55m_hall:0004_0005':[
(VIS,'원래노출됐던 principalwall 왼쪽topcorner가 동일위치에서 계속보인다. 큰cardboard는 after에서영상더왼쪽으로옮겨 이cue를덮지않는다.',None,False),None,None,None,None,None],
'1.23m_hall:0032_0033':[
(VIS,'앞wall 왼쪽topcorner와윗선은 새cardboard상단보다위로 노출되어있다.',None,False),
(VIS,'오른쪽topcorner는 foregroundbox와멀리 떨어져 그대로보인다.',None,False),
(EXT,'원래 lowestwideboard-left corner 전체8px범위가 새cardboard의 불투명앞/옆면 내부로 덮인다. box위쪽rim/side가참조범위보다충분히넓게둘러선다.','foreground cardboard box',True),
(VIS,'오른쪽lowestwideboard 끝은 여전히목재lip/세로외곽이노출되어식별된다.',None,False),None,None],
'1.23m_hall:0024_0025':[
(VIS,'앞wall왼쪽topcorner가 box상단보다위에있고같은woodjunction을볼수있다.',None,False),
(VIS,'오른쪽topcorner와principalwall윗line가그대로보인다.',None,False),
(EXT,'원lowestwidewall-left corner의 전체9px범위는 cardboard의명백한불투명면에덮인다. 원woodlip은이위치에서보이지않는다.','foreground cardboard box',True),
(VIS,'오른쪽lowestwidewall하단끝/lip은box바깥에있어이전과대응되는구조를볼수있다.',None,False),
(UNC,'고정base-left 범위는 분명cardboard면에전부덮이지만 before에서명명된runner-junction근거가독립적으로불충분했다. physicalsamecue E로승격하지않고U. 범위cover사실은별도기록한다.','foreground cardboard box',True),
(UNC,'고정base-right 범위의 support/runner주변이노출되어있지만 before에서upperrunner-end partidentity가이미불분명했다. 같은지점의일반목재노출을namedcue V확정으로바꾸지않고U.',None,False)]}
allrecords=[];pair_reports=[]
reg='Full scenes and unchanged visible wall/background edges show no obvious gross camera/target movement. This is qualitative only, not calibrated subpixel registration. No warp, pose fitting or reference correction.'
for n,row in enumerate(packet['rows']):
 primary=rd(row['before_annotation']['path']);records=[]
 for k,s in enumerate(primary['slots']):
  prev=bmap[(row['pair_id'],s['part_slot'])];a=after[row['pair_id']][k]
  if s['reference_xy']is None:
   assert a is None;state=MISS;evidence='Before reference/uncertainty가 null이므로 after에서보이거나풀렸을가능성과무관하게MISSING으로유지한다. 새위치를만들거나E로채우지않는다.';foreign=None;coverage=None
  else:
   assert a is not None;state,evidence,foreign,coverage=a
   if prev['before_supported_by_RGB']is not True:assert state==UNC
   if state==EXT:assert coverage is True and foreign is not None and prev['before_supported_by_RGB']is True
  rec={**prev,'after_state':state,'after_evidence_KO':evidence,'foreign_blocker':foreign,
   'coverage_entire_frozen_region':coverage,'registration_note':reg,'original_coordinates_uncertainty_preserved':True,
   'after_RGB_observed_independently_primaryafter_quarantined':True,'pixelmask_material_gold':False,'native8_semantic_id':None,'human_gold':False,'training_admitted':False}
  records.append(rec);allrecords.append(rec)
 name=f'{n:02d}_'+row['pair_id'].replace(':','__');folder=R/name
 observed_files=[folder/'BEFORE_FULL_RAW_MARKED.png',folder/'BEFORE_FIXED6_RAW96_DETAILS.png',folder/'AFTER_FULL_RAW_MARKED.png',folder/'AFTER_FIXED6_RAW96_DETAILS.png']
 report={'schema':'V85_SECONDARY_PAIR_INDEPENDENT_FIXED_REFERENCE_REVIEW','completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
  'pair_id':row['pair_id'],'source_split':row['source_split'],'all6_originalprimarybefore_slots':True,
  'primary_before_annotation':row['before_annotation'],'primary_before_seal':row['before_seal'],
  'raw_before_RGB':row['before_RGB'],'raw_after_RGB':row['after_RGB'],
  'secondary_before_support_seal':bind(R/'SECONDARY_BEFORE_SUPPORT_SEAL.json'),'source_RGB_and_paired_detail_actual_views':[bind(p)for p in observed_files],
  'records':records,'state_counts':dict(Counter(x['after_state']for x in records)),
  'primary_after_artifacts_read':0,'sourceproxy_q_masks_depth_ML_current_forward_fit_PnP':0,
  'all_reference_xy_and_uncertainty_exact_original':True,'training_admission':False}
 out=folder/'SECONDARY_FIXED_REFERENCE_REVIEW.json';save(out,report);s=folder/'SECONDARY_PAIR_SEAL.json';save(s,{'completed':True,'PASS':True,'report':bind(out),'files':[bind(p)for p in observed_files],'primaryafter_comparison_not_performed':True})
 pair_reports.append({'pair_id':row['pair_id'],'report':bind(out),'seal':bind(s),'counts':report['state_counts']})
assert len(allrecords)==30
stats={'pairs':5,'slots':30,'original_primary_reference_present':sum(x['reference_xy']is not None for x in allrecords),
 'before_supported_by_RGB':dict(Counter('true'if x['before_supported_by_RGB']is True else'false'if x['before_supported_by_RGB']is False else'null'for x in allrecords)),
 'after_states':dict(Counter(x['after_state']for x in allrecords)),
 'FOREIGN_full_cover_with_supported_before':sum(x['after_state']==EXT for x in allrecords),
 'region_full_cover_but_before_identity_uncertain_kept_U':sum(x['coverage_entire_frozen_region']is True and x['after_state']==UNC for x in allrecords),
 'unique_source_RGB_files':10,'paired_capture_units':5,'physical_bin_instances':1,'human_native8_gold_or_training_rows_admitted':0}
summary={'schema':'V85_SECONDARY5_ACTUAL_RAW_FIXED30_AUDIT','completed':True,'PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
 'policy':bind(R/'REVIEW_POLICY_BEFORE_RGB.json'),'packet':bind(V/'SECONDARY5_REVIEW_PACKET_WITHOUT_PRIMARY_AFTER.json'),
 'summary':stats,'pairs':pair_reports,'all30_records':allrecords,
 'scientific_scope_KO':'고정5쌍/30slot의 독립AI 실제RGB 검토다. before지원은after열기전에봉인했고原19nonnull/11null geometry를보존했다. 모델/마스크/깊이proxy가아닌partline/foreground물체관찰증거이며 human/native8gold가아니다. 원본지원이불명확한2개base는after범위cover/일반목재가보여도namedcue V/E를확정하지않았다. primaryafter동의율은여기서계산하지않으며root가양쪽seal뒤에비교한다. 분모5쌍은한물리bin의반복촬영이고독립object검증이아니다.',
 'no_primary_annotators_judgments_received_or_read':True,'judgments_not_shared_with_primary_annotators':True,'all_primary_after_files_seals_contacts_batchresults_read':0,
 'sourceproxy_q_mask_depth_model_current_newforward_fit_PnP':0,'originalcoords_box_or_missingreference_modified':False,'training_admitted':False,
 'bindings':[bind(R/'SECONDARY_BEFORE_SUPPORT_SEAL.json'),bind(R/'BEFORE_RENDER_RECEIPT.json'),bind(R/'AFTER_RENDER_RECEIPT.json'),bind(Path(__file__)),bind(R/'render_secondary_fixed.py'),bind(R/'write_before_support.py')]}
o=R/'SECONDARY5_REVIEW_SUMMARY_KO.json';save(o,summary)
seal=R/'SECONDARY5_FINAL_REVIEW_SEAL.json';save(seal,{'completed':True,'PASS':True,'summary':bind(o),'pair_reports_and_seals':pair_reports,
 'before_support_seal':bind(R/'SECONDARY_BEFORE_SUPPORT_SEAL.json'),'before_receipt':bind(R/'BEFORE_RENDER_RECEIPT.json'),'after_receipt':bind(R/'AFTER_RENDER_RECEIPT.json'),
 'code':bind(Path(__file__)),'primaryafter_read_or_comparison':False,'original_all30_geometry_retained':True,'new_forward_fit_current_PnP':0})
print(json.dumps({'summary':bind(o),'final_seal':bind(seal),'counts':stats},ensure_ascii=False))
