"""Reviewer raw-RGB observations frozen before any source proxy annotations."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math
R=Path(__file__).resolve().parent;V=R.parent
scope=json.loads((V/'SOURCE_NATIVE8_VISUAL_SCOPE_BEFORE_RGB.json').read_text());targets=json.loads(Path(scope['targets']['path']).read_text())['targets']
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def save(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
S='OBSERVED_STRUCTURAL_CUE_CONDITIONAL_AI';N='NO_IDENTIFIABLE_CUE_OR_AMBIGUOUS_U';I='REFERENCE_IDENTITY_UNRESOLVED_U';O='OUT_OF_FRAME_U'
# Tuple: category, virtual region, direct raw evidence, alternative/identity limit,
# optional observed physical cue range (not replacement), association judgment.
notes={
'G38__G__f36687':[
(S,'영상 앞쪽 넓은 slat의 왼쪽 윗 외곽 부근','갈색 가로 slat 끝의 밝은 윗면이 왼쪽 어두운 세로 외곽으로 꺾이는 작은 junction이 r 근처에서 보인다.','상당히 작은/어두운 pallet이다. 보이는 board 끝과 virtual cuboid corner의 동일성·정확한 재질 pixel은 확정할 수 없다.',[598,342,605,350],'앞 slat 외곽 part와의 국소 대응은 그럴듯하나 native3D corner 의미는 미확정'),
(S,'영상 앞쪽 slat의 오른쪽 윗 외곽 부근','앞 slat 오른쪽 끝에서 상면/짧은 끝면/바닥 배경이 만나는 외곽 변화가 보인다.','r 주변의 blur와 둥근 끝 때문에 정확한 한 점은 불명확하다. 다른 물체가 앞을 덮은 모습은 없다.',[701,333,709,342],'앞 오른쪽 board-end와 conditional 대응 가능'),
(S,'영상 앞쪽 오른쪽 하부 외곽 부근','짙은 앞면/하단 support 끝이 바닥과 나뉘는 오른쪽 외곽 종료 영역을 볼 수 있다.','검은 자체 그림자와 어두운 목재가 섞인다. 상부/하부 virtual vertex를 정확히 어느 물리선에 붙일지는 미확정.',[697,343,705,352],'target 외곽 part region은 관찰되나 semanticD는 미정'),
(N,'영상 앞쪽 왼쪽 하부 virtual extent','r 부근은 아주 어두운 앞면·하단/바닥 접촉 그림자와 가까워 실제 물리 junction을 별개로 식별하기 어렵다.','목재 자체 음영이나 바닥 그림자를 foreign blocker로 부를 수 없다. 같은 왼쪽 board 주변 cue가 존재해도 하단 native identity는 불명확하다.',None,'어두운 접촉 경계/자기음영 모호성'),
(N,'영상 뒤쪽 왼쪽 상부 virtual extent','고정 r는 뒤 slat 끝의 보이는 밝은 윗 윤곽보다 영상 위 바닥 쪽에 놓인다. 가까운 board-end cue는 아래쪽에 있다.','외부 물체가 r를 덮는 것은 보이지 않는다. virtual bbox 여유/불규칙 slat geometry와 실제 part-cue의 불일치 가능성이 있다.',[577,297,585,304],'nearby physical cue와 고정 r는 구분; 거리만으로 D/E를 만들지 않음'),
(N,'영상 뒤쪽 오른쪽 상부 virtual extent','r는 뒤 slat의 오른쪽 끝/윗 윤곽 위쪽 배경 바닥에 가까우며, 가까운 물리 slat 경계는 아래·오른쪽에 보인다.','clear foreign occluder가 없다. r의 정확한 physical corner 대응은 불명확하며 먼 뒤쪽 virtual bbox extent일 수 있다.',[674,290,683,297],'근처 slat junction은 있으나 고정virtual identity 검증 불충분'),
(I,'영상 뒤쪽 오른쪽 하부 virtual extent','뒤쪽 slat/top deck 일부와 검은 slat 틈이 보인다. r에 독립된 하부 corner가 드러난다고 식별하기 어렵다.','자기 상면/board 틈에 투영된 뒤쪽 하부 reference일 가능성; hole이나 자기 표면만으로 S 확정 불가.',None,'own-surface/hole과 virtual lower vertex의 관계 미해결'),
(I,'영상 뒤쪽 왼쪽 하부 virtual extent','외곽의 어두운 세로/상부 slat 접합 근처다. target의 옆면과 slat 끝이 보이지만 별개의 하부 virtual corner는 분리되지 않는다.','가까운 상부 slat corner를 하부 native slot의 physical counterpart로 바꾸면 안 된다. foreign blocker 증거는 없다.',None,'자기면/상부part junction과 하부native identity가 구분되지 않음')],
'G38__G__f16972':[
(S,'영상 앞쪽 왼쪽 top/side extent','파란 molded pallet의 둥근 전면 왼쪽 림이 상부 deck에서 세로 support로 이어지는 외곽이 r 근처에 분명히 보인다.','둥근 모서리라 single vertex는 없다. plastic rim region의 관찰이며 canonical cuboid 꼭짓점의 exact pixel gold는 아니다.',[274,239,281,247],'rounded front-left physical contour 대응 plausible'),
(O,'영상 오른쪽 밖으로 나간 앞쪽 상부 virtual extent','r.x=670.41이 native width640 밖이다. raw padding 검정은 scene의 물체가 아니다.','clipping/truncation이며 external/self 가림이 아니다.',None,'OUT_OF_FRAME'),
(O,'영상 오른쪽 밖으로 나간 앞쪽 하부 virtual extent','r.x=653.27이 native width640 밖이다.','native crop 안의 검정 padding을 occluder/hole로 간주하지 않는다.',None,'OUT_OF_FRAME'),
(S,'영상 앞쪽 왼쪽 하단 extent','둥근 파란 support 하단의 바닥 접촉 외곽/곡선 끝을 r 가까이서 볼 수 있다.','r가 바닥/그림자 인접 pixel일 수 있다. 구조적 하단 region과 exact membership/semanticD를 분리한다.',[276,307,285,315],'rounded bottom-foot contour region 대응 plausible'),
(S,'영상 뒤쪽 왼쪽 상부 extent','pallet far-left deck 외곽이 뒤쪽 윗선에서 왼쪽 옆면으로 꺾이는 뚜렷한 rim junction이다.','rounded molded corner와 virtual extreme의 정확한 한 점 동일성은 별도 문제다.',[240,139,247,147],'far-left top rim physical cue의 conditional 대응'),
(S,'영상 뒤쪽 오른쪽 상부 extent','먼 쪽 deck 외곽선이 오른쪽 비스듬한 outline으로 방향을 바꾸는 영역이 r 근처에 보인다.','미세한 grid pattern은 cue가 아니다. 외곽 turn region을 본 것이며 exact vertex나 mesh label은 확인하지 않았다.',[438,132,448,139],'far-side outer rim turn과의 conditional 대응'),
(I,'영상 뒤쪽 오른쪽 하부 virtual extent','r는 파란 perforated top deck 내부의 규칙적 rib/구멍 pattern 위에 놓인다. 외곽 하부 corner가 직접 드러나는 장면은 아니다.','own top surface가 뒤 하부 reference의 투영을 가릴 수 있으나 RGB만으로 hidden corner나 S를 확정하지 못한다. foreign blocker는 없다.',None,'own deck/hole pattern 대 virtual lower vertex 미해결'),
(I,'영상 뒤쪽 왼쪽 하부 virtual extent','왼쪽 옆면의 좁고 어두운 fork opening/외곽선 근처에 r가 있다.','검은 opening과 자체면/원거리 하부 vertex 중 무엇이 정확한 대응인지 불분명하다. mask0나 black pixels만으로 E/S를 만들 수 없다.',None,'side face/opening/native lower vertex 관계 미해결')],
'G38__G__f29051':[
(S,'영상 왼쪽 아래로 향한 상부/앞쪽 virtual extent','회색 slatted pallet의 outer band가 바닥과 만나는 둥근 왼쪽 아래 contour turn이 r 주위에 보인다.','둥근 band·그림자·작은 수직 두께를 분리하기 어려워 상하 virtual 두점의 정확한 physical mapping은 확정하지 않는다.',[446,467,457,477],'visible outer corner region 대응 plausible'),
(O,'영상 아래로 잘린 virtual extent','r.y=588.56이 native height540 밖이다.','외부 가림이 아니라 truncation. 검정은 inspection padding이다.',None,'OUT_OF_FRAME'),
(O,'영상 아래로 잘린 인접 virtual extent','r.y=587.30이 native height540 밖이다.','외부 가림이나 자체 구멍이라고 부르지 않는다.',None,'OUT_OF_FRAME'),
(I,'영상 왼쪽 아래의 하부 virtual extent','같은 rounded outer band 끝 바로 아래/배경 인접으로 투영되어 별개의 하부 corner를 구별하기 어렵다.','k0 근처 물리 contour 하나를 두 native slots의 gold identity로 복제할 수 없다. own edge/shadow 또는 virtual extent 차이 가능성.',None,'인접상부cue와 native lower slot의 identity 미해결'),
(S,'영상 위쪽 왼쪽 top extent','회색 외곽 rim의 위쪽 왼쪽 둥근 endpoint/방향전환이 r 주변에서 보인다.','r는 contour 바깥 배경 쪽과 가까울 수 있다. 물리 region을 본 것이며 point membership은 별도다. 중앙 cardboard는 이 region을 가리지 않는다.',[518,317,528,327],'outer top rim end의 conditional 대응'),
(S,'영상 위쪽 오른쪽 top extent','긴 outer band의 오른쪽 끝에서 외곽 방향이 바뀌는 rounded board/rim end가 r 가까이에 보인다.','끝면 두께/그림자로 한 점은 모호하다. 중앙 cardboard load가 보이지만 고정 r를 직접 덮는 증거는 없다.',[728,407,737,420],'right outer rim turn region 대응 plausible'),
(I,'영상 위쪽 오른쪽 하부 virtual extent','r가 같은 오른쪽 outer band의 어두운 면/끝부 근처에 놓이며 독립된 하부 corner는 분리되지 않는다.','visible upper rim end를 lower native vertex로 대신하면 안 된다. 자체면/그림자 가능성, foreign blocker 없음.',None,'upper/side part와 native lower identity 미해결'),
(I,'영상 위쪽 왼쪽 하부 virtual extent','r는 왼쪽 위 rim 안쪽/아래 part-line에 가깝다. 주변 target 구조는 보이지만 하부 virtual corner를 직접 식별할 수 없다.','slat gap과 own rim 면 사이 모호함. 고정 r에 중앙 cardboard가 겹치지 않으며 semanticS는 RGB만으로 미확정.',None,'own rim/slot gap 대 lower reference 미해결')]
}
records=[]
for target in targets:
 ident=target['id'];h,w=target['raw_hw']
 assert len(notes[ident])==8
 for k,n in enumerate(notes[ident]):
  category,region,direct,alternative,physical_range,association=n;r=target['fixed_existing_renderer_r8'][k];q=target['preserved_input_q8'][k]
  inside=0<=r[0]<w and 0<=r[1]<h;assert inside==(category!=O)
  # Optional observed region, never an altered reference or query.
  diagnostic=None
  if physical_range:
   lo=physical_range[:2];hi=physical_range[2:]
   dx=max(lo[0]-r[0],0,r[0]-hi[0]);dy=max(lo[1]-r[1],0,r[1]-hi[1]);d=math.hypot(dx,dy)
   diagnostic={'distance_fixed_r_to_observed_region_min_px':d,'region_intersects_fixed4px_disk':d<=4,'not_D_E_S_or_admission_rule':True}
  records.append({'id':ident,'native_k':k,'fixed_original_r':r,'unchanged_original_q':q,'permutation_unchanged':target['perm_v4_dynamic_to_original'],
   'in_native_frame':inside,'raw_evidence_category':category,'virtual_spatial_position_KO':region,'direct_RGB_observation_KO':direct,
   'alternative_or_identity_limit_KO':alternative,'foreign_foreground_blocker_at_fixed_reference':'NOT_OBSERVED'if inside else'NOT_ASSESSABLE_OUT_OF_FRAME',
   'native_part_association_conditional_AI_KO':association,'observed_physical_cue_optional_region_xyxy':physical_range,
   'observed_physical_cue_point_xy':None,'optional_region_diagnostic4px_only':diagnostic,'original_r_or_q_replaced':False,
   'semantic_D_E_S_gold':'UNKNOWN','self_S_from_RGB':'UNKNOWN','pixel_target_membership_GT':'UNKNOWN','human_or_official_verified':False,'training_admitted':False})
assert len(records)==24
counts=Counter(x['raw_evidence_category']for x in records)
report={'schema':'V86_INDEPENDENT_NATIVE24_RAW_RGB_OBSERVATION_SEAL_STAGE','completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
 'scope':'Actual full3 raw source RGB plus all24 unchangedr-centered raw96details inspected. No source annotation fraction/masks/depth/proxy/GTsupport read before this report.',
 'frames':[{'id':t['id'],'image':{'path':t['image'],'sha256':t['image_sha256']},'raw_hw':t['raw_hw'],'asset':t['source_asset']}for t in targets],
 'counts':dict(counts),'raw_observations':records,
 'frame_level_context_KO':{'G38__G__f36687':'갈색 slat pallet이 작고 어둡게 보인다. 주변 원거리 쓰레기통/기둥 등은 있지만 고정corner를 앞에서 덮은 물체는 관찰되지 않는다.','G38__G__f16972':'큰 파란 perforated molded pallet. 오른쪽 두native점을 포함해 바깥으로 잘린 구조가 있다. visible top deck/side openings는 material support와 hiddenvirtualpoint의 관계를 보장하지 않는다.','G38__G__f29051':'비스듬한 grey/slatted pallet 중앙에 cardboard load가 있다. load는 고정inframecorner들을 직접 덮지 않는다. 아래두native점은nativeimage밖.'},
 'interpretation_KO':'구조적 cue가 보이는 사례도 exact r pixel foreground나 canonicalnative identity의 gold를 뜻하지 않는다. virtual cube reference는 material surface 밖에 있을 수 있으며 이관찰을 physicalDIRECT admission gate로 쓰지 않는다. 관찰불가/ownface/hole/shadow/OOB는 자동E/S가 아니다.',
 'direct_full_panels_and_all8_detail_panels_viewed':True,'source_annotation_fraction_mask_depth_proxy_V76_GT_read_before_this_seal':0,
 'original_coords_permutation_order_changed':False,'new_forward_fit_current_PnP':0,
 'bindings':[bind(R/'RAW_REVIEW_POLICY_BEFORE_RGB.json'),bind(R/'RAW_INSPECTION_RENDER_RECEIPT.json'),bind(V/'SOURCE_NATIVE8_VISUAL_SCOPE_BEFORE_RGB.json'),bind(Path(__file__))]}
o=R/'RAW_NATIVE24_OBSERVATIONS.json';save(o,report)
s=R/'RAW_OBSERVATION_SEAL_BEFORE_PROXY.json';save(s,{'completed':True,'PASS':True,'raw_observations':bind(o),'policy':bind(R/'RAW_REVIEW_POLICY_BEFORE_RGB.json'),'render_receipt':bind(R/'RAW_INSPECTION_RENDER_RECEIPT.json'),'no_proxy_annotation_read_before_this_seal':True,'all24_native_slots_retained':True,'new_forward_fit_current_PnP':0})
print(json.dumps({'raw':bind(o),'seal':bind(s),'counts':dict(counts)},ensure_ascii=False))
