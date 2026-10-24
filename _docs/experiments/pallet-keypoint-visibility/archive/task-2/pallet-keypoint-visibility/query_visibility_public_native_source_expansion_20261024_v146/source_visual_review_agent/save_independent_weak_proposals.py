"""Save explicit assistant visual decisions after viewing the completed contacts.

No classifier, automatic pixel/depth label or coordinate-error selection.
"""
from pathlib import Path
from collections import Counter
import hashlib,json

H=Path(__file__).resolve().parent;V=H.parent;S=V/'saved_source64_contact_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
sealpath=S/'CONTACT_RENDER_COMPLETE.json'
assert sha(sealpath)=='ee5884b2c5beec495ef0de2d0c11b1dad7f9dd37d987964ee3192ba863c8d961'
seal=read(sealpath);assert seal['completed']and seal['PASS']
facts=[json.loads(x)for x in check(seal['all512_crop_facts']).read_text().splitlines()]
assert len(facts)==512

# Two characters are native2/native3, in original frame order. These are
# manually written visual proposals, not a rule evaluated against GT pixels.
lower_states=[
 'DD','DD','UU','UE','DE','UU','DE','DE',
 'UD','UD','UU','DD','DD','DD','UD','DE',
 'DD','UD','UU','DE','UU','DD','DE','DE',
 'DD','DD','UU','UU','DD','DD','UU','DD',
 'UD','DE','UD','DD','DD','DD','UD','UD',
 'DE','DD','UE','UD','UU','UU','DE','UU',
 'DE','DD','DD','DD','UU','DD','DD','DD',
 'DD','UD','UU','UD','UU','UU','DD','DE']
assert len(lower_states)==64
different={
 2:('DIFFERENT_YELLOW_BARRIER_FOOT','선택 박스는 왼쪽 작업대의 노란 차단 기둥 받침에 놓입니다. 제공 pose의 목재 조립체는 화면 오른쪽에 따로 있습니다.'),
 10:('DIFFERENT_YELLOW_BARRIER_FOOT','선택은 왼쪽 노란 차단 기둥의 작은 바닥 받침이며 뒤 목재 조립체가 아닙니다.'),
 20:('DIFFERENT_BACKGROUND_HANDTRUCK_OR_WHITE_LOAD_SUPPORT','선택은 목재 통 상단 뒤의 handtruck/흰 용기 받침 부근입니다. 큰 전경 조립체와 native 역할을 동일시할 수 없습니다.'),
 26:('DIFFERENT_BACKGROUND_HANDTRUCK_OR_WHITE_LOAD_SUPPORT','첫 선택의 작은 cuboid는 목재 통 뒤 흰 용기/handtruck 지지 부근입니다. 제공 pose의 앞 목재 조립체 하부 라벨을 복사하지 않습니다.'),
 27:('DIFFERENT_YELLOW_BARRIER_FOOT','선택 박스는 왼쪽 차단 기둥 받침입니다. 오른쪽의 목재 조립체는 별도 개체입니다.'),
 30:('SELECTED_ENTITY_UNRESOLVED_CARD_BOARD_OR_WOOD_ASSEMBLY','선택 cuboid는 앞 cardboard의 상단과 화면 밖 하부까지 걸칩니다. 목재 조립체의 하부를 선택했다고 확정하기 어렵고, 제공 하부 역할도 화면 밖입니다.'),
 44:('DIFFERENT_DISTANT_POST_FOOT','첫 선택은 뒤 작은 기둥 받침입니다. 오른쪽 목재 조립체의 source 역할과 연결하지 않습니다.'),
 45:('DIFFERENT_FOREGROUND_CARDBOARD','첫 선택의 작은 cuboid는 앞 cardboard 하부에 놓입니다. 뒤 목재 조립체의 가림 라벨을 이 상자의 native 라벨로 옮기지 않습니다.'),
 47:('DIFFERENT_FOREGROUND_CARDBOARD','첫 선택은 왼쪽 전경 cardboard에 놓이며 오른쪽 통합 목재 조립체와 다릅니다.'),
 52:('DIFFERENT_RED_PALLET_JACK','선택 박스는 앞 빨간 pallet jack 포크입니다. 제공 pose의 목재 조립체는 오른쪽에 따로 있습니다.'),
 58:('DIFFERENT_RED_PALLET_JACK','첫 선택은 앞 빨간 pallet jack의 포크이며 오른쪽 목재 조립체가 아닙니다.'),
 60:('DIFFERENT_FOREGROUND_CARDBOARD','선택 cuboid는 앞 cardboard 상자에 맞으며 오른쪽 목재 조립체와 별도 개체입니다.'),
 61:('DIFFERENT_DISTANT_POST_FOOT','첫 선택은 뒤 작은 기둥 받침입니다. 오른쪽 목재 조립체의 하부 역할 라벨을 옮기지 않습니다.')}
extra={
 5:'실제 선택은 같은 목재 조립체로 보이지만 제공 T가 SE3 검사를 통과하지 못해 하부 연속 역할이 UNRESOLVED입니다. 사후 허용오차 조정 없이2/3 U를 유지합니다.',
 8:'같은 조립체는 화면 오른쪽에 있고 왼쪽 하부 지지대는 노출되어 있습니다. 원 q3/cuboid 왼쪽은 바닥으로 크게 벗어나지만 보임 D를 좌표 오차로 제외하거나 E로 바꾸지 않습니다.',
 13:'개별 원본/crop 시트 재확인에서 전면에 cardboard가 없고 양쪽 하부 지지대가 노출되어 있습니다. 다른 프레임의 상자 가림을 복사하지 않았습니다.',
 17:'왼쪽의 별도 cardboard는 실제 목재 조립체의 왼쪽 하부와 떨어져 있습니다. 원 q3는 바닥으로 벗어나지만 노출된 왼쪽 지지 역할은 D입니다.',
 18:'같은 목재 조립체이지만 제공 pose의 두 하부 역할이 모두 화면 아래/오른쪽 밖입니다. 화면 안의 상부 널판만으로 하부 역할 D/E를 확정하지 않습니다.',
 36:'왼쪽 cardboard와 선택된 목재 조립체 사이에 바닥 간격이 보입니다. cardboard 존재 자체를 E로 쓰지 않았습니다.',
 37:'왼쪽 cardboard는 목재 조립체와 떨어져 있어 두 하부 역할이 노출되어 있습니다.',
 43:'왼쪽 하부 지지대는 화면 아래 가까이 노출되어 있습니다. 오른쪽 가상 하부 역할은 화면 아래 밖이어서 U이며, 오른쪽 q가 화면 안이라는 이유로 D를 만들지 않습니다.',
 49:'왼쪽 cardboard는 떨어져 있고 조립체의 두 하부 지지대가 노출되어 있습니다.',
 50:'개별 시트에서 오른쪽 하부 지지 구조가 영상 경계 가까이 노출되어 있습니다. 원 q2는 약간 OOB지만 teacher D는 유지하며 모델 입력 보류와 분리합니다. 제한된 가상 역할의 구조적 D이지 화면 경계의 정밀 픽셀 gold가 아닙니다.',
 54:'같은 조립체의 양쪽 하부 지지대가 노출되어 있습니다. 원 q2는 OOB이고 crop 일부가 검지만 이를 teacher E/U로 바꾸지 않습니다. 입력 사용 가능 여부는 원래 flags로 별도 보존합니다.',
 55:'왼쪽 cardboard와 조립체 사이의 간격이 보여 두 하부 역할이 노출되어 있습니다.',
 57:'선택 cuboid 왼쪽은 바닥으로 벗어나지만 오른쪽의 같은 목재 조립체가 분명하며 왼쪽 하부 지지대가 노출되어 있습니다. q3가 바닥 픽셀인 것을 E 근거로 사용하지 않았습니다.',
 62:'왼쪽 cardboard가 별도로 떨어져 있어 같은 조립체의 양 하부 지지 역할이 보입니다.'}
individual=[4,7,13,23,8,50,54,57,33,40,48,63]
views=[bind(S/f'OVERVIEW_{i:02d}_FRAMES_{i*8:02d}_TO_{i*8+7:02d}.png')for i in range(8)]
views += [bind(S/f'FRAME_{i:02d}_WHOLESCENE_AND_FIXED96.png')for i in individual]
rows=[];frames=[]
for i in range(64):
 entity='SAME_INTEGRATED_WOOD_ASSEMBLY_PARTIAL_LOWER_BRIDGE'
 entity_reason='원 선택 박스와 native cuboid가 제공 pose의 같은 목재 통합 조립체 하부를 가리키는 것을 실제 영상 구조로 확인했습니다. 상단 높이와 rear 역할은 인증하지 않습니다.'
 if i in different:entity,entity_reason=different[i]
 if i==5:entity_reason=extra[5]
 frame_evidence=[views[i//8]]+[bind(S/f'FRAME_{i:02d}_WHOLESCENE_AND_FIXED96.png')]if i in individual else[views[i//8]]
 for k in range(8):
  r=dict(facts[i*8+k]);state=lower_states[i][k-2]if k in(2,3)else'U'
  if k not in(2,3):
   why='이번 부분 bridge는 공유 하부2/3만 대상으로 합니다. base upper0/1 및 rear4..7의 역할/가림을 자동 확정하지 않고 U를 유지합니다.'
   certainty='OUTSIDE_PARTIAL_ROLE_SCOPE'
  elif i in different:
   assert state=='U';why=entity_reason;certainty='ENTITY_OR_ROLE_UNRESOLVED'
  elif i==5:
   assert state=='U';why=extra[5];certainty='PROVIDED_T_GEOMETRY_UNRESOLVED'
  elif state=='D':
   why=('오른쪽'if k==2 else'왼쪽')+' 전면 하부 지지대와 바닥 쪽 구조적 경계가 노출되어 있습니다. 같은 조립체의 제한된 가상 하부 역할의 D 제안이며 q 픽셀 소유권이나 mesh 표면/꼭짓점 일치를 D 조건으로 쓰지 않았습니다.'
   certainty='MODERATE_CONDITIONAL_WHOLE_ASSEMBLY_VIRTUAL_ROLE'
  elif state=='E':
   assert k==3
   why='별도 전경 cardboard가 같은 목재 조립체의 전면 왼쪽 하부 지지 역할을 가립니다. 조립체의 전면 널판/지지 구조가 상자 오른쪽으로 이어지고, 상자는 그 앞에 있습니다. q의 상자색·mask0·depth 차 또는 q 오차를 teacher로 쓰지 않았습니다.'
   certainty='MODERATE_CONDITIONAL_WHOLE_ASSEMBLY_VIRTUAL_ROLE'
  else:
   why='같은 목재 조립체이지만 이 하부 역할이 영상 밖/경계 밖에 있어 해당 역할의 D/E/S를 판정할 수 없습니다. 원 q의 입력 유효 여부와는 별개의 teacher U입니다.'
   certainty='EXPECTED_ROLE_OUT_OF_FRAME_OR_UNOBSERVED'
  if k in(2,3)and state in('D','E'):assert r['lower_role_continuity_status']=='PASS'
  r.update(semantic_state=state,semantic_reason_KO=why,assistant_reviewed=True,assistant_reviewer='/root/natural_visibility_sources',
    selected_entity_status=entity,selected_entity_reason_KO=entity_reason,frame_specific_evidence_KO=extra.get(i),
    semantic_confidence=certainty,role_scope='WHOLE_ASSEMBLY_FRONT_LOWER_VIRTUAL_FOOTPRINT_ONLY'if k in(2,3)else'OUTSIDE_LOWER_ROLE_BRIDGE',
    view_evidence=frame_evidence,root_adjudicated=False,human_gold=False,precise_physical_vertex_gold=False,trainready=False,training_now=False,
    source_partial_weak_binary_candidate=state in('D','E'),source_partial_weak_TRAIN_candidate=state in('D','E')and r['split']=='TRAIN_LAB',
    source_partial_weak_forwardable=state in('D','E')and r['original_input_valid']and r['crop_constructed'],
    lower_role_bridge_is_source_specific_provisional=True,full_native8_ontology_certified=False,no_q_error_label_filter=True,
    no_pixel_membership_or_depth_semantic_teacher=True,no_target_reselection_or_nearest_permutation=True,
    current223_V112_or_visibility_model_scores_used=False)
  rows.append(r)
 frames.append(dict(frame_index=i,source_id=facts[i*8]['source_id'],split=facts[i*8]['split'],entity=entity,entity_reason_KO=entity_reason,
   lower2_3_proposal=lower_states[i],extra_evidence_KO=extra.get(i),view_evidence=frame_evidence))
assert len(rows)==512 and[r['global_index']for r in rows]==list(range(512))
assert all(r['semantic_state']=='U'for r in rows if r['native_slot']not in(2,3))
assert all(r['semantic_state']=='U'for r in rows if r['frame_index']==5)
for old,new in zip(facts,rows):
 for key in('query_xy','L','id','native_slot','selected_index','original_input_valid','no_detection','crop_constructed','rgb96_SHA256','valid96_SHA256'):
  assert old[key]==new[key]
p=H/'ALL512_SOURCE_NATIVE_LOWER_WEAK_PROPOSALS.jsonl'
with p.open('x')as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
counts=dict(Counter(r['semantic_state']for r in rows))
summary=dict(completed=True,source_only=True,code=bind(__file__),contact_seal=bind(sealpath),all512_proposals=bind(p),
 all512_original_rows_q_L_firstargmax_input_flags_retained=True,personally_viewed_all8_overviews=True,
 fullscene_context_frames_personally_viewed=64,individual_fullscene_and8crop_sheets_personally_viewed=individual,
 all512_crops_claimed_personally_viewed=False,actually_viewed_files=views,per_frame_evidence=frames,
 semantic_state_counts=counts,by_split={s:dict(Counter(r['semantic_state']for r in rows if r['split']==s))for s in('TRAIN_LAB','VAL_HALL')},
 lower128_counts=dict(Counter(r['semantic_state']for r in rows if r['native_slot']in(2,3))),
 input_status=dict(valid=sum(r['original_input_valid']for r in rows),noquery=sum(r['no_detection']for r in rows),finite_OOB=sum(r['finite_non_sentinel']and not r['original_input_valid']for r in rows)),
 known_teacher_input_unavailable=[r['global_index']for r in rows if r['semantic_state']in('D','E')and not r['original_input_valid']],
 geometry_UNRESOLVED_frame5_lower_U_preserved=True,upper_rear384_U_preserved=True,
 assistant_weak_not_human_gold=True,root_adjudication_pending=True,full_visibility_trainready=False,training_now=False,
 new_model_initializer_encoder_head_fit_current223_V112_PnP_calls_or_reads=0,
 limitations_KO=['전체64개 원본 장면을8개 overview에서 실제 보았고12개 개별 fullscene/crop 시트를 추가 확인했습니다. 모든512 crop을 개별 확인했다고 주장하지 않습니다.',
 '라벨은 제한된 whole-assembly 가상 하부 역할의 assistant weak 제안입니다. 정밀 물리 꼭짓점·사람 gold·전체 native8 ontology를 인증하지 않습니다.',
 '조건부 축 정합과34.5mm scan-AABB outlier 정밀도 한계는 그대로입니다. q 오차로 label을 바꾸거나 성공 crop만 남기지 않았습니다.',
 'LAB/HALL은 원래 TRAIN_LAB/VAL_HALL 그대로이며 이미 개발에 노출된 동일 자료군입니다. 새로운 blind 일반화 검증이나 성능 개선 결과가 아닙니다.'])
sp=H/'SOURCE_VISUAL_REVIEW_COMPLETE.json';sp.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(summary=bind(sp),proposals=bind(p),counts=counts,by_split=summary['by_split'],known_input_U=summary['known_teacher_input_unavailable'])))
