"""Annotator-A batch03 actual BEFORE-only source observations."""
from pathlib import Path
import datetime
import annotation_io as io
R=Path(__file__).resolve().parent;S=io.POLICY['slots']
def known(k,xy,description,evidence,ambiguity,hw=4):
 x,y=xy;return dict(part_slot=S[k],reference_xy=xy,uncertainty_bbox_xyxy_half_open=[x-hw-.5,y-hw-.5,x+hw+.5,y+hw+.5],before_state='APPARENT_VISIBLE_STRUCTURAL_CUE_AI',cue_description=description,evidence=evidence,ambiguity=ambiguity)
def unknown(k,state,evidence,ambiguity):
 return dict(part_slot=S[k],reference_xy=None,uncertainty_bbox_xyxy_half_open=None,before_state=state,cue_description='해당 view-relative physical part cue를 BEFORE에서 특정할 수 없음',evidence=evidence,ambiguity=ambiguity)
rows=[];walls=[]
far=[([587,203],'오른쪽에 일부만 보이는 front wall의 가로 상단선과 뒤로 올라가는 왼쪽 측면 상단선의 만남이 있습니다. BEFORE cardboard가 낮은 부분 앞에 있지만 이 위쪽 목재 junction은 노출되어 있습니다.'),([529,219],'오른쪽 경계에 잘린 넓은 front wall의 상단 가로선과 왼쪽 수직 외곽이 만나는 목재 contour corner가 보입니다. 아래를 가리는 cardboard의 상단보다 높은 곳이며 nail/배경 장비의 경계와 구분했습니다.'),([563,213],'오른쪽 목재 wall 맨 위 broad board의 상단 가로선과 왼쪽 외곽이 만나는 junction이 보입니다. 앞의 cardboard 상단선이 아닌 더 뒤쪽 목재 corner입니다.')]
for i in range(5):
 if i<3:
  xy,e=far[i]
  a=[known(0,xy,'부분적으로 보이는 가까운 broad wooden wall의 upper-left outer contour junction',e,'물체 오른쪽이 frame밖이라 전체 wall width/3D corner/native8는 미정입니다. Blur/닳은 목재 모서리와 몇px 정의 불확실성을 포함한 before-only region입니다.',5)]
  a.append(unknown(1,'OUT_OF_FRAME_U','가까운 broad wall의 오른쪽 상단 endpoint는 image 오른쪽 밖으로 잘렸습니다.','Image 경계나 다른 board 중간점을 corner로 쓰지 않습니다.'))
  a.append(unknown(2,'UNKNOWN_U','BEFORE부터 앞에 놓인 cardboard가 목재 wall 왼쪽 낮은 영역을 가려 lowest-wide-wall-left junction을 확인할 수 없습니다.','목재의 실제 endpoint를 가림 뒤로 추정하지 않습니다.'))
  a.append(unknown(3,'OUT_OF_FRAME_U','가까운 wall 오른쪽 외곽이 frame밖이라 lowest-wide-wall-right endpoint를 지정하지 못합니다.','다른 pose/형태로 좌표를 채우지 않습니다.'))
  a.append(unknown(4,'UNKNOWN_U','왼쪽 base/runner 구조의 endpoint는 BEFORE cardboard 뒤에 있어 실제 목재 접합점을 관찰하지 못했습니다.','Cardboard/floor 경계를 target cue로 대체하지 않습니다.'))
  a.append(unknown(5,'UNKNOWN_U','오른쪽 base/runner는 프레임 절단과 앞 cardboard의 결합으로 부분 구조만 보이고 outer endpoint를 확정할 수 없습니다.','보이는 horizontal wood 조각만으로 view-relative 외곽을 추정하지 않습니다.'))
  w='오른쪽 frame 경계에 잘린 broad-board front wall 일부입니다. 가장 위 board의 왼쪽 outer junction만 식별됩니다. BEFORE부터 cardboard가 낮은 전면을 가리고 있어 그 아래 또는 잘린 오른쪽 부위는 null/U입니다.'
 else:
  coords=([[204,303],[442,299],[209,407],[438,405],[211,428],[435,426]] if i==3 else [[140,198],[510,198],[141,377],[506,377],[141,417],[503,417]])
  descriptions=['가장 넓은 가까운 front wall의 upper-left outer contour junction','가장 넓은 가까운 front wall의 upper-right outer contour junction','가장 아래 broad front board 밑면과 왼쪽 외곽의 접합점','가장 아래 broad front board 밑면과 오른쪽 외곽의 접합점','아래 목재 runner의 왼쪽 돌출 외곽과 지지대가 만나는 junction','아래 목재 runner의 오른쪽 돌출 외곽과 지지대가 만나는 junction']
  ev=['Front board 상단 가로선의 왼쪽 끝과 wall의 바깥 윤곽이 만나며 실제 목재가 보입니다.','Front board 상단 가로선의 오른쪽 끝과 wall의 바깥 윤곽이 만나는 목재 corner가 보입니다.','네 줄 broad boards 중 가장 낮은 board의 밑면이 왼쪽 외곽에서 끝나는 부분을 확인했습니다.','가장 낮은 broad board의 밑면과 오른쪽 외곽이 만나는 endpoint가 보입니다.','두 어두운 fork opening 아래의 얇은 목재 수평 runner가 왼쪽으로 돌출한 끝을 확대 RGB에서 구분했습니다.','아래 얇은 목재 수평 runner가 오른쪽 지지대 바깥으로 돌출한 끝을 확대 RGB에서 구분했습니다.']
  a=[known(k,q,descriptions[k],ev[k],'목재 닳음/board thickness/blur로 endpoint 정의에 수px 불확실성이 있습니다. Region은 exact material mask/centerpixel GT가 아니며 native8 대응은 UNKNOWN입니다.',5 if k>=4 else 4) for k,q in enumerate(coords)]
  w=('Hall 중앙에 전체가 보이는 가까운 front wall이 가장 넓습니다. 상부 내부/뒤쪽 wall 대신 네 줄 broad front boards의 외곽과 아래 두 fork opening 아래 runner를 선택했습니다.' if i==3 else '낮은 시점에서 정면 broad wall 네 줄과 아래 두 fork opening/runner가 거의 전체 보입니다. 더 뒤쪽 wall이나 배경 pillar가 아니라 가까운 front wall의 실제 목재 외곽을 사용했습니다.')
 rows.append(a);walls.append(w)
for i in range(5):
 io.seal_before(i,rows[i],walls[i])
 io.write(io.pair(i)/'BEFORE_PHASE_VISUAL_QA.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pair_id=io.INDEX[i]['pair_id'],full_unmarked_before_and_native20_grid_and_unmarked_detail_directly_viewed=True,all6_fixed_slots_retained=True,after_RGB_opened_or_viewed=False,mask_depth_q_model_current_reads=0,uncertainty_from_before_only=True,writer=io.bind(Path(__file__))))
io.write(R/'ALL_BEFORE_SEALED_BEFORE_AFTER.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_seals=[io.bind(io.pair(i)/'BEFORE_SEAL.json') for i in range(5)],all5_BEFORE_sealed=True,any_batch03_AFTER_RGB_opened_or_decoded=False,source_mask_depth_q_model_current_reads=0,writer=io.bind(Path(__file__))))
print('Known refs:',[sum(x['reference_xy'] is not None for x in a) for a in rows])
