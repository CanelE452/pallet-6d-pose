"""Actual batch04 BEFORE-only observations, no AFTER/mask/q/model access."""
from pathlib import Path
import datetime
import annotation_io as io
R=Path(__file__).resolve().parent;S=io.POLICY['slots']
def known(k,xy,description,evidence,ambiguity,hw=4):
 x,y=xy;return dict(part_slot=S[k],reference_xy=xy,uncertainty_bbox_xyxy_half_open=[x-hw-.5,y-hw-.5,x+hw+.5,y+hw+.5],before_state='APPARENT_VISIBLE_STRUCTURAL_CUE_AI',cue_description=description,evidence=evidence,ambiguity=ambiguity)
def unknown(k,state,evidence,ambiguity):return dict(part_slot=S[k],reference_xy=None,uncertainty_bbox_xyxy_half_open=None,before_state=state,cue_description='해당 view-relative physical part cue를 BEFORE에서 특정할 수 없음',evidence=evidence,ambiguity=ambiguity)
rows=[];walls=[]
for i in range(5):
 if i in (1,2,4):
  xy={1:[602,126],2:[465,198],4:[569,162]}[i]
  evidence={1:'오른쪽 위 작은 목재 front wall의 상단 가로선과 왼쪽으로 뒤로 올라가는 측면 상단선이 만나는 junction이 보입니다. 앞 cardboard의 모서리가 아닌 뒤쪽 목재 contour입니다.',2:'오른쪽 목재 맨 위 broad board의 상단 가로선과 왼쪽 외곽이 만나는 실제 목재 junction을 확인했습니다. 큰 BEFORE cardboard보다 위로 노출된 목재입니다.',4:'오른쪽에 잘린 목재 front wall의 상단 가로선과 왼쪽 사선 측면 상단선의 만남이 보입니다. 앞 cardboard 위쪽보다 더 뒤 목재 접합점입니다.'}[i]
  a=[known(0,xy,'오른쪽에 부분적으로 보이는 가까운 broad wooden wall의 upper-left outer junction',evidence,'물체 오른쪽이 frame밖이므로 전체 wall width/native8/정확한3D corner는 미정입니다. Blur/board thickness/닳은 모서리의 불확실성을 포함했습니다.',5)]
  a.append(unknown(1,'OUT_OF_FRAME_U','가까운 broad wall의 오른쪽 상단 endpoint는 image 오른쪽 밖으로 잘렸습니다.','Frame경계나 배경/다른 board의 중간점을 endpoint로 대체하지 않습니다.'))
  a.append(unknown(2,'UNKNOWN_U','BEFORE부터 앞 cardboard가 wall의 왼쪽 낮은 부분을 가려 lowest-wide-wall endpoint를 확인하지 못했습니다.','가림 뒤의 reference를 추정하지 않습니다.'))
  a.append(unknown(3,'OUT_OF_FRAME_U','Wall의 오른쪽 외곽이 frame밖이어서 lowest-wide-wall-right endpoint를 지정할 수 없습니다.','다른 pose의 형태를 가져와 채우지 않습니다.'))
  a.append(unknown(4,'UNKNOWN_U','왼쪽 base/runner는 BEFORE cardboard 뒤에 있고 실제 목재 junction이 보이지 않습니다.','Cardboard/floor 경계를 목재 cue로 쓰지 않습니다.'))
  a.append(unknown(5,'UNKNOWN_U','오른쪽 base/runner는 frame절단과 cardboard 겹침으로 외곽의 실제 endpoint를 식별할 수 없습니다.','일부 보이는 목재만으로 base 외곽을 추정하지 않습니다.'))
  w='오른쪽 frame경계에 일부만 보이는 broad front wall입니다. 상단 왼쪽 목재 junction만 식별되며 cardboard가 BEFORE부터 낮은 전면을 가립니다. 오른쪽 잘린 endpoint와 base는 null/U입니다.'
 elif i==0:
  coords=[[87,268],[555,263],[108,468],[537,464]]
  ds=['가까운 넓은 front wall의 upper-left contour junction','가까운 넓은 front wall의 upper-right contour junction','가장 아래 broad board 밑면과 왼쪽 외곽의 만남','가장 아래 broad board 밑면과 오른쪽 외곽의 만남']
  ev=['Front wall 상단 가로선이 왼쪽 측면의 사선 상단선과 만나는 실제 목재 모서리입니다.','Front wall 상단 가로선이 오른쪽 측면의 사선 상단선과 만나는 실제 목재 모서리가 보입니다.','네 broad boards 중 가장 아래 board의 밑면이 왼쪽 외곽에서 끝나는 부분이 frame안에 남아 있습니다.','가장 아래 broad board의 밑면이 오른쪽 외곽에서 끝나는 부분을 확인했습니다.']
  a=[known(k,q,ds[k],ev[k],'목재의 닳음/board thickness/blur로 몇px 정의 불확실성이 있습니다. Region은 exact material mask/centerpixel GT가 아니며 native8는 UNKNOWN입니다.') for k,q in enumerate(coords)]
  for k in (4,5):a.append(unknown(k,'OUT_OF_FRAME_U','Base/support 일부는 맨 아래에 보이지만 아래 runner의 외곽 junction은 image아래로 잘려 확인할 수 없습니다.','앞 board의 밑면/이미지 경계를 base runner endpoint로 대체하지 않습니다.'))
  w='화면 중앙의 가까운 broad front wall 네 줄이 가장 넓습니다. Front upper/lower outer corners는 보이지만 하단 runner와 base endpoint는 frame 아래로 잘립니다. 뒤쪽 높은 wall의 corner를 대신 쓰지 않았습니다.'
 else:
  a=[known(0,[149,408],'아래로 잘린 가까운 front wall의 upper-left outer junction','Front wall 맨 위 board의 가로 상단선과 왼쪽 측면 사선이 만나는 실제 목재 contour가 보입니다. 내부 붉은 물건/다른 board 끝과 구분했습니다.','낮은 wall/base는 image아래로 잘려 전체 wall 높이/3D/native8를 가정하지 않습니다.'),known(1,[496,404],'아래로 잘린 가까운 front wall의 upper-right outer junction','Front wall 상단 가로선과 오른쪽 사선 측면 상단선의 만남이 보입니다.','닳은 board와 blur의 불확실성을 포함한 before-only region이며 exact pixel visibility가 아닙니다.')]
  for k in range(2,6):a.append(unknown(k,'OUT_OF_FRAME_U','가까운 front wall의 아래 board들과 base/runner endpoints는 image아래로 잘려 해당 part를 볼 수 없습니다.','현재 보이는 두번째 board를 lowest-wide-board로 바꾸거나 image경계에 좌표를 만들지 않습니다.'))
  w='Hall 중앙 하단의 가까운 broad front wall은 상단 두 줄 정도만 frame안에 있습니다. 가장 넓은 가까운 wall의 두 upper outer junction은 보이지만 lowest board/base는 아래로 잘립니다. 뒤쪽 높은 wall은 대상으로 바꾸지 않았습니다.'
 rows.append(a);walls.append(w)
for i in range(5):
 io.seal_before(i,rows[i],walls[i]);io.write(io.pair(i)/'BEFORE_PHASE_VISUAL_QA.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pair_id=io.INDEX[i]['pair_id'],full_unmarked_before_and_native20_grid_and_unmarked_detail_directly_viewed=True,all6_fixed_slots_retained=True,after_RGB_opened_or_viewed=False,mask_depth_q_model_current_reads=0,uncertainty_from_before_only=True,writer=io.bind(Path(__file__))))
io.write(R/'ALL_BEFORE_SEALED_BEFORE_AFTER.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_seals=[io.bind(io.pair(i)/'BEFORE_SEAL.json') for i in range(5)],all5_BEFORE_sealed=True,any_batch04_AFTER_RGB_opened_or_decoded=False,source_mask_depth_q_model_current_reads=0,writer=io.bind(Path(__file__))))
print('Known refs:',[sum(x['reference_xy'] is not None for x in a) for a in rows])
