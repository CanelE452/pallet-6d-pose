"""Final four assigned pairs: actual BEFORE-only physical cue inspection."""
from pathlib import Path
import datetime
import annotation_io as io
R=Path(__file__).resolve().parent;S=io.POLICY['slots'];N=len(io.INDEX);assert N==4
def known(k,xy,description,evidence,ambiguity,hw=4):
 x,y=xy;return dict(part_slot=S[k],reference_xy=xy,uncertainty_bbox_xyxy_half_open=[x-hw-.5,y-hw-.5,x+hw+.5,y+hw+.5],before_state='APPARENT_VISIBLE_STRUCTURAL_CUE_AI',cue_description=description,evidence=evidence,ambiguity=ambiguity)
def unknown(k,state,evidence,ambiguity):return dict(part_slot=S[k],reference_xy=None,uncertainty_bbox_xyxy_half_open=None,before_state=state,cue_description='해당 view-relative physical part cue를 BEFORE에서 특정할 수 없음',evidence=evidence,ambiguity=ambiguity)
rows=[];walls=[]
for i in range(N):
 if i==0:
  a=[known(0,[440,402],'오른쪽 아래에 잘린 가까운 front wall의 upper-left contour junction','왼쪽 측면의 뒤로 올라가는 상단선과 front board의 오른쪽으로 내려가는 상단선이 만나는 실제 목재 outer junction이 보입니다. 옆 cardboard/내부 붉은 물건의 경계가 아닌 목재 구조입니다.','Wall 오른쪽과 낮은 부분이 frame밖이어서 전체형태/native8 의미는 미정입니다. Blur와 board 두께의 불확실성을 포함했습니다.',5)]
  for k in range(1,6):a.append(unknown(k,'OUT_OF_FRAME_U','목재가 오른쪽 아래로 크게 잘려 해당 upper-right/lowest-wide/base endpoint가 프레임 안에 보이지 않습니다.','현재 보이는 중간 board나 image경계에 endpoint를 만들지 않고, 다른 pose의 구조를 가져오지 않습니다.'))
  w='오른쪽 아래에 부분적으로 보이는 가까운 front wall입니다. 왼쪽 측면의 사선과 front 상단선의 만남 하나가 확인됩니다. 오른쪽 외곽과 아래쪽 lowest board/base는 image밖입니다.'
 elif i==2:
  a=[known(0,[493,311],'오른쪽에 잘린 가까운 front wall의 upper-left contour junction','Front wall 맨 위 broad board의 상단선과 뒤로 올라가는 왼쪽 측면 사선이 만나는 목재 외곽이 보입니다. 앞 cardboard의 상단보다 위에 노출된 구조입니다.','오른쪽 절단과 cardboard로 전체 wall/base형태는 미정입니다. Exact pixel 재질GT/native8가 아니며 닳음과 blur를 포함했습니다.',5)]
  a.append(unknown(1,'OUT_OF_FRAME_U','가까운 wall 오른쪽 상단 endpoint는 image밖입니다.','Image경계로 대체하지 않습니다.'))
  a.append(unknown(2,'UNKNOWN_U','BEFORE부터 왼쪽 아래 foreground cardboard가 wall의 낮은 left endpoint를 가립니다.','보이지 않는 endpoint를 geometry/다른frame으로 추정하지 않습니다.'))
  a.append(unknown(3,'OUT_OF_FRAME_U','Front wall 오른쪽 외곽이 frame밖이라 lowest-wide-wall-right endpoint를 특정하지 못합니다.','보이는 board 중간을 endpoint로 쓰지 않습니다.'))
  a.append(unknown(4,'UNKNOWN_U','Base runner 왼쪽 endpoint는 BEFORE cardboard 뒤에 있고 구조가 드러나지 않습니다.','Carton/floor 윤곽으로 대체하지 않습니다.'))
  a.append(unknown(5,'UNKNOWN_U','오른쪽 base는 부분적으로만 드러나고 frame밖 끝과 cardboard가 겹쳐 outer junction을 확정할 수 없습니다.','어두운 opening/잘린 목재 일부를 실제 base끝으로 가정하지 않습니다.'))
  w='오른쪽으로 잘린 broad-board front wall입니다. 왼쪽 상단 목재 junction만 식별되며 낮은 왼쪽 영역은 이미 cardboard 뒤입니다. 오른쪽 endpoint/base는 전체를 볼 수 없어 U입니다.'
 else:
  coords=([[251,212],[396,210],[253,280],[393,279],[254,291],[391,291]] if i==1 else [[147,225],[503,220],[160,381],[495,378],[166,413],[489,409]])
  desc=['가까운 가장 넓은 front wall의 upper-left outer junction','가까운 가장 넓은 front wall의 upper-right outer junction','가장 아래 broad front board 밑면의 왼쪽 외곽 junction','가장 아래 broad front board 밑면의 오른쪽 외곽 junction','아래 목재 runner의 왼쪽 돌출 외곽과 지지대의 junction','아래 목재 runner의 오른쪽 돌출 외곽과 지지대의 junction']
  ev=['Front wall 가로 상단선과 왼쪽 외곽이 만나는 실제 목재 corner가 보입니다.','Front wall 가로 상단선과 오른쪽 외곽이 만나는 실제 목재 corner가 보입니다.','네 broad boards 중 가장 아래 board의 밑면/왼쪽 외곽이 만나는 곳을 확인했습니다.','가장 아래 broad board 밑면/오른쪽 외곽의 끝을 확인했습니다.','두 어두운 fork opening 아래 이어지는 얇은 수평 목재 runner의 왼쪽 돌출 외곽을 확대 RGB에서 구분했습니다.','아래 수평 목재 runner의 오른쪽 돌출 외곽과 지지대의 만남을 확대 RGB에서 구분했습니다.']
  a=[known(k,q,desc[k],ev[k],'목재 두께/blur/닳은 endpoint 때문에 수px 불확실성이 있습니다. Exact centerpixel mask나 정밀3D/native8 GT가 아닙니다.',3 if i==1 else (5 if k>=4 else 4)) for k,q in enumerate(coords)]
  w=('중앙의 작지만 전체가 보이는 가까운 front wall 네 broad boards가 가장 넓습니다. 뒤쪽 높은 wall과 구분하고 아래 두 fork opening 및 runner의 외곽을 확인했습니다. 해상도가 작아 uncertainty는 RGB blur/part정의의 범위입니다.' if i==1 else 'Hall 중앙에 전체가 보이는 가까운 front wall 네 broad boards와 그 아래 두 fork opening/runner를 선택했습니다. 더 뒤의 높은 wall이나 내부 물건/texture가 아닌 구조 외곽입니다.')
 rows.append(a);walls.append(w)
for i in range(N):
 io.seal_before(i,rows[i],walls[i]);io.write(io.pair(i)/'BEFORE_PHASE_VISUAL_QA.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pair_id=io.INDEX[i]['pair_id'],full_unmarked_before_and_native20_grid_and_unmarked_detail_directly_viewed=True,all6_fixed_slots_retained=True,after_RGB_opened_or_viewed=False,mask_depth_q_model_current_reads=0,uncertainty_from_before_only=True,writer=io.bind(Path(__file__))))
io.write(R/'ALL_BEFORE_SEALED_BEFORE_AFTER.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_seals=[io.bind(io.pair(i)/'BEFORE_SEAL.json') for i in range(N)],all4_BEFORE_sealed=True,any_batch05_AFTER_RGB_opened_or_decoded=False,source_mask_depth_q_model_current_reads=0,writer=io.bind(Path(__file__))))
print('Known refs:',[sum(x['reference_xy'] is not None for x in a) for a in rows])
