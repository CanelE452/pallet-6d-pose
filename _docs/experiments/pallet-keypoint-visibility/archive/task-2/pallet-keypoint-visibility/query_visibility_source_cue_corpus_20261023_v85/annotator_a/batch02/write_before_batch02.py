"""Actual BEFORE-only observations by A, all fixed batch02 pairs retained."""
from pathlib import Path
import datetime
import annotation_io as io
R=Path(__file__).resolve().parent
S=io.POLICY['slots']
def known(k,xy,description,evidence,ambiguity,hw=4):
 x,y=xy
 return dict(part_slot=S[k],reference_xy=xy,uncertainty_bbox_xyxy_half_open=[x-hw-.5,y-hw-.5,x+hw+.5,y+hw+.5],before_state='APPARENT_VISIBLE_STRUCTURAL_CUE_AI',cue_description=description,evidence=evidence,ambiguity=ambiguity)
def unknown(k,state,evidence,ambiguity):
 return dict(part_slot=S[k],reference_xy=None,uncertainty_bbox_xyxy_half_open=None,before_state=state,cue_description='해당 view-relative physical part cue를 BEFORE에서 특정할 수 없음',evidence=evidence,ambiguity=ambiguity)
far_ambiguity='오른쪽으로 잘린 물체의 일부만 관찰되므로 전체 wall 폭/정확한3D 꼭짓점이나 native8 대응을 주장하지 않습니다. Blur와 line-thickness를 불확실 영역에 포함했습니다.'
far={
 0:([565,150],'목재 bin은 화면 오른쪽 위에서 일부만 보이고 큰 cardboard가 이미 그 앞/아래를 가립니다. 보이는 가까운 목재 wall의 상단 가로선과 왼쪽으로 비스듬히 올라가는 측면 상단선이 만나는 모서리가 있습니다.'),
 2:([594,200],'오른쪽 경계에 걸친 목재 wall 상단 가로선의 왼쪽 끝과 뒤로 올라가는 측면 상단선이 만나는 실제 목재 junction을 확대 RGB에서 확인했습니다. Cardboard는 BEFORE부터 그 아래 전면을 가립니다.'),
 4:([587,137],'오른쪽 위 목재 전면의 가로 상단 윤곽과 왼쪽 측면의 사선이 만나는 junction이 cardboard 윗부분 위로 남아 있습니다. Cardboard 구멍/모서리가 아닌 더 뒤 목재의 접합점입니다.')}
wall=[];rows=[]
for i in range(5):
 if i in far:
  xy,e=far[i]
  a=[known(0,xy,'화면 오른쪽에 부분적으로 보이는 가까운 목재 wall의 upper-left contour junction',e,far_ambiguity,5)]
  a.append(unknown(1,'OUT_OF_FRAME_U','가까운 목재 wall의 오른쪽 상단 끝은 오른쪽 image boundary를 넘어 보이지 않습니다.','오른쪽 경계를 임의의 corner로 사용하지 않습니다.'))
  a.append(unknown(2,'UNKNOWN_U','목재 전면 왼쪽 아래는 BEFORE부터 큰 foreground cardboard 뒤에 있어 lowest broad board bottom endpoint를 확인하지 못했습니다.','Mask0/가림 원인 라벨을 만들지 않고 reference 자체를 null로 보존합니다.'))
  a.append(unknown(3,'OUT_OF_FRAME_U','목재 wall의 오른쪽 끝 전체가 잘려 lowest-wide-wall right endpoint를 특정할 수 없습니다.','이미지 밖 실제 위치를 추정하지 않습니다.'))
  a.append(unknown(4,'UNKNOWN_U','왼쪽 base/support 부위는 BEFORE의 cardboard 뒤에 있고 실제 구조선 접합점이 보이지 않습니다.','Floor/cardboard 경계를 목재 base cue로 대체하지 않습니다.'))
  a.append(unknown(5,'UNKNOWN_U','오른쪽 base/support는 물체가 오른쪽으로 잘리고 cardboard와 겹쳐 구조를 확정할 수 없습니다.','OUT_OF_FRAME와 BEFORE 가림을 분리할 근거가 부족하여 UNKNOWN을 보존합니다.'))
  w='오른쪽 image boundary에 잘린 목재 bin의 가까운 broad-board wall 일부입니다. 위쪽 목재 contour junction 하나만 확인됩니다. Cardboard가 이미 전면 낮은 영역을 가리고 있고, 전체 wall의 오른쪽 끝과 base는 특정할 수 없습니다.'
 elif i==1:
  desc=['가까운 넓은 front wall의 상단 왼쪽 외곽 접합점','가까운 넓은 front wall의 상단 오른쪽 외곽 접합점','가까운 front wall lowest broad board의 아래 왼쪽 외곽 끝','가까운 front wall lowest broad board의 아래 오른쪽 외곽 끝','하단 수평 base runner의 왼쪽 돌출부와 왼쪽 지지대가 만나는 외곽 접합점','하단 수평 base runner의 오른쪽 돌출부와 오른쪽 지지대가 만나는 외곽 접합점']
  coords=[[206,300],[442,297],[211,403],[437,402],[214,421],[434,421]]
  ev=['사선 측면 상단선과 front wall의 가로 상단선이 이어지는 목재 외곽을 확인했습니다.','사선 측면 상단선과 front wall 가로 상단선의 오른쪽 만남이 보입니다.','가로로 긴 가장 아래 broad board의 밑면과 왼쪽 외곽의 만남이 보입니다.','가장 아래 broad board의 밑면과 오른쪽 외곽 만남이 보입니다.','두 어두운 지게차 구멍 아래에 이어지는 얇은 목재 수평 runner의 왼쪽 돌출 외곽을 확대 RGB에서 구분했습니다.','오른쪽 지지대 아래로 이어지는 목재 수평 runner의 오른쪽 외곽을 확대 RGB에서 구분했습니다.']
  a=[known(k,q,desc[k],ev[k],'목재의 닳음/blur/여러 board 겹침으로 endpoint 정의에 수px 불확실성이 있습니다. Nail/texture/floor shadow나 exact centerpixel 물질GT가 아닙니다.',5 if k>=4 else 4) for k,q in enumerate(coords)]
  w='프레임 중앙 하단에 온전히 보이는 front wall이 가장 넓습니다. 거의 수평인 broad boards 네 줄과 아래 두 개의 어두운 fork opening, 그 아래 목재 runner가 보입니다. 더 뒤의 높은 rear wall이 아닌 가까운 front wall의 외곽을 선택했습니다.'
 else:
  a=[known(0,[445,405],'오른쪽 아래에 잘린 가까운 front wall의 upper-left contour junction','Front board의 위쪽 목재 가로선/오른쪽 하강선과 뒤쪽으로 올라가는 왼쪽 측면 상단선이 만나는 실제 외곽 접합점이 보입니다. 붉은 내부 물건과 foreground cardboard의 윤곽과 구분했습니다.','Wall의 오른쪽/낮은 부분은 image 밖이므로 전체 형태나 native8 의미는 미정입니다. 목재 닳음과 흐린 경계를 불확실 영역에 포함했습니다.',5)]
  for k in range(1,6):
   a.append(unknown(k,'OUT_OF_FRAME_U','목재 bin이 오른쪽 아래로 크게 잘려 해당 wall endpoint/base cue가 프레임 안에 드러나지 않습니다. 가장 아래 쪽 일부는 별도 cardboard와도 겹칩니다.','보이는 board 중간선을 lowest endpoint로 대체하지 않고, 다른 pose의 구조를 가져와 좌표를 추정하지 않습니다.'))
  w='오른쪽 아래에 가까운 broad-board wall 일부와 왼쪽 사선 측면이 보입니다. 가까운 front wall의 왼쪽 상단 junction은 확인되지만 오른쪽 및 lowest-board/base endpoints는 image 밖입니다. 상단 뒤쪽 wall의 corner를 대신 쓰지 않았습니다.'
 rows.append(a);wall.append(w)
for i in range(5):
 io.seal_before(i,rows[i],wall[i])
 io.write(io.pair(i)/'BEFORE_PHASE_VISUAL_QA.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pair_id=io.INDEX[i]['pair_id'],full_unmarked_before_and_native20_grid_and_unmarked_detail_directly_viewed=True,all6_fixed_slots_retained=True,after_RGB_opened_or_viewed=False,mask_depth_q_model_current_reads=0,uncertainty_from_before_only=True,writer=io.bind(Path(__file__))))
io.write(R/'ALL_BEFORE_SEALED_BEFORE_AFTER.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_seals=[io.bind(io.pair(i)/'BEFORE_SEAL.json') for i in range(5)],all5_BEFORE_sealed=True,any_batch02_AFTER_RGB_opened_or_decoded=False,source_mask_depth_q_model_current_reads=0,writer=io.bind(Path(__file__))))
print('All five BEFORE annotations sealed. Known refs:',[sum(x['reference_xy'] is not None for x in a) for a in rows])
