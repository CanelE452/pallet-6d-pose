"""Record C's actual tools.view_image observations; no image decoding or inference.

Manual judgments below were made from the twelve original fullimage/all8 contact
tool outputs and twelve optional CLEAN96 tool outputs. This is not an image
labeling algorithm. It reads no other observer's values, GT, arrays, or scores.
"""
from pathlib import Path
import collections
import copy
import csv
import datetime
import hashlib
import json
import types

V = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_part_query_support_contract_20261025_v216')
O = V / 'actual_sameframe_observations_c_v1'
ENTRY = V / 'ROOT_ACTUAL_SAMEFRAME_TWO_READER_ANNOTATION_ENTRY.json'
ENTRY_SHA = '9566b0b2419bdb7819d9e824ab281c17b2a733d903f6c9862d76d2507ba80e40'
OPTIONAL_CLEAN = {(0,6),(2,0),(2,3),(2,5),(4,0),(4,3),(5,4),(6,4),(6,6),(8,0),(11,5),(11,6)}

# mode D = weakly addressed visible pallet part; A = explicit-ROI-only local
# feature, never a native target; N = no sufficiently identified local witness.
# Each cell: mode, selected-pallet-instance identified, kind, inclusive raw96
# ROI, independent query-to-feature relation, visible feature, ambiguity.
M = [
 [None,None,None,None,
  ('A',True,'END_CORNER',[17,57,40,76],'OFF','아래쪽에 보이는 팔레트 상판 왼쪽의 둥근 끝 경계','q4 중심은 바닥/케이블 쪽이다. 이 국소 끝과 native4의 의도 부위 대응은 확정하지 못했다.'),
  ('A',True,'END_CORNER',[4,65,31,86],'OFF','crop 좌하단의 팔레트 목재 상판 끝','q5는 목재보다 위의 바닥/벽 경계 쪽이다. 보이는 끝만으로 native5 대응을 가정하지 않는다.'),
  ('D',True,'END_CORNER',[40,27,58,61],'ON','팔레트 오른쪽 상판의 끝면과 바깥 측면이 만나는 경계','원 q6와 q5의 같은 오른쪽 끝 배치 및 상판 연속성을 이용한 약한 대응이다. 정확한 3D 점 인증이 아니다.'),
  ('A',True,'EDGE_JUNCTION',[12,29,42,51],'NEAR','팔레트 왼쪽 상판의 윗면/끝면 경계','목재는 분명 보이지만 native7이 이 경계의 어느 끝을 지정하는지 분리하기 어렵다.')],
 [
  ('A',False,'END_CORNER',[68,0,94,43],'OFF','회색 플라스틱 상자의 아래쪽 모서리','같은 full-frame에서 선택 팔레트라고 식별할 목재 구조가 확인되지 않는다. 상자 모서리는 local feature로만 남긴다.'),
  ('A',False,'END_CORNER',[0,0,46,48],'NEAR','회색 플라스틱 상자의 아래 끝과 측면','검은 받침 프레임과 상자는 보이나 선택 팔레트의 native 부위라고 연결하지 못했다.'),
  ('N',False,'UNKNOWN',None,'UNKNOWN','crop 대부분의 바닥/케이블과 잘린 상자 가장자리','원 native2에 해당하는 선택 팔레트의 국소 feature를 식별하지 못했다.'),
  ('A',False,'END_CORNER',[67,0,95,34],'OFF','crop 우상단 회색 상자 하단 모서리','중심의 검은 프레임을 팔레트 native 가림의 근거로 쓰지 않는다.'),
  ('A',False,'END_CORNER',[28,39,55,68],'NEAR','회색 상자의 위 테두리 왼쪽 코너','보이는 것은 플라스틱 상자이다. 선택 팔레트/native 대응은 UNKNOWN이다.'),
  ('A',False,'END_CORNER',[24,40,50,69],'NEAR','회색 상자의 위 테두리 오른쪽 코너','플라스틱 테두리 코너의 존재를 팔레트 D로 복사하지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[17,21,51,72],'NEAR','회색 상자 테두리와 수직 옆벽의 접합','상자 외곽은 보이지만 의도한 팔레트 부위인지 식별되지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[26,24,61,67],'ON','회색 상자 내부의 두 벽이 만나는 선','상자 내부 feature만 보존하며 native7 팔레트 가시성은 UNKNOWN이다.')],
 [
  ('D',True,'END_CORNER',[47,37,77,67],'NEAR','목재 팔레트의 가까운 왼쪽 상판 끝 코너','같은 상판이 full-frame에서 팔레트의 가까운 왼쪽 끝으로 이어지고 q0도 그 코너 바로 옆에 있다.'),
  None,None,
  ('D',True,'BLOCK_FACE',[41,26,65,68],'ON','가까운 왼쪽 지지 블록의 앞면과 바깥 세로 모서리','q0 아래쪽의 원 q3 및 같은 팔레트의 지지 블록/상판 연결이 이 의도 부위를 약하게 주소화한다.'),
  ('D',True,'END_CORNER',[31,54,64,78],'NEAR','멀리 있는 왼쪽 상판의 끝 코너','q4 쪽 상판의 양쪽 긴 변이 full-frame에서 같은 팔레트로 연속된다. 뒤의 검은 프레임 자체를 가림 정답으로 쓰지 않는다.'),
  ('D',True,'END_CORNER',[24,32,46,61],'NEAR','멀리 있는 오른쪽 상판의 둥근 끝','q5 바로 왼쪽의 둥근 목재 끝과 같은 보드의 연속성이 보인다. 중심이 바닥 쪽이어도 그 사실을 E로 쓰지 않는다.'),
  ('D',True,'END_CORNER',[14,16,40,46],'OFF','팔레트 오른쪽 외곽 상판의 끝/옆면 경계','q5와 q6가 지정한 같은 오른쪽 끝 맥락에서 보드 끝이 원 crop 안에 보인다. q6 자체는 바닥 쪽으로 벗어나 있다.'),
  ('A',True,'EDGE_JUNCTION',[42,40,67,64],'NEAR','평행한 상판 보드 사이 간격의 목재 경계','선택 팔레트의 목재는 보이지만 반복 보드들 중 native7의 특정 끝 대응은 확정하지 못했다.')],
 [
  ('A',False,'EDGE_JUNCTION',[21,41,43,79],'NEAR','나무 상자의 수직 finger-joint와 위 테두리','q8이 작은 상자 윗부분에 몰려 있다. 아래 팔레트의 native 부위로 주소화하지 않는다.'),
  ('A',False,'END_CORNER',[35,27,58,67],'ON','나무 상자의 우상단 코너','이 crop의 나무 상자를 아래 팔레트의 의도 부위로 보지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[30,10,48,52],'NEAR','나무 상자의 수직 맞물림 경계','앞쪽 흰 용기의 테두리가 보이지만 팔레트 native의 외부 가림이라는 연결은 확인되지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[22,22,43,75],'ON','나무 상자 왼쪽의 수직 맞물림 경계','상자와 팔레트는 구분되며 이 feature의 native 팔레트 대응은 UNKNOWN이다.'),
  ('A',False,'OTHER',[60,35,91,74],'OFF','나무 상자 위 테두리의 둥근 손가락 홈 경계','명시 ROI로 지정되는 상자의 홈이다. native4 팔레트 feature라고 승격하지 않는다.'),
  ('A',False,'END_CORNER',[32,42,61,69],'NEAR','나무 상자 우상단 테두리 끝','같은 feature kind가 보여도 팔레트 부위 일치의 근거가 되지 않는다.'),
  ('A',False,'END_CORNER',[36,16,61,63],'NEAR','나무 상자 위/옆 경계 코너','앞쪽 용기와 나무 상자가 함께 보인다. 팔레트 native 가시성은 알 수 없다.'),
  ('A',False,'OTHER',[61,28,90,62],'NEAR','나무 상자의 둥근 홈과 위 테두리','홈은 local feature로만 관찰되며 pallet/native 대응은 확인되지 않는다.')],
 [
  ('D',True,'END_CORNER',[57,39,81,62],'NEAR','팔레트 왼쪽 위 상판의 바깥 끝 코너','full-frame의 앞쪽 상판과 EPAL 지지 블록이 이어지며 원 q0가 그 상판 끝에 가깝다.'),
  ('A',True,'BLOCK_FACE',[57,22,95,65],'ON','오른쪽 EPAL 지지 블록의 보이는 면','오른쪽 실제 끝은 frame/crop 경계 쪽에 있어 native1의 정확한 의도 끝과의 대응은 확정하지 않는다.'),
  ('A',True,'EDGE_JUNCTION',[55,35,92,60],'NEAR','팔레트 오른쪽 아래 보드의 윗 경계','보드의 일부는 보이지만 frame 바깥쪽 끝을 D로 추정하지 않는다.'),
  ('D',True,'END_CORNER',[49,32,69,50],'NEAR','왼쪽 아래 러너 보드와 지지 블록의 바깥 코너','q0 아래의 q3와 팔레트 전면 왼쪽 블록/하부 보드 연결이 같은 가까운 아래 끝을 약하게 식별한다.'),
  ('A',False,'EDGE_JUNCTION',[64,22,83,71],'OFF','파란 상자의 수직 측면 경계','선택 팔레트의 native4 부위를 crop에서 식별하지 못했다. 파란 상자의 존재만으로 E가 아니다.'),
  ('A',True,'EDGE_JUNCTION',[2,75,92,91],'OFF','흰 용기 아래쪽의 팔레트 상판 가장자리','위의 흰 용기는 보이지만 숨은 native5 의도 부위와의 대응이 불명확하다. 임의 E를 부여하지 않는다.'),
  ('A',True,'EDGE_JUNCTION',[1,44,91,64],'NEAR','흰 용기 아래 이어지는 팔레트 긴 보드 경계','목재 경계와 용기 접촉은 보이지만 native6의 특정 부위 가림은 인증하지 못했다.'),
  ('D',True,'EDGE_JUNCTION',[22,55,67,71],'NEAR','팔레트 왼쪽 뒤쪽 상판의 외곽/측면 경계','full-frame에서 파란 상자 앞쪽으로 이어지는 같은 팔레트 상판과 원 q7의 왼쪽 뒤 맥락이 양성 약근거다.')],
 [
  None,
  ('A',True,'END_CORNER',[1,33,26,62],'OFF','crop 왼쪽에 잘려 보이는 팔레트 목재 끝','q1은 검은 받침/케이블과 바닥 쪽이다. 보이는 목재 조각만으로 원 native1 끝을 식별하지 않는다.'),
  ('A',True,'END_CORNER',[0,0,24,42],'OFF','crop 좌상단의 팔레트 외곽 목재/지지 부분','중심의 바닥과 케이블은 native 가림 근거가 아니며 특정 끝 대응은 UNKNOWN이다.'),
  None,
  ('D',True,'END_CORNER',[49,44,74,71],'NEAR','팔레트 멀리 왼쪽 상판의 둥근 끝','같은 팔레트 긴 왼쪽 상판과 원 q4의 뒤쪽 끝 위치가 연속되어 보이는 끝을 주소화한다.'),
  ('A',False,'EDGE_JUNCTION',[50,26,81,65],'NEAR','파란 상자와 나무 상자가 만나는 경계','이 crop의 전경 상자들을 팔레트 native5 부위 또는 그 가림으로 연결할 수 없다.'),
  ('A',False,'END_CORNER',[26,61,62,92],'NEAR','파란 상자 하단의 모서리','보이는 파란 상자의 끝은 팔레트 끝이 아니므로 native6 D/E는 UNKNOWN이다.'),
  ('D',True,'EDGE_JUNCTION',[40,31,63,60],'ON','왼쪽 팔레트 상판의 바깥 옆면/끝 경계','q4와 q7의 같은 왼쪽 상판 끝 맥락 및 full-frame의 긴 보드 연속성으로 약하게 대응된다.')],
 [
  ('A',False,'END_CORNER',[35,19,68,53],'NEAR','앞쪽 작은 나무 상자의 아래 코너','뒤에 세워진 팔레트와 앞 상자를 구분했다. 앞 상자 코너를 팔레트 native0으로 쓰지 않는다.'),
  ('A',False,'END_CORNER',[36,25,64,53],'ON','앞쪽 높은 나무 상자의 위 테두리 코너','q1이 전경 상자 쪽에 있으나 선택 팔레트의 의도 부위 대응/가림은 확인되지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[36,19,61,76],'ON','전경 나무 상자의 바깥 세로 접합','세로 나무 경계만으로 팔레트 native2를 식별하지 않는다.'),
  ('A',False,'END_CORNER',[31,0,62,20],'OFF','crop 윗부분의 작은 나무 상자 하단 코너','q3 중심은 바닥이며 팔레트 native3의 답을 전경 상자에서 복사하지 않는다.'),
  ('D',True,'END_CORNER',[49,44,70,70],'NEAR','세워진 팔레트 위쪽 왼쪽 상판 끝','위쪽 왼쪽 q4/q7 쌍과 full-frame의 같은 상판/옆 보드 연속성이 팔레트의 이 끝을 약하게 주소화한다.'),
  ('D',True,'END_CORNER',[56,58,82,85],'OFF','세워진 팔레트 위쪽 오른쪽 상판 끝','원 q5/q6가 둘러싼 오른쪽 상판의 끝과 긴 보드가 보인다. q5 중심은 끝 위 배경이며 padding은 증거 ROI로 쓰지 않았다.'),
  ('D',True,'END_CORNER',[64,23,86,53],'NEAR','세워진 팔레트 오른쪽 위 상판의 끝면','원 q6 쪽의 끝면/옆면이 같은 팔레트 보드와 이어지고 full-frame의 오른쪽 상단 끝과 일치한다는 약근거다.'),
  ('D',True,'EDGE_JUNCTION',[34,41,68,68],'ON','세워진 팔레트 왼쪽 위 보드의 끝/아래 측면','q4 아래 q7 및 같은 보드의 바깥 경계 연속성으로 이 보이는 왼쪽 끝 부분을 식별했다.')],
 [
  ('A',False,'END_CORNER',[5,9,36,41],'OFF','왼쪽 앞의 작은 나무 상자 하단 코너','full-frame 팔레트는 뒤에 있고 q0 crop은 앞 상자/바닥이다. 팔레트 native 대응은 UNKNOWN이다.'),
  ('A',False,'EDGE_JUNCTION',[29,29,68,61],'ON','앞 나무 상자의 위 가로대와 안쪽 경계','전경 상자의 가로대이며 selected pallet의 native1이라고 식별하지 않았다.'),
  ('A',False,'EDGE_JUNCTION',[31,39,66,65],'ON','앞 나무 상자의 가로대와 내부 개구 경계','나무가 있다는 사실이나 중심이 가로대에 있다는 사실을 팔레트 native2 D로 쓰지 않는다.'),
  ('N',False,'UNKNOWN',None,'UNKNOWN','바닥 텍스처만 보이는 중심 영역','crop 안에서 원 native3의 선택 팔레트/의도 부위 또는 분명한 구조 feature를 식별하지 못했다.'),
  ('A',False,'END_CORNER',[43,26,74,66],'NEAR','왼쪽 작은 나무 상자의 위 모서리','팔레트가 아니라 앞쪽 상자의 코너로만 관찰했다.'),
  ('A',False,'EDGE_JUNCTION',[29,36,64,65],'ON','앞 나무 상자의 안쪽 가로대 접합','원 q/native가 선택 팔레트의 어느 부위를 묻는지 이 전경 상자 feature로 결정할 수 없다.'),
  ('A',False,'OTHER',[24,39,64,61],'ON','앞 나무 상자의 긴 손잡이 구멍 경계','유한한 local cutout feature이나 selected pallet native6의 의도 부위는 아니다.'),
  ('A',False,'EDGE_JUNCTION',[42,26,64,74],'ON','왼쪽 작은 나무 상자의 세로 finger-joint','same-frame 앞 상자와 뒤 팔레트를 구분하며 native7 대응은 UNKNOWN으로 남긴다.')],
 [
  ('A',False,'END_CORNER',[25,14,62,48],'NEAR','앞쪽 큰 나무 상자의 아래 코너','뒤의 팔레트와 분리되는 전경 나무 상자 코너다. native0 팔레트 부위라고 복사하지 않는다.'),
  ('N',False,'UNKNOWN',None,'UNKNOWN','바닥과 crop 왼쪽의 잘린 나무 가장자리','선택 팔레트의 부위도 완전한 국소 코너도 식별되지 않는다.'),
  ('N',False,'UNKNOWN',None,'UNKNOWN','바닥과 왼쪽에 잘린 전경 나무 조각','이 조각만으로 의도 부위/feature를 지정하지 못했다.'),
  ('A',False,'END_CORNER',[18,0,58,25],'OFF','crop 위쪽의 전경 상자 아래 코너','q3 중심은 바닥에 있으며 그 사실을 E로 쓰지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[20,28,60,58],'ON','앞 나무 상자의 세로 코너/가로대 접합','원 q4 부근의 구조는 전경 상자다. 선택 팔레트 native4 매칭은 확인되지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[31,26,68,58],'ON','밝은 가로대와 나무 상자 내부 경계','이 반복 가로대를 팔레트 native5와 동일시하지 않는다.'),
  ('A',False,'OTHER',[16,55,45,78],'NEAR','전경 나무 상자 측면의 둥근 밝은 돌기','작은 유한 feature의 ROI-only 관찰이다. selected pallet/native6 가시성 타깃은 아니다.'),
  ('A',False,'EDGE_JUNCTION',[35,16,68,66],'ON','나무 상자 모서리와 사선 가로대 접합','같은 영상의 뒤 팔레트와 다른 구조이며 native7 대응은 UNKNOWN이다.')],
 [
  ('A',False,'EDGE_JUNCTION',[34,28,63,75],'ON','검은 카트 손잡이의 왼쪽 아래 굽은 접합','선택 팔레트의 의도 부위를 식별하지 못했다. 작은 손잡이 영상은 local feature로만 보존한다.'),
  ('A',False,'EDGE_JUNCTION',[23,23,66,68],'ON','검은 손잡이 오른쪽 아래 굽은 접합','손잡이 외관과 native1 팔레트 의미를 연결하지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[37,18,62,70],'ON','검은 손잡이와 세로 지지대의 접합','팔레트 부위가 보인다는 근거가 아니며 흐린 세부 형상도 제한적이다.'),
  ('A',False,'EDGE_JUNCTION',[25,30,66,70],'ON','손잡이 왼쪽 아래 경계','검은 손잡이 자체의 존재만 관찰하며 native3 visibility는 UNKNOWN이다.'),
  ('A',False,'END_CORNER',[36,38,71,72],'OFF','검은 손잡이 왼쪽 위 굽은 끝','q4는 손잡이 위 바닥 쪽이다. ROI-only 부위를 native corner로 인증하지 않는다.'),
  ('A',False,'END_CORNER',[12,36,53,75],'ON','검은 손잡이 오른쪽 위 굽은 끝','작은 손잡이 feature를 팔레트의 native5 끝으로 읽지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[20,33,62,70],'ON','손잡이 오른쪽 안쪽 굽은 경계','foreground 손잡이는 보이나 선택 팔레트 가림/부위와 연결할 근거는 없다.'),
  ('A',False,'END_CORNER',[40,26,65,62],'ON','손잡이 왼쪽 위 굽은 경계','약한 local 구조 관찰이며 3D native 모서리 gold 또는 팔레트 D가 아니다.')],
 [
  ('A',False,'EDGE_JUNCTION',[31,28,65,70],'ON','검은 손잡이 왼쪽 아래 굽은 접합','이 crop에서 선택 팔레트를 식별하지 못했고 손잡이만 ROI-only로 남긴다.'),
  ('A',False,'EDGE_JUNCTION',[24,27,67,67],'ON','검은 손잡이 오른쪽 아래 접합','손잡이의 국소 경계와 native 팔레트 의미는 별개다.'),
  ('A',False,'EDGE_JUNCTION',[34,16,61,69],'ON','손잡이 오른쪽 세로 지지대 경계','세로 지지대만 보이며 selected pallet/native2를 가정하지 않는다.'),
  ('A',False,'EDGE_JUNCTION',[27,28,65,62],'ON','손잡이 왼쪽 아래 굽은 선','원 q3 부근 feature는 손잡이이고 팔레트 의도 부위는 미식별이다.'),
  ('N',False,'UNKNOWN',None,'UNKNOWN','바닥 텍스처와 crop 끝의 흐린 검은 부분','명확한 구조 feature와 selected pallet/native4를 식별하기 어려웠다.'),
  ('A',False,'END_CORNER',[23,36,61,71],'ON','손잡이 오른쪽 위 굽은 끝','표면이 흐리고 작아 실제 팔레트 native correspondence를 확인할 수 없다.'),
  ('A',False,'EDGE_JUNCTION',[21,34,65,69],'ON','손잡이 안쪽 경계와 오른쪽 지지대 접합','ROI-only local feature이며 팔레트 가시성의 답이 아니다.'),
  ('A',False,'END_CORNER',[40,32,69,70],'NEAR','손잡이 왼쪽 위 굽은 접합','local 손잡이 경계가 보이는 것만 보존하고 native7은 UNKNOWN이다.')],
 [
  ('N',False,'UNKNOWN',None,'UNKNOWN','흰 덮개 표면과 아래 padding','선택 팔레트의 의도 부위나 구조 feature를 원 crop 안에서 식별하지 못했다. 흰 표면을 E 근거로 자동 사용하지 않는다.'),
  None,None,None,
  ('A',False,'EDGE_JUNCTION',[49,31,77,70],'ON','회색 플라스틱 상자의 테두리/세로 리브 접합','보이는 것은 팔레트 위 상자이며 selected pallet native4 자체는 식별되지 않는다.'),
  ('D',True,'END_CORNER',[27,23,48,54],'NEAR','팔레트 오른쪽 긴 상판의 멀리 있는 둥근 끝','full-frame의 흰 덮개 옆 긴 목재 보드가 같은 팔레트로 이어지며 q5/q6의 오른쪽 끝 맥락이 보이는 끝을 약하게 주소화한다.'),
  ('D',True,'EDGE_JUNCTION',[29,28,50,70],'NEAR','같은 오른쪽 긴 팔레트 보드의 바깥 옆면/끝 전이','q5 아래 q6 쪽에서 동일 보드의 바깥 측면이 원 crop에 보인다. 물리 3D keypoint가 아니라 weak intended-part 관찰이다.'),
  ('A',False,'EDGE_JUNCTION',[31,24,61,75],'ON','회색 상자 앞쪽의 세로 리브/테두리 접합','회색 상자 feature를 selected pallet native7 부위 또는 그 가림으로 복사하지 않는다.')]
]


def read(p):
    p = Path(p)
    s = str(p)
    assert 'v112' not in s.lower() and '..' not in p.parts
    assert 'actual_sameframe_observations_a' not in s
    for q in [p, *p.parents]:
        assert not q.is_symlink(), str(q)
    return p.read_bytes()


def binding(p):
    b = read(p)
    return {'path':str(p), 'sha256':hashlib.sha256(b).hexdigest(), 'size_bytes':len(b)}


def load_bound(b):
    raw = read(b['path'])
    assert hashlib.sha256(raw).hexdigest() == b['sha256']
    return json.loads(raw)


def put(name, x):
    p = O / name
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return binding(p)


def main():
    assert binding(ENTRY)['sha256'] == ENTRY_SHA
    e = json.loads(read(ENTRY))
    assert e['approved'] is True and e['training_admission'] is False
    scope = load_bound(e['root_actual_view_scope'])
    assert scope['approved_actual_generated_PNG_views_and_two_reader_SOURCE_annotations'] is True
    template = load_bound(e['templates'])
    split = load_bound(e['split'])
    q = load_bound(e['independent_contract_implementation_QA'])
    assert q['completed'] is True and q['PASS'] is True
    addendum = load_bound(e['exact_pixel_union_addendum'])
    assert addendum['current_native_or_auxiliary_training_admission'] is False
    helper_raw = read(e['helper']['path'])
    assert hashlib.sha256(helper_raw).hexdigest() == e['helper']['sha256']
    helper = types.ModuleType('C_memory_only_contract')
    exec(compile(helper_raw, e['helper']['path'], 'exec'), helper.__dict__)
    whitelist = load_bound(e['generated_image_whitelist_exact'])
    identities = set()
    def collect(x):
        if isinstance(x, dict):
            if isinstance(x.get('path'), str) and isinstance(x.get('sha256'), str):
                identities.add((x['path'], x['sha256']))
            for z in x.values(): collect(z)
        elif isinstance(x, list):
            for z in x: collect(z)
    collect(whitelist)
    base = copy.deepcopy(template['rows'])
    assert len(base) == len(split['rows96']) == 96
    assert len(M) == 12 and all(len(x) == 8 for x in M)
    views = []
    for f in range(12):
        b = base[f*8]['display_bindings']['contact']
        assert (b['path'], b['sha256']) in identities
        assert binding(b['path'])['sha256'] == b['sha256']
        views.append({'binding':b,'position':f,'frame_key':base[f*8]['input']['frame_key'],
                      'tool':'tools.view_image via functions.exec','detail':'original','actually_viewed':True,
                      'batch':'C_CONTACT_BATCH_'+str(f//3+1),'includes':'full-image/q8 display + all8 original crop/clean/valid panels'})
    for f,n in sorted(OPTIONAL_CLEAN):
        b = base[f*8+n]['display_bindings']['clean96']
        assert (b['path'], b['sha256']) in identities
        assert binding(b['path'])['sha256'] == b['sha256']
        views.append({'binding':b,'position':f,'native_slot':n,'frame_key':base[f*8+n]['input']['frame_key'],
                      'tool':'tools.view_image via functions.exec','detail':'original','actually_viewed':True,
                      'batch':'C_OPTIONAL_CLEAN_BATCH_2' if f==11 else 'C_OPTIONAL_CLEAN_BATCH_1'})
    all_view_ids = {helper.binding_identity(x['binding']) for x in views}
    rows = []
    for j, r in enumerate(base):
        f,n = divmod(j,8)
        orig = split['rows96'][j]
        assert all(r['input'][k] == orig[k] for k in r['input'])
        r['reader_id'] = 'C'
        r['observation_complete'] = True
        r['actually_viewed_image_bindings'] = [r['display_bindings']['contact']]
        if (f,n) in OPTIONAL_CLEAN:
            r['actually_viewed_image_bindings'].append(r['display_bindings']['clean96'])
        if not r['input']['original_query_available']:
            assert M[f][n] is None
            r['ambiguity_note'] = 'ORIGINAL_INPUT_U_RETAINED: finite displayed q or visible neighboring material cannot replace original U. All six axes UNKNOWN; no ROI.'
        else:
            mode, identified, kind, box, relation, desc, note = M[f][n]
            r['instance'] = {'status':'YES' if identified else 'UNKNOWN',
                             'evidence':'Full-frame와 이 원 crop의 목재 상판/지지 구조가 같은 선택 팔레트로 이어져 보인다.' if identified else ''}
            r['part'] = {'status':'YES' if mode=='D' else 'UNKNOWN',
                         'description':desc if mode=='D' else '',
                         'evidence':note if mode=='D' else ''}
            r['witness'] = {'status':'VISIBLE' if box is not None else 'UNKNOWN',
                            'kind':kind,'roi':box,'description':desc if box is not None else '',
                            'evidence':'실제 원본 크기 contact의 원 RAW96/clean96 panel에서 이 구조 경계가 보인다. '+desc if box is not None else ''}
            r['addressability'] = {'status':'QUERY_NATIVE_SUPPORTED' if mode=='D' else 'EXPLICIT_ROI_ONLY' if mode=='A' else 'UNKNOWN',
                                   'evidence':note if mode=='D' else '이 feature는 추가 explicit ROI로만 지정했다. 선택 팔레트의 원 native 부위라는 양성 대응 근거는 부족하다.' if mode=='A' else ''}
            r['query_support'] = {'state':relation,
                                  'evidence':('원 cyan 중심과 관찰 feature의 시각적 관계를 '+relation+'으로 독립 기록했다. '+desc+'; 새 수치거리 threshold/보정 q를 쓰지 않았다.') if relation!='UNKNOWN' else ''}
            r['visibility'] = {'state':'D' if mode=='D' else 'UNKNOWN',
                               'evidence':('주소화한 같은 팔레트의 의도 부위가 원 crop 안에서 실제 보인다. '+desc) if mode=='D' else '',
                               'occlusion_evidence':'','evidence_roi':None}
            r['ambiguity_note'] = note + ' Assistant SOURCE weak 관찰이며 physical/native keypoint gold·학습 admission이 아니다.'
        assert r['training_admission'] is False
        assert helper.validate_observation(r, template['rows'][j])
        assert all(helper.binding_identity(b) in all_view_ids for b in r['actually_viewed_image_bindings'])
        rows.append(r)
    counts = {'all_original_rows':96,'frames':12,'original_inputU':sum(not r['input']['original_query_available'] for r in rows),
              'available_rows':sum(r['input']['original_query_available'] for r in rows),
              'internal_source_roles':dict(collections.Counter(r['input']['internal_source_role'] for r in rows)),
              'visibility':dict(collections.Counter(r['visibility']['state'] for r in rows)),
              'witness':dict(collections.Counter(r['witness']['status'] for r in rows)),
              'addressability':dict(collections.Counter(r['addressability']['status'] for r in rows)),
              'query_support':dict(collections.Counter(r['query_support']['state'] for r in rows)),
              'instance':dict(collections.Counter(r['instance']['status'] for r in rows)),
              'part':dict(collections.Counter(r['part']['status'] for r in rows)),
              'actual_distinct_image_tool_views':len(views),'actual_contact_views':12,'actual_optional_clean96_views':len(OPTIONAL_CLEAN)}
    assert counts['original_inputU']==11 and counts['available_rows']==85
    assert counts['internal_source_roles']=={'SOURCE_FIT':64,'SOURCE_DEV':32}
    assert len({r['input']['row_id'] for r in rows})==96
    assert all(r['visibility']['state']=='UNKNOWN' for r in rows if not r['input']['original_query_available'])
    entry_binding = binding(ENTRY)
    code_binding = binding(Path(__file__))
    view_receipt = put('ACTUAL_IMAGE_VIEW_RECEIPT.json',{
        'completed':True,'PASS':True,'reader_id':'C','entry':entry_binding,'views':views,'own_views_only':True,
        'counts':{'distinct_contacts':12,'distinct_optional_clean96':12,'distinct_generated_PNGs':24},
        'no_separate_fullRGB_or_q8_PNG_toolview_claim':'Full image and q8 are included in the twelve viewed contact cards; no separate fullRGB/q8 PNG view is claimed.',
        'no_original_source_RGB_redecode':True,'no_browser_render_claim':True,
        'metadata_membership_prechecks':{'contacts':'deaacb exit0','optional10':'ea9a49 exit0','optional2':'355c1b exit0'}})
    observations = put('ALL96_OBSERVATIONS.json',{
        'completed':True,'PASS':True,'reader_id':'C','entry':entry_binding,'contract':e['contract'],
        'protocol':e['protocol'],'template':e['templates'],'exact_pixel_union_addendum':e['exact_pixel_union_addendum'],
        'rows96':rows,'training_admission':False,'counts':counts,
        'authority':'INDEPENDENT_ASSISTANT_SOURCE_PART_QUERY_WEAK_OBSERVATION_NOT_NATIVE_GOLD',
        'other_reader_new_values_read':0})
    independence = put('INDEPENDENCE_RECEIPT.json',{
        'completed':True,'PASS':True,'reader_id':'C','entry':entry_binding,'own_observations_binding':observations,
        'other_reader_new_observation_files_read':0,'other_reader_new_values_received':0,'shared_schema_only':True,
        'historical_V215_appearance_exposure_disclosed':True,
        'historical_exposure_note':'These frozen V215 derived source images were previously generated/viewed at phase/root level; this C receipt claims only the 24 actual tool views listed now. Earlier task context and shared schema are not fresh-blind evidence. No A V216 new values were received/read.',
        'not_independent_human_readers_or_physical_gold':True,'training_admission':False})
    qa = put('SAVED_OBSERVATION_SCHEMA_QA.json',{
        'completed':True,'PASS':True,'reader_id':'C','entry':entry_binding,'observations':observations,'view_receipt':view_receipt,
        'helper':e['helper'],'template':e['templates'],'split':e['split'],'exact_pixel_union_addendum':e['exact_pixel_union_addendum'],
        'pure_validator_called_all96':True,'all96_exact_order_original_rowID_qL_native_U_flags_component_role_and_fitfalse':True,
        'all96_view_bindings_subset_of_own_actual24_views':True,'all12_contact_frames_actual_viewed':True,
        'all11_original_U_six_axes_UNKNOWN_and_ROI_null':True,'SOURCE_FIT64_DEV32_unchanged':True,
        'all_ROIs_inclusive_integer_0_to95_no_q_containment_or_minimum_area_gate':True,
        'witness_ROI_not_a_corrected_query_or_3D_keypoint':True,'ROI_only_labels_not_copied_to_native':True,
        'no_E_from_other_object_background_queryOFF_or_absence':True,'all_training_admission_false':True,
        'no_pixelproof_or_semantic_samefeature_consensus_claim':True,'counts':counts})
    put('PRE_OPTIONAL_VIEW_METADATA_SHAPE_FAILURE_PRESERVED.json',{
        'completed':True,'preserved_failure':True,'chunk_id':'53424a','exit_code':1,'error':"KeyError: 'path'",
        'stage':'optional PNG membership metadata-only check before any optional image body opened',
        'cause':'Whitelist images contain nested binding records rather than bare path/SHA records.',
        'correction':'Same declared metadata identities collected recursively in memory; no binding followed. Exact optional membership/SHA precheck then passed ea9a49 before view.',
        'source_science_image_or_annotation_modified':False,'producer_or_scientific_rerun':False})
    out = put('COMPLETE.json',{
        'completed':True,'PASS':True,'reader_id':'C','entry':entry_binding,'observations':observations,
        'view_receipt':view_receipt,'schema_QA':qa,'recording_code':code_binding,'independence':independence,
        'counts':counts,'contract':e['contract'],'protocol':e['protocol'],'template':e['templates'],
        'root_scope':e['root_actual_view_scope'],'exact_pixel_union_addendum':e['exact_pixel_union_addendum'],
        'training_admission':False,'own_observation_files_frozen_no_further_writes':True,
        'source_arrays_original_encoded_RGB_GT_teacher_oldlabels_model_scores_or_Git_reads':0,
        'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'completed':True,'PASS':True,'COMPLETE':out,'observations':observations,'schema_QA':qa,'view_receipt':view_receipt,'counts':counts},ensure_ascii=False))


if __name__=='__main__':
    main()
