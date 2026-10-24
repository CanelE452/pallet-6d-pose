#!/usr/bin/env python3
"""A's independent observations after actually viewing the exact15 S0 contacts."""
import collections,csv,hashlib,io,json,os,stat
P='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility'
D=P+'/query_visibility_public_train_intended_part_query_targets_20261025_v218';H=D+'/actual_gt_query_part_observations_a'
STATES={'D':'D_VISIBLE_EXPECTED_PART_WEAK','E':'E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK','S':'SELF_OR_OWN_STRUCTURE_UNOBSERVABLE','W':'WRONG_PART_OR_GEOMETRY_UNRESOLVED','U':'UNKNOWN','O':'OUT_OF_FRAME'}
# Authored from the actual15 displayed contacts; no other observer values consumed.
# [state, part/context evidence, optional ROI, occluder, foreground evidence, uncertainty]
NOTES={
(0,0):{
4:['D','선언 후보의 긴 상판 왼끝 모서리와 위쪽 면의 꺾임이 문맥과 crop에서 이어진다.',[38,38,63,63],'','','왼쪽 crop padding이 있으나 보이는 끝 feature에 대한 약한 관찰이다.'],
5:['D','같은 상판 오른쪽 끝의 자른 면/상면 경계가 crop 안에 보이며 전경 팔레트와 연결된다.',[32,28,58,72],'','','정확한 명목점 위치나 실제 치수 gold는 아니다.'],
6:['W','crop은 상판의 아래 윤곽과 판 사이 틈을 보인다. 기대 BOTTOM_RUNNER 끝을 이 구조로 식별할 수 없다.',None,'','','목재/끝 윤곽 존재를 하단 runner 보임으로 복사하지 않는다.'],
7:['W','참조 부근은 위쪽 상판의 긴 아래 모서리/틈이다. 별도 하단 runner 끝의 대응은 미해결이다.',None,'','','다른 해부학적 부위의 목재를 native D로 쓰지 않는다.']},
(0,1):{
5:['W','후보1의 명목 deck 끝 부근에 전경 상판의 중간 긴 모서리가 보이지만 이 후보의 끝이라는 연결은 불명확하다.',None,'','','두 겹 팔레트/명목 배치의 대응이 남아 있어 보이는 임의 판을 D로 지정하지 않는다.'],
6:['W','crop 중심은 상판 사이 긴 간격과 판면이다. 이 후보의 bottom-runner 끝을 식별하지 못한다.',None,'','','원영상 아래쪽 부분의 후보 경계/부위가 불확실하다.']},
(2,0):{
0:['D','위쪽 팔레트의 가까운 왼쪽 상판 끝과 잘린 코너가 보이고 전경 상판으로 연속된다.',[47,40,75,67],'','','q와 정확히 일치한 3D점 인증은 아니다.'],
3:['W','crop에서 가까운 목재 블록 얼굴은 보이지만 기대 bottom-runner의 끝/수평 하단판과 구별할 수 없다.',None,'','','블록면을 runner 끝으로 바꾸지 않는다.'],
4:['D','선언된 위쪽 팔레트의 먼 왼쪽 deck-board 끝 경계가 crop과 전체 길이 방향 상판에 연결된다.',[36,39,64,62],'','','약한 유한 부위 근거이며 좌표 gold가 아니다.'],
5:['D','위쪽 팔레트의 먼 오른쪽 deck 끝이 바닥과 맞닿는 잘린 모서리로 보인다.',[20,29,50,61],'','','명목 중심 바깥 가까운 실제 끝도 같은 crop의 부위 근거로 기록한다.'],
6:['W','명목 runner 부근에는 위 상판의 옆 윤곽과 더 아래 팔레트의 판이 함께 보인다.',None,'','','상판/아래층 판/runner의 정체성이 불분명하다.'],
7:['W','중심 부근은 위쪽 상판 사이 틈과 그 판의 긴 옆면이다. runner 끝의 식별 근거가 없다.',None,'','','빈 틈/목재를 외부 가림 E로 쓰지 않는다.']},
(2,1):{
0:['W','명목 deck 위치에서 위쪽 팔레트의 가까운 상판 코너가 보이지만 아래 후보의 같은 끝으로 식별하기 어렵다.',None,'','','겹친 두 팔레트의 끝 대응과 명목 치수 차이가 남는다.'],
3:['W','가까운 블록과 위층 구조가 보이나 아래 후보의 하단 runner 끝은 독립적으로 구분되지 않는다.',None,'','','다른 층 블록을 기대 runner로 복사하지 않는다.'],
4:['E','아래층 팔레트는 오른쪽으로 드러난 판/지지구조로 문맥상 이어지고, 선언된 먼 deck 끝 부근은 위층 팔레트의 넓은 상판 아래에 놓인다.',[23,21,73,69],'UPPER_PALLET_OTHER_INSTANCE','crop의 불투명한 밝은 상판은 위층의 다른 팔레트에 속해 아래 후보의 해당 deck 끝 부근을 덮는다.','명목 pose/치수와 실제 끝의 정확 위치는 불확실하며 약한 부위 가림 판단이다.'],
5:['D','아래층으로 오른쪽에 드러난 deck-board의 바깥 끝/자른 모서리가 crop 왼쪽에서 보인다.',[18,25,43,62],'','','위층 끝과 구분되는 아래층의 노출된 판 끝에 한정한 약관찰이다.'],
6:['W','오른쪽 아래 층의 판/블록 옆면은 보이지만 기대 bottom-runner 끝인지 확정할 수 없다.',None,'','','명목점 주변 바닥/판 끝만으로 D/E를 정하지 않는다.'],
7:['W','crop에는 위층 상판과 간격이 주로 보여 아래 후보 runner의 의도 부위가 식별되지 않는다.',None,'','','다른 팔레트와 자기 구조의 영향도 분리되지 않는다.']},
(3,0):{
0:['D','전경 팔레트의 왼쪽 상판 외측 끝과 상면/끝면 모서리가 보인다.',[48,43,78,67],'','','약한 끝 부위 보임이며 정확 3D vertex 인증은 아니다.'],
3:['D','앞쪽 하단 수평 runner의 왼쪽 끝이 바닥 바로 위에서 보이고 같은 팔레트 하단 판으로 이어진다.',[45,41,78,62],'','','블록 자체가 아닌 아래 수평 판 끝의 약한 해부학적 근거다.'],
4:['W','crop 중심에는 긴 앞쪽 deck의 위 모서리가 보이나 먼쪽 끝의 identity는 분명하지 않다.',None,'','','파란 상자와 앞쪽 판이 겹쳐 기대 rear deck 끝을 분리하기 어렵다.'],
5:['E','전경 팔레트 상판은 아래로 연속되고 선언된 먼 오른쪽 deck 끝 부근 위에 회색 플라스틱 통이 놓여 있다.',[18,9,77,49],'GRAY_PLASTIC_BIN','crop의 회색 불투명 통 바닥/옆면이 그 deck 끝 부근을 가리고 있다.','가림 부위의 정확 점/명목 치수까지 확인한 것은 아니다.'],
6:['U','명목 runner 부근은 어두운 fork bay/안쪽 구조여서 어떤 runner 끝인지 식별하지 못한다.',None,'','','어둠/자기 구조/배경을 외부 가림으로 해석하지 않는다.'],
7:['U','crop의 갈색 안쪽 구조는 보이나 기대 runner 끝과 전면 블록/자기 가림의 구분이 불명확하다.',None,'','','자기/외부 또는 실제 틈의 원인을 단정하지 않는다.']},
(4,0):{
0:['D','전경 팔레트 왼쪽 상판 끝이 EPAL 블록 위의 수평 판과 이어지는 코너로 보인다.',[45,42,76,65],'','','보이는 판 끝의 약관찰이며 블록면 자체를 top 끝으로 지정하지 않는다.'],
3:['D','전면 하단 runner의 왼쪽 바깥 끝과 바닥 경계가 같은 팔레트 수평 하단 판으로 이어진다.',[44,41,73,61],'','','실제 끝 부위의 약한 보임이다.'],
4:['W','명목 rear deck 부근에는 앞쪽 긴 판과 파란 상자 하단이 함께 보이나 의도 끝 코너가 식별되지 않는다.',None,'','','상판의 중간 모서리를 rear 끝으로 바꾸지 않는다.'],
5:['E','동일 팔레트 상판의 먼 오른쪽 deck 부근에 회색 통이 상판 위를 차지한다.',[17,12,77,58],'GRAY_PLASTIC_BIN','crop 중심의 불투명 회색 통 면이 기대 deck 끝 부근을 덮는다.','GT명목 부위에 대한 약한 외부 가림으로만 남긴다.'],
6:['W','runner 참조 부근이 앞쪽 긴 상판 밑의 열린 bay에 놓인다. 기대 하단 끝의 물질은 식별되지 않는다.',None,'','','열린 틈/명목 구조 차이를 E로 바꾸지 않는다.'],
7:['W','앞쪽 판과 안쪽 블록/틈이 보일 뿐 기대 rear bottom-runner의 끝과 맞는지 불확실하다.',None,'','','같은 팔레트의 임의 구조를 기대 runner로 대체하지 않는다.']},
(5,0):{
1:['D','가까운 오른쪽 deck 끝의 잘린 모서리가 회색 통 발치의 crop 왼쪽 아래에 보이고 팔레트 상판으로 이어진다.',[15,46,45,76],'','','참조 중심과 실제 코너 사이 어긋남은 남기며 q를 옮기지 않는다.'],
2:['W','crop은 팔레트 오른쪽 블록 옆면과 바닥을 보이나 별도 bottom-runner 끝인지 분리되지 않는다.',None,'','','블록/판/runner의 해부학적 구분이 불명확하다.'],
4:['D','왼쪽 먼 deck-board의 끝이 바닥과 만나는 바깥 코너로 분명히 보인다.',[45,41,74,69],'','','같은 상판 끝의 약관찰이며 native point gold가 아니다.'],
5:['E','대상 팔레트의 상판이 파란 상자 아래로 이어지고 선언된 먼 deck 끝 부근은 상자의 전면 아래에 있다.',[18,18,77,75],'BLUE_PLASTIC_CRATE','crop에서 파란 불투명 상자 면이 기대 deck 끝 부근을 가린다.','정확한 팔레트 치수/점의 ray 순서는 인증하지 않는다.'],
6:['E','같은 팔레트 먼쪽 아래 구조의 선언된 runner 부근이 파란 상자 전면 뒤에 놓인다.',[19,18,75,74],'BLUE_PLASTIC_CRATE','crop의 파란 불투명 면과 상자 하단이 기대 아래 부위의 영상 이웃을 덮는다.','기대 runner의 정확한 물리 점은 보이지 않으며 약한 부위 가림이다.'],
7:['W','crop에 보이는 것은 왼쪽 상판 끝의 옆/아래 모서리다. 하단 runner의 끝으로 동일시할 근거는 없다.',None,'','','DECK와 BOTTOM_RUNNER의 분리를 유지한다.']},
(6,3):{
0:['D','뒤에 세워진 대상 팔레트의 왼쪽 위 deck-board 바깥 끝 코너가 벽과 구분되어 보인다.',[46,40,74,69],'','','원래 detector query와 다른 새GT-family 부위 관찰이다.'],
1:['D','세워진 대상 팔레트의 오른쪽 위 deck-board 자른 끝과 바깥 경계가 보인다.',[22,37,51,69],'','','앞쪽 나무 상자의 모서리와 구분해 뒤 팔레트 상단 끝을 기록한다.'],
2:['S','상단 오른쪽 기대 runner는 세워진 팔레트 뒤쪽 구조인데 영상에는 같은 팔레트의 앞 deck-board 끝면이 불투명하게 겹쳐 보인다.',[30,39,58,68],'','','자기 앞판이 가리는 약한 구조 판단이며 정확 backside 점 인증은 아니다.'],
3:['S','상단 왼쪽 기대 runner의 방향은 같은 팔레트 앞 deck-board 뒤이며 그 앞판/끝면이 해당 이웃을 차지한다.',[44,39,73,69],'','','외부 물체는 이 상단 이웃에 보이지 않으며 자기 구조 불가시성으로만 기록한다.'],
4:['D','대상 팔레트 왼쪽 아래 deck 외곽 끝 일부가 앞 나무 상자 위/왼쪽에서 분리되어 보인다.',[32,19,55,44],'','','부분적으로 노출된 유한 끝 부위다. 앞 상자와 겹친 중심을 정확 점 gold로 쓰지 않는다.'],
5:['E','세워진 팔레트 하단 오른쪽 deck 쪽은 앞의 별도 열린 나무 상자 뒤로 이어진다.',[20,24,77,70],'FOREGROUND_WOOD_CRATE','crop의 불투명 상자 가로판이 선언된 팔레트 하단 deck 끝 이웃을 덮는다.','앞 상자의 목재를 대상 팔레트 D로 복사하지 않는다.'],
6:['E','세워진 팔레트의 선언된 아래 runner 이웃은 앞 나무 상자 뒷판/가로판 뒤에 놓인다.',[20,20,76,70],'FOREGROUND_WOOD_CRATE','crop 중심을 차지하는 별도 상자의 나무 가로판이 대상 하단 부위의 영상 이웃을 가린다.','정확한 숨은 runner 점/자기 구조 순서를 인증한 것은 아니다.'],
7:['S','왼쪽 아래 runner의 투영 이웃에는 대상 팔레트 자신의 앞쪽 deck 판/끝면이 놓여 뒤 구조가 직접 구별되지 않는다.',[31,22,59,56],'','','보이는 앞판을 runner D로 바꾸지 않고 자기 구조 약관찰로 남긴다.']},
(7,3):{
0:['D','세워진 팔레트 오른쪽 위 deck-board 외곽 끝의 상면/끝면 경계가 보인다.',[22,32,52,64],'','','일부 crop padding이 있지만 feature는 원crop 안에서 보인다.'],
1:['D','세워진 대상 팔레트의 오른쪽 아래 deck 끝 코너가 바닥 바로 위의 crop 왼쪽 위에 보인다.',[16,15,45,45],'','','명목 중심의 바닥과 별도로 실제 끝의 약한 근거를 기록한다.'],
2:['W','명목 runner는 바닥/팔레트 옆 끝 이웃에 있으나 보이는 것은 앞 판/블록 경계여서 runner 끝을 분리하지 못한다.',None,'','','바닥에 놓인 참조를 외부 가림으로 쓰지 않는다.'],
3:['W','명목 runner 부근은 상단 끝의 벽/패딩과 deck 가장자리다. 기대 뒤 구조와 정확 대응은 미해결이다.',None,'','','앞 deck 끝을 runner로 승격하지 않는다.'],
4:['D','세워진 팔레트 왼쪽 위 deck-board 바깥 끝이 벽과 만나는 코너로 보인다.',[46,36,76,68],'','','약한 부위 근거다.'],
5:['E','선언된 팔레트의 왼쪽 아래 deck 끝 이웃 앞에 별도 낮은 나무 상자가 놓여 있다.',[17,31,78,68],'FOREGROUND_WOOD_CRATE','crop의 밝은 불투명 상자 상단 판이 대상 팔레트 하단 끝 이웃을 덮는다.','그 밝은 판은 팔레트 material-positive가 아니다.'],
6:['S','하단 runner 이웃에 대상 팔레트의 stamped 앞판과 끝면이 먼저 보이고 뒤쪽 runner가 직접 구별되지 않는다.',[25,18,72,48],'','','별도 상자 가장자리도 가까워 정확 점의 occlusion 순서는 불확실하다.'],
7:['S','세워진 팔레트 왼쪽 위에서 기대 backside runner 쪽은 동일 팔레트 앞 deck-board의 불투명한 끝면 뒤에 있다.',[40,36,71,65],'','','front/rear 이름만이 아니라 실제 앞판의 중첩을 근거로 한 약판단이다.']},
(8,3):{
1:['D','세워진 대상 팔레트의 오른쪽 아래 deck 끝이 crop 왼쪽 위의 바닥과 맞닿는 코너로 보인다.',[15,14,44,45],'','','중심은 바닥 쪽이지만 q를 옮기지 않고 보이는 동일 끝 부위만 기록한다.'],
2:['W','보이는 오른쪽 아래 앞판/블록 윤곽을 bottom-runner 끝과 구분하기 어렵다.',None,'','','같은 목재라는 이유만으로 runner D를 주지 않는다.'],
4:['D','대상 팔레트 왼쪽 위 deck 외곽 코너가 벽과 구분되는 판의 끝으로 보인다.',[46,36,74,68],'','','원crop 안의 약한 유한 feature 관찰이다.'],
5:['E','대상 팔레트 왼쪽 아래 deck 끝 쪽은 앞에 놓인 별도의 열린 나무 상자 뒤에 있다.',[17,20,76,72],'FOREGROUND_WOOD_CRATE','crop 중앙의 상자 테두리/가로판이 기대 팔레트 끝 이웃을 불투명하게 가린다.','앞 상자와 대상 팔레트의 같은 목재를 구분한다.'],
6:['E','대상 팔레트 아래 runner 쪽의 선언된 이웃에 앞 나무 상자의 내부 가로판이 중첩된다.',[18,24,76,73],'FOREGROUND_WOOD_CRATE','crop 중앙의 밝은 별도 상자 판이 그 부위의 영상 이웃을 덮고 있다.','상자 사이 간격/원래 자기 가림의 정확 순서는 gold가 아니다.'],
7:['S','상단 왼쪽 기대 runner 뒤 구조는 동일 팔레트의 앞 deck-board 끝면에 가려 직접 구별되지 않는다.',[39,36,74,66],'','','자기 앞판의 약한 가시 증거이며 runner의 실제 3D 좌표 판정은 아니다.']},
(9,0):{
4:['E','목재 받침 팔레트가 회색 crate 아래와 앞/옆으로 이어지고 선언된 뒤 deck 끝 부근은 crate 바닥 아래에 놓인다.',[18,18,77,75],'GRAY_PLASTIC_CRATE','crop 중앙의 불투명 회색 crate 바닥/안쪽 면이 팔레트 deck 끝 이웃을 가린다.','mask/depth가 아니라 실제 보이는 적재 물체와 팔레트 문맥의 약근거다.'],
5:['W','명목 deck 끝 crop에는 거의 바닥만 있고 실제 목재는 가장자리 일부다. 기대 끝 코너와의 물질 대응은 미해결이다.',None,'','','바닥/부재를 E로 쓰지 않는다.'],
6:['W','crop 왼쪽 목재 옆면과 바닥은 보이나 기대 bottom-runner 끝인지 분리되지 않는다.',None,'','','명목 치수/부위 정합의 불확실성이다.'],
7:['U','crop에는 회색 crate 내부 면이 보이지만 기대 runner가 원래 deck 뒤에서 자기 가림인지와 외부 적재물 가림을 구분하지 못한다.',None,'','','foreign pixel 하나만으로 runner E를 주지 않는다.']},
(10,0):{
1:['W','명목 deck 끝 crop은 바닥이 대부분이고 작은 목재 조각만 가장자리에 있어 같은 끝을 식별하기 어렵다.',None,'','','실제 코너 위치/형상 불확실성을 보존한다.'],
2:['W','명목 runner crop에 목재의 긴 옆 경계와 바닥이 보이나 기대 runner의 끝은 식별되지 않는다.',None,'','','목재 경계/부재를 D/E로 복사하지 않는다.'],
5:['E','선언된 받침 팔레트 deck의 뒤 끝 부근이 회색 crate 아래로 이어진다.',[19,17,77,76],'GRAY_PLASTIC_CRATE','crop의 불투명 회색 crate 안쪽 벽/바닥이 해당 deck 끝 영상 이웃을 차지한다.','적재 crate를 같은 팔레트 부위로 바꾸지 않는다.'],
6:['U','crop에는 회색 crate 면이 있지만 기대 runner와 팔레트 자체 deck/지지구조 및 외부 crate 가림의 순서를 분리할 수 없다.',None,'','','물체@q만으로 native 외부 E가 아니다.']},
(11,0):{
4:['E','목재 받침 팔레트의 선언된 뒤 deck 끝 부근 위에 회색 crate와 금속 통이 적재되어 있다.',[41,17,78,74],'METAL_BIN_AND_GRAY_CRATE','crop 중심 오른쪽의 불투명 검은/금속 통 몸체와 crate 바닥이 기대 deck 끝 이웃을 가린다.','GT명목 끝의 물리 점이나 원 detector 위치를 인증하지 않는다.'],
5:['W','명목 deck 끝은 바닥 쪽에 놓여 crop의 가장자리 목재만으로 실제 끝 코너를 식별할 수 없다.',None,'','','다른 적재물/바닥의 부재를 E로 바꾸지 않는다.'],
6:['W','목재 옆면과 바닥은 보이나 기대 bottom-runner 끝과의 해부학적 연결이 미해결이다.',None,'','','visible material과 기대 runner의 visibility는 다르다.'],
7:['U','회색 crate 외벽은 보이나 기대 runner가 원래 자기 구조 뒤인지 외부 적재물로 가려졌는지 분리되지 않는다.',None,'','','기대 part 주소화/가림 원인 불확실성을 남긴다.']}}
def op(p,flags=os.O_RDONLY):
 assert isinstance(p,str) and 'v112' not in p.lower() and os.path.isabs(p) and os.path.normpath(p)==p
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for c in p.strip('/').split('/')[:-1]:
   n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  z=os.open(p.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd);assert stat.S_ISREG(os.fstat(z).st_mode);return z
 finally:os.close(fd)
def rb(p):
 with os.fdopen(op(p),'rb') as f:return f.read()
def sha(z):return hashlib.sha256(z).hexdigest()
def bind(p,z):return {'path':p,'sha256':sha(z),'size_bytes':len(z)}
def checked(p,h):
 z=rb(p);assert sha(z)==h;return z
def enc(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def new(name,z):
 p=H+'/'+name
 with os.fdopen(op(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(z)
 return bind(p,z)
def newj(name,x):return new(name,enc(x))
def main():
 planb=checked(H+'/BEFORE_VIEWS_EXACT15_CONTACT_SHA_QA.json','c54c3c34f8a496931f12858dd1a22e47c9647bd8e8ad5f89354cef335677bdc8');plan=json.loads(planb)
 ledgerpath=D+'/public_source_gt_query_jitter_v1/ALL360_GENERATED_INPUTS.jsonl';lb=checked(ledgerpath,'37e1aab673d2ae115f0c004e3c401ba3720adf90a71ff30ca2e30bdd33aa5353')
 source=[json.loads(l) for l in lb.splitlines()];s0=[r for r in source if r['normalized_offset']==[0.,0.]];assert len(s0)==120
 contacts={};views=[]
 for i,b in enumerate(plan['fixed15_view_bindings']):
  name=b['path'].rsplit('/',1)[1];f=int(name[1:3]);c=int(name[5:11]);contacts[(f,c)]=b
  views.append({'position':i,'batch':i//3+1,'binding':b,'tool':'tools.view_image','detail':'original','actually_viewed':True,'scope':'one contact fullcontext plus all8 S0 native crops','standalone_RGB96_files_viewed':0})
 rows=[]
 for r in s0:
  f,c,n=r['frame_position'],r['candidate_gt_list_index'],r['native_slot'];key=(f,c)
  if not r['nominal_target_in_frame']:
   assert n not in NOTES.get(key,{})
   v=['O','원영상 canvas 밖의 명목 타깃이다. crop 일부 원영상 support가 있어도 이 타깃을 외부 E로 바꾸지 않는다.',None,'','','OUT_OF_FRAME은 새 GT-family 명목 타깃 flag이며 옛 original inputU와 다른 축이다.']
  else:v=NOTES[key][n]
  state,evidence,roi,occluder,foreign,ambiguity=v
  if state in ('D','E'):assert roi is not None
  if roi is not None:assert len(roi)==4 and all(type(x) is int for x in roi) and 0<=roi[0]<=roi[2]<=95 and 0<=roi[1]<=roi[3]<=95
  if state=='E':assert occluder and foreign
  rows.append({'row_id':r['row_id'],'family':r['family'],'frame_position':f,'candidate_gt_list_index':c,'native_slot':n,'canonical_index':r['canonical_index'],'anatomical_family':r['anatomical_family'],'source_role':r['source_role'],'component_id':r['component_id'],'generated_query_xy':r['generated_query_xy'],'L':r['L'],'generated_query_available':r['generated_query_available'],'nominal_target_in_frame':r['nominal_target_in_frame'],'original_inputU_unchanged':not r['original_input']['original_query_available'],'intended_GT_candidate':'DECLARED_BY_TRAIN_FAMILY_CONSTRUCTION_NOT_GT_BEST','original_detector_instance_correspondence':'UNKNOWN','state':STATES[state],'part_identity_evidence':evidence,'occluder':occluder or None,'foreign_evidence':foreign or None,'optional_roi96':roi,'ambiguity_reason':ambiguity,'actualview_image_binding':contacts[key],'reader_id':'A','authority':'INDEPENDENT_ASSISTANT_SOURCE_INTENDED_PART_WEAK_NOT_PHYSICAL_NATIVE_GOLD','ROI_is_weak_bag_selector_not_segmentation_gold_or_deployed_input':True,'label_transfer_to_jitter':'NOT_AUTOMATIC_REQUIRES_TRANSFORMED_TARGET_CROP_SUPPORT_CHECK','future_fit_role_eligible':r['source_role']=='SOURCE_FIT','SOURCE_DEV_all_descendants_excluded_all_fit':r['source_role']=='SOURCE_DEV','training_admission':False})
 assert len({r['row_id'] for r in rows})==120 and sum(r['state']==STATES['O'] for r in rows)==50
 assert all(r['state'] in STATES.values() and r['training_admission'] is False for r in rows)
 obs=new('ALL120_WEAK_INTENDED_PART_OBSERVATIONS.jsonl',b''.join((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode() for r in rows))
 saved_bytes=rb(obs['path']);assert sha(saved_bytes)==obs['sha256'];saved=[json.loads(l) for l in saved_bytes.splitlines()]
 assert saved==rows and [r['row_id'] for r in saved]==[r['row_id'] for r in s0]
 for a,b in zip(saved,s0):
  assert (a['generated_query_xy'],a['L'],a['generated_query_available'],a['source_role'],a['native_slot'],a['canonical_index'])==(b['generated_query_xy'],b['L'],b['generated_query_available'],b['source_role'],b['native_slot'],b['canonical_index'])
  assert a['actualview_image_binding'] in plan['fixed15_view_bindings']
 flat=[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in rows];s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat);cs=new('ALL120_WEAK_INTENDED_PART_OBSERVATIONS.csv',s.getvalue().encode())
 vr=newj('ACTUAL_IMAGE_VIEW_RECEIPT.json',{'completed':True,'PASS':True,'reader_id':'A','views':views,'exact_contact_images_actually_tool_viewed':15,'all120_native_S0_crop_panels_within_contacts_viewed':True,'standalone_source_context_or_individual360RGB_files_viewed':0,'before_view_plan':bind(H+'/BEFORE_VIEWS_EXACT15_CONTACT_SHA_QA.json',planb),'historical_SOURCE_context_and_GT_diagnostics_exposure_not_blind':True,'authored_prior_weak_pair_builder_not_GT_query_sampler':True,'other_reader_or_root_new_observation_values_read':0})
 indep=newj('INDEPENDENCE_RECEIPT.json',{'completed':True,'PASS':True,'reader_id':'A','other_reader_new_observation_files_read':0,'other_reader_new_values_received':0,'root_new_observation_values_read':0,'shared_schema_only':True,'historical_SOURCE_context_exposure_disclosed':True,'own_observations_binding':obs,'no_source_GT_arrays_models_private_scores_reads':True})
 counts=dict(collections.Counter(r['state'] for r in rows));roles=dict(collections.Counter(r['source_role'] for r in rows))
 q=newj('SAVED_OBSERVATION_SCHEMA_QA.json',{'completed':True,'PASS':True,'all120_exact_S0_row_ids_and_identity':True,'all50_nominal_OOF_preserved_distinct_from_old_U':True,'D_or_E_has_concrete_ROI':True,'E_requires_positive_foreign_evidence_and_occluder':True,'all_actualview_bindings_in_own_receipt':True,'all_SOURCE_roles_preserved_DEV_excluded':True,'physical_gold_training_admission_or_jitter_label_transfer':False,'state_counts':counts,'source_role_counts':roles,'source_GT_model_private_reads':0})
 done=newj('COMPLETE.json',{'completed':True,'PASS':True,'reader_id':'A','code':bind(__file__,rb(__file__)),'scope':plan['scope'],'source_ledger':bind(ledgerpath,lb),'observations':obs,'CSV':cs,'view_receipt':vr,'schema_QA':q,'independence':indep,'rows':120,'contacts_actually_viewed':15,'state_counts':counts,'roles':roles,'native_gold_created':0,'labels_to_other_jitter_states_transferred':0,'training_admission':False,'frozen_no_further_writes':True})
 print(json.dumps({'completed':True,'PASS':True,'complete':done,'observations':obs,'CSV':cs,'view_receipt':vr,'schema_QA':q,'counts':counts,'roles':roles},ensure_ascii=False))
if __name__=='__main__':main()
