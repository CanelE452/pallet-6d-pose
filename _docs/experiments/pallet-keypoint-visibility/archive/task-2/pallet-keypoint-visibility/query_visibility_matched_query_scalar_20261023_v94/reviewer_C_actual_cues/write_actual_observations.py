import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'query_visibility_matched_query_scalar_20261023_v94'
S=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92/rgb96_review'
O=R/'reviewer_C_actual_cues'
def bind(p):
    x=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(x).hexdigest(),bytes=len(x))
def write(p,x):
    assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

# Explicit observations after actually viewing every original context figure and
# every exact-saved96 enlargement. They do not use another reviewer's annotation.
# State = selected-instance auxiliary support; appearance can separately be YES.
NOTES={
0:['No detector query; visible scene objects do not create a replacement query.']*8,
1:['Original query outside native frame; stored black placeholder is factual U.']*8,
2:[
'앞쪽 단일 loaded wood pallet의 deck–왼쪽 block 접합과 fork opening 모서리가 보인다.',
'같은 팔레트 오른쪽 top deck와 수직 block의 접합이 보인다. 오른쪽 노랑/검정 보호대는 별도 물체다.',
'동일 오른쪽 block의 하단/바닥 접합이 crop 상반부에 보인다.',
'동일 왼쪽 block의 하단 외곽이 crop 위쪽에 남는다. query 중심과 단서는 떨어져 있다.',
'같은 팔레트의 좌측 옆 deck plank가 앞쪽 end로 꺾이는 영역이 보이나 여러 plank seam도 함께 있다.',
'carton 표면과 crop 맨 아래의 직선 목재 띠만 있다. 구분되는 팔레트 junction/rim은 충분하지 않다.',
'carton 아래 deck 하단과 fork-opening의 어두운 입구를 가르는 rim 구간이 보인다. 특정 native corner라는 뜻은 아니다.',
'좌측 옆 plank가 앞쪽 end로 꺾이는 접합이 보인다. blur 때문에 좁은 단일 점은 확정하지 않는다.'],
3:[
'Original OOB query U.', '선반의 주황색 beam과 빨간 bin 밑면이다. 팔레트 구조 단서를 보았다고 하지 않는다.',
'선반 밑 그림자와 바닥이다. 별도 팔레트 cue 증거가 없다.', 'Original OOB query U.',
'Original OOB query U.', '빨간 보관 bin과 선반 beam이다. 팔레트가 아니다.',
'선반 beam/그림자이며 팔레트 rim이라는 근거가 없다.', 'Original OOB query U.'],
4:[
'carton 및 종이 라벨만 보인다. 구조 cue가 아니다.',
'bag 아래 dark pallet의 deck/block 같은 접합 후보는 있으나 어둠/blur와 인접 두 팔레트 때문에 경계/소유를 확정하지 못한다.',
'어두운 block–floor 모서리 후보가 있지만 세부 형상과 어느 팔레트 소속인지 불확실하다.',
'carton의 평탄한 표면이다. 팔레트 구조가 아니다.',
'포장 bag 표면과 seam이다. 팔레트 구조가 아니다.',
'인접 빈 pallet의 여러 deck plank와 외곽이 보인다. selected box가 loaded/empty 두 팔레트에 걸쳐 있어 target ownership U.',
'인접 빈 pallet의 deck/front edge 및 opening이 보이지만 selected instance가 어느 것인지 독립 대응이 없다.',
'bag과 묶음 사이 seam이다. 팔레트 cue라고 하지 않는다.'],
5:[
'IBC tank의 metal band/용기 면이다. 팔레트 구조 단서가 아니다.',
'IBC 하부 cage/base의 검은 발 같은 형상이 있으나 별도 pallet인지 integrated IBC base인지 구분되지 않는다.',
'steel rack beam과 그 아래 틈이다. 팔레트 구조가 아니다.',
'IBC 하부 base/foot 후보이나 blur와 integrated container 구조 때문에 pallet cue로 확정하지 못한다.',
'IBC tank 면과 metal band이다. 목재/팔레트 junction을 보았다고 하지 않는다.',
'rack upright와 IBC base가 섞이며 여러 container 중 target ownership이 불명확하다.',
'IBC base의 발/rail 후보이나 material/구조 identity가 불분명하다.',
'container 하부 검은 면과 silver band이다. pallet opening일 가능성을 확정할 수 없다.'],
6:[
'검은 crate의 글자와 면이다. 팔레트 단서가 아니다.',
'carton 아래 어두운 pallet-like base는 있으나 fork/foot 접합을 안정적으로 localize할 수 없다.',
'검은 base와 바닥 경계가 흐리며 단일 cue를 지정할 수 없다.',
'검은 crate와 바닥 경계이다. 팔레트 구조 확인이 아니다.',
'carton/base 경계가 보이지만 distinct pallet structure는 분해되지 않는다.',
'뒤쪽 stacked pallets/랙이 함께 보이며 작은 반복 구조와 target ownership이 불명확하다.',
'멀리 반복되는 pallet-like supports와 바닥이다. 어느 target/cue인지 불명확하다.',
'carton 아래 dark base는 보이나 흰 포장 조각/그림자와 실제 foot 경계가 구분되지 않는다.'],
7:[
'판층과 어두운 개구부 후보가 있지만 tiny/blur로 구조 identity가 불명확하다.',
'carton 면이 대부분이다. distinct pallet cue가 없다.',
'carton 측면과 바닥이다. 팔레트 구조가 없다.',
'상단 어두운 물체 조각과 바닥만 남는다. distinct pallet cue가 없다.',
'deck-like 얇은 층/검은 틈 후보이나 반사 금속/wood 판과 실제 pallet support를 분리하기 어렵다.',
'나무판처럼 보이는 층이 반복되지만 cue junction을 안정적으로 지정할 수 없다.',
'판층 끝/어두운 support 후보는 보이지만 작은 원본 해상도에서 기하 구조가 확정되지 않는다.',
'바닥과 상단 어두운 조각이다. distinct pallet cue를 확인하지 못한다.'],
8:[
'선택된 왼쪽 단일 wood pallet의 top deck–front block 접합이 보인다.',
'같은 pallet 우측 전면 block과 deck의 turn이 보인다.',
'같은 block 하단 외곽과 floor 접합이 보인다. 아래쪽 black padding은 support 아래에 있다.',
'동일 pallet 왼쪽 block의 바닥 접합/세로 외곽이 보인다.',
'회색 container 면과 아래 직선 deck 띠이다. distinct junction은 분명하지 않다.',
'목재 측면과 어두운 적재물 경계만 있어 구체 cue turn을 확인하지 못한다.',
'대각선 목재 면/바닥 경계이며 distinct opening/junction을 분리하지 못한다.',
'deck 아래 fork-opening의 상단 rim과 support 세로선이 만나는 영역이 보인다.'],
9:[
'blue/grey industrial bin의 integrated foot와 rack 보호대다. 별도 flat pallet라고 하지 않는다.',
'grey industrial bin의 면/성형선이다. pallet cue가 아니다.',
'grey bin의 integrated foot와 바닥이다. 별도 팔레트라는 근거가 없다.',
'노란 rack 보호대와 바닥이다.',
'blue bin의 모서리/하부 발이다. pallet cue가 아니다.',
'작은 흰 파편/라벨과 bin 형상이 흐려서 별도 pallet structure를 판정할 수 없다.',
'grey bin의 lower molded edge다. 팔레트 구조라는 근거가 없다.',
'바닥과 bin 발 일부이다. pallet cue가 아니다.'],
10:['No detector query; no replacement crop or label invented.']*8,
11:[
'Original OOB query U.',
'rack upright와 노란 보호대/beam이다. 팔레트 구조가 아니다.',
'rack 보호대 및 beam과 바닥이다. 팔레트 구조가 아니다.',
'Original OOB query U.',
'metal bin과 그 아래 sheet/cardboard가 주로 보인다. distinct pallet structure 확인은 불충분하다.',
'여러 rack bay의 나무 pallet opening이 보이나 selected box가 선반 여러 칸을 포괄해 ownership 불명확하다.',
'반복 pallet supports/beam이 함께 보이며 특정 selected pallet의 cue라는 독립 대응이 없다.',
'metal bin 앞 sheet 밑의 나무 edge 후보가 crop 아래에 있으나 cut/blur와 target association이 불명확하다.'],
12:['Original OOB query U; stored black crop is not semantic absence.']*8,
13:[
'white crates 아래 선택된 단일 wood pallet의 왼쪽 deck/block 접합이 보인다. blur uncertainty를 넓게 둔다.',
'동일 pallet 오른쪽 block/deck corner가 둥글게 흐려져 보인다. pixel point gold는 아니다.',
'동일 오른쪽 support의 하단/바닥 turn이 crop 위쪽에 보인다.',
'동일 왼쪽 support의 하단 외곽이 보인다. crop 중심 아래쪽은 바닥이다.',
'노란 보호대와 rack post가 보인다. pallet cue가 아니다.',
'crate 아래 pallet deck 하단–fork opening rim이 보인다. 특정 native endpoint는 아니다.',
'동일 pallet의 fork-opening rim/옆 support 경계가 보이며 blur로 넓은 지역만 특정한다.',
'노란 보호대와 바닥이다. pallet structure는 아니다.'],
14:[
'노란 보호대 면이다. pallet structure가 아니다.',
'멀리 있는 적재물/붉은 base 후보가 심하게 blurred되어 physical cue를 localize할 수 없다.',
'base와 바닥이 흐리며 pallet support identity를 판정할 수 없다.',
'노란 보호대와 바닥이다.',
'노란 보호대 면이다.',
'멀리 있는 적재물 하단이 원본 tiny pixels로 퍼져 구체 cue가 확인되지 않는다.',
'붉고 검은 base 후보가 blurred되어 pallet structure 여부가 불명확하다.',
'노란 보호대와 그 foot이다. pallet cue가 아니다.'],
15:[
'wood pallet end block과 upper plank의 접합은 보인다. selected box가 rack 양쪽 인접 pallets를 포함해 ownership U.',
'rack post/steel foot 및 shelf contents이다. 독립 pallet cue를 특정하지 못한다.',
'steel rack footing과 그 아래 floor이다. pallet 구조가 아니다.',
'노란 보호대 옆 wood block 하단이 보이지만 selected target이 어느 pallet인지 불명확하다.',
'wood pallet fork opening/blocks가 보이지만 여러 인접 pallet/적재물 ownership이 불명확하다.',
'green shrinkwrap 아래 wooden centre support 및 opening rim이 보인다. selected box와 physical instance 대응이 확정되지 않는다.',
'동일 loaded wood pallet의 centre block/rim이 보이지만 selected box가 복수 물체를 포함해 ownership U.',
'목재 blocks와 deck가 반복되어 보인다. 인접 다른 pallet일 수 있어 selected-instance support는 U.'],
16:[
'white cart/door face다. pallet cue가 아니다.',
'red wheeled cart의 metal bottom/캐스터 주변이다. pallet structure가 아니다.',
'cart의 붉은 바닥과 그림자이다. pallet 구조가 아니다.',
'white cart face다. pallet cue가 아니다.',
'red cart의 frame/foot이다. pallet structure가 아니다.',
'red cart metal bottom이다. pallet cue가 아니다.',
'cart bottom/바닥 그림자다. pallet structure가 아니다.',
'cart 주변 wheel/base 후보이나 pallet가 아니다.'],
17:[
'grey plastic pallet fork-opening upper rim과 support가 명확히 보인다. selected box는 복수 인접 pallet를 포괄하므로 owner U.',
'grey pallet deck/support junction이 보이지만 옆 pallet와 경계가 달라 동일 selected instance라고 확정 못한다.',
'grey pallet support 아래 turn과 바닥이 보인다. selected owner 불명확하다.',
'plastic pallet support face와 fork-opening rim이 보이지만 복수 pallet ownership U.',
'인접 grey pallets의 개구부/발이 함께 보여 여러 가능한 cue가 있다.',
'노란 보호대 옆 grey fork opening이 보인다. 특정 selected instance와 cue identity는 U.',
'여러 grey pallet supports가 반복된다. visible structure는 있으나 instance ownership U.',
'grey pallet의 명확한 rim/block junction이 보이지만 원래 selected box에 복수 pallet가 들어있다.'],
18:[
'grey plastic pallet의 upper-deck/fork-opening boundary가 보인다. selected box는 post/옆 red crate까지 포괄해 owner U.',
'red storage crate 면/라벨이다. pallet cue가 아니다.',
'red crate의 모서리/격자면이다. pallet cue가 아니다.',
'grey pallet block 하단 turn이 보이지만 selected target association이 불명확하다.',
'grey pallet의 deck/opening rim이 crop 아래에 보인다. 인접 crate/pallet과 ownership 불명확하다.',
'노란 보호대와 red crate 일부이다. pallet cue가 아니다.',
'노란 보호대와 red crate 면이다. pallet cue가 아니다.',
'grey plastic pallet support/opening이 보이나 selected instance를 독립 확정하지 못한다.'],
19:[
'carton/wooden crate 면과 black padding이다. distinct pallet structure가 없다.',
'shrinkwrapped load 아래 wood block/fork opening이 보인다. selected box가 여러 load/pallet를 가로질러 owner U.',
'동일 가까운 wood pallet의 block 하단/opening이 보이나 선택된 target identity는 불명확하다.',
'carton 밑 crop 경계에 wood edge 후보만 남으며 온전한 cue region을 확인하지 못한다.',
'load 아래 wood block/edge가 아래경계에서 잘려 local cue와 ownership 모두 제한된다.',
'여러 팔레트 foot가 원근 반복된다. 어느 physical instance가 selected target인지 U.',
'반복된 wood fork openings/foot가 있으나 selected instance 대응이 없다.',
'앞쪽 wood pallet의 support/rim이 보이지만 selected box에는 복수 적재물/팔레트가 들어있다.'],
}

# Explicit confident local auxiliary observations (all native semantics UNKNOWN).
YES={
(2,0):('DECK_BLOCK_JUNCTION',[31,33,64,65],4),(2,1):('DECK_BLOCK_JUNCTION',[22,35,58,70],5),
(2,2):('SUPPORT_FOOT_FLOOR_TURN',[32,40,60,65],5),(2,3):('SUPPORT_FOOT_FLOOR_TURN',[34,7,65,29],5),
(2,4):('SIDE_DECK_END_JUNCTION',[46,65,74,88],6),(2,6):('FORK_OPENING_RIM',[18,49,84,73],5),
(2,7):('SIDE_DECK_END_JUNCTION',[43,38,78,70],5),
(8,0):('DECK_BLOCK_JUNCTION',[36,28,70,61],5),(8,1):('DECK_BLOCK_JUNCTION',[37,30,71,67],5),
(8,2):('SUPPORT_FOOT_FLOOR_TURN',[43,32,74,64],5),(8,3):('SUPPORT_FOOT_FLOOR_TURN',[38,32,68,63],5),
(8,7):('FORK_OPENING_RIM_BLOCK_JUNCTION',[36,39,65,73],5),
(13,0):('DECK_BLOCK_JUNCTION',[34,32,71,70],8),(13,1):('DECK_BLOCK_JUNCTION',[34,35,68,75],8),
(13,2):('SUPPORT_FOOT_FLOOR_TURN',[34,22,68,56],8),(13,3):('SUPPORT_FOOT_FLOOR_TURN',[31,28,67,61],8),
(13,5):('FORK_OPENING_RIM',[24,42,79,67],8),(13,6):('FORK_OPENING_RIM_BLOCK_JUNCTION',[10,27,69,60],8),
}

# Visible pallet structure, but selected-instance ownership is unresolved.
OWNERSHIP_U={
(4,5):('REPEATED_DECK_PLANK_EDGE',[23,49,68,82],7),(4,6):('DECK_OPENING_FRONT_EDGE',[21,30,72,65],7),
(11,5):('REPEATED_FORK_OPENINGS',[40,50,67,72],8),(11,6):('REPEATED_PALLET_SUPPORTS',[19,12,42,40],8),
(15,0):('DECK_BLOCK_JUNCTION',[17,39,43,77],6),(15,3):('SUPPORT_FOOT_FLOOR_TURN',[20,18,42,45],6),
(15,4):('FORK_OPENING_BLOCK',[45,35,74,70],6),(15,5):('CENTRE_SUPPORT_OPENING_RIM',[43,66,68,91],6),
(15,6):('CENTRE_SUPPORT_OPENING_RIM',[40,43,67,70],6),(15,7):('REPEATED_BLOCK_DECK_SUPPORT',[42,23,73,53],7),
(17,0):('PLASTIC_FORK_OPENING_RIM',[26,58,65,87],4),(17,1):('PLASTIC_DECK_SUPPORT_JUNCTION',[40,43,77,77],4),
(17,2):('PLASTIC_SUPPORT_FLOOR_TURN',[40,20,76,49],4),(17,3):('PLASTIC_FORK_SUPPORT_JUNCTION',[29,35,62,74],4),
(17,4):('PLASTIC_FORK_SUPPORT_JUNCTION',[47,45,80,80],5),(17,5):('PLASTIC_FORK_OPENING_RIM',[53,35,90,70],5),
(17,6):('REPEATED_PLASTIC_SUPPORTS',[47,14,73,40],5),(17,7):('PLASTIC_RIM_BLOCK_JUNCTION',[45,40,75,69],4),
(18,0):('PLASTIC_DECK_OPENING_BOUNDARY',[25,27,68,70],6),(18,3):('PLASTIC_SUPPORT_FLOOR_TURN',[23,19,65,62],6),
(18,4):('PLASTIC_DECK_OPENING_RIM',[43,52,77,85],6),(18,7):('PLASTIC_SUPPORT_OPENING',[31,28,69,68],5),
(19,1):('WOOD_BLOCK_OPENING',[34,50,67,86],5),(19,2):('WOOD_BLOCK_OPENING_FLOOR',[20,14,55,48],5),
(19,5):('REPEATED_WOOD_FORK_SUPPORTS',[29,41,49,63],7),(19,6):('REPEATED_WOOD_FORK_SUPPORTS',[18,20,45,44],7),
(19,7):('WOOD_SUPPORT_RIM',[39,54,72,90],5),
}

# NO = no observed distinct pallet cue in this stored crop, not E/S/negative GT.
NO={
2:{5},3:{1,2,5,6},4:{0,3,4,7},5:{0,2,4},6:{0,3},7:{1,2,3,7},
8:{5,6},9:{0,1,2,3,4,6,7},11:{1,2},13:{4,7},14:{0,3,4,7},
15:{1,2},16:set(range(8)),18:{1,2,5,6},19:{0},
}

FRAME_NOTES={
0:'No detection. Several scene objects visible, but no input query/crop can be manufactured.',
1:'All selected queries outside native image; fullcontext alone cannot replace missing stored crops.',
2:'One front loaded wood pallet is context-compatible with the selected box. Cue support is regional and blurred, not native corner gold.',
3:'Selected box/context lands on storage rack/bin, not an identified pallet.',
4:'Selected rectangle spans loaded bag pallet and neighbouring empty pallet; physical structure exists but target ownership is not unique.',
5:'IBC bases/cages and rack beam; integrated IBC base vs independent pallet is unresolved.',
6:'Dark black crates/carton bases and tiny distant repeated pallets; native crop resolution limits cue localization.',
7:'Very small, blurred layered pallet-like base; distinct physical cue identity not stable enough.',
8:'One selected left wood pallet supports bins; deck/block/opening regions are visible. Material presence alone was not used.',
9:'Industrial bins and integrated molded feet, with rack post; no independently identified flat pallet.',
10:'No detection/query. All eight placeholders stay U.',
11:'Selected rectangle spans shelf bays and multiple pallet-like bases; rack/bins dominate several queries.',
12:'All original query centers are outside native frame, even though scene contains pallets.',
13:'Selected right wood pallet under white crates is context-compatible; visual uncertainty is broad due to blur.',
14:'Selected remote base is extremely blurred; rack/bollard and indistinct shape do not certify pallet cue.',
15:'Actual wooden pallet structures exist across adjacent loaded pallets and rack; selected box does not uniquely bind an instance.',
16:'Selected source object is wheeled red cart/door base, not a pallet.',
17:'Multiple adjacent grey plastic pallets have clear fork/foot structures; selected rectangle spans them, so ownership stays U.',
18:'Grey pallet next to yellow post/red crates has visible structure, but selected object association is ambiguous.',
19:'Several left-side loaded wood pallets lie inside the selected rectangle; visible fork supports do not resolve which instance is intended.',
}

def main():
    rows=[json.loads(x) for x in (S/'ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
    assert len(rows)==160 and set(NOTES)==set(range(20)) and all(len(x)==8 for x in NOTES.values())
    source_manifest=json.loads((S/'RGB_PROVENANCE_AND_FIGURES.json').read_text())
    raw_manifest=json.loads((O/'RAW_SHEET_RENDER_MANIFEST.json').read_text())
    receipts=[];out=[]
    for fi in range(20):
        receipts.append(dict(frame_index=fi,original_context_figure=source_manifest['frames'][fi]['figure'],raw96_sheet=raw_manifest['frames'][fi]['derived_raw_sheet'],actual_view_tool='tools.view_image',actual_fullcontext_viewed=True,actual_all8_saved96_panels_viewed=True,reviewer_C_frame_observation=FRAME_NOTES[fi]))
        for k in range(8):
            src=rows[fi*8+k]
            factual=bool(src['query_center_valid'])
            spec=YES.get((fi,k)) or OWNERSHIP_U.get((fi,k))
            state='UNKNOWN';appearance='UNKNOWN';ownership='UNRESOLVED_BLUR_OR_STRUCTURE'
            if not factual:
                ownership='NO_QUERY' if not src['query_exists'] else 'OOB_OR_INVALID_QUERY'
            elif (fi,k) in YES:
                state=appearance='YES';ownership='CONTEXT_COMPATIBLE_SINGLE_PALLET_AI'
            elif (fi,k) in OWNERSHIP_U:
                appearance='YES';ownership='MULTIPLE_OR_UNBOUND_SELECTED_INSTANCES'
            elif k in NO.get(fi,set()):
                state=appearance='NO';ownership='NO_PALLET_STRUCTURE_OBSERVED_IN_CROP'
            kind,bbox,unc=spec if spec else (None,None,None)
            if bbox:
                assert all(0<=z<=96 for z in bbox) and bbox[0]<bbox[2] and bbox[1]<bbox[3]
                centre=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]
                centrality=math.hypot(centre[0]-47.5,centre[1]-47.5)
                ub=[max(0,bbox[0]-unc),max(0,bbox[1]-unc),min(96,bbox[2]+unc),min(96,bbox[3]+unc)]
            else:centre=centrality=ub=None
            out.append(dict(input_id=src['input_id'],frame_index=fi,native_slot=k,original_q=src['query_xy'],original_L=src['L'],split=src['split'],image_sha256=src['image_sha256'],source_validity=src['input_status'],query_center_valid=factual,
                target_ownership_confidence=ownership,visible_aux_structuralcue=state,visible_pallet_structure_appearance=appearance,cue_kind=kind,crop_support_bbox_xyxy96_or_null=bbox,support_uncertainty_px=unc,support_uncertainty_bbox_xyxy96_or_null=ub,bbox_is_region_not_occupied_mask=True,
                multiplecue_flag=((fi,k) in OWNERSHIP_U or (fi,k) in {(2,4),(2,6),(8,7)}),cue_region_center_xy96=centre,cue_region_distance_from_query_center_px96=centrality,
                notes=NOTES[fi][k],native_expected_part_state='UNKNOWN',native8_semantic_mapping='UNKNOWN',native_human_gold=False,official_gold=False,external_E_generated=False,self_S_generated=False,negative_training_label=False,training_admitted=False,
                evidence_tier='AI_INDEPENDENT_ACTUAL_PUBLIC_SAVED_CROP_OBSERVATION_CONDITIONAL',original_q_L_and_slot_unchanged=True))
    assert sum(x['query_center_valid'] for x in out)==122
    write(O/'ACTUAL_ALL20_FIGURE_AND_SAVED96_VIEW_RECEIPTS.json',dict(completed=True,all20=True,all160=True,valid122=True,receipts=receipts,no_B_new_annotations_read=True))
    p=O/'ALL160_C_ACTUAL_QUERY_CUE_OBSERVATIONS.jsonl'
    assert not p.exists();p.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out))
    write(O/'RAW_OBSERVATION_SEAL_BEFORE_MARKED_GALLERY.json',dict(completed=True,policy=bind(O/'POLICY_BEFORE_ACTUAL_CROP_PIXELS.json'),observations=bind(p),view_receipts=bind(O/'ACTUAL_ALL20_FIGURE_AND_SAVED96_VIEW_RECEIPTS.json'),all_original_q_L=True,no_new_GT_or_fit=True))
    # Derived mark-only sheets, preserving raw-neighbour panel directly beside it.
    a=np.load(S/'INPUT_RGB96_ONLY.npz',allow_pickle=False)['rgb96'].reshape(20,8,3,96,96)
    fdir=O/'raw_marked_saved96_gallery';fdir.mkdir()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    gallery=[]
    for fi in range(20):
        canvas=Image.new('RGB',(4*432,2*286+42),'#f4f4f4');d=ImageDraw.Draw(canvas)
        d.text((10,8),f'FRAME {fi:02d}: raw left / support+uncertainty right | AI conditional | native expectedpart UNKNOWN',fill='black',font=font)
        for k in range(8):
            x=(k%4)*432;y=42+(k//4)*286;r=out[fi*8+k]
            raw=Image.fromarray(a[fi,k].transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST)
            canvas.paste(raw,(x+8,y+35));canvas.paste(raw,(x+224,y+35))
            d.text((x+8,y+2),f'k{k} aux={r["visible_aux_structuralcue"]} appearance={r["visible_pallet_structure_appearance"]}',fill='black',font=small)
            d.text((x+8,y+18),'selected owner '+r['target_ownership_confidence'][:35],fill='black',font=small)
            b=r['crop_support_bbox_xyxy96_or_null'];ub=r['support_uncertainty_bbox_xyxy96_or_null']
            if b:
                color='#00b050' if r['visible_aux_structuralcue']=='YES' else '#ff9500'
                d.rectangle([x+224+2*ub[0],y+35+2*ub[1],x+224+2*ub[2],y+35+2*ub[3]],outline='#f5d000',width=1)
                d.rectangle([x+224+2*b[0],y+35+2*b[1],x+224+2*b[2],y+35+2*b[3]],outline=color,width=2)
            d.text((x+8,y+235),(r['cue_kind'] or r['source_validity'])[:48],fill='#333333',font=small)
            d.text((x+8,y+253),'NO/UNKNOWN != external/self; training admitted false',fill='#555555',font=small)
        f=fdir/f'frame_{fi:02d}_raw_marked.png';canvas.save(f);gallery.append(dict(frame_index=fi,figure=bind(f)))
    valid=[x for x in out if x['query_center_valid']]
    summary=dict(schema='V94_C_ALL20_ACTUAL_QUERY_LOCAL_PHYSICAL_CUE_REVIEW_KO',completed=True,frames=20,all_original_slots=160,valid_crop_slots=122,no_query_U=16,OOB_invalid_U=22,
      valid_aux_observations=dict(Counter(x['visible_aux_structuralcue'] for x in valid)),all_aux_observations=dict(Counter(x['visible_aux_structuralcue'] for x in out)),valid_structure_appearance=dict(Counter(x['visible_pallet_structure_appearance'] for x in valid)),
      aux_YES_distinct_frames=sorted({x['frame_index'] for x in valid if x['visible_aux_structuralcue']=='YES'}),ownership_U_but_visible_pallet_structure=sum(x['visible_pallet_structure_appearance']=='YES' and x['visible_aux_structuralcue']=='UNKNOWN' for x in out),
      bbox_region_rows=sum(x['crop_support_bbox_xyxy96_or_null'] is not None for x in out),human_native_gold_added=0,native_expected_part_all_UNKNOWN=True,negative_or_E_S_labels_generated=0,training_admitted=False,
      scope_KO='모든20 context와160저장crop panel을 실제 view_image로 확인했다. 122valid를 빼거나 q/L을 바꾸지 않았다. 구조가 보여도 복수 selected-instance ownership U를 따로 보존했다. NO는 이 crop에서 구조를 관찰하지 못했다는 appearance 기록일 뿐 native E/S 또는 학습 negative가 아니다.',
      prior_coarse_comparison='V92 near-query coarse categories and this whole-crop structural/ownership review have different definitions. No accuracy/CM or gold agreement claim.',
      independence='Reviewer B new V94 records and model outputs not read. Prior V92 visual exposure disclosed; two AI observations are not human verification.',
      gallery=gallery,observations=bind(p),raw_seal=bind(O/'RAW_OBSERVATION_SEAL_BEFORE_MARKED_GALLERY.json'),new_model_forward_fit_current_PnP=0,goal_status='active',fully_solved=False)
    write(O/'ACTUAL_CUE_REVIEW_SUMMARY_KO.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['gallery','scope_KO','prior_coarse_comparison','independence']},ensure_ascii=False))

if __name__=='__main__':main()
