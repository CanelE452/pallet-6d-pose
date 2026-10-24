"""Freeze only actual RGB-based observations, no mask/model data selection."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
OBS=[
 {'xy':[409,450],'bbox':[404.5,446.5,414.5,455.5],
  'part':'Outer upper deck/end-support corner of the left loaded flat wood pallet',
  'evidence':'Horizontal laminated wood deck terminates into a visible vertical wood end/support face. Cardboard is above/left, purple crates behind; neither supplies this cue.',
  'uncertainty':'Retain priorxy. The native deck edge is slightly soft; bbox includes top strip thickness and end-face turn, not just one exact material pixel.',
  'visibility':'The two wood edge directions and end face around the cue are exposed. No distinct foreground object crosses the local wood turning region in this image.',
  'confidence':'medium-high','duplicate':'Likely same left loaded pallet seen in STORAGE_HAWK0/1 and STORAGE_D4551; no calibrated crosssensor or temporal point matching established.'},
 {'xy':[945,615],'bbox':[938.5,607.5,952.5,620.5],
  'part':'Lower outer bevel/laminated edge turn of the wood support block under the right loaded carton',
  'evidence':'Two wood block faces converge above the grayfloor; laminated bottom wood edge visibly turns. It is below the AprilTag/carton and separate from the paper marker.',
  'uncertainty':'Retain priorxy. Wider than prior generic±4guess to include layered wood bevel and blur at the floor edge. Region is a visual corner cue, not a certified3D vertex.',
  'visibility':'The wood body sides and layered lower border are directly observable; cast shadow nearby is not selected as the cue.',
  'confidence':'medium','duplicate':'Same recording family, physical-pallet identity across other views/capture unresolved.'},
 {'xy':[945,470],'bbox':[939.5,459.5,955.5,478.5],
  'part':'Outer left wood topdeck/endboard turning region below purple crate stack',
  'evidence':'Pale wood boards and wood end-board form a low pallet structure below purple plastic ribs. Candidate is on the wood corner region, not on the purple crate corner.',
  'uncertainty':'Retain priorxy, but prior±4guess is too precise for blur/rounded board layers. The larger vertical span includes the top-rim turn visible above oldxy; exact single vertex is unverified.',
  'visibility':'The outer wood end and several topboard ends are exposed at the left side of the loaded stack. The region is visually supported though soft.',
  'confidence':'medium-low','duplicate':'Possibly same purple-crate wood pallet seen in later Storage and Galileo scenes. Do not treat as a new independent capturefamily.'},
 {'xy':[375,960],'bbox':[368.5,953.5,383.5,966.5],
  'part':'Near lower outer corner of left wood pallet support block, HAWKview',
  'evidence':'Large visible wood block has two faces and a layered lower edge ending above the floor. Green steel rack base is left of it and black-yellow floor tape right/below.',
  'uncertainty':'Wood bevel and motion/optical softness spread the lower turning region across15x13native cells. Representative integerxy is not an exact recovered3D vertex.',
  'visibility':'Wood block end remains exposed beneath the carton. Rack/painted floor are adjacent but do not cover the chosen wood corner region.',
  'confidence':'medium','duplicate':'Likely same physical left pallet as legacyP0 but a different local support cue and sensorview. All sensor/timestamp siblings remain one coarsefamily.'},
 {'xy':[171,298],'bbox':[166.5,293.5,176.5,302.5],
  'part':'Front-right lower wood support/deck edge turn under the left carton stack, Galileo2',
  'evidence':'Low wood pallet with a visible fork opening supports the cartons. The lower wood front border turns into the right side border near the selected block; blue metalcart is left of this region.',
  'uncertainty':'Native640RGB is small/soft.10x9cell bbox includes wood lower-edge blur and floor-side turn; exactpixel may straddle the contour.',
  'visibility':'The front-right wood support and wood lower edge are exposed, rather than hidden behind the blue carrier. Cardboard load is clearly above.',
  'confidence':'medium','duplicate':'Galileo0/1 are near-duplicate sameview. Site/object relation to2023Storage remains unresolved; groupallcoarsefamily.'},
 {'xy':[298,478],'bbox':[292.5,472.5,304.5,484.5],
  'part':'Far/right topdeck-end corner of left loaded flat wood pallet, D455slot1',
  'evidence':'Light horizontal wooddeck continues out from under cardboard then terminates into a visible vertical wood support face. Purplecrate behind and large transportcrate right are separate.',
  'uncertainty':'Motion softness and layered topboard round the apparent corner;12x12cell range spans the visible deck/endface junction.',
  'visibility':'Deck end is outside the cardboard silhouette and unobstructed in the image. The cue is above the support wood, not on the carton.',
  'confidence':'medium','duplicate':'Strong likely reobservation of legacyP0 under camera motion. It is a per-frame2Dreference, not new independent groundtruth.'},
 {'xy':[514,852],'bbox':[507.5,845.5,520.5,858.5],
  'part':'Far/right lower wood end-support corner of left loaded pallet, HAWKslot1',
  'evidence':'Dark wood end face and lighter side face converge at the layered bottom edge above grayfloor. The shape is the wood pallet support, below cardboard and apart from purple plastic.',
  'uncertainty':'Slight blur and layered plywood bevel require13x13native cells around the lower edge turn. Shadow and floor are included as uncertainty context, not labeled wood pixels.',
  'visibility':'Both wood faces and their bottom edge are visible without an external object covering the local support end.',
  'confidence':'medium','duplicate':'Same left loaded pallet/time neighborhood as the other Storageleft cues, with unresolved exact crosssensor/time landmark correspondence.'},
 {'xy':None,'bbox':None,
  'part':'Attempted upper wooddeck/carton contact corner in Galileo2slot1',
  'evidence':'The small topdeck strip and lower cardboard border overlap in this soft nativeRGB. A unique pallet-only upper corner cannot be fixed independently of the carton seam.',
  'uncertainty':'U/null: no arbitrary nearest wood/slat point invented. The clearer lower support is likely a near-duplicate of NEW1; no extra distinctreference is manufactured.',
  'visibility':'Wood pallet is present in this frame, but presence does not certify this attempted named upper cue.',
  'confidence':'low','duplicate':'Near-duplicate Galileo0frame; no new independent evidence count.'},
]
def main():
    dest=OUT/'RGB_CUE_OBSERVATIONS_LOCK.json';assert not dest.exists()
    regions=json.loads((OUT/'CANDIDATE_FRAME_VISUAL_ORDER.json').read_text())
    rows=[]
    for row,ob in zip(regions['records'],OBS):
        if ob['xy'] is not None:
            x,y=ob['xy'];b=ob['bbox'];assert b[0]<=x<b[2] and b[1]<=y<b[3]
        rows.append({'record_id':row['record_id'],'cue_refID':'AI_R2B_V71:'+row['frame']['stream']+':'+str(row['frame']['sensor_slot']).zfill(2)+':'+row['record_id'],
            'source_frame':row['frame'],'legacy_candidate':row['legacy_point'],'kind':row['kind'],
            'associated_cue_xy':ob['xy'],'uncertainty_bbox_half_open_xyxy':ob['bbox'],
            'proposal_state':'AI_VISIBLE_CUE_PROPOSAL' if ob['xy'] is not None else 'U_UNRESOLVED',
            'named_wood_cue':ob['part'],'wood_target_association':row['legacy_target'],
            'actual_RGB_evidence':ob['evidence'],'uncovered_cue_evidence':ob['visibility'],'visual_uncertainty_reason':ob['uncertainty'],
            'AI_confidence_only':ob['confidence'],'near_duplicate_or_identity_caveat':ob['duplicate'],
            'native8_mapping':'UNKNOWN','canonical_semantic_keypoint_id':None,'human_verified':False,'official_annotation':False,
            'evidence_tier':'AI_VISUAL_NONHUMAN_CONDITIONAL_TEACHER_PROPOSAL','semantic_gold':False,'GT_admitted':False,'training_admitted':False,
            'E_or_S_generated':False,'crossframe_reference_correspondence_certified':False,'same_coarse_family_group':'NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED'})
    result={'schema':'V71_R2B_VISIBLE_WOOD_CUE_OBSERVATIONS_LOCK_V1','status':'FROZEN_FROM_ACTUAL_RGB_ONLY_BEFORE_ANY_DOWNSTREAM_LABEL_MODEL_OR_COMPOSITION',
        'source_frames_existing_only':True,'counts':{'total_records':8,'visible_cue_AI_proposals':7,'U_null':1,'legacy_rechecked':3,'new_resolved':4,'new_attempted_U':1,'human_gold':0,'semantic_native8_known':0,'E_S_generated':0,'training_admitted':0},
        'reference_definition':'A named2D visible wood-pallet corner/edge turning REGION, with integer representativexy and explicit half-open continuous uncertainty box. Not necessarily an exact occupied materialpixel or cuboidvertex. Each cue remains fixed for future separately authorized jitter/cover experiments.',
        'legacy_policy':'Three priorxy retained exactly; visual uncertainty bbox is an additive review and may widen prior generic4pxguess from actual blur/bevel evidence. OriginalV67records/seals unchanged.',
        'selection_limit':'First5 non-prior frames in frozen round-robin order were inspected. Fourclear additional woodcues proposed; the fifth attempted uppercue remainsU/null. Later favorable images are not substituted after hitting8records.',
        'grouping':'All48 images belong to one unresolvedcoarsefamily; adjacent frames/sensors and plausible repeatedphysicalpallets are NOT independent TRAIN/VAL families. No split constructed.',
        'provenance_warning':'AI observation is not a human annotation or official semanticGT. No SAM/visiblemask/score-based coordinate adjustment. No naturalE correspondence or S truth generated.',
        'bindings':[{'path':str(p),'sha256':sha(p)} for p in [OUT/'RGB_REVIEW_PROTOCOL.json',OUT/'CANDIDATE_FRAME_VISUAL_ORDER.json',OUT/'FIXED48_RGB_OVERVIEW_OUTPUTS.json',Path(__file__)]],
        'records':rows}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'path':str(dest),'sha256':sha(dest),'counts':result['counts']}))
if __name__=='__main__':main()
