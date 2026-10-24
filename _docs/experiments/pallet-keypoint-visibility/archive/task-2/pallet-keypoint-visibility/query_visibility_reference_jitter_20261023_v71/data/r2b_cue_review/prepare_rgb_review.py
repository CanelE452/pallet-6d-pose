"""Existing48 public RGB only, no new intake/masks/models/GT admission."""
from pathlib import Path
import argparse,hashlib,json
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;V71=OUT.parents[1];BASE=V71.parent
PUB=BASE/'public_real_sources_20261023'
PRIOR=BASE/'query_visibility_visible_preservation_20261023_v67/source_audit/R2B_REFERENCE_ANCHOR_CANDIDATES'
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=lambda n:ImageFont.truetype(F,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def prepare():
    dest=OUT/'RGB_REVIEW_PROTOCOL.json';assert not dest.exists()
    paths=[PUB/d/'RGB_FIXED_MANIFEST.jsonl' for d in ['r2b_storage_v3','r2b_galileo2_v1']]
    storage=[json.loads(s) for s in paths[0].read_text().splitlines()];gal=[json.loads(s) for s in paths[1].read_text().splitlines()]
    streams=[('STORAGE_D455',sorted([r for r in storage if r['sensor']=='D455_COLOR'],key=lambda r:r['sensor_slot'])),('STORAGE_HAWK_LEFT',sorted([r for r in storage if r['sensor']=='HAWK_LEFT'],key=lambda r:r['sensor_slot'])),('GALILEO2_D455',sorted(gal,key=lambda r:r['sensor_slot']))]
    assert all(len(rows)==16 for name,rows in streams)
    order=[]
    for slot in range(16):
        for name,rows in streams:order.append({'review_order':len(order),'stream':name,**rows[slot]})
    prior=json.loads((PRIOR/'CANDIDATE_POINTS.json').read_text())
    result={'schema':'V71_PUBLIC_R2B_WOOD_CUE_RGB_REVIEW_PROTOCOL_V1','status':'FROZEN_BEFORE_NEW_COORDINATE_LOCALIZATION',
       'purpose':'Expand actual public flat wood-pallet2D visual corner/edge cue proposals, distinct from boxes, purple plastic crates, wood transport crates, current reference slots or mask occupancy.',
       'frame_review_order':'Existing fixed sensor_slot0..15, round-robin STORAGE_D455, STORAGE_HAWK_LEFT, GALILEO2_D455. All48 reviewed as fixed RGB overview pages. Existing3 prior P0/P8/P15 checked first; new cues use this fixed order, at most1 new cue per non-prior frame, max5 additional/max8 records total. If no clearly associated wood-pallet cue, retain U/null rather than relocate onto a load or shadow.',
       'new_source_RGB_references_only':'New xy/visual uncertainty ranges frozen after viewing actual nativeRGB and before any downstream use. No SAM/mask/model-based relocation or radius selection.',
       'candidate_order_guard':'Existing3 sourceframes reserved for prior recheck, no extra newcue on them. First clear unambiguous wood cue per subsequent fixed-order frame may be proposed. Ambiguous attempted frames can be retained as U without inventing coordinates; no E generated without exact same-reference cross-view correspondence.',
       'prior_uncertainty_recheck':'Preserve priorxy and provenance; reassess whether its visible landmark and prior±4px support are plausible. New visual-region bbox is an additive AI proposal, never an edit to prior3 evidence. U keeps new associatedxy/bboxnull.',
       'global_family':'NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED; same/related scene risk. No TRAIN/VAL split or claim of recording independence. Sensors/nearby timestamps are near-duplicate evidence, not independent ground-truth samples.',
       'native8_mapping':'UNKNOWN for every cue. This is a new explicitly named2D operational reference, not canonical8semantic GT.',
       'provenance_tier':'AI_VISUAL_NONHUMAN_CONDITIONAL_TEACHER_PROPOSAL_ONLY; HUMAN_GOLD0; GT_ADMISSION0; TRAINING0',
       'source_E_S_states':'No E or S labels admitted/generated; apparent visibility alone does not infer hidden reference correspondence.',
       'read_scope':{'allowed':'Existing48 publicRGB, prior3 proposal metadata and V69 cue sanity lessons only','new_source_intake':False,'source_or_current_model_inference_fit':False,'current223_or_private_data':False,'SAM_GTmask_depth_features_weights_or_PnP':False},
       'bindings':[bind(p) for p in paths+[PRIOR/'CANDIDATE_POINTS.json',PRIOR/'SEAL.json',PUB/'r2b_storage_v3/INTAKE_SEAL.json',PUB/'r2b_galileo2_v1/FAMILY_REVIEW_SEAL.json',BASE/'query_visibility_reference_semantic_20261023_v69/source_cue_reference_proposals/FROZEN_CUE_OWNERSHIP_SANITY.json',Path(__file__)]],
       'prior3_records':prior['records'],'fixed48_review_order':order}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'path':str(dest),'sha256':sha(dest),'frames':len(order)}))
def render():
    pp=OUT/'RGB_REVIEW_PROTOCOL.json';p=json.loads(pp.read_text())
    for b in p['bindings']:assert sha(b['path'])==b['sha256']
    dest=OUT/'FIXED48_OVERVIEW_PAGES';dest.mkdir(exist_ok=True);outputs=[]
    for page in range(4):
        panel=Image.new('RGB',(1500,1190),(22,24,27));d=ImageDraw.Draw(panel)
        d.text((12,8),f'Public fixed48 RGB / page{page+1}/4 / one unresolved coarsefamily / AI review only',font=font(22),fill='white')
        for local,row in enumerate(p['fixed48_review_order'][page*12:(page+1)*12]):
            assert sha(row['image'])==row['image_sha256'];im=Image.open(row['image']).convert('RGB');assert im.size==(row['width'],row['height'])
            im.thumbnail((486,250),Image.Resampling.LANCZOS);x=(local%3)*500+7;y=(local//3)*285+55
            d.text((x,y),f"{row['review_order']:02d} {row['stream']} slot{row['sensor_slot']:02d} idx{row['index_in_sorted_topic']}",font=font(16),fill='white');panel.paste(im,(x,y+25))
        path=dest/f'FIXED48_RGB_PAGE_{page+1}.png';panel.save(path);outputs.append(bind(path))
    (OUT/'FIXED48_RGB_OVERVIEW_OUTPUTS.json').write_text(json.dumps({'protocol':bind(pp),'pages':outputs,'source_RGB_unique_decodes':48,'all_image_SHA_shape_parity':True,'mask_model_current_reads':0},indent=2)+'\n')
    print(json.dumps({'pages':[r['path'] for r in outputs]}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--render',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    elif a.render:render()
    else:ap.error('choose mode')
