"""Review-only AI-proposed source landmarks, never accepted labels or model inputs."""
from pathlib import Path
import hashlib,json,textwrap
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))

CANDIDATES=[
 dict(slot=0,detail_box=[285,415,440,535],target='Low wooden pallet under the large foreground LEFT cardboard carton. The large wooden transport crate on the RIGHT is a different object.',
  points=[
   dict(id='P0',xy=[409.,450.],proposal='D_PROPOSED',part='Rightmost visible outer top-deck corner of the left wooden pallet',coordinate_uncertainty_px=4.,
        evidence='Visible horizontal wood deck strip ends at the right outer wood face; a vertical wood block/edge descends below it. This is below the cardboard base, not a carton corner.',
        caveat='Pallet is cropped by the left image edge. Complete pallet cuboid orientation and native8 semantic slot are not established.'),
   dict(id='U0',xy=[372.,456.],proposal='U',part='Carton/wood-deck contact seam',coordinate_uncertainty_px=None,
        evidence='Cardboard bottom edge and wood top strip touch/overlap. A unique full-pallet corner is not visually established.',caveat='Do not certify D/E/S from this seam.'),
   dict(id='R0',xy=[1066.,517.],proposal='REJECTED_WRONG_PART',part='Bottom corner/foot of the separate large wooden transport crate on the right',coordinate_uncertainty_px=None,
        evidence='Connected to the large upright crate body; no separate flat-pallet identity established.',caveat='Wood material alone does not make this a flat-pallet reference corner.')]),
 dict(slot=8,detail_box=[850,470,1030,640],target='Low wooden pallet carrying the large foreground RIGHT cardboard box/AprilTag.',
  points=[
   dict(id='P8',xy=[945.,615.],proposal='D_PROPOSED',part='Near lower outer corner of the visible wooden pallet support block',coordinate_uncertainty_px=4.,
        evidence='Two visible wood block faces converge near the floor; the laminated bottom edge turns at this point. It lies below the box/marker and is connected to the low pallet structure.',
        caveat='This is a local wood-base landmark candidate. The pallet extends beyond the right image edge; full cuboid slot/orientation remains unmapped.'),
   dict(id='U8',xy=[912.,506.],proposal='U',part='Marker/cardboard and wood top-deck junction',coordinate_uncertainty_px=None,
        evidence='Paper/box lower edge overlaps the upper deck region. The apparent corner may belong to the attached marker or box rather than the pallet.',caveat='Do not use as a certified pallet semantic reference.'),
   dict(id='R8',xy=[779.,443.],proposal='REJECTED_WRONG_PART',part='Lower cardboard-box corner above the rear wood support',coordinate_uncertainty_px=None,
        evidence='The visible vertical edge belongs to the loaded cardboard box, above the pallet wood.',caveat='A load corner cannot substitute for a pallet-base corner.')]),
 dict(slot=15,detail_box=[905,435,1095,560],target='Low wooden flat pallet beneath the purple plastic crate stack on the RIGHT.',
  points=[
   dict(id='P15',xy=[945.,470.],proposal='D_PROPOSED',part='Leftmost visible outer top-deck/end-board corner of the wooden pallet',coordinate_uncertainty_px=4.,
        evidence='Light wood top boards terminate against the vertical wood end board; the low deck is distinct from the purple crate stack above.',
        caveat='Boundary is soft and partly rounded at native resolution; exact pixel is uncertain. Rear/right pallet extent is clipped, so current-style8-slot mapping is not established.'),
   dict(id='U15',xy=[946.,530.],proposal='U',part='Lower left visible wood end-board boundary',coordinate_uncertainty_px=None,
        evidence='Wood edge/floor shading and board thickness overlap; a unique lower structural corner cannot be fixed confidently.',caveat='Possible alternative local landmark only; no D/E/S certainty.'),
   dict(id='R15',xy=[1057.,460.],proposal='REJECTED_WRONG_PART',part='Lower purple plastic crate edge where it meets the wood deck',coordinate_uncertainty_px=None,
        evidence='Purple material and crate ribs continue upward; this is the loaded crate, not the wood pallet.',caveat='Do not transfer crate-corner visibility to the pallet reference.')])]

def mark(im,points,box=None):
    draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,18)
    if box is None:box=[0,0,im.width,im.height]
    colors={'D_PROPOSED':'#52ee88','U':'#ffd34e','REJECTED_WRONG_PART':'#ff7373'}
    for p in points:
        x,y=p['xy'];x-=box[0];y-=box[1]
        if not(0<=x<im.width and 0<=y<im.height):continue
        color=colors[p['proposal']];draw.ellipse([x-6,y-6,x+6,y+6],outline=color,width=2);draw.line([x-11,y,x+11,y],fill=color,width=1);draw.line([x,y-11,x,y+11],fill=color,width=1)
        label=f"{p['id']} ({p['xy'][0]:.0f},{p['xy'][1]:.0f})"
        xx=max(2,min(x+10,im.width-180));yy=max(2,min(y+8,im.height-25))
        draw.text([xx,yy],label,font=font,fill=color,stroke_width=2,stroke_fill='black')
    return im

def main():
    assert not(ROOT/'CANDIDATE_POINTS.json').exists()
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());rows=[]
    for candidate in CANDIDATES:
        source=next(r for r in protocol['selected_images']if r['sensor_slot']==candidate['slot']);assert sha(source['image'])==source['image_sha256']
        image=Image.open(source['image']).convert('RGB');assert image.size==(1280,720)
        for p in candidate['points']:
            p.update(certified_semantic_state=None,annotation_provenance='AI_VISUAL_PROPOSAL_CONDITIONAL_TEACHER',human_verified=False,
              official_semantic_annotation=False,native8_semantic_id=None,GT_admission='HOLD')
        annotated=mark(image.copy(),candidate['points']);afile=ROOT/f"ANNOTATED_CONTEXT_slot{candidate['slot']:02d}.png";annotated.save(afile)
        box=candidate['detail_box'];raw=image.crop(box);rawfile=ROOT/f"NATIVE_DETAIL_RGB_slot{candidate['slot']:02d}.png";raw.save(rawfile)
        detail=mark(raw.copy(),candidate['points'],box);detailfile=ROOT/f"ANNOTATED_NATIVE_DETAIL_slot{candidate['slot']:02d}.png";detail.save(detailfile)
        rows.append({**candidate,'source':source,'annotated_context':ref(afile),'native_detail_RGB':ref(rawfile),'annotated_native_detail':ref(detailfile),
           'coordinate_rule':'Original1280x720pixel centers; crop-coordinate=original_xy-(left,top), no estimated coordinate repair.',
           'semantic_scope':'Local visually proposed wood-pallet-base landmark only. No asserted correspondence to current evaluation reference slots.'})
    result=dict(schema='R2B_FIXED_THREE_AI_REFERENCE_ANCHOR_CANDIDATES_V1',protocol=ref(ROOT/'PROTOCOL.json'),code=ref(Path(__file__)),
      goal='Reviewable public source annotation proposals to distinguish semantic reference visibility from initial-query occupancy.',
      human_confirmations=0,official_semantic_point_annotations=0,accepted_GT_points=0,training_admission='HOLD',
      all_candidate_states_uncertified=True,source_model_scores_current_RGB_GT_coordinates_or_weights_used=False,
      note='D_PROPOSED means an AI reviewer judges the local wood landmark apparently observable. It is not DIRECT_VISIBLE ground truth. U/rejected candidates are retained to expose ambiguity and wrong-part confusions.',
      required_review=['Confirm flat-pallet target instance versus box/crate/load.',
       'Confirm point is a stable named wood-base landmark and coordinate uncertainty.',
       'Specify whether/how it maps to an operational full-pallet semantic keypoint identity; do not force an8slot mapping.',
       'If accepted in a future conditional-teacher protocol, preserve exact provenance. Only an actual human/official annotation can upgrade provenance to humanverified/official.',
       'No natural before/after correspondence across the3frames is claimed; same-pixel subtraction is invalid under moving camera.'],
      records=rows,new_inference_training_GTcache_or_PnP=0)
    (ROOT/'CANDIDATE_POINTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    make_sheet(rows)
    seal=dict(schema='REVIEW_ONLY_CANDIDATE_ARTIFACT_SEAL_V1',candidate_manifest=ref(ROOT/'CANDIDATE_POINTS.json'),contact_sheet=ref(ROOT/'REFERENCE_ANCHOR_REVIEW.png'),
      protocol=ref(ROOT/'PROTOCOL.json'),accepted_GT_points=0,admission='HOLD',new_models_training_current_data_PnP=0)
    (ROOT/'SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
    print('AI_ANCHOR_REVIEW_READY',sha(ROOT/'SEAL.json'),sha(ROOT/'CANDIDATE_POINTS.json'))

def make_sheet(rows):
    W,H=1500,510;canvas=Image.new('RGB',(W,120+3*H),'#18232c');d=ImageDraw.Draw(canvas)
    title=ImageFont.truetype(FONT,24);font=ImageFont.truetype(FONT,16);small=ImageFont.truetype(FONT,14)
    d.text((15,10),'R2B public RGB: AI-proposed reference anchors for HUMAN REVIEW (not GT)',font=title,fill='white')
    d.text((15,46),'Green D_PROPOSED = apparently visible local wood landmark | Yellow U = ambiguous | Red R = wrong object/part',font=font,fill='#d7e4ed')
    d.text((15,74),'Fixed D455 slots0/8/15 before review | No model scores/current data | Human confirmed0, acceptedGT0, admissionHOLD',font=font,fill='#ffd34e')
    for j,r in enumerate(rows):
        y=120+j*H;source=r['source']
        d.text((12,y),f"D455 slot{r['slot']} / topic index{source['index_in_sorted_topic']} / bag msg{source['bag_message_id']}",font=title,fill='white')
        full=Image.open(r['annotated_context']['path']);full.thumbnail((850,445),Image.Resampling.LANCZOS);canvas.paste(full,(0,y+40))
        detail=Image.open(r['annotated_native_detail']['path']);detail.thumbnail((590,315),Image.Resampling.NEAREST)
        # Upscale only the display, never native coordinates or PNG evidence.
        scale=min(590/detail.width,315/detail.height);detail=detail.resize((int(detail.width*scale),int(detail.height*scale)),Image.Resampling.NEAREST)
        canvas.paste(detail,(875,y+40))
        p=next(p for p in r['points']if p['proposal']=='D_PROPOSED')
        text=f"{p['id']} original xy={tuple(p['xy'])}, uncertainty +/-{p['coordinate_uncertainty_px']:.0f}px. {p['part']}. Native8 slot UNKNOWN; conditional AI proposal only."
        for k,line in enumerate(textwrap.wrap(text,width=67)):d.text((875,y+365+20*k),line,font=small,fill='#d7e4ed')
    canvas.save(ROOT/'REFERENCE_ANCHOR_REVIEW.png')

if __name__=='__main__':main()
