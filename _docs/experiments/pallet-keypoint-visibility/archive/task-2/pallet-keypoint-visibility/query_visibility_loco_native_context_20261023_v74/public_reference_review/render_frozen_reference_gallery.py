"""Render already frozen RGB references; never modify xy/ranges or read models."""
from pathlib import Path
import hashlib,json,textwrap
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def text(d,xy,s,size=16,fill='white'):
    d.text(xy,s,font=ImageFont.truetype(FONT,size),fill=fill,stroke_width=1,stroke_fill='black')
def thumb(im,size):
    z=Image.new('RGB',size,'#171c24');im=im.copy();im.thumbnail(size,Image.Resampling.LANCZOS);z.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return z
def draw_refs(im,refs):
    z=im.copy();d=ImageDraw.Draw(z)
    for j,r in enumerate(refs):
        color=['#ffff00','#ff66ff','#00ffff','#88ff88'][j];x,y=r['reference_xy'];d.rectangle(r['uncertainty_bbox_half_open_xyxy'],outline=color,width=2)
        d.line((x-10,y,x+10,y),fill=color,width=2);d.line((x,y-10,x,y+10),fill=color,width=2);text(d,(x+8,y+4),f'R{j}',max(14,im.width//80),color)
    return z
def main():
    lp=R/'RGB_REFERENCE_PROPOSAL_LEDGER.json';gp=R/'RGB_REFERENCE_GEOMETRY_LOCK.json'
    assert sha(lp)=='d6cca39fdd4424e14fc4b45b9118b3399b61d3d9da6a315f691c5f8ccf794aa0' and sha(gp)=='632bfb81bc3107d231045e9917707b730e0b3fdc7469f1564becbb78fc2b2006'
    ledger=json.loads(lp.read_text());frames=ledger['frames'];folder=R/'FROZEN_REFERENCE_GALLERY';folder.mkdir(exist_ok=False);cards=[];refcards=[]
    for f in frames:
        n=f['selection_index'];im=Image.open(f['image']).convert('RGB');refs=f['reference_proposals'];marked=draw_refs(im,refs)
        locator=marked.copy();d=ImageDraw.Draw(locator);x,y,w,h=f['representative_annotation']['bbox'];d.rectangle((x,y,x+w,y+h),outline='#00ffff',width=max(2,im.width//500));
        b=f['expanded_bbox_native_xyxy'];rawcrop=im.crop(b);markcrop=marked.crop(b)
        card=Image.new('RGB',(1170,760),'#101722');d=ImageDraw.Draw(card)
        text(d,(12,10),f"{n:02d} subset{f['source_subset']} image{f['image_id']} ann{f['representative_annotation']['id']} / sourcepalletinstances={f['official_pallet_instance_count']}",20)
        text(d,(12,47),f"Frozen representative only | {f['frame_reference_state']} | native8UNKNOWN | AIproposal, nohumangold",15)
        card.paste(thumb(locator,(550,330)),(10,90));card.paste(rawcrop.resize((280,280),Image.Resampling.NEAREST),(590,90));card.paste(markcrop.resize((280,280),Image.Resampling.NEAREST),(880,90))
        text(d,(590,375),'raw expandedbbox crop | same fixed reference marks',15)
        if refs:
            for j,r in enumerate(refs):
                xy=r['reference_xy'];bound=r['uncertainty_bbox_half_open_xyxy'];text(d,(15,445+j*100),f"R{j} xy={xy}; uncertainty={bound}",17)
                for v,line in enumerate(textwrap.wrap(r['reference_role_2D']+' | '+r['actual_RGB_evidence'],67)):text(d,(15,472+j*100+v*19),line,13)
                x,y=xy;patch=im.crop((x-48,y-48,x+48,y+48));markpatch=marked.crop((x-48,y-48,x+48,y+48))
                c=Image.new('RGB',(550,320),'#101722');dr=ImageDraw.Draw(c);text(dr,(10,10),f"frame{n:02d} R{j} ({x},{y}), raw96 | frozenrange/cross",15);c.paste(patch.resize((240,240),Image.Resampling.NEAREST),(10,50));c.paste(markpatch.resize((240,240),Image.Resampling.NEAREST),(290,50));refcards.append(c)
                card.paste(patch.resize((210,210),Image.Resampling.NEAREST),(620+j*270,425))
        else:
            for k,line in enumerate(textwrap.wrap('U: '+f['U_reason'],95)):text(d,(15,450+k*26),line,17)
            text(d,(15,640),'No point invented; no alternate instance chosen; U is not E/S/absence.',17,'#ffcc88')
        text(d,(15,712),'Cyan=unchanged officialbbox. Yellow/magenta=frozen RGB reference uncertainty; actual raw stays beside marks.',14)
        cp=folder/f'{n:02d}_FRAME_RAW_MARKED_REFERENCE.png';card.save(cp);cards.append(card)
    pages=[]
    for pg in range(5):
        z=Image.new('RGB',(2340,1520),'#101722')
        for j in range(4):z.paste(cards[pg*4+j],((j%2)*1170,(j//2)*760))
        p=R/f'FROZEN_REFERENCE_PAGE_{pg+1}.png';z.save(p);pages.append(bind(p))
    sheet=Image.new('RGB',(1100,1600),'#101722')
    for k,c in enumerate(refcards):sheet.paste(c,((k%2)*550,(k//2)*320))
    sh=R/'TEN_FROZEN_REFERENCE_NATIVE_RAW_MARKED.png';sheet.save(sh)
    assert sha(lp)=='d6cca39fdd4424e14fc4b45b9118b3399b61d3d9da6a315f691c5f8ccf794aa0' and sha(gp)=='632bfb81bc3107d231045e9917707b730e0b3fdc7469f1564becbb78fc2b2006'
    report=dict(schema='V74_ACTUAL_FIXED20_SOURCE_REFERENCE_FEASIBILITY_RESULT',completed=True,counts=ledger['counts'],
      frames=20,source_bbox_instances_across20frames=sum(f['official_pallet_instance_count'] for f in frames),representative_entity_perframe=1,
      actual_RGB_reference_frames=[0,1,8,16,17],actual_wood_appearance_reference_frames=[0,1,8],actual_grayplastic_appearance_reference_frames=[16,17],material_tags_are_AI_observations_not_official_categories=True,
      scope='Actual2Dnativegeometry/ranges supplied, not justgeneralreview. No modeloutput, predictedq, ownershipmask, score or currentdata read. Original20/largestbbox retained incl15U.',
      operational_visibility='10 conditionalvisiblephysicalbase/support/deckedge references on5frames, eachtiedtothefrozenofficialbbox. Native8/cuboidvertexcorrespondence remainsUNKNOWN. No naturalE/S created.',
      feasibility_limit='The fixedlargestbbox often selects an elevated/blurred/cut or stack/group region. 15Uframes do not imply theirotherpallets areabsent or occluded. Do not substitute a betterlookinginstance or count10referencesas10independentcaptures.',
      possible_next_source_data='Afterrootvisualreview/ownershipsanity andseparatecontract, hold thesevisible2Drefs fixedwhilevaryinginputq andaddingopaqueother-instancecover overtheentirecueuncertaintyrange. No physicalmeshDIRECT orPnP prerequisite; no permissiontoautoadmit here.',
      admission='HOLD_ROOT_REFERENCE_REVIEW_AND_CAPTURE_GROUP_CONTRACT',
      geometry_unchanged=True,ledger=bind(lp),geometry_lock=bind(gp),source_pages=pages,native_reference_sheet=bind(sh),
      newmodel_forward_fit_current_PnP=0)
    rp=R/'ACTUAL_REFERENCE_FEASIBILITY_RESULT.json';assert not rp.exists();rp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    seal=R/'REFERENCE_REVIEW_SEAL.json';assert not seal.exists();files=sorted(p for p in R.rglob('*') if p.is_file())
    seal.write_text(json.dumps(dict(schema='V74_PUBLIC_RGB_REFERENCE_REVIEW_SEAL_V1',completed=True,bindings=[bind(p) for p in files],native8_mapping='UNKNOWN',training_admission=report['admission'],model_outputs_or_masks_or_current_or_newforward_fit_PnP=0),indent=2)+'\n')
    print(json.dumps(dict(report=bind(rp),seal=bind(seal),native_sheet=bind(sh),counts=ledger['counts'])),flush=True)
if __name__=='__main__':main()
