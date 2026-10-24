"""Freeze largest officialpallet instance then render source-only RGB review."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;BASE=R.parents[1]
OLD=BASE/'query_visibility_public_pallet_loco_20261023_v73/data/intake'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(name,obj):
    p=R/name;assert not p.exists(),p;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def text(d,xy,s,size=17):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill='white',stroke_width=1,stroke_fill='black')
def thumb(im,hw):
    z=Image.new('RGB',hw,'#151923');im=im.copy();im.thumbnail(hw,Image.Resampling.LANCZOS);z.paste(im,((hw[0]-im.width)//2,(hw[1]-im.height)//2));return z
def main():
    source=[json.loads(x) for x in (OLD/'FIXED_20_RGB_MANIFEST.jsonl').read_text().splitlines()]
    coco=json.loads((OLD/'annotations/loco-all-v1.json').read_text());anns={r['id']:r for r in coco['annotations']}
    selected=[]
    for r in source:
        candidates=[anns[i] for i in r['pallet_annotation_ids']]
        a=min(candidates,key=lambda x:(-x['bbox'][2]*x['bbox'][3],x['id']))
        x,y,w,h=a['bbox'];side=1.25*max(w,h);cx=x+w/2;cy=y+h/2
        bounds=[math.floor(cx-side/2),math.floor(cy-side/2),math.ceil(cx+side/2),math.ceil(cy+side/2)]
        selected.append(dict(selection_index=r['selection_index'],review_id=f"LOCO_V74:{r['selection_index']:02d}:ann{a['id']}",source_subset=r['source_subset'],image_id=r['image_id'],image=r['image'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],official_path=r['archive_member'],official_pallet_instance_count=len(candidates),representative_annotation=a,representative_area_xywh=w*h,expanded_bbox_native_xyxy=bounds,display_crop_side=224))
    assert len(selected)==20
    protocol=dict(schema='V74_PUBLIC_LOCO_PHYSICAL_REFERENCE_REVIEW_PROTOCOL_V1',frozen_UTC=datetime.now(timezone.utc).isoformat(),source_selection=bind(OLD/'FIXED_20_RGB_SELECTION_LOCK.json'),source_manifest=bind(OLD/'FIXED_20_RGB_MANIFEST.jsonl'),official_COCO=bind(OLD/'annotations/loco-all-v1.json'),
      entity_selection='Per eachof20unchangedframes, largest width*height officialcategory7bbox; exactarea ties lowestannotationID. Freeze before reading anymodeloutputs. Everyframe retained inclblur/tiny/stack/group.',
      crop='Square centered on representativebbox, side1.25*max(width,height), integerfloorleft/top andceilright/bottom; raw224display andfullnativecrop. Out-of-image context is black reviewpadding, never a semanticcue.',
      reference_rule='At most4perrepresentativeentity. First seek exposednear/front pallet base/edge physicalturn endpoints: lower-image-left, lower-image-right, thenupper-image-left/right endpoints if a visiblepalletsupport/deckface edge can be associated withthe same singlepallet. Image-left/right are2Droles, not physical/canonicalnativeIDs. Do not copybboxcorners merely because they are rectangles. An interioropening/supportedge may be proposed onlyif its identity/associationis explicit; otherwiseU.',
      certainty='FixedRGBvisualcuexy plus independentlyvisualized localizationuncertaintybbox; positiveisAIconditionalVISIBLE_REFERENCEonly, not human/officialcornergold. Stack/group ambiguity, blur, cropcut, or unlocalizableendpoint =>U/null, notE. No hiddenpoint completion or recentering byGTmask/model.',
      geometry_freeze_order='Freeze reviewledger xy/ranges before anyownershipmask/sanity or readmodeloutputs. Originalrepresentativebbox untouched.',
      forbidden='No modeloutput, score, predictedq, current223RGB/GT, SAM/ownershipmask, training orPnP usedforselection/geometry.',
      native8_mapping='UNKNOWN',human_gold=False,natural_external_self_labels_generated=False,training_admission='HOLD_ROOT_REVIEW_AND_SOURCE_GROUP_CONTRACT')
    write('REFERENCE_REVIEW_PROTOCOL.json',protocol)
    write('REPRESENTATIVE_INSTANCE_LOCK.json',dict(schema='V74_FIXED20_LARGEST_OFFICIAL_PALLET_INSTANCE_LOCK',frozen_before_model_outputs=True,selection_rule=protocol['entity_selection'],records=selected,protocol=bind(R/'REFERENCE_REVIEW_PROTOCOL.json')))
    for name in ['RAW_224','RAW_NATIVE','REVIEW_CARDS','FULLFRAME_LOCATORS']:(R/name).mkdir(exist_ok=False)
    cards=[]
    for r in selected:
        n=r['selection_index'];assert sha(r['image'])==r['image_sha256'];im=Image.open(r['image']).convert('RGB')
        crop=im.crop(r['expanded_bbox_native_xyxy']);crop.save(R/'RAW_NATIVE'/f'{n:02d}.png');crop.resize((224,224),Image.Resampling.LANCZOS).save(R/'RAW_224'/f'{n:02d}.png')
        marked=im.copy();d=ImageDraw.Draw(marked);x,y,w,h=r['representative_annotation']['bbox'];d.rectangle([x,y,x+w,y+h],outline='#00ffff',width=max(2,im.width//500));d.rectangle(r['expanded_bbox_native_xyxy'],outline='#ffff00',width=max(1,im.width//700));marked.save(R/'FULLFRAME_LOCATORS'/f'{n:02d}.png')
        card=Image.new('RGB',(1150,720),'#101722');d=ImageDraw.Draw(card)
        text(d,(15,10),f"{n:02d} subset{r['source_subset']} image{r['image_id']} largest ann{r['representative_annotation']['id']} / palletinstances={r['official_pallet_instance_count']}",20)
        text(d,(15,45),f"Officialbbox={r['representative_annotation']['bbox']} expandednative={r['expanded_bbox_native_xyxy']}",16)
        card.paste(thumb(marked,(550,450)),(10,100));card.paste(crop.resize((560,560),Image.Resampling.NEAREST),(580,100))
        text(d,(15,570),'Fullframe cyan=officialbbox; yellow=context crop',17)
        text(d,(15,605),'Right=raw source crop, no model/mask/cue marks',17)
        text(d,(15,640),'Coordinates use original nativeRGB. Native8 UNKNOWN.',17)
        f=R/'REVIEW_CARDS'/f'{n:02d}.png';card.save(f);cards.append(card)
    pages=[]
    for pg in range(5):
        sheet=Image.new('RGB',(2300,1440),'#101722')
        for j in range(4):sheet.paste(cards[pg*4+j],((j%2)*1150,(j//2)*720))
        q=R/f'RAW_REVIEW_PAGE_{pg+1}.png';sheet.save(q);pages.append(bind(q))
    write('REVIEW_RENDER_RECEIPT.json',dict(completed=True,frames=20,modeloutputs_read=0,ownership_masks_read=0,source_RGB_only=True,native8_mapping='UNKNOWN',lock=bind(R/'REPRESENTATIVE_INSTANCE_LOCK.json'),pages=pages,newmodel_training_current_PnP=0))
    print(json.dumps(dict(lock=bind(R/'REPRESENTATIVE_INSTANCE_LOCK.json'),protocol=bind(R/'REFERENCE_REVIEW_PROTOCOL.json'),pages=pages)),flush=True)
if __name__=='__main__':main()
