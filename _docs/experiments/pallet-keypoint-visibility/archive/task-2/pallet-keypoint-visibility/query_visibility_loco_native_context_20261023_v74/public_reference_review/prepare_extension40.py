"""Freeze next SHA ranks4..11 perLOCOsubset; RGB review only, no new refs."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,math,re,zipfile
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;BASE=R.parents[1]
OLD=BASE/'query_visibility_public_pallet_loco_20261023_v73/data/intake'
OUT=R/'EXTENSION_40'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(name,obj):
    p=OUT/name;assert not p.exists(),p;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def text(d,xy,s,size=16):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill='white',stroke_width=1,stroke_fill='black')
def thumb(im,size):
    a=Image.new('RGB',size,'#181d25');im=im.copy();im.thumbnail(size,Image.Resampling.LANCZOS);a.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return a
def main():
    OUT.mkdir(exist_ok=False)
    a=json.loads((OLD/'annotations/loco-all-v1.json').read_text());pallet=defaultdict(list)
    for x in a['annotations']:
        if x['category_id']==7:pallet[x['image_id']].append(x)
    orig=json.loads((OLD/'FIXED_20_RGB_SELECTION_LOCK.json').read_text())['selected'];oldpaths={r['archive_member'] for r in orig}
    selected=[]
    for subset in range(1,6):
        pool=[i for i in a['images'] if f'/subset-{subset}/' in i['path'] and pallet[i['id']]]
        def order(i):return hashlib.sha256(f"LOCO-v1|{subset}|{i['id']}|{i['path']}".encode()).hexdigest()
        ranked=sorted(pool,key=order)
        assert {r['archive_member'] for r in orig if r['source_subset']==subset}=={i['path'].lstrip('/') for i in ranked[:4]}
        for rank in range(4,12):
            im=ranked[rank];ann=min(pallet[im['id']],key=lambda r:(-r['bbox'][2]*r['bbox'][3],r['id']))
            x,y,w,h=ann['bbox'];side=1.25*max(w,h);cx=x+w/2;cy=y+h/2
            b=[math.floor(cx-side/2),math.floor(cy-side/2),math.ceil(cx+side/2),math.ceil(cy+side/2)]
            selected.append(dict(extension_index=len(selected),source_subset=subset,source_SHA_rank=rank,selection_SHA256=order(im),
              review_id=f"LOCO_V74_EXT:subset{subset}:rank{rank}:ann{ann['id']}",official_image_metadata=im,
              archive_member=im['path'].lstrip('/'),image_id=im['id'],raw_hw=[im['height'],im['width']],official_pallet_instance_count=len(pallet[im['id']]),
              representative_annotation=ann,expanded_bbox_native_xyxy=b,display_crop_side=224,
              root_authorized_source_split='TRAIN' if subset in [2,3,5] else 'VALIDATION',
              source_capture_group=f'LOCO_SUBSET_{subset}',native8_mapping='UNKNOWN',reference_annotations_created=False))
    assert len(selected)==40 and len({r['archive_member'] for r in selected})==40 and not oldpaths&{r['archive_member'] for r in selected}
    protocol=dict(schema='V74_NEXT40_LOCO_SOURCE_REFERENCE_EXTENSION_PROTOCOL_V1',frozen_UTC=datetime.now(timezone.utc).isoformat(),
      selection='Sameofficialpalletimagepool+SHA256(LOCO-v1|subset|imageid|officialpath) asoriginal20; ranks4..11 inclusive pereachsubset. No RGB/score/bboxsize/blur/qualityfilter. Exactly40additionalframes; originalsunchanged.',
      representative='Largestofficialpalletbbox width*height, ties lowestannotationID, identicaltooriginal20rule. Do notchooseanotherinstanceafterview.',
      rendering='1.25-expandedbbox square, raw224+nativecrop/fullframelocator; all40 includingblur/stack/tiny.',
      source_split={'TRAIN_subsets':[2,3,5],'VALIDATION_subsets':[1,4],'authority':'RootexplicitV74sourcefreeze; wholecapturegroups/sessionlineage retained; do notrandomsplitneighborframes.','subject_group_audit':'Bind finalV74sourcegroupaudit beforefuturefit; no claimCOCOdataset_idis siteidentity.'},
      annotation='No referencecoordinates, masks, D/E/S, visibleteacherlabels or canonicalnative8createdinthisextensionselection/extraction step. LaterannotationmustuseRGBonlyindependentlyofmodeloutputs.',
      model_current_fit_PnP=0,training_admission='HOLD_REFERENCE_ANNOTATION_AND_CAPTURE_GROUP_CONTRACT',
      original20_lock=bind(OLD/'FIXED_20_RGB_SELECTION_LOCK.json'),source_COCO=bind(OLD/'annotations/loco-all-v1.json'),previous20_reference_geometry=bind(R/'RGB_REFERENCE_GEOMETRY_LOCK.json'))
    write('EXTENSION_PROTOCOL.json',protocol)
    write('EXTENSION_40_SELECTION_LOCK.json',dict(schema='V74_METADATA_ONLY_NEXT40_SOURCE_RGB_AND_REPRESENTATIVE_LOCK',frozen_before_any_new40_RGB_decode_or_view=True,protocol=bind(OUT/'EXTENSION_PROTOCOL.json'),N=40,per_subset=8,records=selected))
    for s in ['RGB_ORIGINAL','RAW_224','RAW_NATIVE','REVIEW_CARDS','FULLFRAME_LOCATORS']:(OUT/s).mkdir()
    manifests=[];cards=[]
    with zipfile.ZipFile(OLD/'dataset.zip') as z:
        for r in selected:
            n=r['extension_index'];data=z.read(r['archive_member']);dest=OUT/'RGB_ORIGINAL'/f"{n:02d}_subset{r['source_subset']}_rank{r['source_SHA_rank']:02d}_id{r['image_id']}.jpg";dest.write_bytes(data)
            im=Image.open(dest).convert('RGB');assert [im.height,im.width]==r['raw_hw']
            crop=im.crop(r['expanded_bbox_native_xyxy']);crop.save(OUT/'RAW_NATIVE'/f'{n:02d}.png');crop.resize((224,224),Image.Resampling.LANCZOS).save(OUT/'RAW_224'/f'{n:02d}.png')
            mark=im.copy();d=ImageDraw.Draw(mark);x,y,w,h=r['representative_annotation']['bbox'];d.rectangle((x,y,x+w,y+h),outline='#00ffff',width=max(2,im.width//500));d.rectangle(r['expanded_bbox_native_xyxy'],outline='#ffff00',width=max(1,im.width//700));mark.save(OUT/'FULLFRAME_LOCATORS'/f'{n:02d}.png')
            card=Image.new('RGB',(1150,720),'#101722');d=ImageDraw.Draw(card)
            text(d,(12,10),f"EXT{n:02d} subset{r['source_subset']} SHA-rank{r['source_SHA_rank']} {r['root_authorized_source_split']} image{r['image_id']} ann{r['representative_annotation']['id']}",20)
            text(d,(12,45),f"bbox={r['representative_annotation']['bbox']} expandednative={r['expanded_bbox_native_xyxy']} palletinstances={r['official_pallet_instance_count']}",15)
            card.paste(thumb(mark,(550,450)),(10,100));card.paste(crop.resize((560,560),Image.Resampling.NEAREST),(580,100))
            text(d,(15,570),'Metadata selection only; actualraw expandedcrop at right',17);text(d,(15,610),'No reference/D/E/S labels yet. Native8 UNKNOWN.',17)
            cp=OUT/'REVIEW_CARDS'/f'{n:02d}.png';card.save(cp);cards.append(card)
            manifests.append(dict(**r,image=str(dest),image_sha256=hashlib.sha256(data).hexdigest(),raw_native_crop=str(OUT/'RAW_NATIVE'/f'{n:02d}.png'),raw224=str(OUT/'RAW_224'/f'{n:02d}.png'),review_card=str(cp)))
    mf=OUT/'EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl';mf.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifests))
    pages=[]
    for pg in range(10):
        sheet=Image.new('RGB',(2300,1440),'#101722')
        for j in range(4):sheet.paste(cards[pg*4+j],((j%2)*1150,(j//2)*720))
        f=OUT/f'EXTENSION_REVIEW_PAGE_{pg+1:02d}.png';sheet.save(f);pages.append(bind(f))
    # RGB-only initializer manifest deliberately excludes annotation/reference/contextcrop fields.
    rgb=[dict(id=f"LOCO_V74_EXT:{r['extension_index']:02d}:{r['image_id']}",image=r['image'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],source_subset=r['source_subset'],split=r['root_authorized_source_split']) for r in manifests]
    inputfile=OUT/'INPUTS_RGB_ONLY.jsonl';inputfile.write_text(''.join(json.dumps(r)+'\n' for r in rgb))
    receipt=dict(schema='V74_ACTUAL_NEXT40_PUBLIC_SOURCE_RGB_EXTENSION_RECEIPT',completed=True,additional_RGB=40,original_frames_unchanged=20,total_available_selected_source_frames=60,
      selected_source_split=dict(Counter(r['root_authorized_source_split'] for r in selected)),original_overlap=0,selection_lock=bind(OUT/'EXTENSION_40_SELECTION_LOCK.json'),manifest=bind(mf),RGB_only_manifest=bind(inputfile),pages=pages,
      all_native_HW_match_official_metadata=True,source_archive=bind(OLD/'dataset.zip'),reference_labels_created=0,model_forward_fit_current_PnP=0,training_admission=protocol['training_admission'])
    write('EXTENSION_40_RESULT.json',receipt)
    files=sorted(p for p in OUT.rglob('*') if p.is_file());write('EXTENSION_40_SEAL.json',dict(schema='V74_ACTUAL_NEXT40_SOURCE_RGB_REVIEW_SEAL',completed=True,bindings=[bind(p) for p in files],code=bind(Path(__file__)),source_only=True,newmodel_forward_fit_current_PnP=0))
    print(json.dumps(dict(result=bind(OUT/'EXTENSION_40_RESULT.json'),seal=bind(OUT/'EXTENSION_40_SEAL.json'),manifest=bind(mf),RGB_only_manifest=bind(inputfile),selection=bind(OUT/'EXTENSION_40_SELECTION_LOCK.json'))),flush=True)
if __name__=='__main__':main()
