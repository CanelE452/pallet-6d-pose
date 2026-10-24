"""CRC/hash every ZIP member without extraction; decode only frozen twenty RGB."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,re,time,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(name,obj):
    p=OUT/name;assert not p.exists(),p;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def label(d,xy,s,size=17):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill='white',stroke_width=1,stroke_fill='black')
def thumb(im,size):
    out=Image.new('RGB',size,'#171b23');im=im.copy();im.thumbnail(size,Image.Resampling.LANCZOS);out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out
def main():
    started=time.monotonic();lockpath=OUT/'FIXED_20_RGB_SELECTION_LOCK.json';lock=json.loads(lockpath.read_text())
    assert sha(lockpath)=='333486dc92050c92f6ab87e2fb0bc1f0299f530f6bfba7e58136b8e8c92846bc'
    assert lock['frozen_before_any_RGB_decode_or_view'] and len(lock['selected'])==20
    coco=json.loads((OUT/'annotations/loco-all-v1.json').read_text());annotated={i['path'].lstrip('/') for i in coco['images']}
    denyfile=BASE/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json'
    deny=json.loads(denyfile.read_text());assert deny['image_count']==223;denysha=set(deny['image_sha256'])
    rawdir=OUT/'FIXED_20_RGB';markdir=OUT/'FIXED_20_BBOX_OVERLAYS';rawdir.mkdir(exist_ok=False);markdir.mkdir(exist_ok=False)
    hashes=[];overlaps=[];crc_count=0;selected_data={}
    with zipfile.ZipFile(OUT/'dataset.zip') as z:
        names=set(z.namelist());assert len(names)==len(z.infolist())
        wanted={r['archive_member'] for r in lock['selected']};assert wanted<=names
        alljpg={i.filename for i in z.infolist() if i.filename.lower().endswith('.jpg')}
        assert len(alljpg)==5593 and annotated<=alljpg
        unannotated=alljpg-annotated;assert len(unannotated)==496
        for info in z.infolist():
            if info.is_dir():continue
            h=hashlib.sha256();keep=[] if info.filename in wanted else None
            with z.open(info) as f:
                for block in iter(lambda:f.read(1048576),b''):
                    h.update(block)
                    if keep is not None:keep.append(block)
            crc_count+=1
            if keep is not None:selected_data[info.filename]=b''.join(keep)
            if info.filename in alljpg:
                digest=h.hexdigest();row=dict(archive_member=info.filename,bytes=info.file_size,CRC32=f'{info.CRC:08x}',sha256=digest,official_COCO_image_present=info.filename in annotated)
                hashes.append(row)
                if digest in denysha:overlaps.append(row)
        assert len(selected_data)==20
    with (OUT/'ALL_5593_RGB_FILE_SHA256.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(hashes[0]));w.writeheader();w.writerows(hashes)
    write('ARCHIVE_CRC_AND_SHA_AUDIT.json',dict(PASS=True,archive=bind(OUT/'dataset.zip'),all_non_directory_members_CRC_read_PASS=crc_count,all_JPEG_members=5593,COCO_metadata_images=5097,unannotated_JPEGs=496,
      annotated_missing_archive=0,unannotated_by_subset=dict(Counter(re.search(r'/subset-(\d+)/',x).group(1) for x in unannotated)),
      annotation_absence_is_not_negative_label=True,unannotated_training_state='UNLABELLED_UNKNOWN; no automaticbackground/negativeassignment',
      exact_private_SHA_overlap_count=len(overlaps),overlaps=overlaps,denylist=bind(denyfile),private_image_or_GT_read=False,
      RGB_decode_calls_before_selection_lock=0,full_archive_extracted=False,hash_inventory=bind(OUT/'ALL_5593_RGB_FILE_SHA256.csv'),seconds=time.monotonic()-started))
    assert not overlaps,'Source exactfileoverlap: preserveevidenceandHOLD'
    selected=[];raws=[];marks=[]
    for r in lock['selected']:
        n=r['selection_index'];dest=rawdir/f"{n:02d}_subset{r['source_subset']}_id{r['image_id']}.jpg";data=selected_data[r['archive_member']];dest.write_bytes(data)
        im=Image.open(dest).convert('RGB');meta=r['official_image_metadata'];assert im.size==(meta['width'],meta['height'])
        mark=im.copy();d=ImageDraw.Draw(mark);stroke=max(1,round(im.width/800))
        for b in r['pallet_bboxes_xywh']:
            x,y,w,h=b;d.rectangle((x,y,x+w,y+h),outline='#39ff14',width=stroke)
        marked=markdir/(dest.stem+'.png');mark.save(marked)
        selected.append(dict(**r,image=str(dest),image_sha256=sha(dest),RGB_pixels_sha256=hashlib.sha256(np.ascontiguousarray(np.asarray(im)).tobytes()).hexdigest(),raw_hw=[im.height,im.width],overlay=bind(marked),semantic_native8_D_E_S=None,annotation_tier='OFFICIAL_MANUAL_INSTANCE_BBOX_ONLY',training_admission='HOLD_PENDING_NATIVE8_CUE_SCENE_INDEPENDENT_CONTRACT'))
        raws.append(im);marks.append(mark)
    mf=OUT/'FIXED_20_RGB_MANIFEST.jsonl';mf.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    pages=[]
    for group in range(5):
        page=Image.new('RGB',(1600,1100),'#101722');d=ImageDraw.Draw(page)
        label(d,(15,12),f'LOCO official subset{group+1}: fixed SHA-selected4, raw RGB | official PALLET bbox (category7)',23)
        label(d,(15,47),'Selection frozen before RGB view. Rectangles are instance boxes, not native8/keypoint or visible/amodal pixel GT.',16)
        for j in range(4):
            n=group*4+j;r=selected[n];x=(j%2)*800;y=90+(j//2)*500
            label(d,(x+12,y),f"{n:02d} image_id={r['image_id']} pallets={len(r['pallet_annotation_ids'])} {r['raw_hw']}",16)
            page.paste(thumb(raws[n],(390,420)),(x+5,y+32));page.paste(thumb(marks[n],(390,420)),(x+405,y+32))
            label(d,(x+12,y+460),r['archive_member'][-84:],12)
        dest=OUT/f'FIXED_20_CONTACT_SUBSET_{group+1}.png';page.save(dest);pages.append(bind(dest))
    combined=Image.new('RGB',(2400,1900),'#101722');d=ImageDraw.Draw(combined)
    label(d,(15,12),'LOCO fixed20: actual RGB + official pallet bbox | four per original subset | no new model / semanticGT',25)
    for n in range(20):
        x=(n%4)*600;y=65+(n//4)*365;r=selected[n]
        label(d,(x+8,y),f"{n:02d} subset{r['source_subset']} image{r['image_id']} palletbboxN={len(r['pallet_annotation_ids'])}",13)
        combined.paste(thumb(raws[n],(292,300)),(x+4,y+30));combined.paste(thumb(marks[n],(292,300)),(x+304,y+30))
    contact=OUT/'FIXED_20_ACTUAL_RGB_CONTACT_SHEET.png';combined.save(contact)
    write('FIXED_20_EXTRACTION_RECEIPT.json',dict(completed=True,selection_lock=bind(lockpath),extracted_original_RGB=20,decoded_RGB=20,additional_RGB_decoded=0,
      all_native_dimensions_match_COCO=True,source_raw_archive_bytes_preserved=True,manifest=bind(mf),contact_sheet=bind(contact),subset_contact_sheets=pages,
      same_sourcegroup_wholeframes_before_future_split=True,labels_not_promoted_to_semanticGT=True,training_admission='HOLD_PENDING_NATIVE8_CUE_SCENE_INDEPENDENT_CONTRACT',newmodel_fit_current_PnP=0,seconds=time.monotonic()-started))
    print(json.dumps(dict(completed=True,CRC=bind(OUT/'ARCHIVE_CRC_AND_SHA_AUDIT.json'),manifest=bind(mf),contact=bind(contact),pages=pages),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
