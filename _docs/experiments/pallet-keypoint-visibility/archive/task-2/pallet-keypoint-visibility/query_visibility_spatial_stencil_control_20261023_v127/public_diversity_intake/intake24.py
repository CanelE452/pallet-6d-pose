"""Metadata lock first, then exact registered PUBLIC TRAIN JPEG extraction/view cards."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,datetime,hashlib,io,json,math,zipfile

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
H=Path(__file__).resolve().parent
V73=P/'query_visibility_public_pallet_loco_20261023_v73/data/intake'
V74=P/'query_visibility_loco_native_context_20261023_v74'
SOURCE=V74/'source_groups/INPUT_ONLY_SOURCE_GROUP_MANIFEST.jsonl'
ANN=V73/'annotations/loco-sub2-v1-train.json'
ZIP=V73/'dataset.zip'
F20=V73/'FIXED_20_RGB_MANIFEST.jsonl'
E40=V74/'public_reference_review/EXTENSION_40/EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl'
C32=P/'query_visibility_flat_pallet_cue_source_20261023_v122/input_preparation/CUES32_INPUT_ONLY.jsonl'
SIX=P/'query_visibility_boundary_query_source_20261023_v125/public_loco_take_intake/SELECTION_LOCK_BEFORE_PIXELS.json'

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,v):
    with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def select():
    assert not(H/'SELECTION_LOCK_BEFORE_DECODE.json').exists()
    source=[r for r in rows(SOURCE)if r['subset']==2];assert len(source)==1109 and all(r['official_split']=='TRAIN'for r in source)
    prior=rows(F20)+rows(E40);old6=read(SIX)['rows'];cues=[r for r in rows(C32)if r['source']=='LOCO']
    excluded_members={r['archive_member']for r in prior if r['source_subset']==2}|{r['original_row']['archive_member']for r in old6}
    excluded_decoded={r.get('RGB_pixels_sha256',r.get('decoded_RGB_sha256'))for r in prior}|{r['original_row']['decoded_RGB_SHA256']for r in old6}|{r['decoded_RGB_sha256']for r in cues}
    excluded_jpeg={r.get('image_sha256')for r in prior}|{r['original_row']['jpeg_file_SHA256']for r in old6}|{r['original_JPEG']['sha256']for r in cues}
    d=read(ANN);pallet=next(c['id']for c in d['categories']if c['name']=='pallet');assert pallet==7
    images={x['path'].lstrip('/'):x for x in d['images']};aa=defaultdict(list)
    for a in d['annotations']:
        if a['category_id']==pallet and a['bbox'][2]>0 and a['bbox'][3]>0:aa[a['image_id']].append(a)
    groups=defaultdict(list)
    for r in source:groups[r['path_capture_date_take']].append(r)
    assert sorted(groups)==['2019-11-25_0','2019-11-25_1','2019-11-25_2','2019-11-25_3','2019-11-25_5','2019-11-25_6']
    selected=[];groupinfo=[];chosen_decoded=set()
    with zipfile.ZipFile(ZIP)as z:
        for take,rs in sorted(groups.items()):
            streams=defaultdict(list)
            for r in rs:
                if r['archive_member']in excluded_members or r['decoded_RGB_SHA256']in excluded_decoded or r['jpeg_file_SHA256']in excluded_jpeg:continue
                im=images.get(r['archive_member'])
                if im is not None and aa[im['id']]:streams[r['path_stream_key']].append(r)
            eligible={k:v for k,v in streams.items()if len(v)>=4}
            assert eligible,'Fixed take has no stream with4 unused officially annotated pallet originals'
            kinect={k:v for k,v in eligible.items()if'/Kinect/'in k}
            candidates=kinect if kinect else eligible
            stream=min(candidates,key=lambda k:(-len(candidates[k]),k))
            seq=sorted(candidates[stream],key=lambda r:(r['timestamp_filename_number'],r['archive_member']))
            positions=[math.floor(j*(len(seq)-1)/3)for j in range(4)]
            assert len(set(positions))==4
            lo=seq[0]['timestamp_filename_number'];hi=seq[-1]['timestamp_filename_number']
            groupinfo.append(dict(take=take,all_source_rows=len(rs),eligible_stream_counts={k:len(v)for k,v in sorted(eligible.items())},selected_stream=stream,Kinect_preferred=bool(kinect),fallback_reason=None if kinect else'No Kinect stream in this take; largest eligible stream, lexicographic tie.',selected_ordinals=positions,sequence_count=len(seq),clock_kind=seq[0]['timestamp_kind'],timestamp_min=lo,timestamp_max=hi))
            for ordinal in positions:
                r=seq[ordinal];im=images[r['archive_member']];anns=sorted(aa[im['id']],key=lambda a:a['id'])
                target=min(anns,key=lambda a:(-a['bbox'][2]*a['bbox'][3],a['id']))
                assert r['decoded_RGB_SHA256']not in chosen_decoded,'No silent duplicate replacement';chosen_decoded.add(r['decoded_RGB_SHA256'])
                info=z.getinfo(r['archive_member']);i=len(selected)
                selected.append(dict(selection_index=i,take=take,camera_stream=stream,ordinal=ordinal,sequence_count=len(seq),normalized_temporal_position=(r['timestamp_filename_number']-lo)/(hi-lo)if hi>lo else 0,original_source_row=r,official_image_metadata=im,all_official_pallet_annotations=anns,crop_annotation=target,crop_selection='Largest official pallet bbox area, lowest annotationID tie. Display crop only, no initializer/GT-best query or cue choice.',zip=dict(member=info.filename,CRC32=f'{info.CRC:08x}',uncompressed_bytes=info.file_size,compressed_bytes=info.compress_size,compression_type=info.compress_type)))
    assert len(selected)==24 and len(chosen_decoded)==24
    lock=dict(completed_before_decode=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),code=bind(__file__),source=bind(SOURCE),official_annotations=bind(ANN),exclusions=dict(fixed20=bind(F20),extension40=bind(E40),V122_cues=bind(C32),V125_six=bind(SIX),excluded_subset2_members=len(excluded_members)),selection_rule='Four frames/take: prefer Kinect stream with >=4 unused officially pallet-annotated frames, otherwise largest eligible stream; tie stream path. Sort timestamp within one stream, select floor(j*(N-1)/3),j0..3. No pixel/visibility/material/model screening. Abort duplicate decoded image rather than replace after pixels.',archive=dict(path=str(ZIP),bytes=ZIP.stat().st_size,prior_whole_archive_audit=bind(V73/'ARCHIVE_CRC_AND_SHA_AUDIT.json'),advertised_previous_sha256='f3629d989071b824edc63e000f45ccc5826cfbb201a338f16ed87ab3e10bcdcf'),groups=groupinfo,rows=selected,all_original_fold='TRAIN',new_labels_queries_or_training_admission=False,source_limitations=['All six takes remain one official subset/environment/date; conservatively one source cluster for future folds.','New relative to listed review/training originals, not never exposed: V74 already inventoried all5593 source images.','Timestamp units are not verified across sensor streams. Temporal spacing is only within one selected stream.','Official pallet annotations are manual bbox-only (isbbox); no instance pixel mask or structural-cue gold is inferred.','Largest official bbox is solely a deterministic inspection crop, not an initializer target/cue association.'])
    write(H/'SELECTION_LOCK_BEFORE_DECODE.json',lock)
    print(json.dumps(dict(completed=True,selection_lock=bind(H/'SELECTION_LOCK_BEFORE_DECODE.json'),groups=groupinfo),ensure_ascii=False),flush=True)

def extract_render():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    lockpath=H/'SELECTION_LOCK_BEFORE_DECODE.json';lock=read(lockpath);assert lock['code']==bind(__file__)and lock['source']==bind(SOURCE)and lock['official_annotations']==bind(ANN)
    assert not(H/'EXTRACTION_AND_RENDER_COMPLETE.json').exists()
    for name in('RGB_ORIGINAL','PALLET_CROPS_NATIVE','figures'):(H/name).mkdir(exist_ok=False)
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';title=ImageFont.truetype(font,24);normal=ImageFont.truetype(font,17);small=ImageFont.truetype(font,14)
    receipts=[];contacts={};cards=[];W,HH=740,750
    def fit(im,size):
        c=im.copy();c.thumbnail(size,Image.Resampling.LANCZOS);return c
    with zipfile.ZipFile(ZIP)as z:
        for s in lock['rows']:
            i=s['selection_index'];r=s['original_source_row'];zi=z.getinfo(s['zip']['member'])
            assert f'{zi.CRC:08x}'==s['zip']['CRC32']and zi.file_size==s['zip']['uncompressed_bytes']
            raw=z.read(zi);assert hashlib.sha256(raw).hexdigest()==r['jpeg_file_SHA256']
            a=np.array(Image.open(io.BytesIO(raw)).convert('RGB'),np.uint8,copy=True)
            assert list(a.shape[:2])==r['raw_hw']and hashlib.sha256(a.tobytes()).hexdigest()==r['decoded_RGB_SHA256']
            orig=H/'RGB_ORIGINAL'/f'{i:02d}_{s["take"]}.jpg';orig.write_bytes(raw)
            b=s['crop_annotation']['bbox'];x,y,w,h=b;Hn,Wn=a.shape[:2];bounds=[max(0,math.floor(x)),max(0,math.floor(y)),min(Wn,math.ceil(x+w)),min(Hn,math.ceil(y+h))];assert bounds[2]>bounds[0]and bounds[3]>bounds[1]
            im=Image.fromarray(a);crop=im.crop(bounds);cf=H/'PALLET_CROPS_NATIVE'/f'{i:02d}_ann{s["crop_annotation"]["id"]}.png';crop.save(cf)
            panel=Image.new('RGB',(W,HH),'#f8fafc');dr=ImageDraw.Draw(panel)
            dr.text((16,10),f'{i:02d} | {s["take"]} | TRAIN',font=title,fill='#123753')
            dr.text((16,43),s['camera_stream'].split(s['take']+'/')[-1]+f' | time rank {s["ordinal"]}/{s["sequence_count"]-1}',font=normal,fill='#34566d')
            full=fit(im,(708,399));panel.paste(full,(16+(708-full.width)//2,74+(399-full.height)//2))
            dr.text((16,478),f'Raw frame {Wn}x{Hn}; official pallet bboxes={len(s["all_official_pallet_annotations"])}',font=normal,fill='#34566d')
            view=fit(crop,(530,190));panel.paste(view,(16+(530-view.width)//2,512+(190-view.height)//2))
            marked=fit(im,(174,190));md=ImageDraw.Draw(marked);sx=marked.width/Wn;sy=marked.height/Hn;md.rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='#ff3344',width=3);panel.paste(marked,(550+(174-marked.width)//2,512+(190-marked.height)//2))
            dr.text((16,706),f'Raw bbox crop ann{s["crop_annotation"]["id"]}; separate red-box thumbnail',font=small,fill='#34566d')
            dr.text((16,728),'No cue points / D-E labels / native identity / initializer / model calls',font=small,fill='#8a5d27')
            card=H/'figures'/f'CASE_{i:02d}.png';panel.save(card);cards.append(bind(card))
            take=s['take'];contacts.setdefault(take,Image.new('RGB',(2*W,2*HH),'#dfe8ef'));k=i%4;contacts[take].paste(panel,((k%2)*W,(k//2)*HH))
            receipts.append(dict(selection_index=i,take=take,original_member=r['archive_member'],image_id=s['official_image_metadata']['id'],original_JPEG=bind(orig),native_hw=list(a.shape[:2]),decoded_RGB_sha256=r['decoded_RGB_SHA256'],ZIP_CRC32_verified=True,file_and_decoded_SHA_exact_V74=True,representative_bbox_annotation=s['crop_annotation'],crop_native_bounds_half_open=bounds,crop_native=bind(cf),card=bind(card),camera_stream=s['camera_stream'],timestamp_raw=r['timestamp_filename_raw'],timestamp_number=r['timestamp_filename_number'],timestamp_kind=r['timestamp_kind'],all_pallet_annotation_ids=[a['id']for a in s['all_official_pallet_annotations']],original_fold='TRAIN'))
    sheets=[]
    for i,(take,contact)in enumerate(sorted(contacts.items())):
        f=H/'figures'/f'CONTACT_TAKE_{i:02d}.png';contact.save(f);sheets.append(dict(take=take,file=bind(f)))
    html=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>PUBLIC LOCO TRAIN24 intake</title><style>body{font-family:system-ui;max-width:1500px;margin:24px auto;background:#f4f7fa;color:#17364c}img{max-width:100%}section{background:white;padding:16px;margin:20px 0}a{color:#1264a3}</style><h1>공개 LOCO TRAIN24 원본 다양성 확인</h1><p>디코딩 전 고정: 6 take × 4장. 기존 fixed20/ext40/V122/V125 원본 제외. 같은 subset·날짜·환경의 상관된 TRAIN 자료이며 새 holdout이 아닙니다. 공식 bbox는 물체 영역 주석이며 cue/D/E/native gold가 아닙니다.</p>']
    for s in sheets:html.append(f'<section><h2>{s["take"]}</h2><img src="figures/{Path(s["file"]["path"]).name}"></section>')
    html.append('<h2>Original frames and native bbox crops</h2><ul>')
    for r in receipts:html.append(f'<li>{r["selection_index"]:02d} <a href="RGB_ORIGINAL/{Path(r["original_JPEG"]["path"]).name}">original</a> · <a href="PALLET_CROPS_NATIVE/{Path(r["crop_native"]["path"]).name}">native bbox crop</a> · <a href="figures/{Path(r["card"]["path"]).name}">card</a></li>')
    html.append('</ul></html>');(H/'GALLERY_KO.html').write_text('\n'.join(html))
    write(H/'EXTRACTION_AND_RENDER_COMPLETE.json',dict(completed=True,PASS=True,selection_lock=bind(lockpath),code=bind(__file__),frames=24,takes=6,all_original_fold='TRAIN',all24_CRC_JPEG_SHA_decoded_SHA_exact=True,rows=receipts,contact_sheets=sheets,gallery=bind(H/'GALLERY_KO.html'),model_encoder_head_fit_new_labels_current_V112_PnP=0,old_files_modified=False))
    print(json.dumps(dict(completed=True,receipt=bind(H/'EXTRACTION_AND_RENDER_COMPLETE.json'),sheets=sheets),ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['select','extract-render'],required=True);args=ap.parse_args();(select if args.stage=='select'else extract_render)()
