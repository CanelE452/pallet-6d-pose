"""LOCO TRAIN subset2: prospective metadata-only12 selection, then exact intake.

--prepare never opens an RGB/ZIP member. --entry extracts only the twelve
preselected original JPEGs and shows whole frames plus official bbox metadata.
Native-slot placeholders remain UNREVIEWED; no reference/visibility labels.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import html
import io
import json
import sys
import time
import zipfile
import zlib
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_PUBLIC_NATIVE_FEASIBILITY_POLICY_BEFORE_BODY_READS.json'
POLICY_SHA='59addd73ff75fbaf926c54d4640834327d3401fbd6a604a1097237372f816d5c'
INTAKE=P/'query_visibility_public_pallet_loco_20261023_v73/data/intake'
GROUPS=P/'query_visibility_loco_native_context_20261023_v74/source_groups/INPUT_ONLY_SOURCE_GROUP_MANIFEST.jsonl'
FINDINGS=P/'query_visibility_training_duration_control_20261023_v124/public_intake/FINDINGS.json'
ALL=INTAKE/'annotations/loco-all-v1.json';SUB=INTAKE/'annotations/loco-sub2-v1-train.json'
FIXED20=INTAKE/'FIXED_20_RGB_SELECTION_LOCK.json'
EXT40=P/'query_visibility_loco_native_context_20261023_v74/public_reference_review/EXTENSION_40/EXTENSION_40_SELECTION_LOCK.json'
ARCHIVE=INTAKE/'dataset.zip';ARCHIVE_AUDIT=INTAKE/'ARCHIVE_CRC_AND_SHA_AUDIT.json';INVENTORY=INTAKE/'ARCHIVE_INVENTORY.json'
LICENSE=INTAKE/'LICENSE_CC0_OFFICIAL.txt'
SELECTION=H/'SELECTION12_BEFORE_RGB.json';PROTOCOL=H/'PROTOCOL_BEFORE_RGB.json';GENERATED=H/'GENERATED_SELECTION_CHECKS.json'
OUT=V/'loco_fixed12_intake_v1'
TAKES=tuple('2019-11-25_'+str(n)for n in(0,1,2,3,5,6))
PREFIX='V140_LOCO_NATIVE_FEASIBILITY|'
FORBIDDEN=('initializer_forward','visibility_head_forward','fit','label_creation','native_reference_creation','nearest_mapping','model_checkpoint_read','private_current223','V112','PnP')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def digest(member):return hashlib.sha256((PREFIX+member).encode('utf-8')).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def lines(path):return[json.loads(line)for line in Path(path).read_text().splitlines()if line]
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def select(candidates,excluded_members,excluded_ids,excluded_sub_ids=(),takes=TAKES):
    chosen=[];counts={}
    for take in takes:
        pool=[r for r in candidates if r['take']==take and r['has_official_pallet_instance']and r['archive_member']not in excluded_members and r['all_COCO_image_id']not in excluded_ids and r['official_sub2_train_image_id']not in excluded_sub_ids]
        pool.sort(key=lambda r:(digest(r['archive_member']),r['archive_member']))
        assert len(pool)>=2,take
        counts[take]=len(pool)
        for rank,r in enumerate(pool[:2]):chosen.append(dict(r,selection_index=len(chosen),rank_within_take=rank,selection_SHA256=digest(r['archive_member'])))
    return chosen,counts


def metadata_selection():
    coco=read(ALL);sub=read(SUB);manifest=lines(GROUPS)
    assert next(c['id']for c in coco['categories']if c['name']=='pallet')==7
    assert next(c['id']for c in sub['categories']if c['name']=='pallet')==7
    annotations=collections.defaultdict(list);subannotations=collections.defaultdict(list)
    for a in coco['annotations']:
        if a['category_id']==7:annotations[a['image_id']].append(a)
    for a in sub['annotations']:
        if a['category_id']==7:subannotations[a['image_id']].append(a)
    subimages={r['path'].lstrip('/'):r for r in sub['images']};allimages={r['path'].lstrip('/'):r for r in coco['images']}
    old20=read(FIXED20)['selected'];old40=read(EXT40)['records'];old=old20+old40
    assert len(old20)==20 and len(old40)==40
    excluded_members={r['archive_member']for r in old};excluded_ids={r['image_id']for r in old}
    excluded_sub_ids={subimages[member]['id']for member in excluded_members if member in subimages}
    assert len(excluded_members)==len(excluded_ids)==60
    inventory={r['name']:r for r in read(INVENTORY)['files']}
    raw=[r for r in manifest if r['subset']==2];assert len(raw)==1109
    assert set(r['path_capture_date_take']for r in raw)==set(TAKES)
    candidates=[]
    for r in raw:
        assert r['official_split']=='TRAIN'
        member=r['archive_member']
        if member not in allimages:continue
        image=allimages[member];sim=subimages[member]
        assert image['path']==sim['path']and(image['height'],image['width'])==(sim['height'],sim['width'])
        assert r['raw_hw']==[image['height'],image['width']]
        aa=annotations[image['id']];sa=subannotations[sim['id']]
        # Different official exports use different image/annotation IDs. Pair by
        # exact original member only; verify their pallet bbox multisets agree.
        assert sorted(tuple(a['bbox'])for a in aa)==sorted(tuple(a['bbox'])for a in sa)
        candidates.append(dict(archive_member=member,source_id=r['source_id'],take=r['path_capture_date_take'],subset=2,official_split='TRAIN',
            all_COCO_image_id=image['id'],official_sub2_train_image_id=sim['id'],raw_hw=r['raw_hw'],has_official_pallet_instance=bool(aa),
            expected_raw_JPEG_sha256=r['jpeg_file_SHA256'],expected_decoded_RGB_sha256=r['decoded_RGB_SHA256'],
            zip_member_metadata={k:inventory[member][k]for k in('compressed_bytes','uncompressed_bytes','CRC32','method')},
            official_pallet_instances=[{k:a[k]for k in('id','image_id','category_id','bbox','iscrowd','isbbox')}for a in sorted(aa,key=lambda a:a['id'])]))
    assert len(candidates)==1070 and sum(r['has_official_pallet_instance']for r in candidates)==1022
    chosen,counts=select(candidates,excluded_members,excluded_ids,excluded_sub_ids)
    assert len(chosen)==12 and len({r['archive_member']for r in chosen})==12
    return dict(completed=True,frozen_before_any_V140_RGB_read=True,rule="SHA256(UTF8('V140_LOCO_NATIVE_FEASIBILITY|'+member)), smallest2 per take; no pixel/quality/score filtering",
        official_split='TRAIN',subset=2,takes=list(TAKES),raw_subset2_count=1109,COCO_subset2_images=1070,COCO_pallet_images=1022,
        excluded_old20_plus_extension40_members=sorted(excluded_members),excluded_all_COCO_image_ids=sorted(excluded_ids),excluded_official_sub2_train_image_ids=sorted(excluded_sub_ids),
        candidates_after_exclusion_per_take=counts,selected=chosen,all_rows_retained_after_selection=True,material_labels_available=False,
        native8_reference_or_visibility_available=False,target_instance_association='UNREVIEWED; all official pallet boxes preserved, none selected',
        fold_warning='All six temporal takes remain one existing official TRAIN environment; not six independent environments. Old VAL unchanged.',
        raw_already_observed_in_prior_inventory=True,new_to_fixed20_plus_extension40=True,RGB_model_label_private_reads=0)


def generated_checks():
    candidates=[]
    for take in TAKES:
        for i in range(5):candidates.append(dict(take=take,archive_member=take+'/'+str(i)+'.jpg',all_COCO_image_id=len(candidates),official_sub2_train_image_id=1000+len(candidates),has_official_pallet_instance=i!=4))
    exmember={TAKES[0]+'/0.jpg'};exid={6};exsub={1010}
    a,counts=select(candidates,exmember,exid,exsub);b,_=select(list(reversed(candidates)),exmember,exid,exsub)
    assert a==b and len(a)==12 and all(r['archive_member']not in exmember and r['all_COCO_image_id']not in exid and r['official_sub2_train_image_id']not in exsub and r['has_official_pallet_instance']for r in a)
    for take in TAKES:
        selected=[r for r in a if r['take']==take]
        pool=[r for r in candidates if r['take']==take and r['has_official_pallet_instance']and r['archive_member']not in exmember and r['all_COCO_image_id']not in exid and r['official_sub2_train_image_id']not in exsub]
        assert [r['selection_SHA256']for r in selected]==sorted(digest(r['archive_member'])for r in pool)[:2]
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_metadata_only=True,two_per_six_take_exact_SHA_order=True,
        original_member_and_all_COCO_id_exclusions=True,official_pallet_instance_required=True,input_order_invariant=True,no_pixels_or_new_labels=True))


def prepare():
    assert POLICY.exists()and sha(POLICY)==POLICY_SHA,'Root policy must exist before prospective selection'
    generated_checks();selection=metadata_selection();write(SELECTION,selection)
    audit=read(ARCHIVE_AUDIT);assert audit['PASS']and audit['all_JPEG_members']==5593
    assert ARCHIVE.stat().st_size==769055500 and audit['archive']['path']==str(ARCHIVE)
    sources={name:bind(path)for name,path in dict(source_groups=GROUPS,official_all_COCO=ALL,official_sub2_train_COCO=SUB,old_fixed20=FIXED20,old_extension40=EXT40,
        cached_archive_audit=ARCHIVE_AUDIT,cached_archive_inventory=INVENTORY,prior_public_findings=FINDINGS,official_CC0_license=LICENSE).items()}
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),selection=bind(SELECTION),sources=sources,
        archive=audit['archive'],archive_bytes=769055500,archive_whole_SHA_inherited_not_rehashed=True,output=str(OUT),selection_images=12,
        pending_review_slots=96,pixels_read_during_preparation=0,model_and_label_creation=0,
        outputs='Exact12original JPEGs,6wholeframe/raw-vs-officialbbox review sheets,96UNREVIEWED slot placeholders, intake seal and index.',
        official_bbox_scope='Class/instance metadata only; not native correspondence or D/E/S/U truth.')
    write(PROTOCOL,p)
    template=dict(approved_exact12_LOCO_intake=False,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],generated=p['generated'],selection=p['selection'],**{k:False for k in FORBIDDEN})
    path=H/'ROOT_EXACT12_INTAKE_ENTRY_TEMPLATE.json';write(path,template)
    print(json.dumps(dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),selection=bind(SELECTION),generated=bind(GENERATED),template=bind(path),
        images=12,pending_slots=96,pixels_read=0,selected=[dict(take=r['take'],all_COCO_image_id=r['all_COCO_image_id'],member=r['archive_member'])for r in selection['selected']])),flush=True)


def font(n):
    from PIL import ImageFont
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)


def run(entry):
    e=read(entry);assert e['approved_exact12_LOCO_intake']is True and all(e[k]is False for k in FORBIDDEN)
    assert sha(POLICY)==POLICY_SHA and same(e['code'],bind(__file__))and same(e['protocol'],bind(PROTOCOL))
    p=read(PROTOCOL)
    for key in('code','policy','generated','selection'):assert same(e[key],p[key]);check(p[key])
    for b in p['sources'].values():check(b)
    assert read(GENERATED)['PASS']and not OUT.exists()and str(OUT)==p['output']
    assert str(ARCHIVE)==p['archive']['path']and ARCHIVE.stat().st_size==p['archive_bytes']
    selection=read(SELECTION);assert selection==metadata_selection(),'Metadata selection changed; no automatic replacement'
    OUT.mkdir();rgbdir=OUT/'RGB_ORIGINAL';rgbdir.mkdir();start=time.monotonic();receipts=[];images={}
    from PIL import Image,ImageDraw,ImageOps
    write(OUT/'INPUT_BINDINGS_BEFORE_RGB_READ.json',dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),selection=p['selection'],archive=p['archive'],
        inherited_archive_SHA_not_rehashed=True,exact_member_CRC_rawSHA_decodedSHA_will_be_verified=True,RGB_reads=0))
    with zipfile.ZipFile(ARCHIVE)as archive:
        for row in selection['selected']:
            member=row['archive_member'];zi=archive.getinfo(member);meta=row['zip_member_metadata']
            assert zi.file_size==meta['uncompressed_bytes']and zi.compress_size==meta['compressed_bytes']and f'{zi.CRC:08x}'==meta['CRC32']and zi.compress_type==meta['method']
            raw=archive.read(member);assert len(raw)==zi.file_size and f'{zlib.crc32(raw)&0xffffffff:08x}'==meta['CRC32']
            assert hashlib.sha256(raw).hexdigest()==row['expected_raw_JPEG_sha256']
            image=Image.open(io.BytesIO(raw)).convert('RGB');rgb=np.asarray(image)
            assert list(rgb.shape[:2])==row['raw_hw']and array_sha(rgb)==row['expected_decoded_RGB_sha256']
            path=rgbdir/f'{row["selection_index"]:02d}_take{row["take"][-1]}_allid{row["all_COCO_image_id"]}.jpg';path.write_bytes(raw)
            assert sha(path)==row['expected_raw_JPEG_sha256']
            images[row['selection_index']]=image
            receipts.append(dict(selection_index=row['selection_index'],source_id=row['source_id'],archive_member=member,take=row['take'],subset=2,official_split='TRAIN',
                all_COCO_image_id=row['all_COCO_image_id'],official_sub2_train_image_id=row['official_sub2_train_image_id'],raw_JPEG=bind(path),raw_hw=row['raw_hw'],
                CRC32=meta['CRC32'],CRC_verified=True,decoded_RGB_sha256=array_sha(rgb),official_pallet_instances=row['official_pallet_instances'],
                target_instance_association='UNREVIEWED',native8_mapping='UNREVIEWED',native8_reference_created=False,visibility_labels_created=False))
    sheets=[]
    for take in TAKES:
        rows=[r for r in selection['selected']if r['take']==take];assert len(rows)==2
        canvas=Image.new('RGB',(1640,1120),(247,248,250));d=ImageDraw.Draw(canvas)
        d.text((14,8),f'LOCO official TRAIN subset2 | {take} | deterministic2 original frames',font=font(25),fill=(20,25,30))
        d.text((14,44),'Left: unchanged whole frame. Right: official pallet boxes only. Native8/visibility/target association: UNREVIEWED.',font=font(20),fill=(30,35,40))
        for ri,row in enumerate(rows):
            i=row['selection_index'];rawimage=images[i];overlay=rawimage.copy();od=ImageDraw.Draw(overlay)
            for ann in row['official_pallet_instances']:
                x,y,w,h=ann['bbox'];assert np.isfinite([x,y,w,h]).all()and w>0 and h>0
                od.rectangle((x,y,x+w,y+h),outline=(20,235,55),width=max(2,round(rawimage.width/700)))
                od.text((max(0,min(x,rawimage.width-90)),max(0,min(y,rawimage.height-25))),str(ann['id']),font=font(max(13,round(rawimage.width/100))),fill=(20,235,55),stroke_width=1,stroke_fill=(0,0,0))
            top=95+ri*500
            d.text((14,top),f'#{i:02d} allCOCO {row["all_COCO_image_id"]} / sub2TRAIN {row["official_sub2_train_image_id"]} | HW {row["raw_hw"]} | pallet instances {len(row["official_pallet_instances"])}',font=font(19),fill=(20,25,30))
            for ci,im in enumerate((rawimage,overlay)):
                panel=ImageOps.contain(im,(800,450),Image.Resampling.LANCZOS);x=12+ci*820+(800-panel.width)//2; y=top+34+(450-panel.height)//2
                canvas.paste(panel,(x,y))
        path=OUT/f'WHOLEFRAME_OFFICIAL_BOXES_{take}.png';canvas.save(path)
        sheets.append(dict(take=take,sheet=bind(path),selection_indices=[r['selection_index']for r in rows]))
    placeholders=[]
    for row in receipts:
        for slot in range(8):placeholders.append(dict(selection_index=row['selection_index'],source_id=row['source_id'],official_split='TRAIN',slot_index=slot,
            review_status='UNREVIEWED',target_instance_annotation_id=None,native_correspondence=None,reference_xy=None,visibility_state=None,
            annotation_created=False,initializer_or_nearest_mapping_used=False))
    pending=OUT/'NATIVE_SLOT_REVIEW_PLACEHOLDERS96.jsonl'
    with pending.open('x')as f:
        for row in placeholders:f.write(json.dumps(row,allow_nan=False)+'\n')
    assert len(placeholders)==96
    manifest=OUT/'EXACT12_INPUT_MANIFEST.json';write(manifest,dict(completed=True,rows=receipts,selection=p['selection'],new_native_or_visibility_annotations=0))
    index=OUT/'INDEX.html';parts=['<!doctype html><meta charset="utf-8"><title>LOCO fixed12 input feasibility</title><style>body{font:17px sans-serif;margin:24px;background:#f7f8fa}img{max-width:100%}</style><h1>LOCO fixed12 input feasibility</h1><p>Two SHA-ranked previously unused images per six official TRAIN subset2 date/takes. All remain one official TRAIN environment. Official boxes are instance metadata, not native8 or visibility gold. All96 review placeholders are UNREVIEWED; no initializer/model/nearest mapping used.</p>']
    for s in sheets:parts.append('<h2>'+html.escape(s['take'])+'</h2><img src="'+html.escape(Path(s['sheet']['path']).name)+'">')
    index.write_text('\n'.join(parts)+'\n')
    seal=dict(completed=True,PASS=True,code=bind(__file__),policy=p['policy'],protocol=bind(PROTOCOL),entry=bind(entry),selection=p['selection'],archive=p['archive'],
        inherited_archive_whole_SHA_not_rehashed=True,only_selected12_members_extracted=True,all12_CRC_rawSHA_decodedSHA_HW_exact=True,images=12,official_split='TRAIN',subset=2,takes=list(TAKES),
        input_manifest=bind(manifest),native_slot_placeholders=bind(pending),unreviewed_slots=96,new_native_or_visibility_labels=0,all_original_frames_preserved=True,
        original_input_JPEGs_unchanged=True,sheets=sheets,index=bind(index),initializer_head_fit_checkpoint_private_V112_PnP_nearest_mapping_calls=0,seconds=time.monotonic()-start)
    write(OUT/'INTAKE_COMPLETE.json',seal)
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'INTAKE_COMPLETE.json'),sheets=sheets,images=12,UNREVIEWED_slots=96)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --prepare after root policy, or explicit root --entry')
