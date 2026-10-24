"""Public LOCO source RGB only: deterministic inventory, hashes, family review.

No model, source label masks, current data, training, or PnP. JPEGs are streamed
from the immutable ZIP. Only reduced thumbnails are persisted outside the ZIP.
"""
from pathlib import Path,PurePosixPath
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import hashlib
import io
import json
import re
import time
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]
V73=PARENT/'query_visibility_public_pallet_loco_20261023_v73/data/intake'
ARCHIVE=V73/'dataset.zip'
PROTOCOL=HERE/'CAPTURE_AUDIT_PROTOCOL.json'
LUT=np.array([int(i).bit_count() for i in range(256)],np.uint8)
COS=np.cos(np.pi/32*(np.arange(32)[None,:]+.5)*np.arange(32)[:,None])


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def bind(path):return dict(path=str(path),sha256=sha(path))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def jsonl(path,value):Path(path).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in value))
def csvwrite(path,value):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(value[0]));w.writeheader();w.writerows(value)


def temporal(path):
    parts=PurePosixPath(path).parts;subset=int(parts[1].split('-')[-1]);stem=PurePosixPath(path).stem
    try:number=float(stem.replace(',','.'))
    except ValueError:number=None
    epoch=None;kind='UNPARSED_FILENAME'
    if number is not None:
        if 1.4e9<=number<=2.1e9:epoch=number;kind='PLAUSIBLE_UNIX_SECONDS'
        elif 1.4e12<=number<=2.1e12:epoch=number/1000;kind='PLAUSIBLE_UNIX_MILLISECONDS'
        else:kind='SENSOR_CLOCK_UNVERIFIED_UNIT'
    capture=next((x for x in parts[2:-1] if re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d+',x)),None)
    session='LOCO:subset-'+str(subset)+':'+(capture if capture else 'NO_SESSION_METADATA')
    return dict(subset=subset,official_split='VALIDATION' if subset in (1,4) else 'TRAIN',
        path_capture_family=session,path_stream_key='/'.join(parts[:-1]),path_capture_date_take=capture,
        timestamp_filename_raw=stem,timestamp_filename_number=number,timestamp_kind=kind,
        timestamp_epoch_seconds=epoch,timestamp_UTC=datetime.datetime.fromtimestamp(epoch,datetime.timezone.utc).isoformat() if epoch else None)


def freeze():
    assert not PROTOCOL.exists()
    annpaths=sorted((V73/'annotations').glob('*.json'));assert len(annpaths)==6
    bindings=[bind(ARCHIVE),bind(V73/'INTAKE_SEAL.json'),bind(V73/'ALL_5593_RGB_FILE_SHA256.csv')]+[bind(p) for p in annpaths]
    code=bind(__file__)
    with zipfile.ZipFile(ARCHIVE) as z:
        names=sorted(i.filename for i in z.infolist() if not i.is_dir())
    assert len(names)==5593 and all(PurePosixPath(x).suffix.lower() in ('.jpg','.jpeg') for x in names)
    meta=[dict(index=i,archive_member=name,source_id='LOCO:'+name,**temporal(name)) for i,name in enumerate(names)]
    jsonl(HERE/'TEMPORAL_INPUT_METADATA.jsonl',meta)
    write(PROTOCOL,dict(schema='V74_LOCO_SOURCE_CAPTURE_HASH_AUDIT_V1',frozen_before_RGB_decode=True,code=code,bindings=bindings,metadata=bind(HERE/'TEMPORAL_INPUT_METADATA.jsonl'),
        scope='Input-only public-source capture evidence; not a semantic or learned-model evaluation.',
        RGB_decode='PIL decode original JPEG to RGB without EXIF transpose; SHA256 decoded uint8 RGB bytes plus record nativeHW. Stream ZIP, retain no full JPEG filesystem copy.',
        reduced_thumbnail='128x80 RGB via PIL LANCZOS, aspect-ratio warped only for hash review; native geometry untouched.',
        perceptual_hash='pHash64:32x32 LANCZOS gray, fixed unnormalised cosine DCT, top8x8 coefficients >median excludingDC; dHash64:9x8 gray, right>left. Hex big-endian bit ordering.',
        candidate_rule='Compare all source pairs. Strongnear flag pHashHamming<=8 AND dHashHamming<=10 AND RGBthumbnailMAE/255<=.08. These are fixed review flags, not verified duplicate or independence thresholds.',
        gallery_selection='Top2 by (2*pHashHamming+dHashHamming,pHashHamming,dHashHamming,indexA,indexB) for each of10different-subset pairs, constrained distinct pair; additionally report all cross-split strongnear components. Never filter by GT/model outcomes.',
        session_rule='Whole path date_take across all cameras grouped together within subset. Missing session metadata=>whole subset is one unresolved capture family. Timestamps from filename remain sensor-clock UNKNOWN unless plausible Unix seconds/ms.',
        future_split_rule='No random image split. Preserve exact/strongnear connected components and whole path-session groups, report any official TRAIN/VAL collision. These rules do not prove absence of same-site revisits.',
        semantic_labels_queries_current_inputs_used=False,decode_workers=4,new_model_forward_training_PnP=0))
    print(json.dumps(dict(frozen=True,protocol=bind(PROTOCOL))),flush=True)


def bits64(binary):return int.from_bytes(np.packbits(np.asarray(binary).ravel()).tobytes(),'big')


def decode_one(args):
    z,rec,expected=args;raw=z.read(rec['archive_member']);assert hashlib.sha256(raw).hexdigest()==expected
    with Image.open(io.BytesIO(raw)) as im:
        rgb=im.convert('RGB');arr=np.asarray(rgb);h,w=arr.shape[:2]
        decoded=hashlib.sha256(arr.tobytes()).hexdigest()
        thumbnail=np.asarray(rgb.resize((128,80),Image.Resampling.LANCZOS)).copy()
        small=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),np.float64)
        dh=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    co=(COS@small@COS.T)[:8,:8];phash=bits64(co>np.median(co.ravel()[1:]));dhash=bits64(dh[:,1:]>dh[:,:-1])
    return dict(rec,jpeg_file_SHA256=expected,decoded_RGB_SHA256=decoded,raw_hw=[h,w],pHash64=f'{phash:016x}',dHash64=f'{dhash:016x}',
        grayscale_std=float(small.std()),thumbnail_index=rec['index']),thumbnail


def ham(a,b):
    xor=np.bitwise_xor(a[:,None],b[None,:])
    return LUT[xor.view(np.uint8).reshape(*xor.shape,8)].sum(2).astype(np.uint8)


class DSU:
    def __init__(self,n):self.p=list(range(n))
    def root(self,a):
        while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.p[max(a,b)]=min(a,b)


def gallery(name,pairs,records,thumb):
    f=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    im=Image.new('RGB',(1260,100+len(pairs)*260),(248,249,251));d=ImageDraw.Draw(im)
    d.text((20,15),name,font=f(25),fill=(20,30,45));d.text((20,52),'Fixed hash-ranked candidates; resemblance alone does not establish capture independence or identity.',font=f(17),fill=(60,65,80))
    for n,p in enumerate(pairs):
        y=95+n*260
        for col,k in enumerate([p['indexA'],p['indexB']]):
            x=20+col*620;r=records[k]
            tile=Image.fromarray(thumb[k]).resize((320,200),Image.Resampling.NEAREST);im.paste(tile,(x,y))
            d.text((x+330,y),f"subset{r['subset']} {r['official_split']}\n{r['raw_hw']}\n{r['timestamp_UTC'] or r['timestamp_kind']}\nindex{k}",font=f(16),fill=(20,35,50))
            d.text((x,y+205),r['archive_member'][-77:],font=f(13),fill=(35,45,60))
        d.text((20,y+226),f"pHash={p['pHashHamming']} dHash={p['dHashHamming']} RGB_MAE={p['thumbnail_MAE']:.4f} strongnear={p['strongnear']}",font=f(17),fill=(25,45,70))
    path=HERE/(name+'.png');im.save(path);return bind(path)


def run():
    assert PROTOCOL.exists() and not (HERE/'CAPTURE_AUDIT_RESULT.json').exists()
    prot=json.loads(PROTOCOL.read_text());assert prot['code']==bind(__file__)
    for b in prot['bindings']+[prot['metadata']]:assert sha(b['path'])==b['sha256']
    meta=rows(HERE/'TEMPORAL_INPUT_METADATA.jsonl');start=time.monotonic()
    with (V73/'ALL_5593_RGB_FILE_SHA256.csv').open() as f:expected={r['archive_member']:r['sha256'] for r in csv.DictReader(f)}
    records=[];thumbnail=np.zeros((5593,80,128,3),np.uint8)
    with zipfile.ZipFile(ARCHIVE) as z,ThreadPoolExecutor(max_workers=4) as pool:
        for i,(rec,thumb) in enumerate(pool.map(decode_one,((z,r,expected[r['archive_member']]) for r in meta))):
            records.append(rec);thumbnail[i]=thumb
            if (i+1)%500==0:print(json.dumps(dict(decoded=i+1,seconds=time.monotonic()-start)),flush=True)
    jsonl(HERE/'RGB_TEMPORAL_HASH_MANIFEST.jsonl',records);np.savez_compressed(HERE/'THUMBNAILS_INPUT_ONLY.npz',rgb128x80=thumbnail)
    allc=json.loads((V73/'annotations/loco-all-v1.json').read_text());co={r['path'].lstrip('/'):r for r in allc['images']}
    assert all(r['raw_hw']==[co[r['archive_member']]['height'],co[r['archive_member']]['width']] for r in records if r['archive_member'] in co)
    for p in sorted((V73/'annotations').glob('loco-sub*.json')):
        doc=json.loads(p.read_text());official='VALIDATION' if '-val' in p.name else 'TRAIN'
        assert all(records[next(i for i,r in enumerate(records) if r['archive_member']==im['path'].lstrip('/'))]['official_split']==official for im in doc['images'])
    ph=np.array([int(r['pHash64'],16) for r in records],np.uint64);dh=np.array([int(r['dHash64'],16) for r in records],np.uint64)
    subsets=np.array([r['subset'] for r in records]);strong=[];candidates=[]
    for s1 in range(1,6):
        ia=np.flatnonzero(subsets==s1)
        for s2 in range(s1+1,6):
            ib=np.flatnonzero(subsets==s2);best=[]
            for at in range(0,len(ia),64):
                ix=ia[at:at+64];pa=ham(ph[ix],ph[ib]);da=ham(dh[ix],dh[ib]);rank=pa.astype(np.int16)*2+da
                # Keep bounded local top2; global sorting later is exact top2.
                order=np.argsort(rank.ravel(),kind='stable')[:2]
                positions=set((int(j//len(ib)),int(j%len(ib))) for j in order)
                positions.update(map(tuple,np.argwhere((pa<=8)&(da<=10))))
                for a,b in sorted(positions):
                    i,j=int(ix[a]),int(ib[b]);mae=float(np.abs(thumbnail[i].astype(np.int16)-thumbnail[j].astype(np.int16)).mean()/255)
                    row=dict(indexA=i,indexB=j,subsetA=s1,subsetB=s2,splitA=records[i]['official_split'],splitB=records[j]['official_split'],pHashHamming=int(pa[a,b]),dHashHamming=int(da[a,b]),thumbnail_MAE=mae,
                        strongnear=bool(pa[a,b]<=8 and da[a,b]<=10 and mae<=.08))
                    best.append(row)
                    if row['strongnear']:strong.append(row)
            best.sort(key=lambda r:(2*r['pHashHamming']+r['dHashHamming'],r['pHashHamming'],r['dHashHamming'],r['indexA'],r['indexB']))
            candidates.extend(best[:2])
    candidates.sort(key=lambda r:(r['subsetA'],r['subsetB'],2*r['pHashHamming']+r['dHashHamming'],r['indexA'],r['indexB']))
    jsonl(HERE/'FIXED_NEAR_CANDIDATES.jsonl',candidates)
    jsonl(HERE/'CROSS_SUBSET_STRONG_NEAR_PAIRS.jsonl',strong)
    dsu=DSU(len(records));exact=defaultdict(list)
    for i,r in enumerate(records):exact[(tuple(r['raw_hw']),r['decoded_RGB_SHA256'])].append(i)
    exact_duplicates=[v for v in exact.values() if len(v)>1]
    for group in exact_duplicates:
        for j in group[1:]:dsu.join(group[0],j)
    for row in strong:dsu.join(row['indexA'],row['indexB'])
    sessions=defaultdict(list)
    for i,r in enumerate(records):sessions[r['path_capture_family']].append(i)
    for group in sessions.values():
        for j in group[1:]:dsu.join(group[0],j)
    components=defaultdict(list)
    for i in range(len(records)):components[dsu.root(i)].append(i)
    component_rows=[]
    for key,ids in sorted(components.items()):
        splits=sorted({records[i]['official_split'] for i in ids});ss=sorted({records[i]['subset'] for i in ids})
        component_rows.append(dict(group_id=f'LOCO_INPUT_EVIDENCE_GROUP:{key:06d}',n=len(ids),subsets=ss,official_splits=splits,cross_official_split=len(splits)>1,members=ids))
    mapping={i:g['group_id'] for g in component_rows for i in g['members']}
    groupmanifest=[dict(r,evidence_group=mapping[i],official_annotation_present=r['archive_member'] in co,proposed_group_scope='pathsession+exactdecoded+fixedstrongnear; same-site revisit not ruled out') for i,r in enumerate(records)]
    jsonl(HERE/'INPUT_ONLY_SOURCE_GROUP_MANIFEST.jsonl',groupmanifest);write(HERE/'EVIDENCE_GROUP_COMPONENTS.json',dict(groups=component_rows))
    cross=[r for r in candidates if r['splitA']!=r['splitB']]
    figures=[]
    for k in range(0,len(cross),6):figures.append(gallery(f'CROSS_SPLIT_FIXED_CANDIDATES_{k//6+1}',cross[k:k+6],records,thumbnail))
    figures.append(gallery('SUBSET_1_5_FIXED_CANDIDATES',[r for r in candidates if (r['subsetA'],r['subsetB'])==(1,5)],records,thumbnail))
    session_table=[]
    for name,ids in sorted(sessions.items()):
        ts=[records[i]['timestamp_epoch_seconds'] for i in ids if records[i]['timestamp_epoch_seconds'] is not None]
        session_table.append(dict(path_capture_family=name,images=len(ids),subsets=sorted({records[i]['subset'] for i in ids}),official_splits=sorted({records[i]['official_split'] for i in ids}),stream_count=len({records[i]['path_stream_key'] for i in ids}),plausible_UNIX_n=len(ts),min_epoch=min(ts) if ts else None,max_epoch=max(ts) if ts else None))
    write(HERE/'PATH_CAPTURE_SESSION_TABLE.json',dict(sessions=session_table))
    result=dict(schema='V74_LOCO_ACTUAL_SOURCE_CAPTURE_AUDIT_V1',completed=True,protocol=bind(PROTOCOL),decoded_source_images=len(records),full_JPEG_filesystem_extraction=0,
        official_split=dict(TRAIN_subsets=[2,3,5],VALIDATION_subsets=[1,4]),native_resolutions=dict(Counter(str(r['raw_hw']) for r in records)),timestamp_kind=dict(Counter(r['timestamp_kind'] for r in records)),
        exact_decoded_RGB_duplicate_groups=exact_duplicates,cross_subset_strongnear_pairs=len(strong),cross_official_split_strongnear_pairs=sum(r['splitA']!=r['splitB'] for r in strong),
        path_capture_families=len(sessions),evidence_groups=len(component_rows),evidence_groups_crossing_official_split=[g for g in component_rows if g['cross_official_split']],
        fixed_candidate_pairs=len(candidates),figures=figures,visual_review='PENDING_ADDITIVE_ACTUAL_REVIEW',
        limitations=['No hash threshold establishes physical-site independence.','Different dates/sensors may revisit the same warehouse. Generic rack/pallet resemblance cannot certify same capture.','Official dataset_id/image_id are re-export linkage IDs, not physical capture-family labels.','Preserve whole path-session and cross-split evidence components before any future split; unresolved physical-site relations require conservative handling.'],
        outputs=[bind(HERE/x) for x in ['TEMPORAL_INPUT_METADATA.jsonl','RGB_TEMPORAL_HASH_MANIFEST.jsonl','THUMBNAILS_INPUT_ONLY.npz','FIXED_NEAR_CANDIDATES.jsonl','CROSS_SUBSET_STRONG_NEAR_PAIRS.jsonl','INPUT_ONLY_SOURCE_GROUP_MANIFEST.jsonl','EVIDENCE_GROUP_COMPONENTS.json','PATH_CAPTURE_SESSION_TABLE.json']],
        current_data_model_forward_training_PnP=0,seconds=time.monotonic()-start)
    write(HERE/'CAPTURE_AUDIT_RESULT.json',result)
    print(json.dumps(dict(completed=True,result=bind(HERE/'CAPTURE_AUDIT_RESULT.json'),strong=len(strong),crosssplit=result['cross_official_split_strongnear_pairs'],seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);args=p.parse_args()
    if args.stage=='freeze':freeze()
    else:run()
